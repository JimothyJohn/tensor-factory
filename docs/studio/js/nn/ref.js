// The detector's training math on the CPU, in plain JavaScript: forward, backward, Adam.
//
// This is the reference the WebGPU trainer is checked against, and what trains when a
// browser has no WebGPU. It is the same network as `tensor_factory_train.model.TinyDetector`
// with a presence head: conv3x3 + batch norm + ReLU blocks (four at stride 2, then
// `context` at stride 1), a 1x1 conv to four edge heatmaps, a marginal soft-argmax with a
// learnable gain, and one objectness logit from pooled mean + max features. Parameter
// names are torch's `state_dict` names, so weights move between the two unchanged.
// `tests/nn_ref.test.mjs` pins outputs, loss and every gradient to values torch computed.
//
// Tensors are `{data: Float64Array, shape: number[]}`, NCHW.

export const BN_EPS = 1e-5;
export const BN_MOMENTUM = 0.1;

export function tensor(shape, data) {
  const n = shape.reduce((a, b) => a * b, 1);
  const out = data ? Float64Array.from(data) : new Float64Array(n);
  if (out.length !== n) throw new Error(`tensor: ${out.length} values for shape [${shape}]`);
  return { data: out, shape: shape.slice() };
}

/** Channel plan: [3, c, 2c, 4c, 4c, (4c)*context], and each block's stride. */
export function plan(width, context) {
  if (!Number.isInteger(width) || width < 1) throw new Error(`width must be >= 1, got ${width}`);
  if (!Number.isInteger(context) || context < 0) {
    throw new Error(`context must be >= 0, got ${context}`);
  }
  const chans = [3, width, 2 * width, 4 * width, 4 * width];
  for (let i = 0; i < context; i++) chans.push(4 * width);
  const blocks = [];
  for (let i = 0; i + 1 < chans.length; i++) {
    blocks.push({ cin: chans[i], cout: chans[i + 1], stride: i < 4 ? 2 : 1 });
  }
  return blocks;
}

// mulberry32: small, seedable, good enough for weight init and shuffling.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fresh parameters, initialized the way torch initializes Conv2d / Linear / BatchNorm2d. */
export function initParams({ width = 16, context = 0, seed = 0 } = {}) {
  const rand = rng(seed);
  const uniform = (shape, bound) => {
    const t = tensor(shape);
    for (let i = 0; i < t.data.length; i++) t.data[i] = (rand() * 2 - 1) * bound;
    return t;
  };
  const p = {};
  plan(width, context).forEach((b, i) => {
    const bound = 1 / Math.sqrt(b.cin * 9);
    p[`features.${i}.0.weight`] = uniform([b.cout, b.cin, 3, 3], bound);
    p[`features.${i}.0.bias`] = uniform([b.cout], bound);
    p[`features.${i}.1.weight`] = tensor([b.cout], new Float64Array(b.cout).fill(1));
    p[`features.${i}.1.bias`] = tensor([b.cout]);
    p[`features.${i}.1.running_mean`] = tensor([b.cout]);
    p[`features.${i}.1.running_var`] = tensor([b.cout], new Float64Array(b.cout).fill(1));
  });
  const c4 = 4 * width;
  p["heat.weight"] = uniform([4, c4, 1, 1], 1 / Math.sqrt(c4));
  p["heat.bias"] = uniform([4], 1 / Math.sqrt(c4));
  p["log_gain"] = tensor([1]);
  p["presence_head.weight"] = uniform([1, 2 * c4], 1 / Math.sqrt(2 * c4));
  p["presence_head.bias"] = uniform([1], 1 / Math.sqrt(2 * c4));
  return p;
}

/** Names the optimizer updates: everything except batch norm's running statistics. */
export function trainable(params) {
  return Object.keys(params).filter((k) => !k.includes("running_"));
}

export function configOf(params) {
  let blocks = 0;
  while (params[`features.${blocks}.0.weight`]) blocks++;
  if (blocks < 4) throw new Error("not a detector: fewer than four feature blocks");
  return { width: params["features.0.0.weight"].shape[0], context: blocks - 4 };
}

function outSize(n, k, stride, pad) {
  return Math.floor((n + 2 * pad - k) / stride) + 1;
}

export function conv2d(x, w, b, stride, pad) {
  const [n, cin, h, wd] = x.shape;
  const [cout, , k] = w.shape;
  const ho = outSize(h, k, stride, pad);
  const wo = outSize(wd, k, stride, pad);
  const y = tensor([n, cout, ho, wo]);
  for (let i = 0; i < n; i++) {
    for (let co = 0; co < cout; co++) {
      const yBase = (i * cout + co) * ho * wo;
      for (let oy = 0; oy < ho; oy++) {
        for (let ox = 0; ox < wo; ox++) {
          let acc = b.data[co];
          for (let ci = 0; ci < cin; ci++) {
            const xBase = (i * cin + ci) * h * wd;
            const wBase = (co * cin + ci) * k * k;
            for (let ky = 0; ky < k; ky++) {
              const iy = oy * stride - pad + ky;
              if (iy < 0 || iy >= h) continue;
              for (let kx = 0; kx < k; kx++) {
                const ix = ox * stride - pad + kx;
                if (ix < 0 || ix >= wd) continue;
                acc += x.data[xBase + iy * wd + ix] * w.data[wBase + ky * k + kx];
              }
            }
          }
          y.data[yBase + oy * wo + ox] = acc;
        }
      }
    }
  }
  return y;
}

/** Gradients of a conv: returns {dx, dw, db} for upstream `dy`. */
export function conv2dBackward(x, w, dy, stride, pad) {
  const [n, cin, h, wd] = x.shape;
  const [cout, , k] = w.shape;
  const [, , ho, wo] = dy.shape;
  const dx = tensor(x.shape);
  const dw = tensor(w.shape);
  const db = tensor([cout]);
  for (let i = 0; i < n; i++) {
    for (let co = 0; co < cout; co++) {
      const yBase = (i * cout + co) * ho * wo;
      for (let oy = 0; oy < ho; oy++) {
        for (let ox = 0; ox < wo; ox++) {
          const g = dy.data[yBase + oy * wo + ox];
          if (g === 0) continue;
          db.data[co] += g;
          for (let ci = 0; ci < cin; ci++) {
            const xBase = (i * cin + ci) * h * wd;
            const wBase = (co * cin + ci) * k * k;
            for (let ky = 0; ky < k; ky++) {
              const iy = oy * stride - pad + ky;
              if (iy < 0 || iy >= h) continue;
              for (let kx = 0; kx < k; kx++) {
                const ix = ox * stride - pad + kx;
                if (ix < 0 || ix >= wd) continue;
                dw.data[wBase + ky * k + kx] += g * x.data[xBase + iy * wd + ix];
                dx.data[xBase + iy * wd + ix] += g * w.data[wBase + ky * k + kx];
              }
            }
          }
        }
      }
    }
  }
  return { dx, dw, db };
}

function forEachChannel(shape, fn) {
  const [n, c, h, w] = shape;
  const hw = h * w;
  for (let ch = 0; ch < c; ch++) {
    for (let i = 0; i < n; i++) fn(ch, (i * c + ch) * hw, hw);
  }
}

/**
 * Forward pass. `train: true` uses batch statistics in the batch norms and returns the
 * updated running statistics in `running`; `train: false` uses the stored ones.
 * Returns {box [B,4], presence [B,1], cache} where `cache` is what `backward` needs.
 */
export function forward(params, x, { train = false } = {}) {
  const { width, context } = configOf(params);
  const blocks = plan(width, context);
  const layers = [];
  const running = {};
  let f = x;
  blocks.forEach((blk, i) => {
    const w = params[`features.${i}.0.weight`];
    const conv = conv2d(f, w, params[`features.${i}.0.bias`], blk.stride, 1);
    const [n, c, h, wd] = conv.shape;
    const count = n * h * wd;
    const gamma = params[`features.${i}.1.weight`].data;
    const beta = params[`features.${i}.1.bias`].data;
    const mean = new Float64Array(c);
    const variance = new Float64Array(c);
    if (train) {
      forEachChannel(conv.shape, (ch, base, hw) => {
        for (let j = 0; j < hw; j++) mean[ch] += conv.data[base + j];
      });
      for (let ch = 0; ch < c; ch++) mean[ch] /= count;
      forEachChannel(conv.shape, (ch, base, hw) => {
        for (let j = 0; j < hw; j++) {
          const d = conv.data[base + j] - mean[ch];
          variance[ch] += d * d;
        }
      });
      for (let ch = 0; ch < c; ch++) variance[ch] /= count;
      // torch keeps the *unbiased* variance in running_var
      const unbias = count > 1 ? count / (count - 1) : 1;
      const rm = params[`features.${i}.1.running_mean`].data;
      const rv = params[`features.${i}.1.running_var`].data;
      running[`features.${i}.1.running_mean`] = tensor(
        [c],
        rm.map((v, ch) => (1 - BN_MOMENTUM) * v + BN_MOMENTUM * mean[ch]),
      );
      running[`features.${i}.1.running_var`] = tensor(
        [c],
        rv.map((v, ch) => (1 - BN_MOMENTUM) * v + BN_MOMENTUM * variance[ch] * unbias),
      );
    } else {
      mean.set(params[`features.${i}.1.running_mean`].data);
      variance.set(params[`features.${i}.1.running_var`].data);
    }
    const xhat = tensor(conv.shape);
    const out = tensor(conv.shape);
    forEachChannel(conv.shape, (ch, base, hw) => {
      const inv = 1 / Math.sqrt(variance[ch] + BN_EPS);
      for (let j = 0; j < hw; j++) {
        const z = (conv.data[base + j] - mean[ch]) * inv;
        xhat.data[base + j] = z;
        const v = gamma[ch] * z + beta[ch];
        out.data[base + j] = v > 0 ? v : 0;
      }
    });
    layers.push({ input: f, xhat, out, variance, stride: blk.stride });
    f = out;
  });

  const heat = conv2d(f, params["heat.weight"], params["heat.bias"], 1, 0);
  const gain = Math.exp(params["log_gain"].data[0]);
  const [n, c, h, wd] = f.shape;
  const hw = h * wd;
  const prob = tensor(heat.shape);
  const box = tensor([n, 4]);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 4; k++) {
      const base = (i * 4 + k) * hw;
      let peak = -Infinity;
      for (let j = 0; j < hw; j++) peak = Math.max(peak, heat.data[base + j] * gain);
      let total = 0;
      for (let j = 0; j < hw; j++) {
        const e = Math.exp(heat.data[base + j] * gain - peak);
        prob.data[base + j] = e;
        total += e;
      }
      let pos = 0;
      for (let j = 0; j < hw; j++) {
        prob.data[base + j] /= total;
        pos += prob.data[base + j] * coord(k, j, h, wd);
      }
      box.data[i * 4 + k] = pos;
    }
  }

  // pooled = [mean over HW, max over HW] per channel -> one objectness logit
  const pooled = tensor([n, 2 * c]);
  const pw = params["presence_head.weight"].data;
  const presence = tensor([n, 1]);
  for (let i = 0; i < n; i++) {
    let logit = params["presence_head.bias"].data[0];
    for (let ch = 0; ch < c; ch++) {
      const base = (i * c + ch) * hw;
      let sum = 0;
      let max = -Infinity;
      for (let j = 0; j < hw; j++) {
        sum += f.data[base + j];
        if (f.data[base + j] > max) max = f.data[base + j];
      }
      pooled.data[i * 2 * c + ch] = sum / hw;
      pooled.data[i * 2 * c + c + ch] = max;
      logit += pw[ch] * (sum / hw) + pw[c + ch] * max;
    }
    presence.data[i] = logit;
  }
  return { box, presence, running, cache: { layers, features: f, heat, prob, pooled, gain } };
}

// Position of heatmap cell j along the axis edge k is read from: x for x1/x2, y for y1/y2.
function coord(k, j, h, w) {
  if (k % 2 === 0) return w > 1 ? (j % w) / (w - 1) : 0;
  return h > 1 ? Math.floor(j / w) / (h - 1) : 0;
}

/**
 * The training loss and its gradient with respect to the model's outputs.
 * `boxes` is [B,4] targets, `hasBox` is B flags: 1 = the object is in this frame (it has a
 * box and its objectness target is 1), 0 = a negative (no box loss, objectness target 0).
 * Smooth L1 (beta 1) over the boxes of positives + binary cross-entropy on the logit.
 */
export function loss(out, boxes, hasBox, { boxWeight = 1, presenceWeight = 1 } = {}) {
  const n = hasBox.length;
  const dBox = tensor([n, 4]);
  const dPresence = tensor([n, 1]);
  let positives = 0;
  for (let i = 0; i < n; i++) positives += hasBox[i] ? 1 : 0;
  let boxLoss = 0;
  for (let i = 0; i < n; i++) {
    if (!hasBox[i]) continue;
    for (let k = 0; k < 4; k++) {
      const d = out.box.data[i * 4 + k] - boxes.data[i * 4 + k];
      const a = Math.abs(d);
      boxLoss += a < 1 ? 0.5 * d * d : a - 0.5;
      dBox.data[i * 4 + k] = ((a < 1 ? d : Math.sign(d)) * boxWeight) / (positives * 4);
    }
  }
  if (positives) boxLoss /= positives * 4;
  let bce = 0;
  for (let i = 0; i < n; i++) {
    const z = out.presence.data[i];
    const t = hasBox[i] ? 1 : 0;
    bce += Math.max(z, 0) - z * t + Math.log1p(Math.exp(-Math.abs(z)));
    dPresence.data[i] = ((1 / (1 + Math.exp(-z)) - t) * presenceWeight) / n;
  }
  bce /= n;
  return { value: boxWeight * boxLoss + presenceWeight * bce, boxLoss, bce, dBox, dPresence };
}

/** Gradients of every trainable parameter, given the loss gradients at the outputs. */
export function backward(params, out, dBox, dPresence) {
  const { layers, features: f, heat, prob, gain } = out.cache;
  const [n, c, h, wd] = f.shape;
  const hw = h * wd;
  const grads = {};
  const df = tensor(f.shape);

  // presence head -> pooled mean and max
  const pw = params["presence_head.weight"].data;
  const dpw = tensor([1, 2 * c]);
  const dpb = tensor([1]);
  for (let i = 0; i < n; i++) {
    const g = dPresence.data[i];
    dpb.data[0] += g;
    for (let ch = 0; ch < c; ch++) {
      dpw.data[ch] += g * out.cache.pooled.data[i * 2 * c + ch];
      dpw.data[c + ch] += g * out.cache.pooled.data[i * 2 * c + c + ch];
      const base = (i * c + ch) * hw;
      const max = out.cache.pooled.data[i * 2 * c + c + ch];
      let ties = 0;
      for (let j = 0; j < hw; j++) if (f.data[base + j] === max) ties++;
      // torch's amax shares the gradient equally between tied maxima
      const gMax = (g * pw[c + ch]) / ties;
      const gMean = (g * pw[ch]) / hw;
      for (let j = 0; j < hw; j++) {
        df.data[base + j] += gMean + (f.data[base + j] === max ? gMax : 0);
      }
    }
  }
  grads["presence_head.weight"] = dpw;
  grads["presence_head.bias"] = dpb;

  // soft-argmax -> heatmap and gain
  const dHeat = tensor(heat.shape);
  let dGain = 0;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 4; k++) {
      const g = dBox.data[i * 4 + k];
      if (g === 0) continue;
      const base = (i * 4 + k) * hw;
      const pos = out.box.data[i * 4 + k];
      for (let j = 0; j < hw; j++) {
        const dz = g * prob.data[base + j] * (coord(k, j, h, wd) - pos);
        dHeat.data[base + j] = dz * gain;
        dGain += dz * heat.data[base + j];
      }
    }
  }
  grads["log_gain"] = tensor([1], [dGain * gain]);

  const heatBack = conv2dBackward(f, params["heat.weight"], dHeat, 1, 0);
  grads["heat.weight"] = heatBack.dw;
  grads["heat.bias"] = heatBack.db;
  for (let j = 0; j < df.data.length; j++) df.data[j] += heatBack.dx.data[j];

  // feature blocks, last to first: ReLU -> batch norm -> conv
  let dOut = df;
  for (let i = layers.length - 1; i >= 0; i--) {
    const L = layers[i];
    const [bn, bc, bh, bw] = L.out.shape;
    const count = bn * bh * bw;
    const gamma = params[`features.${i}.1.weight`].data;
    const dGamma = tensor([bc]);
    const dBeta = tensor([bc]);
    const dAct = tensor(L.out.shape);
    forEachChannel(L.out.shape, (ch, base, n2) => {
      for (let j = 0; j < n2; j++) {
        const g = L.out.data[base + j] > 0 ? dOut.data[base + j] : 0;
        dAct.data[base + j] = g;
        dBeta.data[ch] += g;
        dGamma.data[ch] += g * L.xhat.data[base + j];
      }
    });
    const dConv = tensor(L.out.shape);
    forEachChannel(L.out.shape, (ch, base, n2) => {
      const scale = gamma[ch] / Math.sqrt(L.variance[ch] + BN_EPS) / count;
      for (let j = 0; j < n2; j++) {
        dConv.data[base + j] =
          scale *
          (count * dAct.data[base + j] - dBeta.data[ch] - L.xhat.data[base + j] * dGamma.data[ch]);
      }
    });
    grads[`features.${i}.1.weight`] = dGamma;
    grads[`features.${i}.1.bias`] = dBeta;
    const back = conv2dBackward(L.input, params[`features.${i}.0.weight`], dConv, L.stride, 1);
    grads[`features.${i}.0.weight`] = back.dw;
    grads[`features.${i}.0.bias`] = back.db;
    dOut = back.dx;
  }
  return grads;
}

/** Adam with torch's defaults. `state` is created on first use and carried between steps. */
export function adamStep(params, grads, state, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
  state.t = (state.t || 0) + 1;
  state.m ||= {};
  state.v ||= {};
  const c1 = 1 - beta1 ** state.t;
  const c2 = 1 - beta2 ** state.t;
  for (const name of Object.keys(grads)) {
    const p = params[name].data;
    const g = grads[name].data;
    const m = (state.m[name] ||= new Float64Array(p.length));
    const v = (state.v[name] ||= new Float64Array(p.length));
    for (let i = 0; i < p.length; i++) {
      m[i] = beta1 * m[i] + (1 - beta1) * g[i];
      v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
      p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
    }
  }
  return state;
}

/** One optimizer step on a batch; mutates `params` and `state`. Returns the loss parts. */
export function trainStep(params, state, x, boxes, hasBox, opts = {}) {
  const out = forward(params, x, { train: true });
  const l = loss(out, boxes, hasBox, opts);
  const grads = backward(params, out, l.dBox, l.dPresence);
  adamStep(params, grads, state, opts);
  for (const [name, value] of Object.entries(out.running)) params[name] = value;
  return { loss: l.value, boxLoss: l.boxLoss, bce: l.bce };
}
