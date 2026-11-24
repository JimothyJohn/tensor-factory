// The detector's training step on WebGPU: the same math as ref.js, as compute shaders.
//
// No library: each operation is a small WGSL kernel written "gather" style (one invocation
// per output element, looping over what it needs), so there are no atomics and no
// scatter. A training step is one command buffer: forward with batch statistics, loss
// gradients, backward, Adam. Nothing is read back during a step; the per-sample loss is
// copied into a history buffer and read once per epoch.
//
// Checked against ref.js (which is pinned to torch) by tests/gpu/run.py in a real browser.
// Parameters keep torch's names; batch norm running stats live in one buffer per block,
// laid out [mean x C][var x C].

import { BN_EPS, BN_MOMENTUM, configOf, plan as blockPlan } from "./ref.js";

const HEADER = `
struct U { a: vec4u, b: vec4u, c: vec4u, f: vec4f };
@group(0) @binding(0) var<uniform> u: U;
`;
// u.c.z = number of elements this kernel computes, u.c.w = invocations per dispatch row.
const MAIN = `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.y * u.c.w + gid.x;
  if (idx >= u.c.z) { return; }
`;

function kernel(bindings, body) {
  const decl = bindings
    .map(([name, access], i) => {
      const mode = access === "rw" ? "read_write" : "read";
      return `@group(0) @binding(${i + 1}) var<storage, ${mode}> ${name}: array<f32>;`;
    })
    .join("\n");
  return { bindings, code: `${HEADER}${decl}\n${MAIN}${body}\n}` };
}

// a = (N, Cin, H, W), b = (Cout, Ho, Wo, stride), c.x = pad, c.y = K
const KERNELS = {
  conv: kernel([["x", "r"], ["w", "r"], ["bias", "r"], ["y", "rw"]], `
  let cin = u.a.y; let h = i32(u.a.z); let wd = i32(u.a.w);
  let cout = u.b.x; let ho = u.b.y; let wo = u.b.z; let k = u.c.y;
  let ox = idx % wo; let oy = (idx / wo) % ho; let co = (idx / (wo * ho)) % cout;
  let n = idx / (wo * ho * cout);
  var acc = bias[co];
  for (var ci = 0u; ci < cin; ci++) {
    let xb = (n * cin + ci) * u32(h * wd);
    let wb = (co * cin + ci) * k * k;
    for (var ky = 0u; ky < k; ky++) {
      let iy = i32(oy * u.b.w + ky) - i32(u.c.x);
      if (iy < 0 || iy >= h) { continue; }
      for (var kx = 0u; kx < k; kx++) {
        let ix = i32(ox * u.b.w + kx) - i32(u.c.x);
        if (ix < 0 || ix >= wd) { continue; }
        acc += x[xb + u32(iy * wd + ix)] * w[wb + ky * k + kx];
      }
    }
  }
  y[idx] = acc;`),

  // a = (N, C, H, W); f.x = momentum, f.y = unbiased-variance factor. One invocation per channel.
  bnStats: kernel([["conv", "r"], ["stats", "rw"], ["running", "rw"]], `
  let n = u.a.x; let c = u.a.y; let h = u.a.z; let wd = u.a.w;
  let count = f32(n * h * wd);
  var total = 0.0;
  for (var i = 0u; i < n; i++) {
    var plane = 0.0;
    for (var yy = 0u; yy < h; yy++) {
      var row = 0.0;
      let base = ((i * c + idx) * h + yy) * wd;
      for (var xx = 0u; xx < wd; xx++) { row += conv[base + xx]; }
      plane += row;
    }
    total += plane;
  }
  let mean = total / count;
  var sq = 0.0;
  for (var i = 0u; i < n; i++) {
    var plane = 0.0;
    for (var yy = 0u; yy < h; yy++) {
      var row = 0.0;
      let base = ((i * c + idx) * h + yy) * wd;
      for (var xx = 0u; xx < wd; xx++) { let d = conv[base + xx] - mean; row += d * d; }
      plane += row;
    }
    sq += plane;
  }
  let variance = sq / count;
  stats[idx] = mean;
  stats[c + idx] = variance;
  running[idx] = (1.0 - u.f.x) * running[idx] + u.f.x * mean;
  running[c + idx] = (1.0 - u.f.x) * running[c + idx] + u.f.x * variance * u.f.y;`),

  // a = (N, C, H, W); f.x = eps. stats is the batch's (training) or the running ones (eval).
  bnRelu: kernel(
    [["conv", "r"], ["stats", "r"], ["gamma", "r"], ["beta", "r"], ["xhat", "rw"], ["act", "rw"]],
    `
  let c = u.a.y;
  let ch = (idx / (u.a.z * u.a.w)) % c;
  let z = (conv[idx] - stats[ch]) * inverseSqrt(stats[c + ch] + u.f.x);
  xhat[idx] = z;
  act[idx] = max(gamma[ch] * z + beta[ch], 0.0);`,
  ),

  // a = (N, 4, H, W). One invocation per (sample, edge).
  softArgmax: kernel([["heat", "r"], ["logGain", "r"], ["prob", "rw"], ["box", "rw"]], `
  let h = u.a.z; let wd = u.a.w; let hw = h * wd;
  let k = idx % 4u; let base = idx * hw;
  let gain = exp(logGain[0]);
  var peak = heat[base] * gain;
  for (var j = 1u; j < hw; j++) { peak = max(peak, heat[base + j] * gain); }
  var total = 0.0;
  for (var j = 0u; j < hw; j++) {
    let e = exp(heat[base + j] * gain - peak);
    prob[base + j] = e;
    total += e;
  }
  var pos = 0.0;
  for (var j = 0u; j < hw; j++) {
    let p = prob[base + j] / total;
    prob[base + j] = p;
    var coord = 0.0;
    if (k % 2u == 0u) { if (wd > 1u) { coord = f32(j % wd) / f32(wd - 1u); } }
    else { if (h > 1u) { coord = f32(j / wd) / f32(h - 1u); } }
    pos += p * coord;
  }
  box[idx] = pos;`),

  // a = (N, C, H, W). One invocation per (sample, channel): mean, max, and how many cells tie the max.
  pool: kernel([["feat", "r"], ["pooled", "rw"], ["ties", "rw"]], `
  let c = u.a.y; let hw = u.a.z * u.a.w;
  let n = idx / c; let ch = idx % c; let base = idx * hw;
  var total = 0.0; var top = feat[base]; var count = 0.0;
  for (var j = 0u; j < hw; j++) { total += feat[base + j]; top = max(top, feat[base + j]); }
  for (var j = 0u; j < hw; j++) { if (feat[base + j] == top) { count += 1.0; } }
  pooled[n * 2u * c + ch] = total / f32(hw);
  pooled[n * 2u * c + c + ch] = top;
  ties[idx] = count;`),

  // a = (N, 2C). One invocation per sample.
  presence: kernel([["pooled", "r"], ["pw", "r"], ["pb", "r"], ["logit", "rw"]], `
  let c2 = u.a.y;
  var acc = pb[0];
  for (var j = 0u; j < c2; j++) { acc += pw[j] * pooled[idx * c2 + j]; }
  logit[idx] = acc;`),

  // f = (boxWeight, presenceWeight, positives, N). One invocation per sample.
  lossGrad: kernel(
    [["box", "r"], ["tgt", "r"], ["hasBox", "r"], ["logit", "r"], ["dBox", "rw"], ["dLogit", "rw"], ["lossOut", "rw"]],
    `
  let positive = hasBox[idx] > 0.5;
  var boxLoss = 0.0;
  for (var k = 0u; k < 4u; k++) {
    var grad = 0.0;
    if (positive) {
      let d = box[idx * 4u + k] - tgt[idx * 4u + k];
      let a = abs(d);
      if (a < 1.0) { boxLoss += 0.5 * d * d; grad = d; } else { boxLoss += a - 0.5; grad = sign(d); }
      grad = grad * u.f.x / (u.f.z * 4.0);
    }
    dBox[idx * 4u + k] = grad;
  }
  if (positive) { boxLoss = boxLoss / (u.f.z * 4.0); }
  let z = logit[idx];
  let t = select(0.0, 1.0, positive);
  let bce = max(z, 0.0) - z * t + log(1.0 + exp(-abs(z)));
  dLogit[idx] = (1.0 / (1.0 + exp(-z)) - t) * u.f.y / u.f.w;
  lossOut[idx] = u.f.x * boxLoss + u.f.y * bce / u.f.w;`,
  ),

  // a = (N, 4, H, W). Elementwise over the heatmap.
  dHeat: kernel([["prob", "r"], ["box", "r"], ["dBox", "r"], ["logGain", "r"], ["dHeatOut", "rw"]], `
  let h = u.a.z; let wd = u.a.w; let hw = h * wd;
  let nk = idx / hw; let j = idx % hw; let k = nk % 4u;
  var coord = 0.0;
  if (k % 2u == 0u) { if (wd > 1u) { coord = f32(j % wd) / f32(wd - 1u); } }
  else { if (h > 1u) { coord = f32(j / wd) / f32(h - 1u); } }
  dHeatOut[idx] = dBox[nk] * prob[idx] * (coord - box[nk]) * exp(logGain[0]);`),

  // a = (N, 4, H, W). A single invocation: d loss / d log_gain = sum(dHeat * heat).
  dGain: kernel([["dHeatIn", "r"], ["heat", "r"], ["res", "rw"]], `
  let total = u.a.x * 4u; let hw = u.a.z * u.a.w;
  var acc = 0.0;
  for (var nk = 0u; nk < total; nk++) {
    var part = 0.0;
    for (var j = 0u; j < hw; j++) { part += dHeatIn[nk * hw + j] * heat[nk * hw + j]; }
    acc += part;
  }
  res[idx] = acc;`),

  // a = (N, 2C). Invocations 0..2C-1 are the weight gradient, 2C is the bias gradient.
  dPresence: kernel([["dLogit", "r"], ["pooled", "r"], ["gw", "rw"], ["gb", "rw"]], `
  let n = u.a.x; let c2 = u.a.y;
  var acc = 0.0;
  if (idx < c2) {
    for (var i = 0u; i < n; i++) { acc += dLogit[i] * pooled[i * c2 + idx]; }
    gw[idx] = acc;
  } else {
    for (var i = 0u; i < n; i++) { acc += dLogit[i]; }
    gb[0] = acc;
  }`),

  // a = (N, C, H, W). Gradient reaching the last feature map from both heads.
  dFeat: kernel(
    [["feat", "r"], ["pooled", "r"], ["ties", "r"], ["dLogit", "r"], ["pw", "r"], ["dHeatIn", "r"], ["heatW", "r"], ["dOut", "rw"]],
    `
  let c = u.a.y; let hw = u.a.z * u.a.w;
  let j = idx % hw; let nc = idx / hw; let ch = nc % c; let n = nc / c;
  let g = dLogit[n];
  var acc = g * pw[ch] / f32(hw);
  if (feat[idx] == pooled[n * 2u * c + c + ch]) { acc += g * pw[c + ch] / ties[nc]; }
  for (var k = 0u; k < 4u; k++) { acc += dHeatIn[(n * 4u + k) * hw + j] * heatW[k * c + ch]; }
  dOut[idx] = acc;`,
  ),

  // a = (N, C, H, W). One invocation per channel: sums[ch] = dBeta, sums[C + ch] = dGamma.
  bnSums: kernel([["dOut", "r"], ["act", "r"], ["xhat", "r"], ["sums", "rw"]], `
  let n = u.a.x; let c = u.a.y; let h = u.a.z; let wd = u.a.w;
  var dBeta = 0.0; var dGamma = 0.0;
  for (var i = 0u; i < n; i++) {
    var pb = 0.0; var pg = 0.0;
    for (var yy = 0u; yy < h; yy++) {
      var rb = 0.0; var rg = 0.0;
      let base = ((i * c + idx) * h + yy) * wd;
      for (var xx = 0u; xx < wd; xx++) {
        if (act[base + xx] > 0.0) { rb += dOut[base + xx]; rg += dOut[base + xx] * xhat[base + xx]; }
      }
      pb += rb; pg += rg;
    }
    dBeta += pb; dGamma += pg;
  }
  sums[idx] = dBeta;
  sums[c + idx] = dGamma;`),

  // a = (N, C, H, W); f.x = eps. Gradient at the conv output, through ReLU and batch norm.
  dConv: kernel(
    [["dOut", "r"], ["act", "r"], ["xhat", "r"], ["sums", "r"], ["gamma", "r"], ["stats", "r"], ["dConvOut", "rw"]],
    `
  let c = u.a.y; let count = f32(u.a.x * u.a.z * u.a.w);
  let ch = (idx / (u.a.z * u.a.w)) % c;
  let dAct = select(0.0, dOut[idx], act[idx] > 0.0);
  let scale = gamma[ch] * inverseSqrt(stats[c + ch] + u.f.x) / count;
  dConvOut[idx] = scale * (count * dAct - sums[ch] - xhat[idx] * sums[c + ch]);`,
  ),

  // conv dims. One invocation per weight.
  convDw: kernel([["x", "r"], ["dy", "r"], ["dw", "rw"]], `
  let n = u.a.x; let cin = u.a.y; let h = i32(u.a.z); let wd = i32(u.a.w);
  let cout = u.b.x; let ho = u.b.y; let wo = u.b.z; let k = u.c.y;
  let kx = idx % k; let ky = (idx / k) % k; let ci = (idx / (k * k)) % cin; let co = idx / (k * k * cin);
  var acc = 0.0;
  for (var i = 0u; i < n; i++) {
    var plane = 0.0;
    let xb = (i * cin + ci) * u32(h * wd);
    let yb = (i * cout + co) * ho * wo;
    for (var oy = 0u; oy < ho; oy++) {
      let iy = i32(oy * u.b.w + ky) - i32(u.c.x);
      if (iy < 0 || iy >= h) { continue; }
      var row = 0.0;
      for (var ox = 0u; ox < wo; ox++) {
        let ix = i32(ox * u.b.w + kx) - i32(u.c.x);
        if (ix < 0 || ix >= wd) { continue; }
        row += dy[yb + oy * wo + ox] * x[xb + u32(iy * wd + ix)];
      }
      plane += row;
    }
    acc += plane;
  }
  dw[idx] = acc;`),

  // conv dims. One invocation per output channel.
  convDb: kernel([["dy", "r"], ["db", "rw"]], `
  let n = u.a.x; let cout = u.b.x; let hw = u.b.y * u.b.z;
  var acc = 0.0;
  for (var i = 0u; i < n; i++) {
    var plane = 0.0;
    for (var j = 0u; j < hw; j++) { plane += dy[(i * cout + idx) * hw + j]; }
    acc += plane;
  }
  db[idx] = acc;`),

  // conv dims. One invocation per input element.
  convDx: kernel([["dy", "r"], ["w", "r"], ["dx", "rw"]], `
  let cin = u.a.y; let h = u.a.z; let wd = u.a.w;
  let cout = u.b.x; let ho = i32(u.b.y); let wo = i32(u.b.z); let k = u.c.y; let stride = i32(u.b.w);
  let ix = idx % wd; let iy = (idx / wd) % h; let ci = (idx / (wd * h)) % cin; let n = idx / (wd * h * cin);
  var acc = 0.0;
  for (var ky = 0u; ky < k; ky++) {
    let ty = i32(iy + u.c.x) - i32(ky);
    if (ty < 0 || ty % stride != 0) { continue; }
    let oy = ty / stride;
    if (oy >= ho) { continue; }
    for (var kx = 0u; kx < k; kx++) {
      let tx = i32(ix + u.c.x) - i32(kx);
      if (tx < 0 || tx % stride != 0) { continue; }
      let ox = tx / stride;
      if (ox >= wo) { continue; }
      for (var co = 0u; co < cout; co++) {
        acc += dy[(n * cout + co) * u32(ho * wo) + u32(oy * wo + ox)] * w[((co * cin + ci) * k + ky) * k + kx];
      }
    }
  }
  dx[idx] = acc;`),

  // f = (lr, 1 - beta1^t, 1 - beta2^t, eps). Elementwise.
  adam: kernel([["p", "rw"], ["grad", "r"], ["m", "rw"], ["v", "rw"]], `
  let gr = grad[idx];
  let m1 = 0.9 * m[idx] + 0.1 * gr;
  let v1 = 0.999 * v[idx] + 0.001 * gr * gr;
  m[idx] = m1; v[idx] = v1;
  p[idx] = p[idx] - u.f.x * (m1 / u.f.y) / (sqrt(v1 / u.f.z) + u.f.w);`),
};

const ROW = 16384 * 64; // invocations per dispatch row (16384 workgroups of 64)

function outSize(n, k, stride, pad) {
  return Math.floor((n + 2 * pad - k) / stride) + 1;
}

export async function requestDevice() {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const info = adapter.info || {};
  return { device, name: [info.vendor, info.architecture].filter(Boolean).join(" ") || "gpu" };
}

export class GpuTrainer {
  /** `params` are ref.js tensors (torch names); they are uploaded and then live on the GPU. */
  constructor(device, params, { size, boxWeight = 1, presenceWeight = 1, lr = 1e-3 } = {}) {
    this.device = device;
    this.size = size;
    this.opts = { boxWeight, presenceWeight, lr };
    const { width, context } = configOf(params);
    this.blocks = blockPlan(width, context);
    this.shapes = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, v.shape.slice()]));
    this.pipelines = {};
    for (const [name, k] of Object.entries(KERNELS)) {
      this.pipelines[name] = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: k.code }), entryPoint: "main" },
      });
    }
    this.buf = {}; // parameter, gradient and Adam state buffers, by torch name
    this.step = 0;
    for (const [name, t] of Object.entries(params)) {
      if (name.includes("running_")) continue;
      this.buf[name] = this._buffer(t.data.length, t.data);
      this.buf[`grad:${name}`] = this._buffer(t.data.length);
      this.buf[`m:${name}`] = this._buffer(t.data.length);
      this.buf[`v:${name}`] = this._buffer(t.data.length);
    }
    this.blocks.forEach((b, i) => {
      const both = new Float64Array(2 * b.cout);
      both.set(params[`features.${i}.1.running_mean`].data, 0);
      both.set(params[`features.${i}.1.running_var`].data, b.cout);
      this.buf[`running:${i}`] = this._buffer(2 * b.cout, both);
    });
    this.plans = new Map();
  }

  _buffer(length, data) {
    const buffer = this.device.createBuffer({
      size: Math.max(4, length) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (data) this.device.queue.writeBuffer(buffer, 0, Float32Array.from(data));
    return buffer;
  }

  /** One kernel call: a uniform, a bind group and how many invocations to dispatch. */
  _op(kernelName, buffers, total, { a = [0, 0, 0, 0], b = [0, 0, 0, 0], pad = 0, k = 0, f = [0, 0, 0, 0] } = {}) {
    const uniform = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const raw = new ArrayBuffer(64);
    new Uint32Array(raw, 0, 12).set([...a, ...b, pad, k, total, ROW]);
    new Float32Array(raw, 48, 4).set(f);
    this.device.queue.writeBuffer(uniform, 0, raw);
    const pipeline = this.pipelines[kernelName];
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        ...buffers.map((buffer, i) => ({ binding: i + 1, resource: { buffer } })),
      ],
    });
    const groups = Math.ceil(total / 64);
    const gx = Math.min(groups, ROW / 64);
    return { pipeline, group, gx, gy: Math.ceil(groups / gx), uniform, raw };
  }

  /** Buffers and kernel calls for one batch size; built once and reused. */
  _plan(n) {
    if (this.plans.has(n)) return this.plans.get(n);
    const P = { n, forward: [], evalForward: [], backward: [], adam: [] };
    const B = this.buf;
    P.x = this._buffer(n * 3 * this.size * this.size);
    P.target = this._buffer(n * 4);
    P.hasBox = this._buffer(n);
    let input = P.x;
    let h = this.size;
    const layers = [];
    this.blocks.forEach((blk, i) => {
      const ho = outSize(h, 3, blk.stride, 1);
      const count = n * blk.cout * ho * ho;
      const L = {
        input,
        conv: this._buffer(count),
        xhat: this._buffer(count),
        act: this._buffer(count),
        stats: this._buffer(2 * blk.cout),
        sums: this._buffer(2 * blk.cout),
        dOut: this._buffer(count),
        dConv: this._buffer(count),
        dims: { a: [n, blk.cin, h, h], b: [blk.cout, ho, ho, blk.stride], pad: 1, k: 3 },
        bn: { a: [n, blk.cout, ho, ho] },
        count,
        blk,
        i,
      };
      const w = B[`features.${i}.0.weight`];
      const bias = B[`features.${i}.0.bias`];
      const gamma = B[`features.${i}.1.weight`];
      const beta = B[`features.${i}.1.bias`];
      const conv = this._op("conv", [input, w, bias, L.conv], count, L.dims);
      const perSample = n * ho * ho;
      P.forward.push(
        conv,
        this._op("bnStats", [L.conv, L.stats, B[`running:${i}`]], blk.cout, {
          ...L.bn,
          f: [BN_MOMENTUM, perSample > 1 ? perSample / (perSample - 1) : 1, 0, 0],
        }),
        this._op("bnRelu", [L.conv, L.stats, gamma, beta, L.xhat, L.act], count, { ...L.bn, f: [BN_EPS, 0, 0, 0] }),
      );
      P.evalForward.push(
        conv,
        this._op("bnRelu", [L.conv, B[`running:${i}`], gamma, beta, L.xhat, L.act], count, {
          ...L.bn,
          f: [BN_EPS, 0, 0, 0],
        }),
      );
      layers.push(L);
      input = L.act;
      h = ho;
    });
    const last = layers[layers.length - 1];
    const c = last.blk.cout;
    const hw = h * h;
    P.heat = this._buffer(n * 4 * hw);
    P.prob = this._buffer(n * 4 * hw);
    P.dHeat = this._buffer(n * 4 * hw);
    P.box = this._buffer(n * 4);
    P.pooled = this._buffer(n * 2 * c);
    P.ties = this._buffer(n * c);
    P.logit = this._buffer(n);
    P.dBox = this._buffer(n * 4);
    P.dLogit = this._buffer(n);
    P.lossOut = this._buffer(n);
    const heatDims = { a: [n, c, h, h], b: [4, h, h, 1], pad: 0, k: 1 };
    const heads = [
      this._op("conv", [last.act, B["heat.weight"], B["heat.bias"], P.heat], n * 4 * hw, heatDims),
      this._op("softArgmax", [P.heat, B["log_gain"], P.prob, P.box], n * 4, { a: [n, 4, h, h] }),
      this._op("pool", [last.act, P.pooled, P.ties], n * c, { a: [n, c, h, h] }),
      this._op("presence", [P.pooled, B["presence_head.weight"], B["presence_head.bias"], P.logit], n, {
        a: [n, 2 * c, 0, 0],
      }),
    ];
    P.forward.push(...heads);
    P.evalForward.push(...heads);

    P.lossOp = this._op("lossGrad", [P.box, P.target, P.hasBox, P.logit, P.dBox, P.dLogit, P.lossOut], n);
    P.backward.push(
      P.lossOp,
      this._op("dHeat", [P.prob, P.box, P.dBox, B["log_gain"], P.dHeat], n * 4 * hw, { a: [n, 4, h, h] }),
      this._op("dGain", [P.dHeat, P.heat, B["grad:log_gain"]], 1, { a: [n, 4, h, h] }),
      this._op("dPresence", [P.dLogit, P.pooled, B["grad:presence_head.weight"], B["grad:presence_head.bias"]], 2 * c + 1, {
        a: [n, 2 * c, 0, 0],
      }),
      this._op("convDw", [last.act, P.dHeat, B["grad:heat.weight"]], 4 * c, heatDims),
      this._op("convDb", [P.dHeat, B["grad:heat.bias"]], 4, heatDims),
      this._op(
        "dFeat",
        [last.act, P.pooled, P.ties, P.dLogit, B["presence_head.weight"], P.dHeat, B["heat.weight"], last.dOut],
        last.count,
        { a: [n, c, h, h] },
      ),
    );
    for (let i = layers.length - 1; i >= 0; i--) {
      const L = layers[i];
      const name = `features.${i}`;
      P.backward.push(
        this._op("bnSums", [L.dOut, L.act, L.xhat, L.sums], L.blk.cout, L.bn),
        this._op("dConv", [L.dOut, L.act, L.xhat, L.sums, B[`${name}.1.weight`], L.stats, L.dConv], L.count, {
          ...L.bn,
          f: [BN_EPS, 0, 0, 0],
        }),
        // sums holds [dBeta, dGamma]: copied into the two gradient buffers after the pass
        this._op("convDw", [L.input, L.dConv, B[`grad:${name}.0.weight`]], L.blk.cout * L.blk.cin * 9, L.dims),
        this._op("convDb", [L.dConv, B[`grad:${name}.0.bias`]], L.blk.cout, L.dims),
      );
      if (i > 0) {
        const below = layers[i - 1];
        P.backward.push(
          this._op("convDx", [L.dConv, B[`${name}.0.weight`], below.dOut], below.count, L.dims),
        );
      }
    }
    P.layers = layers;
    for (const name of Object.keys(this.shapes)) {
      if (name.includes("running_")) continue;
      const length = this.shapes[name].reduce((x, y) => x * y, 1);
      P.adam.push(this._op("adam", [B[name], B[`grad:${name}`], B[`m:${name}`], B[`v:${name}`]], length));
    }
    P.readback = this.device.createBuffer({
      size: n * 5 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.plans.set(n, P);
    return P;
  }

  _run(pass, ops) {
    for (const op of ops) {
      pass.setPipeline(op.pipeline);
      pass.setBindGroup(0, op.group);
      pass.dispatchWorkgroups(op.gx, op.gy);
    }
  }

  _setFloats(op, f) {
    new Float32Array(op.raw, 48, 4).set(f);
    this.device.queue.writeBuffer(op.uniform, 0, op.raw);
  }

  _upload(P, x, boxes, hasBox) {
    const q = this.device.queue;
    q.writeBuffer(P.x, 0, x instanceof Float32Array ? x : Float32Array.from(x));
    if (boxes) q.writeBuffer(P.target, 0, Float32Array.from(boxes));
    if (hasBox) q.writeBuffer(P.hasBox, 0, Float32Array.from(hasBox));
  }

  /**
   * One optimizer step. `x` is N*3*S*S floats (NCHW), `boxes` N*4, `hasBox` N flags.
   * With `lossInto` (a GPUBuffer) the per-sample loss is copied there at `lossOffset`.
   */
  trainStep(x, boxes, hasBox, { lossInto = null, lossOffset = 0, update = true } = {}) {
    const n = hasBox.length;
    const P = this._plan(n);
    this._upload(P, x, boxes, hasBox);
    const positives = hasBox.reduce((a, v) => a + (v ? 1 : 0), 0);
    const { boxWeight, presenceWeight, lr } = this.opts;
    this._setFloats(P.lossOp, [boxWeight, presenceWeight, positives, n]);
    if (update) {
      this.step += 1;
      const f = [lr, 1 - 0.9 ** this.step, 1 - 0.999 ** this.step, 1e-8];
      for (const op of P.adam) this._setFloats(op, f);
    }
    const enc = this.device.createCommandEncoder();
    let pass = enc.beginComputePass();
    this._run(pass, P.forward);
    this._run(pass, P.backward);
    pass.end();
    for (const L of P.layers) {
      const c = L.blk.cout * 4;
      enc.copyBufferToBuffer(L.sums, 0, this.buf[`grad:features.${L.i}.1.bias`], 0, c);
      enc.copyBufferToBuffer(L.sums, c, this.buf[`grad:features.${L.i}.1.weight`], 0, c);
    }
    if (update) {
      pass = enc.beginComputePass();
      this._run(pass, P.adam);
      pass.end();
    }
    if (lossInto) enc.copyBufferToBuffer(P.lossOut, 0, lossInto, lossOffset * 4, n * 4);
    this.device.queue.submit([enc.finish()]);
    return P;
  }

  /** Eval-mode forward (running statistics). Resolves to {box: Float32Array N*4, presence: N}. */
  async predict(x, n) {
    const P = this._plan(n);
    this._upload(P, x);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this._run(pass, P.evalForward);
    pass.end();
    enc.copyBufferToBuffer(P.box, 0, P.readback, 0, n * 16);
    enc.copyBufferToBuffer(P.logit, 0, P.readback, n * 16, n * 4);
    this.device.queue.submit([enc.finish()]);
    await P.readback.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(P.readback.getMappedRange().slice(0));
    P.readback.unmap();
    return { box: all.slice(0, n * 4), presence: all.slice(n * 4, n * 5) };
  }

  /** Copy any GPU buffer back as a Float32Array. */
  async read(buffer, length) {
    const staging = this.device.createBuffer({
      size: length * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, length * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** A buffer the per-sample losses of an epoch are collected in. */
  lossBuffer(samples) {
    return this._buffer(samples);
  }

  /** The current parameters as ref.js tensors (torch names), e.g. to export or checkpoint. */
  async params() {
    const out = {};
    for (const [name, shape] of Object.entries(this.shapes)) {
      if (name.includes("running_")) continue;
      const length = shape.reduce((x, y) => x * y, 1);
      out[name] = { shape: shape.slice(), data: Float64Array.from(await this.read(this.buf[name], length)) };
    }
    for (let i = 0; i < this.blocks.length; i++) {
      const c = this.blocks[i].cout;
      const both = await this.read(this.buf[`running:${i}`], 2 * c);
      out[`features.${i}.1.running_mean`] = { shape: [c], data: Float64Array.from(both.slice(0, c)) };
      out[`features.${i}.1.running_var`] = { shape: [c], data: Float64Array.from(both.slice(c)) };
    }
    return out;
  }

  /** Gradients of the last step, by parameter name (for tests). */
  async grads() {
    const out = {};
    for (const [name, shape] of Object.entries(this.shapes)) {
      if (name.includes("running_")) continue;
      out[name] = await this.read(this.buf[`grad:${name}`], shape.reduce((x, y) => x * y, 1));
    }
    return out;
  }
}
