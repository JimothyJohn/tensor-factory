/* eslint-env worker */
/* global tf, importScripts */
// Continuous in-browser trainer (classic Web Worker). Holds the tiny detector,
// trains it on the approved labeled set forever, streams live val metrics, and
// serves the best-so-far weights for auto-labeling. WebGPU → WebGL → CPU.
//
// Architecture mirrors tensor_factory_train.TinyDetector: stride-2 conv stack →
// 4-channel edge heatmap → marginal soft-argmax with a learnable gain → xyxy box,
// plus a global-pool presence logit. Box loss is masked to positives; negatives
// (present=0) train only the presence head — same contract as the Python trainer.

importScripts("../vendor/tf.min.js", "../vendor/tf-backend-webgpu.min.js");

let SIZE = 192;
let WIDTH = 12;
let BATCH = 8;
let LR = 2e-3;

const HUBER_DELTA = 0.03;
const REGRESS_FACTOR = 1.25; // val err this much above best ⇒ flag regression
const STEPS_PER_EVAL = 12;

let model = null; // {convs, head, pres, logGain} of tf.Variables
let served = null; // structured clone of best weights (for prediction)
let optimizer = null;
let varList = [];

const data = new Map(); // frameId -> {imgData, box:[4]|null, present:0|1, split:'train'|'val'}
let insertOrder = 0; // deterministic ~20% val split, independent of frame-id base
let running = false;
let epoch = 0;
let bestErr = Infinity;
const errHistory = [];

// --- model construction ----------------------------------------------------
function heVar(shape, fanIn) {
  return tf.variable(tf.randomNormal(shape, 0, Math.sqrt(2 / fanIn)));
}

function buildModel() {
  const chans = [WIDTH, WIDTH * 2, WIDTH * 4, WIDTH * 8];
  let inC = 3;
  const convs = [];
  for (const outC of chans) {
    convs.push({
      f: heVar([3, 3, inC, outC], 3 * 3 * inC),
      b: tf.variable(tf.zeros([outC])),
    });
    inC = outC;
  }
  const head = { f: heVar([1, 1, inC, 4], inC), b: tf.variable(tf.zeros([4])) };
  const pres = { w: heVar([inC * 2, 1], inC * 2), b: tf.variable(tf.zeros([1])) };
  const logGain = tf.variable(tf.scalar(0)); // exp(0)=1 → plain softmax at init
  model = { convs, head, pres, logGain };
  varList = [
    ...convs.flatMap((c) => [c.f, c.b]),
    head.f,
    head.b,
    pres.w,
    pres.b,
    logGain,
  ];
  optimizer = tf.train.adam(LR);
}

// --- forward pass ----------------------------------------------------------
// tf.softmax only supports the last axis; soft-argmax needs softmax over W/H.
function softmaxAxis(logits, axis) {
  const m = tf.max(logits, axis, true);
  const e = tf.exp(tf.sub(logits, m));
  return tf.div(e, tf.sum(e, axis, true));
}

function softArgmax(heat, gain) {
  // heat: [B,H,W,4] in channel order [x1,y1,x2,y2]
  const [B, H, W] = heat.shape;
  const xs = tf.linspace(0, 1, W).reshape([1, W, 1]);
  const ys = tf.linspace(0, 1, H).reshape([1, H, 1]);
  const xheat = tf.gather(heat, [0, 2], 3); // [B,H,W,2]
  const xprob = softmaxAxis(tf.mul(tf.sum(xheat, 1), gain), 1); // over W
  const xcoord = tf.sum(tf.mul(xprob, xs), 1); // [B,2] -> x1,x2
  const yheat = tf.gather(heat, [1, 3], 3);
  const yprob = softmaxAxis(tf.mul(tf.sum(yheat, 2), gain), 1); // over H
  const ycoord = tf.sum(tf.mul(yprob, ys), 1); // [B,2] -> y1,y2
  const x1 = xcoord.slice([0, 0], [B, 1]);
  const x2 = xcoord.slice([0, 1], [B, 1]);
  const y1 = ycoord.slice([0, 0], [B, 1]);
  const y2 = ycoord.slice([0, 1], [B, 1]);
  return tf.concat([x1, y1, x2, y2], 1); // [B,4]
}

function forward(W, x) {
  let h = x;
  for (const c of W.convs) {
    h = tf.relu(tf.add(tf.conv2d(h, c.f, [2, 2], "same"), c.b));
  }
  const heat = tf.add(tf.conv2d(h, W.head.f, [1, 1], "same"), W.head.b);
  const box = softArgmax(heat, tf.exp(W.logGain));
  const feat = tf.concat([tf.mean(h, [1, 2]), tf.max(h, [1, 2])], 1);
  const presence = tf.add(tf.matMul(feat, W.pres.w), W.pres.b);
  return { box, presence };
}

function huber(pred, target) {
  // 0.5·min(|d|,δ)² + δ·relu(|d|−δ) — equals Huber, fully differentiable
  // (avoids tf.where/tf.less, which break the gradient tape).
  const ad = tf.abs(tf.sub(pred, target));
  const quad = tf.mul(0.5, tf.square(tf.minimum(ad, HUBER_DELTA)));
  const lin = tf.mul(HUBER_DELTA, tf.relu(tf.sub(ad, HUBER_DELTA)));
  return tf.add(quad, lin);
}

function totalLoss(box, presence, batch) {
  const boxErr = tf.sum(huber(box, batch.ybox), 1, true); // [B,1]
  const denom = tf.add(tf.sum(batch.mask), 1e-6);
  const boxLoss = tf.div(tf.sum(tf.mul(boxErr, batch.mask)), denom);
  const presLoss = tf.losses.sigmoidCrossEntropy(batch.ypres, presence);
  return tf.add(boxLoss, presLoss);
}

// --- weight snapshot (served / best) ---------------------------------------
function cloneStruct(W) {
  return {
    convs: W.convs.map((c) => ({ f: tf.keep(c.f.clone()), b: tf.keep(c.b.clone()) })),
    head: { f: tf.keep(W.head.f.clone()), b: tf.keep(W.head.b.clone()) },
    pres: { w: tf.keep(W.pres.w.clone()), b: tf.keep(W.pres.b.clone()) },
    logGain: tf.keep(W.logGain.clone()),
  };
}
function disposeStruct(W) {
  if (!W) return;
  W.convs.forEach((c) => {
    c.f.dispose();
    c.b.dispose();
  });
  W.head.f.dispose();
  W.head.b.dispose();
  W.pres.w.dispose();
  W.pres.b.dispose();
  W.logGain.dispose();
}
function snapshotServed() {
  const prev = served;
  served = cloneStruct(model);
  disposeStruct(prev);
}

// --- data ------------------------------------------------------------------
function tensorFromImageData(imgData) {
  return tf.browser.fromPixels(imgData).toFloat().div(255);
}

function makeBatch(entries) {
  return tf.tidy(() => {
    const imgs = entries.map((e) => tensorFromImageData(e.imgData));
    return {
      x: tf.stack(imgs),
      ybox: tf.tensor2d(entries.map((e) => e.box || [0, 0, 0, 0])),
      ypres: tf.tensor2d(entries.map((e) => [e.present])),
      mask: tf.tensor2d(entries.map((e) => [e.present])),
    };
  });
}

function sample(arr, n) {
  if (arr.length <= n) return arr;
  const out = [];
  const used = new Set();
  while (out.length < n) {
    const i = Math.floor(Math.random() * arr.length);
    if (!used.has(i)) {
      used.add(i);
      out.push(arr[i]);
    }
  }
  return out;
}

function entriesBy(split) {
  return [...data.values()].filter((e) => e.split === split);
}

// --- training + eval -------------------------------------------------------
function trainStep() {
  const train = entriesBy("train");
  if (!train.length) return null;
  const batch = makeBatch(sample(train, BATCH));
  const cost = optimizer.minimize(
    () => {
      const { box, presence } = forward(model, batch.x);
      return totalLoss(box, presence, batch);
    },
    true,
    varList,
  );
  const c = cost.dataSync()[0];
  cost.dispose();
  batch.x.dispose();
  batch.ybox.dispose();
  batch.ypres.dispose();
  batch.mask.dispose();
  return c;
}

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function centerErrPx(a, b) {
  const acx = (a[0] + a[2]) / 2;
  const acy = (a[1] + a[3]) / 2;
  const bcx = (b[0] + b[2]) / 2;
  const bcy = (b[1] + b[3]) / 2;
  return Math.hypot(acx - bcx, acy - bcy) * SIZE;
}

function baselineErr(train, val) {
  const pos = train.filter((e) => e.present && e.box);
  const vpos = val.filter((e) => e.present && e.box);
  if (!pos.length || !vpos.length) return null;
  const mean = [0, 1, 2, 3].map((i) => pos.reduce((s, e) => s + e.box[i], 0) / pos.length);
  return median(vpos.map((e) => centerErrPx(e.box, mean)));
}

function evaluate() {
  const val = entriesBy("val");
  const train = entriesBy("train");
  if (val.length < 2) return null;
  return tf.tidy(() => {
    const x = tf.stack(val.map((e) => tensorFromImageData(e.imgData)));
    const { box, presence } = forward(model, x);
    const boxes = box.arraySync();
    const pres = tf.sigmoid(presence).dataSync();
    const errs = [];
    let correct = 0;
    val.forEach((e, i) => {
      if (e.present && e.box) errs.push(centerErrPx(boxes[i], e.box));
      const pred = pres[i] >= 0.5 ? 1 : 0;
      if (pred === e.present) correct++;
    });
    return {
      err: median(errs),
      presenceAcc: correct / val.length,
      baseline: baselineErr(train, val),
      gain: Math.exp(model.logGain.dataSync()[0]),
      valCount: val.length,
      trainCount: train.length,
    };
  });
}

function suspects() {
  // approved positives the current model fits worst — likely bad labels
  const pos = entriesBy("train").concat(entriesBy("val")).filter((e) => e.present && e.box);
  if (pos.length < 4) return [];
  return tf.tidy(() => {
    const x = tf.stack(pos.map((e) => tensorFromImageData(e.imgData)));
    const { box } = forward(model, x);
    const boxes = box.arraySync();
    return pos
      .map((e, i) => ({ frameId: e.frameId, err: centerErrPx(boxes[i], e.box) }))
      .sort((a, b) => b.err - a.err)
      .slice(0, 5);
  });
}

async function loop() {
  while (running) {
    const train = entriesBy("train");
    if (train.length < 3 || entriesBy("val").length < 2) {
      postMessage({ type: "status", text: `need data: ${train.length} train / ${entriesBy("val").length} val (≥3 / ≥2)` });
      await sleep(800);
      continue;
    }
    let cost = 0;
    for (let i = 0; i < STEPS_PER_EVAL && running; i++) {
      const c = trainStep();
      if (c != null) cost = c;
      await sleep(0);
    }
    epoch++;
    const m = evaluate();
    if (m && m.err != null) {
      errHistory.push(m.err);
      if (errHistory.length > 120) errHistory.shift();
      let regressed = false;
      if (m.err < bestErr) {
        bestErr = m.err;
        snapshotServed();
      } else if (bestErr < Infinity && m.err > bestErr * REGRESS_FACTOR) {
        regressed = true;
      }
      postMessage({
        type: "metrics",
        epoch,
        loss: cost,
        err: m.err,
        bestErr,
        baseline: m.baseline,
        presenceAcc: m.presenceAcc,
        gain: m.gain,
        regressed,
        suspects: regressed ? suspects() : [],
        valCount: m.valCount,
        trainCount: m.trainCount,
        backend: tf.getBackend(),
        history: [...errHistory],
      });
    }
    await sleep(0);
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// --- prediction (auto-label) ----------------------------------------------
async function predict(frameId, blob) {
  const weights = served || model;
  if (!weights) return;
  const bmp = await createImageBitmap(blob);
  const oc = new OffscreenCanvas(SIZE, SIZE);
  const ctx = oc.getContext("2d");
  ctx.drawImage(bmp, 0, 0, bmp.width, bmp.height, 0, 0, SIZE, SIZE);
  bmp.close();
  const imgData = ctx.getImageData(0, 0, SIZE, SIZE);
  const result = tf.tidy(() => {
    const x = tensorFromImageData(imgData).expandDims(0);
    const { box, presence } = forward(weights, x);
    return { box: box.dataSync(), score: tf.sigmoid(presence).dataSync()[0] };
  });
  postMessage({
    type: "prediction",
    frameId,
    box: Array.from(result.box),
    score: result.score,
    present: result.score >= 0.5,
  });
}

// --- export ----------------------------------------------------------------
function exportWeights() {
  const tensors = {};
  model.convs.forEach((c, i) => {
    tensors[`conv${i}.f`] = { shape: c.f.shape, data: Array.from(c.f.dataSync()) };
    tensors[`conv${i}.b`] = { shape: c.b.shape, data: Array.from(c.b.dataSync()) };
  });
  tensors["head.f"] = { shape: model.head.f.shape, data: Array.from(model.head.f.dataSync()) };
  tensors["head.b"] = { shape: model.head.b.shape, data: Array.from(model.head.b.dataSync()) };
  tensors["pres.w"] = { shape: model.pres.w.shape, data: Array.from(model.pres.w.dataSync()) };
  tensors["pres.b"] = { shape: model.pres.b.shape, data: Array.from(model.pres.b.dataSync()) };
  tensors["logGain"] = { shape: [], data: Array.from(model.logGain.dataSync()) };
  postMessage({
    type: "weights",
    meta: { size: SIZE, width: WIDTH, arch: "TinyDetector", framework: "tfjs", bestErr },
    tensors,
  });
}

// --- decode a sample's image to ImageData (for the train cache) ------------
async function decodeImageData(blob) {
  const bmp = await createImageBitmap(blob);
  const oc = new OffscreenCanvas(SIZE, SIZE);
  const ctx = oc.getContext("2d");
  ctx.drawImage(bmp, 0, 0, bmp.width, bmp.height, 0, 0, SIZE, SIZE);
  bmp.close();
  return ctx.getImageData(0, 0, SIZE, SIZE);
}

// --- message handling ------------------------------------------------------
async function init(opts) {
  if (opts.size) SIZE = opts.size;
  if (opts.width) WIDTH = opts.width;
  if (opts.batch) BATCH = opts.batch;
  if (opts.lr) LR = opts.lr;
  let backend = "cpu";
  for (const b of ["webgpu", "webgl", "cpu"]) {
    try {
      if (await tf.setBackend(b)) {
        await tf.ready();
        backend = b;
        break;
      }
    } catch (_e) {
      /* try next */
    }
  }
  buildModel();
  postMessage({ type: "ready", backend, size: SIZE, width: WIDTH });
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case "init":
        await init(msg.opts || {});
        break;
      case "upsert":
        for (const s of msg.samples) {
          const imgData = await decodeImageData(s.blob);
          const existing = data.get(s.frameId);
          const split = existing ? existing.split : (insertOrder++ % 5 === 4 ? "val" : "train");
          data.set(s.frameId, { frameId: s.frameId, imgData, box: s.box, present: s.present, split });
        }
        break;
      case "remove":
        data.delete(msg.frameId);
        break;
      case "reset":
        data.clear();
        insertOrder = 0;
        epoch = 0;
        bestErr = Infinity;
        errHistory.length = 0;
        disposeStruct(served);
        served = null;
        break;
      case "predict":
        await predict(msg.frameId, msg.blob);
        break;
      case "pause":
        running = false;
        break;
      case "resume":
        if (!running) {
          running = true;
          loop();
        }
        break;
      case "export":
        exportWeights();
        break;
    }
  } catch (e) {
    postMessage({ type: "error", text: String(e && e.message ? e.message : e) });
  }
};
