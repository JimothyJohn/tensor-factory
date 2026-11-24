// Training in the browser. Same interface and the same metrics as the backend client in
// trainer.js (init / upsert / predict / reset / pause / resume + callbacks), so the rest of
// the Studio does not care which one it has. Here the labeled frames never leave the
// page: they are resized with PIL's arithmetic (nn/data.js), trained on the GPU through
// WebGPU (nn/gpu.js), and the result is written as the same int8 ONNX the Python trainer
// exports (nn/onnx.js).
//
// A round mirrors what the backend asks `tensor_factory_train.fit` for: start from fresh
// weights, hold 20% out, Adam at 1e-3, batches of 16, the checkpoint with the best
// held-out score wins, and it replaces the served model only if it beats the best so far.

import { baselineCenterErr, checkpointScore, resizeBilinear, resizeNearest, score, shuffled, split, writeChw } from "./nn/data.js";
import { exportOnnx, templateName } from "./nn/onnx.js";
import { initParams, rng } from "./nn/ref.js";

export const DEFAULTS = {
  size: 224,
  width: 16,
  context: 4,
  resize: "mixed",
  epochs: 20,
  batch: 16,
  valFrac: 0.2,
  minPositives: 4,
};
const MAX_CONTEXT = 8; // templates ship for 0..8 context blocks

/** Decode an image blob to RGBA bytes with the browser's own decoder. */
async function decodeBlob(blob) {
  const bitmap = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close?.();
  return { pixels: data, width: canvas.width, height: canvas.height, channels: 4 };
}

async function fetchTemplate(name) {
  const r = await fetch(new URL(`../models/${name}`, import.meta.url));
  if (!r.ok) throw new Error(`could not load ${name} (HTTP ${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}

export class LocalTrainer {
  /**
   * `makeEngine()` returns a fresh engine (GpuEngine or RefEngine); two are made, one that
   * trains and one that holds the served model. `decode` and `loadTemplate` are
   * replaceable so the loop can run outside a browser.
   */
  constructor({ makeEngine, backend = "webgpu", decode = decodeBlob, loadTemplate = fetchTemplate, seed = 0, ...rest } = {}) {
    const { onReady, onMetrics, onStatus, onError, ...config } = rest;
    this.cb = { onReady, onMetrics, onStatus, onError };
    this.cfg = { ...DEFAULTS, ...config };
    if (!["bilinear", "nearest", "mixed"].includes(this.cfg.resize)) {
      throw new Error(`resize must be bilinear, nearest or mixed, got ${this.cfg.resize}`);
    }
    if (!(this.cfg.context >= 0 && this.cfg.context <= MAX_CONTEXT)) {
      throw new Error(`context must be 0..${MAX_CONTEXT}, got ${this.cfg.context}`);
    }
    this.makeEngine = makeEngine;
    this.backend = backend;
    this.decode = decode;
    this.loadTemplate = loadTemplate;
    this.seed = seed;
    this.samples = new Map(); // frameId -> {id, box|null, bilinear, nearest}
    this.order = []; // upsert order, to name suspects after a regression
    this.ready = false;
    this.running = true;
    this._dirty = false;
    this._busy = null;
    this._clear();
  }

  _clear() {
    this.served = null; // parameters of the model in use
    this.globalBest = Infinity;
    this.idsAtBest = new Set();
    this.history = [];
    this.generation = (this.generation || 0) + 1; // bumps on reset: a round in flight gives up
  }

  config() {
    const { size, width, context, resize } = this.cfg;
    return { size, width, context, resize };
  }

  async init() {
    this.trainEngine = this.makeEngine();
    this.serveEngine = this.makeEngine();
    this.ready = true;
    this.cb.onReady?.({ backend: this.backend, classes: null });
    this._status("idle");
  }

  _status(text) {
    this.cb.onStatus?.(text);
  }

  async setClasses(names) {
    if (Array.isArray(names) && names.length > 1) {
      this.cb.onError?.(
        "Browser training finds one kind of object. For several classes start the Python backend.",
      );
    }
    return names;
  }

  /** samples: [{frameId, blob, box:[x1,y1,x2,y2]|null, present:0|1}] */
  async upsert(samples) {
    for (const s of samples) {
      try {
        const img = await this.decode(s.blob);
        const { size } = this.cfg;
        const box = s.present && s.box ? s.box.map(Number) : null;
        if (box && !(box.length === 4 && box.every((v) => v >= 0 && v <= 1) && box[2] > box[0] && box[3] > box[1])) {
          throw new Error("box must be x1,y1,x2,y2 in 0..1");
        }
        this.samples.set(s.frameId, {
          id: s.frameId,
          box,
          bilinear: resizeBilinear(img.pixels, img.width, img.height, size, img.channels),
          nearest: resizeNearest(img.pixels, img.width, img.height, size, img.channels),
        });
        this.order = this.order.filter((id) => id !== s.frameId).concat(s.frameId);
        this._dirty = true;
      } catch (e) {
        this.cb.onError?.(`frame ${s.frameId} was not added: ${e.message}`);
      }
    }
    this._kick();
  }

  counts() {
    let positives = 0;
    for (const s of this.samples.values()) if (s.box) positives++;
    return { positives, negatives: this.samples.size - positives, total: this.samples.size };
  }

  async predict(frameId, blob) {
    const none = { ready: false, present: false, score: 0, box: null, cls: null, clsName: null, clsScore: null };
    if (!this.served) return none;
    try {
      const { size } = this.cfg;
      const img = await this.decode(blob);
      const x = new Float32Array(3 * size * size);
      writeChw(x, 0, resizeBilinear(img.pixels, img.width, img.height, size, img.channels), size);
      const out = await this.serveEngine.predict(x, 1);
      const p = 1 / (1 + Math.exp(-out.presence[0]));
      const present = p >= 0.5;
      return { ...none, ready: true, present, score: p, box: present ? Array.from(out.box.slice(0, 4)) : null };
    } catch {
      return none;
    }
  }

  async reset() {
    this.samples.clear();
    this.order = [];
    this._dirty = false;
    this._clear();
    this._status("session cleared");
  }

  async pause() {
    this.running = false;
  }

  async resume() {
    this.running = true;
    this._dirty = this._dirty || this.samples.size > 0;
    this._kick();
  }

  /** The served model as int8 ONNX bytes; null before anything is trained. */
  async modelBytes() {
    if (!this.served) return null;
    const template = await this.loadTemplate(templateName(this.served));
    return exportOnnx(template, this.served, { size: this.cfg.size });
  }

  /** Resolves when no round is running or pending (for tests and for export). */
  async idle() {
    while (this._busy) await this._busy;
  }

  _kick() {
    if (this._busy || !this.ready) return;
    this._busy = this._loop().finally(() => {
      this._busy = null;
    });
  }

  async _loop() {
    while (this._dirty && this.running) {
      this._dirty = false;
      const { positives } = this.counts();
      if (positives < this.cfg.minPositives) {
        this._status(`need ≥${this.cfg.minPositives} positives (${positives} now)`);
        return;
      }
      try {
        await this._round();
      } catch (e) {
        this._status(`train error: ${e.message}`);
        this.cb.onError?.(`training failed: ${e.message}`);
        return;
      }
    }
  }

  _batch(items, { train, rand }) {
    const { size, resize } = this.cfg;
    const n = items.length;
    const x = new Float32Array(n * 3 * size * size);
    const boxes = new Float32Array(n * 4);
    const hasBox = new Float32Array(n);
    items.forEach((s, i) => {
      const nearest = resize === "nearest" || (resize === "mixed" && train && rand() < 0.5);
      writeChw(x, i, nearest ? s.nearest : s.bilinear, size);
      if (s.box) {
        boxes.set(s.box, i * 4);
        hasBox[i] = 1;
      }
    });
    return { x, boxes, hasBox };
  }

  async _predictAll(engine, items) {
    const preds = [];
    for (let i = 0; i < items.length; i += this.cfg.batch) {
      const chunk = items.slice(i, i + this.cfg.batch);
      const out = await engine.predict(this._batch(chunk, { train: false }).x, chunk.length);
      chunk.forEach((_, j) => {
        preds.push({ box: Array.from(out.box.slice(j * 4, j * 4 + 4)), presence: out.presence[j] });
      });
    }
    return preds;
  }

  async _round() {
    const { size, width, context, epochs, batch, valFrac } = this.cfg;
    const generation = this.generation;
    const items = [...this.samples.values()];
    const ids = new Set(items.map((s) => s.id));
    const { train, val } = split(items, valFrac, this.seed);
    const baseline = baselineCenterErr(train, val, size);
    const rand = rng(this.seed + 1);
    const engine = this.trainEngine;
    engine.load(initParams({ width, context, seed: this.seed }));
    let best = { score: -Infinity, err: null, params: null };
    let last = null;
    for (let epoch = 1; epoch <= epochs; epoch++) {
      const order = shuffled(train, rand);
      for (let i = 0; i < order.length; i += batch) {
        while (!this.running) await new Promise((r) => setTimeout(r, 100));
        if (generation !== this.generation) return; // the session was cleared mid-round
        const b = this._batch(order.slice(i, i + batch), { train: true, rand });
        await engine.trainStep(b.x, b.boxes, b.hasBox);
      }
      const loss = await engine.epochLoss();
      const s = val.length ? score(val, await this._predictAll(engine, val), size) : { err: null, presenceAcc: null, valBoxes: 0 };
      const isBest = val.length > 0 && checkpointScore(s, size) > best.score;
      if (isBest) best = { score: checkpointScore(s, size), err: s.err, params: await engine.params() };
      if (generation !== this.generation) return;
      if (s.err != null) {
        this.history.push(s.err);
        this.history = this.history.slice(-120);
      }
      last = {
        phase: "training",
        epoch,
        epochs,
        loss,
        err: s.err,
        baseline,
        presenceAcc: s.presenceAcc,
        classAcc: null,
        numClasses: 1,
        gain: await engine.gain(),
        trainCount: train.length,
        valCount: val.length,
        valBoxes: s.valBoxes,
        sources: [],
        regressed: false,
        suspects: [],
      };
      this._emit(last);
    }
    if (generation !== this.generation || !last) return;
    // Keep-best: a round replaces the served model only if it beats the best so far.
    let regressed = false;
    if (best.params && best.err != null && best.err < this.globalBest) {
      this.served = best.params;
      this.serveEngine.load(best.params);
      this.globalBest = best.err;
      this.idsAtBest = ids;
    } else if (this.globalBest < Infinity) {
      regressed = true;
    }
    const suspects = regressed
      ? this.order.filter((id) => !this.idsAtBest.has(id)).reverse().slice(0, 5).map((frameId) => ({ frameId }))
      : [];
    this._emit({ ...last, phase: "idle", regressed, suspects });
  }

  _emit(m) {
    this.cb.onMetrics?.({
      ...m,
      backend: this.backend,
      running: this.running,
      bestErr: this.globalBest === Infinity ? null : this.globalBest,
      history: this.history.slice(),
      hasModel: this.served !== null,
      config: this.config(),
    });
  }
}
