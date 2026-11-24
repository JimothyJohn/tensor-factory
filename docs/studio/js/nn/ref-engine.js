// The CPU reference behind the same interface as GpuEngine (gpu.js). It is what the
// training loop's tests run on in node; in a browser it is far too slow for a real model
// (seconds per step at 224 px), so the Studio only offers it WebGPU or the Python backend.

import { forward, tensor, trainStep } from "./ref.js";

function clone(params) {
  return Object.fromEntries(
    Object.entries(params).map(([k, v]) => [k, { shape: v.shape.slice(), data: Float64Array.from(v.data) }]),
  );
}

export class RefEngine {
  constructor({ size }) {
    this.size = size;
  }

  load(params) {
    this.p = clone(params);
    this.state = {};
    this.sum = 0;
    this.samples = 0;
  }

  async trainStep(x, boxes, hasBox) {
    const n = hasBox.length;
    const r = trainStep(
      this.p,
      this.state,
      tensor([n, 3, this.size, this.size], x),
      tensor([n, 4], boxes),
      Array.from(hasBox),
    );
    this.sum += r.loss * n;
    this.samples += n;
  }

  async epochLoss() {
    const mean = this.samples ? this.sum / this.samples : null;
    this.sum = 0;
    this.samples = 0;
    return mean;
  }

  async predict(x, n) {
    const out = forward(this.p, tensor([n, 3, this.size, this.size], x), { train: false });
    return { box: Float32Array.from(out.box.data), presence: Float32Array.from(out.presence.data) };
  }

  async params() {
    return clone(this.p);
  }

  async gain() {
    return Math.exp(this.p["log_gain"].data[0]);
  }
}
