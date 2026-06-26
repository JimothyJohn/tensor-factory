// Main-thread controller for the training worker. Owns the worker, forwards
// labeled samples, surfaces metrics, and turns prediction requests into promises.

export class Trainer {
  constructor({ onReady, onMetrics, onStatus, onError } = {}) {
    this.worker = new Worker("js/trainer.worker.js");
    this.ready = false;
    this.running = false;
    this._predPromises = new Map(); // frameId -> {resolve}
    this._weightsResolve = null;

    this.worker.onmessage = (ev) => {
      const m = ev.data;
      switch (m.type) {
        case "ready":
          this.ready = true;
          this.backend = m.backend;
          onReady?.(m);
          break;
        case "metrics":
          onMetrics?.(m);
          break;
        case "status":
          onStatus?.(m.text);
          break;
        case "prediction": {
          const p = this._predPromises.get(m.frameId);
          if (p) {
            this._predPromises.delete(m.frameId);
            p.resolve(m);
          }
          break;
        }
        case "weights":
          this._weightsResolve?.(m);
          this._weightsResolve = null;
          break;
        case "error":
          onError?.(m.text);
          break;
      }
    };
  }

  init(opts = {}) {
    this.worker.postMessage({ type: "init", opts });
  }

  /** samples: [{frameId, blob, box:[x1,y1,x2,y2]|null, present:0|1}] */
  upsert(samples) {
    if (samples.length) this.worker.postMessage({ type: "upsert", samples });
  }

  remove(frameId) {
    this.worker.postMessage({ type: "remove", frameId });
  }

  reset() {
    this.worker.postMessage({ type: "reset" });
  }

  resume() {
    this.running = true;
    this.worker.postMessage({ type: "resume" });
  }

  pause() {
    this.running = false;
    this.worker.postMessage({ type: "pause" });
  }

  /** Resolve with {box, score, present} for a frame using the best served weights. */
  predict(frameId, blob) {
    return new Promise((resolve) => {
      this._predPromises.set(frameId, { resolve });
      this.worker.postMessage({ type: "predict", frameId, blob });
    });
  }

  /** Resolve with {meta, tensors} of the current trained weights. */
  exportWeights() {
    return new Promise((resolve) => {
      this._weightsResolve = resolve;
      this.worker.postMessage({ type: "export" });
    });
  }
}
