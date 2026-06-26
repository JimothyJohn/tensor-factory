// Client for the tensor-factory-studio backend. Same interface the app used for the old
// in-browser trainer (init / upsert / predict / reset / pause / resume + callbacks), so
// app.js is unchanged — only the engine moved: training now runs server-side on torch
// (MPS/CUDA) and the canonical int8 ONNX falls out of tensor-factory-train.

const POLL_MS = 1500;

export class Trainer {
  constructor({ onReady, onMetrics, onStatus, onError } = {}) {
    this.cb = { onReady, onMetrics, onStatus, onError };
    this.ready = false;
    this.running = false;
    this.backend = "?";
    this._timer = null;
  }

  async init() {
    try {
      const s = await fetch("/status").then((r) => r.json());
      this.ready = true;
      this.running = s.running;
      this.backend = s.backend;
      this.cb.onReady?.({ backend: s.backend });
      this._poll();
    } catch (e) {
      this.cb.onError?.(`backend unreachable: ${e.message}`);
    }
  }

  _poll() {
    clearInterval(this._timer);
    this._timer = setInterval(async () => {
      try {
        const m = await fetch("/metrics").then((r) => r.json());
        this.backend = m.backend || this.backend;
        this.running = !!m.running;
        if (m.epoch != null) this.cb.onMetrics?.(m);
        else this.cb.onStatus?.(m.status || m.phase || "");
      } catch (e) {
        this.cb.onError?.(`metrics poll failed: ${e.message}`);
      }
    }, POLL_MS);
  }

  /** samples: [{frameId, blob, box:[x1,y1,x2,y2]|null, present:0|1}] */
  async upsert(samples) {
    for (const s of samples) {
      const params = new URLSearchParams({ id: String(s.frameId), present: s.present ? "1" : "0" });
      if (s.present && s.box) params.set("box", s.box.join(","));
      try {
        await fetch(`/samples?${params}`, { method: "POST", body: s.blob });
      } catch (e) {
        this.cb.onError?.(`upload failed: ${e.message}`);
      }
    }
  }

  async predict(frameId, blob) {
    try {
      const r = await fetch("/predict", { method: "POST", body: blob }).then((x) => x.json());
      return { ready: !!r.ready, present: !!r.present, score: r.score ?? 0, box: r.box || null };
    } catch {
      return { ready: false, present: false, score: 0, box: null };
    }
  }

  async reset() {
    await fetch("/reset", { method: "POST" }).catch(() => {});
  }

  async resume() {
    this.running = true;
    await fetch("/resume", { method: "POST" }).catch(() => {});
  }

  async pause() {
    this.running = false;
    await fetch("/pause", { method: "POST" }).catch(() => {});
  }
}
