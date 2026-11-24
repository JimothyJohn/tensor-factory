// Decide where training runs, and build the trainer for it.
//
// The Studio trains in the browser on WebGPU when it can: nothing to install, frames stay
// on the machine. The Python backend (`uv run tensor-factory-studio`) is optional and is
// used for what a browser cannot do (writing a model to a board, GroundingDINO
// auto-label, datasets on disk). It also does the training when the browser has no
// WebGPU, or when it was started with `--source` data that only it can read.

import { DEFAULTS, LocalTrainer } from "./local.js";
import { Trainer } from "./trainer.js";

/** "browser" | "backend" | "none". `override` is the page's ?trainer= value, if any. */
export function pickMode({ gpu, backend, override = null }) {
  const sources = backend && Number(backend.sources) > 0;
  if (override === "backend" && backend) return "backend";
  if (override === "browser" && gpu) return "browser";
  if (sources) return "backend";
  if (gpu) return "browser";
  if (backend) return "backend";
  return "none";
}

/** Why that mode, in a sentence a person can act on. */
export function modeLine(mode, { gpu, backend }) {
  if (mode === "browser") {
    return backend
      ? "Training in this browser on your GPU. The local helper is connected for flashing boards."
      : "Training in this browser on your GPU. Nothing is uploaded.";
  }
  if (mode === "backend") {
    if (backend && Number(backend.sources) > 0) return "Training on the local backend, with its --source datasets.";
    return gpu ? "Training on the local backend." : "This browser has no WebGPU, so the local backend is training.";
  }
  return "This browser has no WebGPU and no local backend is running. Use a current Chrome or Edge, or run `uv run tensor-factory-studio` in the repo.";
}

function intIn(value, lo, hi) {
  if (value == null || !/^\d+$/.test(String(value))) return null;
  const n = Number(value);
  return n >= lo && n <= hi ? n : null;
}

/**
 * Training settings for browser mode: the defaults, then the backend's (so its flags
 * apply either way), then the page's own ?size=&context=&resize=&epochs= if valid.
 */
export function localConfig(search, backend) {
  const cfg = { size: DEFAULTS.size, width: DEFAULTS.width, context: DEFAULTS.context, resize: DEFAULTS.resize, epochs: DEFAULTS.epochs };
  const from = backend && backend.config ? backend.config : {};
  const q = new URLSearchParams(search || "");
  const pick = (key, lo, hi) => intIn(q.get(key), lo, hi) ?? intIn(from[key], lo, hi);
  cfg.size = pick("size", 32, 640) ?? cfg.size;
  cfg.width = pick("width", 2, 64) ?? cfg.width;
  cfg.context = pick("context", 0, 8) ?? cfg.context;
  cfg.epochs = intIn(q.get("epochs"), 1, 500) ?? cfg.epochs;
  const resize = q.get("resize") ?? from.resize;
  if (["bilinear", "nearest", "mixed"].includes(resize)) cfg.resize = resize;
  return cfg;
}

/** The backend's /status, or null when there is no backend behind this page. */
export async function probeBackend(timeoutMs = 1500) {
  try {
    const r = await fetch("/status", { signal: AbortSignal.timeout(timeoutMs) });
    if (!r.ok) return null;
    const s = await r.json();
    return s && typeof s.backend === "string" && s.counts ? s : null;
  } catch {
    return null;
  }
}

/**
 * Probe, pick, build. Returns {mode, trainer|null, backend, line}; `trainer.init()` is
 * left to the caller.
 */
export async function connect(callbacks, { search = globalThis.location?.search ?? "" } = {}) {
  const [{ GpuEngine, requestDevice }, backend] = await Promise.all([import("./nn/gpu.js"), probeBackend()]);
  const gpu = await requestDevice().catch(() => null);
  const override = new URLSearchParams(search).get("trainer");
  const mode = pickMode({ gpu: !!gpu, backend, override });
  const line = modeLine(mode, { gpu: !!gpu, backend });
  if (mode === "browser") {
    const cfg = localConfig(search, backend);
    const trainer = new LocalTrainer({
      ...callbacks,
      ...cfg,
      backend: `webgpu · ${gpu.name}`,
      makeEngine: () => new GpuEngine(gpu.device, { size: cfg.size }),
    });
    return { mode, trainer, backend, line };
  }
  if (mode === "backend") return { mode, trainer: new Trainer(callbacks), backend, line };
  return { mode, trainer: null, backend, line };
}
