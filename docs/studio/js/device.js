// Device panel: put the trained model on a board. Pure helpers first (unit-tested in
// node), then a small client for the backend's /devices, /deploy and /model routes.

/** What the status line under the Flash button says for a /deploy state. */
export function deployLine(state) {
  if (!state || typeof state !== "object") return { tone: "", text: "" };
  if (state.state === "running") {
    return { tone: "", text: `Working: ${state.step || "starting"}…` };
  }
  if (state.state === "error") {
    return { tone: "loses", text: String(state.error || "deploy failed") };
  }
  if (state.state !== "done" || !state.result) return { tone: "", text: "" };
  const r = state.result;
  if (!r.selfTest) {
    return { tone: "loses", text: "The board loaded the model but its self-test failed." };
  }
  const parts = ["On the board"];
  if (typeof r.netMs === "number") parts.push(`${r.netMs.toFixed(1)} ms per inference`);
  if (typeof r.bytes === "number") parts.push(`${Math.round(r.bytes / 1024)} KB`);
  let text = parts.join(" · ") + ".";
  let tone = "beats";
  const f = r.frame;
  if (f && typeof f.ms === "number") {
    text += ` One of your frames, file to box: ${f.ms.toFixed(0)} ms.`;
    if (typeof f.iou === "number") {
      const agree = f.iou >= 0.7;
      text += agree
        ? ` Board and desktop agree (overlap ${f.iou.toFixed(2)}).`
        : ` Board and desktop disagree (overlap ${f.iou.toFixed(2)}).`;
      if (!agree) tone = "loses";
    }
  } else if (r.frameError) {
    text += ` Could not run a frame: ${r.frameError}.`;
    tone = "unsure";
  }
  return { tone, text };
}

/** Options for the port picker; ports that look like a P4's USB bridge come first. */
export function portOptions(ports) {
  if (!Array.isArray(ports)) return [];
  const rank = (p) => (/usbmodem|ttyACM/.test(p) ? 0 : 1);
  return ports
    .filter((p) => typeof p === "string" && p)
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map((p) => ({ value: p, label: p.replace(/^\/dev\/(cu\.)?/, "") }));
}

export const DOWNLOADS = {
  p4: { name: "tinydetector.p4.bin", label: "ESP32-P4 image" },
  pyz: { name: "edge.pyz", label: "Linux edge.pyz" },
  onnx: { name: "tinydetector.onnx", label: "ONNX" },
};

/** What to do with a downloaded file, shown after the download. */
export function howToUse(format) {
  if (format === "p4") {
    return "Write it with: esptool --chip esp32p4 -p PORT write-flash 0xB10000 tinydetector.p4.bin";
  }
  if (format === "pyz") {
    return "Copy it to the device and run: python3 edge.pyz --host 0.0.0.0  (then POST an image to /detect)";
  }
  return "int8 ONNX: runs with onnxruntime or tensor-factory's own runtime.";
}

const POLL_MS = 1000;
export const CALIB_FRAMES = 32;

export class Device {
  constructor({ onState, onDevices } = {}) {
    this.cb = { onState, onDevices };
    this._timer = null;
  }

  async refresh() {
    try {
      const d = await fetch("/devices").then((r) => r.json());
      this.cb.onDevices?.(d);
      return d;
    } catch {
      this.cb.onDevices?.({ ports: [], p4Tools: false, hint: "backend unreachable" });
      return null;
    }
  }

  /** Start a deploy to `port`, then poll until it finishes. */
  async deploy(port) {
    const r = await fetch(`/deploy?port=${encodeURIComponent(port)}`, { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      this.cb.onState?.({ state: "error", error: body.error || `HTTP ${r.status}` });
      return;
    }
    this.cb.onState?.(body);
    clearInterval(this._timer);
    this._timer = setInterval(async () => {
      try {
        const s = await fetch("/deploy").then((x) => x.json());
        this.cb.onState?.(s);
        if (s.state !== "running") clearInterval(this._timer);
      } catch {
        /* keep polling: a deploy outlives one dropped request */
      }
    }, POLL_MS);
  }

  /**
   * Hand a browser-trained model (ONNX bytes) to the backend so it can deploy it, with up
   * to `CALIB_FRAMES` labeled frames (image blobs): the P4 quantizer calibrates on them
   * and the first is tried on the board.
   */
  async uploadModel(bytes, frames = []) {
    const post = async (url, body) => {
      const r = await fetch(url, { method: "POST", body });
      if (!r.ok) {
        const payload = await r.json().catch(() => ({}));
        throw new Error(payload.error || `HTTP ${r.status}`);
      }
    };
    await post("/model", bytes);
    for (const [i, blob] of frames.slice(0, CALIB_FRAMES).entries()) await post(`/calib?i=${i}`, blob);
  }

  /** Fetch a model file; returns {blob, name} or throws with the backend's message. */
  async download(format) {
    const r = await fetch(`/model?format=${encodeURIComponent(format)}`);
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${r.status}`);
    }
    return { blob: await r.blob(), name: DOWNLOADS[format].name };
  }
}
