// Runs in a real browser (see run.py): the whole browser path on WebGPU. PNG blobs are
// decoded by the browser, resized, trained on at the real model size, and the result is
// exported as ONNX. Leaves {report, onnx (base64), probe} on window.localReport; run.py
// saves the model and the probe frame so the Python runtime can be asked the same question.
import { LocalTrainer } from "../../js/local.js";
import { GpuEngine, requestDevice } from "../../js/nn/gpu.js";
import { rng } from "../../js/nn/ref.js";

const W = 320;
const H = 240;

async function frame(i, withBox = true) {
  const rand = rng(1000 + i);
  const canvas = new OffscreenCanvas(W, H);
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W, H);
  for (let j = 0; j < img.data.length; j += 4) {
    img.data[j] = rand() * 90;
    img.data[j + 1] = rand() * 90;
    img.data[j + 2] = rand() * 90;
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const s = 50 + Math.floor(rand() * 60); // the object changes size, which is what context blocks are for
  const x = Math.floor(rand() * (W - s));
  const y = Math.floor(rand() * (H - s));
  if (withBox) {
    ctx.fillStyle = "rgb(250, 240, 230)";
    ctx.fillRect(x, y, s, s);
  }
  return {
    frameId: i,
    blob: await canvas.convertToBlob({ type: "image/png" }),
    box: withBox ? [x / W, y / H, (x + s) / W, (y + s) / H] : null,
    present: withBox ? 1 : 0,
  };
}

async function main() {
  const gpu = await requestDevice();
  if (!gpu) return { error: "no WebGPU adapter" };
  const metrics = [];
  const errors = [];
  const t = new LocalTrainer({
    makeEngine: () => new GpuEngine(gpu.device, { size: 224 }),
    backend: `webgpu · ${gpu.name}`,
    epochs: 14,
    onMetrics: (m) => metrics.push(m),
    onError: (e) => errors.push(e),
  });
  await t.init();
  const samples = [];
  for (let i = 0; i < 96; i++) samples.push(await frame(i));
  for (let i = 0; i < 24; i++) samples.push(await frame(500 + i, false));
  const t0 = performance.now();
  await t.upsert(samples);
  await t.idle();
  const seconds = (performance.now() - t0) / 1000;
  const last = metrics.at(-1);
  const probe = await frame(9001);
  const empty = await frame(9002, false);
  const hit = await t.predict(0, probe.blob);
  const miss = await t.predict(0, empty.blob);
  const bytes = await t.modelBytes();
  const b64 = async (blob) => {
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = "";
    for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
    return btoa(s);
  };
  return {
    report: {
      adapter: gpu.name,
      seconds,
      epochs: metrics.filter((m) => m.phase === "training").length,
      err: last?.bestErr,
      baseline: last?.baseline,
      presenceAcc: last?.presenceAcc,
      valBoxes: last?.valBoxes,
      firstLoss: metrics[0]?.loss,
      lastLoss: last?.loss,
      errors,
      probeBox: hit.box,
      probeTruth: probe.box,
      probePresent: hit.present,
      emptyPresent: miss.present,
      onnxBytes: bytes?.length ?? 0,
    },
    onnx: bytes ? await b64(new Blob([bytes])) : null,
    probe: await b64(probe.blob),
  };
}

main()
  .catch((e) => ({ error: String(e?.stack || e) }))
  .then((r) => {
    window.localReport = r;
    document.getElementById("out").textContent = JSON.stringify(r.report || r, null, 2);
  });
