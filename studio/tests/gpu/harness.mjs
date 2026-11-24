// Runs in a real browser (see run.py): the WebGPU trainer against the CPU reference on the
// torch fixture. Leaves a JSON report on window.report.
import { GpuTrainer, requestDevice } from "../../js/nn/gpu.js";
import { backward, forward, loss, tensor, trainStep } from "../../js/nn/ref.js";

function worst(actual, expected) {
  let w = 0;
  for (let i = 0; i < expected.length; i++) {
    const err = Math.abs(actual[i] - expected[i]) / Math.max(1, Math.abs(expected[i]));
    if (!(err <= w)) w = err; // also catches NaN
  }
  return actual.length === expected.length ? w : Infinity;
}

async function main() {
  // `loose` holds float32 results after several chained steps: same check, wider tolerance.
  const report = { checks: {}, loose: {} };
  const gpu = await requestDevice();
  if (!gpu) return { ...report, error: "no WebGPU adapter" };
  report.adapter = gpu.name;
  gpu.device.addEventListener?.("uncapturederror", (e) => {
    report.gpuError = String(e.error?.message || e.error);
  });
  const fx = await fetch("../fixtures/torch_step.json").then((r) => r.json());
  const { size, batch, steps, boxWeight, presenceWeight } = fx.config;
  const load = () =>
    Object.fromEntries(Object.entries(fx.params).map(([k, v]) => [k, tensor(v.shape, v.data)]));
  const x = tensor([batch, 3, size, size]);
  for (let i = 0; i < x.data.length; i++) x.data[i] = ((i * 7919) % 1000) / 1000;
  const boxes = [0.1, 0.2, 0.6, 0.7, 0, 0, 0, 0, 0.35, 0.05, 0.95, 0.55];
  const hasBox = [1, 0, 1];
  const opts = { size, boxWeight, presenceWeight };

  // 1. gradients of one step, no update
  {
    const t = new GpuTrainer(gpu.device, load(), opts);
    const P = t.trainStep(x.data, boxes, hasBox, { update: false });
    const got = await t.grads();
    for (const [name, want] of Object.entries(fx.grads)) {
      report.checks[`grad ${name}`] = worst(got[name], want.data);
    }
    report.checks["train box"] = worst(await t.read(P.box, batch * 4), fx.train.box.data);
    report.checks["train presence"] = worst(await t.read(P.logit, batch), fx.train.presence.data);
    const perSample = await t.read(P.lossOut, batch);
    report.checks["loss"] = worst([perSample.reduce((a, v) => a + v, 0)], [fx.train.loss]);
    const after = await t.params();
    for (const [name, want] of Object.entries(fx.running)) {
      report.checks[name] = worst(after[name].data, want.data);
    }
  }

  // 2. three Adam steps, then eval mode
  {
    const t = new GpuTrainer(gpu.device, load(), opts);
    for (let s = 0; s < steps; s++) t.trainStep(x.data, boxes, hasBox);
    const after = await t.params();
    // A bias that feeds a batch norm, and the heatmap bias under a softmax, have a gradient
    // that is zero in exact arithmetic: what each runtime computes is rounding noise, Adam
    // turns the noise into full-size steps of either sign, and the function does not
    // change. Those (and the running means that follow them) are not compared; the
    // eval-mode outputs below are, and they depend on everything that matters.
    const noise = (name) => /\.0\.bias$|^heat\.bias$|running_mean$/.test(name);
    for (const [name, want] of Object.entries(fx.after.params)) {
      if (!noise(name)) report.checks[`after ${name}`] = worst(after[name].data, want.data);
    }
    const out = await t.predict(x.data, batch);
    report.loose["eval box"] = worst(out.box, fx.after.eval.box.data);
    report.loose["eval presence"] = worst(out.presence, fx.after.eval.presence.data);
  }

  // 3. a different batch size reuses the same weights (the last batch of an epoch is short)
  {
    const params = load();
    const t = new GpuTrainer(gpu.device, params, opts);
    const one = tensor([1, 3, size, size], x.data.slice(0, 3 * size * size));
    const out = await t.predict(one.data, 1);
    const want = forward(params, one, { train: false });
    report.checks["batch of 1 box"] = worst(out.box, want.box.data);
    report.checks["batch of 1 presence"] = worst(out.presence, want.presence.data);
  }

  // 4. an all-negative batch: no box gradient, presence still learns
  {
    const params = load();
    const t = new GpuTrainer(gpu.device, params, opts);
    t.trainStep(x.data, boxes, [0, 0, 0], { update: false });
    const got = await t.grads();
    const out = forward(params, x, { train: true });
    const l = loss(out, tensor([3, 4], boxes), [0, 0, 0], { boxWeight, presenceWeight });
    const want = backward(params, out, l.dBox, l.dPresence);
    report.checks["negatives grad heat.weight"] = worst(got["heat.weight"], want["heat.weight"].data);
    report.checks["negatives grad presence"] = worst(
      got["presence_head.weight"],
      want["presence_head.weight"].data,
    );
    report.checks["negatives grad conv0"] = worst(
      got["features.0.0.weight"],
      want["features.0.0.weight"].data,
    );
  }

  // 5. speed at the real size: 224 px, width 16, context 4, batch 16
  {
    const { initParams } = await import("../../js/nn/ref.js");
    const t = new GpuTrainer(gpu.device, initParams({ width: 16, context: 4, seed: 1 }), { size: 224 });
    const n = 16;
    const big = new Float32Array(n * 3 * 224 * 224).map((_, i) => ((i * 31) % 255) / 255);
    const tb = new Array(n * 4).fill(0).map((_, i) => [0.2, 0.3, 0.7, 0.8][i % 4]);
    const hb = new Array(n).fill(1);
    t.trainStep(big, tb, hb);
    await t.read(t.buf["log_gain"], 1);
    const t0 = performance.now();
    for (let s = 0; s < 10; s++) t.trainStep(big, tb, hb);
    const lg = await t.read(t.buf["log_gain"], 1);
    report.stepMs = (performance.now() - t0) / 10;
    report.finite = Number.isFinite(lg[0]);
  }
  // keep ref's trainStep import honest: the reference must still run here
  trainStep(load(), {}, x, tensor([3, 4], boxes), hasBox, { boxWeight, presenceWeight });
  return report;
}

main()
  .catch((e) => ({ error: String(e?.stack || e) }))
  .then((report) => {
    window.report = report;
    document.getElementById("out").textContent = JSON.stringify(report, null, 2);
  });
