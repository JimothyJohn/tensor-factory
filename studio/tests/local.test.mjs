// The browser training loop end to end, on the CPU reference engine with a tiny model:
// it has to learn to find a bright square, report the same metrics the backend does, keep
// the best model, and export an ONNX file.
import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";

import { LocalTrainer } from "../js/local.js";
import { RefEngine } from "../js/nn/ref-engine.js";
import { rng } from "../js/nn/ref.js";

const SIZE = 32;

// A "blob" here is already the decoded image; decode just hands it back.
function frame(i, { box = true } = {}) {
  const rand = rng(100 + i);
  const pixels = new Uint8Array(SIZE * SIZE * 3);
  for (let j = 0; j < pixels.length; j++) pixels[j] = Math.floor(rand() * 60);
  const x = 2 + ((i * 5) % 16);
  const y = 4 + ((i * 3) % 14);
  if (box) {
    for (let yy = y; yy < y + 10; yy++) {
      for (let xx = x; xx < x + 10; xx++) pixels.fill(255, (yy * SIZE + xx) * 3, (yy * SIZE + xx) * 3 + 3);
    }
  }
  return {
    frameId: i,
    blob: { pixels, width: SIZE, height: SIZE, channels: 3 },
    box: box ? [x / SIZE, y / SIZE, (x + 10) / SIZE, (y + 10) / SIZE] : null,
    present: box ? 1 : 0,
  };
}

function trainer(overrides = {}) {
  const metrics = [];
  const statuses = [];
  const errors = [];
  const t = new LocalTrainer({
    makeEngine: () => new RefEngine({ size: SIZE }),
    backend: "cpu-test",
    decode: async (blob) => blob,
    loadTemplate: async (name) => new Uint8Array(fs.readFileSync(new URL(`../models/${name}`, import.meta.url))),
    size: SIZE,
    width: 4,
    context: 1,
    epochs: 12,
    batch: 8,
    resize: "mixed",
    onMetrics: (m) => metrics.push(m),
    onStatus: (s) => statuses.push(s),
    onError: (e) => errors.push(e),
    ...overrides,
  });
  return { t, metrics, statuses, errors };
}

const dataset = () => [
  ...Array.from({ length: 24 }, (_, i) => frame(i)),
  ...Array.from({ length: 8 }, (_, i) => frame(200 + i, { box: false })),
];

test("a round trains, reports the backend's metrics, and serves a model that finds the square", async () => {
  const { t, metrics, errors } = trainer();
  await t.init();
  await t.upsert(dataset());
  await t.idle();
  assert.deepEqual(errors, []);
  const training = metrics.filter((m) => m.phase === "training");
  assert.equal(training.length, 12);
  const m = metrics.at(-1);
  assert.equal(m.phase, "idle");
  for (const key of [
    "epoch", "epochs", "loss", "err", "baseline", "presenceAcc", "gain", "trainCount", "valCount",
    "valBoxes", "sources", "regressed", "suspects", "backend", "running", "bestErr", "history",
    "hasModel", "config",
  ]) {
    assert.ok(key in m, `metrics lack ${key}`);
  }
  assert.equal(m.trainCount + m.valCount, 32);
  assert.equal(m.valBoxes, 5); // 20% of 24 positives
  assert.deepEqual(m.config, { size: SIZE, width: 4, context: 1, resize: "mixed" });
  assert.equal(m.hasModel, true);
  assert.ok(training[0].loss > training.at(-1).loss, "loss did not fall");
  assert.ok(m.bestErr < m.baseline, `not localizing: ${m.bestErr} vs baseline ${m.baseline}`);
  assert.ok(Number.isFinite(m.gain) && m.gain > 0);
  JSON.stringify(m);

  const hit = await t.predict(999, frame(7).blob);
  assert.equal(hit.ready, true);
  assert.equal(hit.present, true);
  const want = frame(7).box;
  const off = Math.hypot((hit.box[0] + hit.box[2] - want[0] - want[2]) / 2, (hit.box[1] + hit.box[3] - want[1] - want[3]) / 2);
  assert.ok(off * SIZE < m.baseline, `prediction is ${off * SIZE}px off`);
});

test("the served model exports as an ONNX file that declares the trained size", async () => {
  const { t } = trainer({ epochs: 2 });
  await t.init();
  assert.equal(await t.modelBytes(), null);
  await t.upsert(dataset());
  await t.idle();
  const bytes = await t.modelBytes();
  assert.ok(bytes instanceof Uint8Array && bytes.length > 5000);
  const text = Buffer.from(bytes).toString("latin1");
  for (const name of ["image", "box", "presence", "heat.weight_quantized", "log_gain"]) {
    assert.ok(text.includes(name), `export lacks ${name}`);
  }
});

test("too few positives waits instead of training", async () => {
  const { t, metrics, statuses } = trainer();
  await t.init();
  await t.upsert([frame(1), frame(2), frame(3), frame(300, { box: false })]);
  await t.idle();
  assert.equal(metrics.length, 0);
  assert.match(statuses.at(-1), /need ≥4 positives \(3 now\)/);
  assert.deepEqual(await t.predict(1, frame(1).blob), {
    ready: false, present: false, score: 0, box: null, cls: null, clsName: null, clsScore: null,
  });
});

test("a worse round does not replace the served model, and names the newest frames", async () => {
  const { t, metrics } = trainer({ epochs: 6 });
  await t.init();
  await t.upsert(dataset());
  await t.idle();
  const before = metrics.at(-1).bestErr;
  const served = t.served;
  // mislabel a batch of new frames: boxes nowhere near the square
  const bad = Array.from({ length: 16 }, (_, i) => ({ ...frame(500 + i), box: [0.8, 0.8, 0.95, 0.95] }));
  await t.upsert(bad);
  await t.idle();
  const after = metrics.at(-1);
  if (after.regressed) {
    assert.equal(t.served, served);
    assert.equal(after.bestErr, before);
    assert.ok(after.suspects.length > 0 && after.suspects.length <= 5);
    assert.ok(after.suspects.every((s) => s.frameId >= 500));
  } else {
    assert.ok(after.bestErr < before); // it may honestly improve; then it must be served
    assert.notEqual(t.served, served);
  }
});

test("re-labeling a frame replaces it instead of adding a second copy", async () => {
  const { t } = trainer({ epochs: 1 });
  await t.init();
  await t.upsert([frame(1)]);
  await t.upsert([{ ...frame(1), box: null, present: 0 }]);
  assert.deepEqual(t.counts(), { positives: 0, negatives: 1, total: 1 });
});

test("reset forgets the data and the model", async () => {
  const { t } = trainer({ epochs: 2 });
  await t.init();
  await t.upsert(dataset());
  await t.idle();
  await t.reset();
  assert.deepEqual(t.counts(), { positives: 0, negatives: 0, total: 0 });
  assert.equal(t.served, null);
  assert.equal((await t.predict(1, frame(1).blob)).ready, false);
});

test("pause holds a round until resume", async () => {
  const { t, metrics } = trainer({ epochs: 2 });
  await t.init();
  await t.pause();
  await t.upsert(dataset());
  await t.idle();
  assert.equal(metrics.length, 0);
  await t.resume();
  await t.idle();
  assert.equal(metrics.at(-1).phase, "idle");
});

test("hostile samples are reported and skipped, never trained on", async () => {
  const { t, errors } = trainer({
    decode: async (blob) => {
      if (blob === "garbage") throw new Error("not an image");
      return blob;
    },
  });
  await t.init();
  await t.upsert([
    { frameId: 1, blob: "garbage", box: [0.1, 0.1, 0.5, 0.5], present: 1 },
    { ...frame(2), box: [0.5, 0.5, 0.2, 0.2] }, // inverted
    { ...frame(3), box: [0, 0, 2, 2] }, // outside the image
    { ...frame(4), box: [NaN, 0, 1, 1] },
    { ...frame(5), blob: { pixels: new Uint8Array(5), width: SIZE, height: SIZE, channels: 3 } },
  ]);
  await t.idle();
  assert.equal(errors.length, 5);
  assert.equal(t.counts().total, 0);
});

test("unsupported settings are refused up front", () => {
  assert.throws(() => trainer({ resize: "bicubic" }), /resize/);
  assert.throws(() => trainer({ context: 9 }), /context/);
  assert.throws(() => trainer({ context: -1 }), /context/);
});

test("more than one class is said to need the backend", async () => {
  const { t, errors } = trainer();
  await t.init();
  assert.deepEqual(await t.setClasses(["a", "b"]), ["a", "b"]);
  assert.match(errors[0], /one kind of object/);
});
