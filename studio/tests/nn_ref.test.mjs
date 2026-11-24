// The JavaScript reference trainer against torch. `fixtures/torch_step.json` is what
// TinyDetector + autograd + Adam produced in float64 (see make_torch_fixture.py); these
// tests fail if any output, gradient or update differs.
import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";

import {
  adamStep,
  backward,
  configOf,
  conv2d,
  conv2dBackward,
  forward,
  initParams,
  loss,
  plan,
  tensor,
  trainable,
  trainStep,
} from "../js/nn/ref.js";

const fx = JSON.parse(fs.readFileSync(new URL("./fixtures/torch_step.json", import.meta.url)));
const { size, batch, steps, boxWeight, presenceWeight } = fx.config;

function load(dict) {
  return Object.fromEntries(Object.entries(dict).map(([k, v]) => [k, tensor(v.shape, v.data)]));
}

function batchInputs() {
  const n = batch * 3 * size * size;
  const x = tensor([batch, 3, size, size]);
  for (let i = 0; i < n; i++) x.data[i] = ((i * 7919) % 1000) / 1000;
  const boxes = tensor([3, 4], [0.1, 0.2, 0.6, 0.7, 0, 0, 0, 0, 0.35, 0.05, 0.95, 0.55]);
  return { x, boxes, hasBox: [1, 0, 1] };
}

function close(actual, expected, tol, what) {
  assert.equal(actual.length, expected.length, `${what}: length`);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    const scale = Math.max(1, Math.abs(expected[i]));
    worst = Math.max(worst, Math.abs(actual[i] - expected[i]) / scale);
  }
  assert.ok(worst <= tol, `${what}: worst error ${worst.toExponential(2)} > ${tol}`);
}

test("training-mode forward and loss match torch", () => {
  const { x, boxes, hasBox } = batchInputs();
  const out = forward(load(fx.params), x, { train: true });
  close(out.box.data, fx.train.box.data, 1e-10, "box");
  close(out.presence.data, fx.train.presence.data, 1e-10, "presence");
  const l = loss(out, boxes, hasBox, { boxWeight, presenceWeight });
  close([l.value], [fx.train.loss], 1e-10, "loss");
  for (const [name, want] of Object.entries(fx.running)) {
    close(out.running[name].data, want.data, 1e-10, name);
  }
});

test("every gradient matches torch autograd", () => {
  const { x, boxes, hasBox } = batchInputs();
  const params = load(fx.params);
  const out = forward(params, x, { train: true });
  const l = loss(out, boxes, hasBox, { boxWeight, presenceWeight });
  const grads = backward(params, out, l.dBox, l.dPresence);
  assert.deepEqual(Object.keys(grads).sort(), Object.keys(fx.grads).sort());
  assert.deepEqual(Object.keys(grads).sort(), trainable(params).sort());
  for (const [name, want] of Object.entries(fx.grads)) {
    assert.deepEqual(grads[name].shape, want.shape, name);
    close(grads[name].data, want.data, 1e-9, `grad ${name}`);
  }
});

test("Adam steps land on torch's parameters, and eval mode on torch's outputs", () => {
  const { x, boxes, hasBox } = batchInputs();
  const params = load(fx.params);
  const state = {};
  const losses = [];
  for (let s = 0; s < steps; s++) {
    losses.push(trainStep(params, state, x, boxes, hasBox, { boxWeight, presenceWeight }).loss);
  }
  close(losses, fx.after.losses, 1e-9, "losses");
  for (const [name, want] of Object.entries(fx.after.params)) {
    close(params[name].data, want.data, 1e-8, `param ${name}`);
  }
  const out = forward(params, x, { train: false });
  close(out.box.data, fx.after.eval.box.data, 1e-8, "eval box");
  close(out.presence.data, fx.after.eval.presence.data, 1e-8, "eval presence");
});

test("a batch with no positives has no box loss and still trains presence", () => {
  const { x, boxes } = batchInputs();
  const params = load(fx.params);
  const out = forward(params, x, { train: true });
  const l = loss(out, boxes, [0, 0, 0], { boxWeight, presenceWeight });
  assert.equal(l.boxLoss, 0);
  assert.ok(l.dBox.data.every((v) => v === 0));
  assert.ok(Number.isFinite(l.value) && l.value > 0);
  const grads = backward(params, out, l.dBox, l.dPresence);
  assert.ok(grads["heat.weight"].data.every((v) => v === 0));
  assert.ok(grads["presence_head.weight"].data.some((v) => v !== 0));
});

test("conv gradients agree with finite differences", () => {
  const x = tensor([2, 2, 5, 4]);
  const w = tensor([3, 2, 3, 3]);
  const b = tensor([3]);
  let s = 1;
  for (const t of [x, w, b]) for (let i = 0; i < t.data.length; i++) t.data[i] = Math.sin(s++ * 1.3);
  for (const stride of [1, 2]) {
    const y = conv2d(x, w, b, stride, 1);
    const dy = tensor(y.shape, y.data.map((_, i) => Math.cos(i)));
    const { dx, dw, db } = conv2dBackward(x, w, dy, stride, 1);
    const f = () => conv2d(x, w, b, stride, 1).data.reduce((a, v, i) => a + v * dy.data[i], 0);
    for (const [t, g] of [[x, dx], [w, dw], [b, db]]) {
      for (const i of [0, 1, t.data.length - 1]) {
        const keep = t.data[i];
        t.data[i] = keep + 1e-6;
        const hi = f();
        t.data[i] = keep - 1e-6;
        const lo = f();
        t.data[i] = keep;
        assert.ok(Math.abs((hi - lo) / 2e-6 - g.data[i]) < 1e-6, `stride ${stride} index ${i}`);
      }
    }
  }
});

test("fresh parameters have torch's shapes and a reproducible seed", () => {
  const a = initParams({ width: 4, context: 1, seed: 3 });
  const b = initParams({ width: 4, context: 1, seed: 3 });
  const c = initParams({ width: 4, context: 1, seed: 4 });
  for (const [name, want] of Object.entries(fx.params)) assert.deepEqual(a[name].shape, want.shape, name);
  assert.deepEqual(Object.keys(a).sort(), Object.keys(fx.params).sort());
  assert.deepEqual(a["heat.weight"].data, b["heat.weight"].data);
  assert.notDeepEqual(a["heat.weight"].data, c["heat.weight"].data);
  assert.deepEqual(configOf(a), { width: 4, context: 1 });
  const bound = 1 / Math.sqrt(3 * 9);
  assert.ok(a["features.0.0.weight"].data.every((v) => Math.abs(v) <= bound));
});

test("bad shapes and configs are refused", () => {
  assert.throws(() => plan(0, 0), /width/);
  assert.throws(() => plan(4, -1), /context/);
  assert.throws(() => plan(4, 1.5), /context/);
  assert.throws(() => tensor([2, 2], [1, 2, 3]), /3 values/);
  assert.throws(() => configOf({}), /not a detector/);
  assert.equal(plan(16, 4).length, 8);
  assert.deepEqual(plan(16, 4).map((b) => b.stride), [2, 2, 2, 2, 1, 1, 1, 1]);
});

test("Adam leaves parameters without a gradient alone", () => {
  const params = { a: tensor([2], [1, 2]), b: tensor([1], [5]) };
  adamStep(params, { a: tensor([2], [0.5, -0.5]) }, {});
  assert.deepEqual(Array.from(params.b.data), [5]);
  assert.ok(params.a.data[0] < 1 && params.a.data[1] > 2);
});
