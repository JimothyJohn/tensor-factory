import assert from "node:assert/strict";
import { test } from "node:test";

import {
  baselineCenterErr,
  checkpointScore,
  median,
  resizeBilinear,
  resizeNearest,
  score,
  shuffled,
  split,
  writeChw,
} from "../js/nn/data.js";
import { rng } from "../js/nn/ref.js";

const items = (pos, neg) => [
  ...Array.from({ length: pos }, (_, i) => ({ id: i, box: [0.1, 0.1, 0.5, 0.5] })),
  ...Array.from({ length: neg }, (_, i) => ({ id: 1000 + i, box: null })),
];

test("split holds out a share of positives and of negatives, and loses nothing", () => {
  const { train, val } = split(items(40, 10), 0.2, 0);
  assert.equal(val.filter((s) => s.box).length, 8);
  assert.equal(val.filter((s) => !s.box).length, 2);
  assert.equal(train.length + val.length, 50);
  const ids = new Set([...train, ...val].map((s) => s.id));
  assert.equal(ids.size, 50);
});

test("split does not depend on arrival order, and changes with the seed", () => {
  const a = split(items(30, 6), 0.2, 0);
  const b = split(items(30, 6).reverse(), 0.2, 0);
  const c = split(items(30, 6), 0.2, 1);
  const ids = (s) => s.val.map((x) => x.id).sort((x, y) => x - y);
  assert.deepEqual(ids(a), ids(b));
  assert.notDeepEqual(ids(a), ids(c));
});

test("split keeps a lone sample for training and holds nothing out at fraction 0", () => {
  assert.equal(split(items(1, 1), 0.2).val.length, 0);
  assert.equal(split(items(10, 10), 0).val.length, 0);
  assert.deepEqual(split([], 0.2), { train: [], val: [] });
  assert.equal(split(items(2, 0), 0.2).val.length, 1); // two is the least that can be scored
});

test("median", () => {
  assert.equal(median([]), null);
  assert.equal(median([3]), 3);
  assert.equal(median([4, 1, 3]), 3);
  assert.equal(median([4, 1, 3, 2]), 2.5);
});

test("baseline is the error of always guessing the mean training box", () => {
  const train = [{ box: [0, 0, 0.2, 0.2] }, { box: [0.8, 0.8, 1, 1] }, { box: null }];
  const val = [{ box: [0.4, 0.4, 0.6, 0.6] }, { box: [0.4, 0.0, 0.6, 0.2] }, { box: null }];
  // mean train center (0.5, 0.5); val centers (0.5, 0.5) and (0.5, 0.1): errors 0 and 0.4
  assert.ok(Math.abs(baselineCenterErr(train, val, 100) - 20) < 1e-9);
  assert.equal(baselineCenterErr([{ box: null }], val, 100), null);
  assert.equal(baselineCenterErr(train, [{ box: null }], 100), null);
});

test("score: center error over boxed frames, presence over all of them", () => {
  const val = [{ box: [0.2, 0.2, 0.4, 0.4] }, { box: [0.5, 0.5, 0.7, 0.7] }, { box: null }, { box: null }];
  const preds = [
    { box: [0.2, 0.2, 0.4, 0.4], presence: 3 }, // exact, right
    { box: [0.5, 0.6, 0.7, 0.8], presence: -1 }, // 0.1 off in y, called absent: wrong
    { box: [0, 0, 1, 1], presence: -2 }, // negative, right; its box is ignored
    { box: [0, 0, 1, 1], presence: 0.5 }, // negative called present: wrong
  ];
  const s = score(val, preds, 200);
  assert.ok(Math.abs(s.err - 10) < 1e-9);
  assert.equal(s.presenceAcc, 0.5);
  assert.equal(s.valBoxes, 2);
  assert.deepEqual(score([], [], 200), { err: null, presenceAcc: null, valBoxes: 0 });
});

test("checkpoint score: presence leads, a closer box breaks the tie", () => {
  const size = 224;
  assert.ok(checkpointScore({ err: 50, presenceAcc: 1 }, size) > checkpointScore({ err: 1, presenceAcc: 0.5 }, size));
  assert.ok(checkpointScore({ err: 5, presenceAcc: 1 }, size) > checkpointScore({ err: 9, presenceAcc: 1 }, size));
  assert.equal(checkpointScore({ err: null, presenceAcc: null }, size), 0);
});

test("shuffled is a permutation and is reproducible", () => {
  const src = [1, 2, 3, 4, 5, 6, 7];
  const a = shuffled(src, rng(5));
  assert.deepEqual(a, shuffled(src, rng(5)));
  assert.deepEqual(a.slice().sort(), src);
  assert.deepEqual(src, [1, 2, 3, 4, 5, 6, 7]);
});

test("resize takes canvas RGBA and drops alpha", () => {
  const rgba = new Uint8Array(4 * 4 * 4);
  for (let i = 0; i < 16; i++) rgba.set([10, 20, 30, 7], i * 4);
  for (const fn of [resizeBilinear, resizeNearest]) {
    const out = fn(rgba, 4, 4, 2, 4);
    assert.deepEqual(Array.from(out), [10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]);
  }
});

test("resize refuses a buffer that is not the stated image", () => {
  assert.throws(() => resizeBilinear(new Uint8Array(10), 4, 4, 2), /not a 4x4x3 image/);
  assert.throws(() => resizeNearest(new Uint8Array(48), 0, 4, 2), /not a 0x4x3 image/);
});

test("writeChw lays packed RGB out as planes in [0, 1]", () => {
  const dst = new Float32Array(2 * 3 * 4);
  writeChw(dst, 1, Uint8Array.from([255, 0, 51, 0, 255, 51, 0, 0, 51, 255, 255, 51]), 2);
  assert.ok(dst.slice(0, 12).every((v) => v === 0));
  assert.deepEqual(Array.from(dst.slice(12, 16)), [1, 0, 0, 1]); // R plane
  assert.deepEqual(Array.from(dst.slice(16, 20)), [0, 1, 0, 1]); // G plane
  assert.ok(dst.slice(20).every((v) => Math.abs(v - 0.2) < 1e-6)); // B plane
});
