import assert from "node:assert/strict";
import { test } from "node:test";

import {
  area,
  clamp01,
  decodeUint8,
  encodeUint8,
  maxErrorPx,
  orderClamp,
  toCocoBbox,
} from "../js/codec.js";

test("clamp01 bounds to [0,1]", () => {
  assert.equal(clamp01(-0.5), 0);
  assert.equal(clamp01(1.5), 1);
  assert.equal(clamp01(0.3), 0.3);
});

test("orderClamp repairs reversed and out-of-range coords", () => {
  assert.deepEqual(orderClamp({ x1: 0.9, y1: 1.4, x2: 0.3, y2: -0.2 }), {
    x1: 0.3,
    y1: 0,
    x2: 0.9,
    y2: 1,
  });
});

test("encode/decode round-trips under 1px at 480 for random boxes", () => {
  // deterministic LCG so the fuzz is reproducible
  let s = 12345;
  const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  for (let i = 0; i < 2000; i++) {
    const box = orderClamp({ x1: rnd(), y1: rnd(), x2: rnd(), y2: rnd() });
    const back = decodeUint8(encodeUint8(box));
    for (const k of ["x1", "y1", "x2", "y2"]) {
      assert.ok(Math.abs(back[k] - box[k]) * 480 < 1, `${k} drift too large`);
    }
  }
});

test("encodeUint8 emits four bytes in 0..255", () => {
  const u = encodeUint8({ x1: 0, y1: 0.25, x2: 1, y2: 0.5 });
  assert.equal(u.length, 4);
  for (const v of u) assert.ok(Number.isInteger(v) && v >= 0 && v <= 255);
  assert.deepEqual(u, [0, 64, 255, 128]);
});

test("encodeUint8 order-clamps a reversed box before encoding", () => {
  // y reversed (0.5 > 0.25) -> orderClamp swaps -> same as the ordered box above
  assert.deepEqual(encodeUint8({ x1: 0, y1: 0.5, x2: 1, y2: 0.25 }), [0, 64, 255, 128]);
});

test("decodeUint8 repairs order and clamps", () => {
  const b = decodeUint8([255, 255, 0, 0]); // reversed
  assert.ok(b.x1 <= b.x2 && b.y1 <= b.y2);
});

test("toCocoBbox converts normalized xyxy to pixel xywh", () => {
  assert.deepEqual(toCocoBbox({ x1: 0.25, y1: 0.5, x2: 0.75, y2: 1 }, 480, 480), [120, 240, 240, 240]);
});

test("maxErrorPx under budget at 480", () => {
  assert.ok(maxErrorPx(480) < 1);
});

test("area is non-negative and zero for degenerate", () => {
  assert.equal(area({ x1: 0.2, y1: 0.2, x2: 0.1, y2: 0.5 }), 0);
  assert.ok(Math.abs(area({ x1: 0, y1: 0, x2: 0.5, y2: 0.5 }) - 0.25) < 1e-9);
});
