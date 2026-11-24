import assert from "node:assert/strict";
import { test } from "node:test";

import { configLine, FEW_BOXES, verdict } from "../js/verdict.js";

test("beating the constant predictor reads as finding the object", () => {
  const v = verdict({ err: 10, baseline: 25, valBoxes: 400 });
  assert.equal(v.tone, "beats");
  assert.match(v.text, /^Finding the object: 60% closer/);
  assert.doesNotMatch(v.text, /rough/);
});

test("not beating it is called out, even when the error looks small", () => {
  const v = verdict({ err: 4.1, baseline: 4.0, valBoxes: 400, config: { context: 4 } });
  assert.equal(v.tone, "loses");
  assert.match(v.text, /^Not finding the object yet/);
  assert.match(v.text, /too few labels/);
});

test("a tie is not a win", () => {
  assert.equal(verdict({ err: 20, baseline: 20, valBoxes: 100 }).tone, "loses");
});

test("with no context blocks the advice is to add them", () => {
  const v = verdict({ err: 27, baseline: 25, valBoxes: 400, config: { context: 0 } });
  assert.match(v.text, /--context 4/);
});

test("a small held-out set downgrades a win to unsure and says how small", () => {
  const v = verdict({ err: 5, baseline: 25, valBoxes: FEW_BOXES - 1 });
  assert.equal(v.tone, "unsure");
  assert.match(v.text, /only 29 held-out frames/);
  assert.match(verdict({ err: 5, baseline: 25, valBoxes: 1 }).text, /only 1 held-out frame,/);
  assert.equal(verdict({ err: 5, baseline: 25, valBoxes: FEW_BOXES }).tone, "beats");
});

test("an older backend that does not report the box count still gets a verdict", () => {
  assert.equal(verdict({ err: 5, baseline: 25 }).tone, "beats");
});

test("no numbers, no verdict", () => {
  for (const m of [undefined, null, {}, { err: null, baseline: 20 }, { err: 5, baseline: null }, "x"]) {
    assert.deepEqual(verdict(m), { tone: "", text: "" });
  }
});

test("a perfect score against a zero baseline does not divide by zero", () => {
  const v = verdict({ err: 0, baseline: 0, valBoxes: 100 });
  assert.equal(v.tone, "loses");
  assert.doesNotMatch(v.text, /NaN|Infinity/);
});

test("config line", () => {
  assert.equal(configLine({ size: 224, width: 16, context: 4, resize: "mixed" }), "224 px · 4 context · mixed resize");
  assert.equal(configLine({ size: 480 }), "480 px");
  assert.equal(configLine(undefined), "");
  assert.equal(configLine({}), "");
});
