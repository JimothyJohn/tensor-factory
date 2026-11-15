import assert from "node:assert/strict";
import { test } from "node:test";

import { sourceRows } from "../js/sources.js";

test("one row per source, formatted for the table", () => {
  const rows = sourceRows(
    [
      { name: "studio", train: 40, val: 10, err: 12.34, presenceAcc: 0.96, classAcc: null },
      { name: "real_ds", train: 88, val: 22, err: 31.0, presenceAcc: 1, classAcc: null },
    ],
    20,
  );
  assert.deepEqual(rows[0].cells, ["studio", "40", "10", "12.3px", "96%"]);
  assert.deepEqual(rows[1].cells, ["real_ds", "88", "22", "31.0px", "100%"]);
});

test("a source is toned against the constant-predictor floor", () => {
  const [good, bad] = sourceRows(
    [
      { name: "a", train: 1, val: 1, err: 5, presenceAcc: null },
      { name: "b", train: 1, val: 1, err: 50, presenceAcc: null },
    ],
    20,
  );
  assert.equal(good.tone, "beats");
  assert.equal(bad.tone, "loses");
});

test("no held-out box shows a dash, never a perfect 0", () => {
  const [neg] = sourceRows([{ name: "negatives", train: 60, val: 15, err: null, presenceAcc: 0.8 }], 20);
  assert.equal(neg.cells[3], "—");
  assert.equal(neg.tone, "");
});

test("an error of exactly 0 is a real score, not a missing one", () => {
  const [row] = sourceRows([{ name: "mock", train: 4, val: 1, err: 0, presenceAcc: null }], 20);
  assert.equal(row.cells[3], "0.0px");
  assert.equal(row.tone, "beats");
});

test("missing or malformed payloads render nothing instead of throwing", () => {
  assert.deepEqual(sourceRows(undefined, 20), []);
  assert.deepEqual(sourceRows(null, null), []);
  assert.deepEqual(sourceRows("nope", 20), []);
  const [row] = sourceRows([{}], undefined);
  assert.deepEqual(row.cells, ["?", "0", "0", "—", "—"]);
});

test("names stay plain strings (the table writes them as text)", () => {
  const [row] = sourceRows([{ name: "<img src=x onerror=alert(1)>", train: 1, val: 0 }], null);
  assert.equal(row.cells[0], "<img src=x onerror=alert(1)>");
});
