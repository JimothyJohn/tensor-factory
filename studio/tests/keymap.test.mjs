import assert from "node:assert/strict";
import { test } from "node:test";

import { KEYMAP } from "../js/keymap.js";

test("every binding has keys, an id, and a label", () => {
  for (const b of KEYMAP) {
    assert.ok(Array.isArray(b.keys) && b.keys.length > 0, `binding ${b.id} has keys`);
    assert.ok(typeof b.id === "string" && b.id, "binding has id");
    assert.ok(typeof b.label === "string" && b.label, `binding ${b.id} has label`);
  }
});

test("no physical key is bound to two different actions", () => {
  const seen = new Map();
  for (const b of KEYMAP) {
    for (const k of b.keys) {
      assert.ok(!seen.has(k), `key "${k}" bound to both ${seen.get(k)} and ${b.id}`);
      seen.set(k, b.id);
    }
  }
});

test("the core left-hand workflow keys are present", () => {
  const ids = new Set(KEYMAP.map((b) => b.id));
  for (const id of ["prev", "next", "accept", "commit", "skip", "negative"]) {
    assert.ok(ids.has(id), `missing core binding: ${id}`);
  }
  // A/D navigation and Space accept are the muscle-memory anchors
  const byKey = new Map(KEYMAP.flatMap((b) => b.keys.map((k) => [k, b.id])));
  assert.equal(byKey.get("a"), "prev");
  assert.equal(byKey.get("d"), "next");
  assert.equal(byKey.get(" "), "accept");
});
