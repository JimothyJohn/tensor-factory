import assert from "node:assert/strict";
import { test } from "node:test";

import { hamming, isNovel } from "../js/dhash.js";

test("hamming counts differing bits", () => {
  assert.equal(hamming(0n, 0n), 0);
  assert.equal(hamming(0b1011n, 0b1110n), 2);
  assert.equal(hamming(0n, (1n << 64n) - 1n), 64);
});

test("hamming is symmetric and zero only for equal hashes", () => {
  const a = 0xdeadbeefn;
  const b = 0x1234abcdn;
  assert.equal(hamming(a, b), hamming(b, a));
  assert.equal(hamming(a, a), 0);
});

test("isNovel: empty known set is always novel", () => {
  assert.equal(isNovel(123n, [], 6), true);
});

test("isNovel: rejects when within threshold of any known hash", () => {
  const known = [0b0000n, 0b1111n];
  assert.equal(isNovel(0b0001n, known, 1), false); // 1 bit from 0000, not > 1
  assert.equal(isNovel(0b0011n, known, 1), true); // 2 from 0000, 2 from 1111 -> >1 both
});

test("isNovel: threshold 0 keeps anything not bit-identical", () => {
  assert.equal(isNovel(5n, [5n], 0), false);
  assert.equal(isNovel(5n, [4n], 0), true);
});
