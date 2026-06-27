import assert from "node:assert/strict";
import { test } from "node:test";

import { mediaErrorMessage } from "../js/video.js";

test("maps MediaError codes to actionable messages", () => {
  assert.match(mediaErrorMessage({ error: { code: 3 } }), /decoded|codec/i);
  assert.match(mediaErrorMessage({ error: { code: 4 } }), /not supported|format/i);
  assert.match(mediaErrorMessage({ error: { code: 2 } }), /network/i);
  assert.match(mediaErrorMessage({ error: { code: 1 } }), /abort/i);
});

test("falls back to a generic message for unknown/missing error", () => {
  assert.match(mediaErrorMessage({ error: null }), /could not open/i);
  assert.match(mediaErrorMessage({ error: { code: 99 } }), /could not open/i);
  assert.match(mediaErrorMessage({}), /could not open/i);
});
