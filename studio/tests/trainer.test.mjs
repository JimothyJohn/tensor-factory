import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { Trainer } from "../js/trainer.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function stubFetch(impl) {
  const calls = [];
  globalThis.fetch = async (url, opts) => {
    calls.push({ url, opts });
    return impl(url, opts);
  };
  return calls;
}

test("upsert posts a positive with id, present=1, and box query + blob body", async () => {
  const calls = stubFetch(async () => ({ ok: true, json: async () => ({}) }));
  const t = new Trainer({});
  await t.upsert([{ frameId: 7, blob: "BLOB", box: [0.1, 0.2, 0.3, 0.4], present: 1 }]);
  assert.equal(calls.length, 1);
  const u = new URL("http://x" + calls[0].url);
  assert.equal(u.searchParams.get("id"), "7");
  assert.equal(u.searchParams.get("present"), "1");
  assert.equal(u.searchParams.get("box"), "0.1,0.2,0.3,0.4");
  assert.equal(calls[0].opts.method, "POST");
  assert.equal(calls[0].opts.body, "BLOB");
});

test("upsert posts a negative with present=0 and no box", async () => {
  const calls = stubFetch(async () => ({ ok: true, json: async () => ({}) }));
  const t = new Trainer({});
  await t.upsert([{ frameId: 9, blob: "B", box: null, present: 0 }]);
  const u = new URL("http://x" + calls[0].url);
  assert.equal(u.searchParams.get("present"), "0");
  assert.equal(u.searchParams.has("box"), false);
});

test("upsert surfaces network failure via onError, doesn't throw", async () => {
  globalThis.fetch = async () => {
    throw new Error("down");
  };
  const errors = [];
  const t = new Trainer({ onError: (m) => errors.push(m) });
  await t.upsert([{ frameId: 1, blob: "B", box: null, present: 0 }]); // must not reject
  assert.equal(errors.length, 1);
  assert.match(errors[0], /down/);
});

test("predict parses the backend shape including class fields", async () => {
  stubFetch(async () => ({
    json: async () => ({
      ready: true,
      present: true,
      score: 0.87,
      box: [0, 0, 1, 1],
      cls: 2,
      clsName: "washer",
      clsScore: 0.91,
    }),
  }));
  const t = new Trainer({});
  const r = await t.predict(3, "B");
  assert.deepEqual(r, {
    ready: true,
    present: true,
    score: 0.87,
    box: [0, 0, 1, 1],
    cls: 2,
    clsName: "washer",
    clsScore: 0.91,
  });
});

test("predict defaults class fields to null for single-class models", async () => {
  stubFetch(async () => ({ json: async () => ({ ready: true, present: true, score: 0.5, box: [0, 0, 1, 1] }) }));
  const t = new Trainer({});
  const r = await t.predict(3, "B");
  assert.equal(r.cls, null);
  assert.equal(r.clsName, null);
  assert.equal(r.clsScore, null);
});

test("predict resolves to not-ready on fetch failure (never throws)", async () => {
  globalThis.fetch = async () => {
    throw new Error("boom");
  };
  const t = new Trainer({});
  const r = await t.predict(1, "B");
  assert.deepEqual(r, {
    ready: false,
    present: false,
    score: 0,
    box: null,
    cls: null,
    clsName: null,
    clsScore: null,
  });
});

test("upsert sends the class index on positives", async () => {
  const calls = stubFetch(async () => ({ ok: true, json: async () => ({}) }));
  const t = new Trainer({});
  await t.upsert([{ frameId: 4, blob: "B", box: [0.1, 0.2, 0.3, 0.4], present: 1, cls: 2 }]);
  const u = new URL("http://x" + calls[0].url);
  assert.equal(u.searchParams.get("cls"), "2");
});

test("upsert omits cls on negatives", async () => {
  const calls = stubFetch(async () => ({ ok: true, json: async () => ({}) }));
  const t = new Trainer({});
  await t.upsert([{ frameId: 5, blob: "B", box: null, present: 0, cls: 3 }]);
  const u = new URL("http://x" + calls[0].url);
  assert.equal(u.searchParams.has("cls"), false);
});

test("setClasses POSTs the class list as JSON and returns the server's list", async () => {
  const calls = stubFetch(async () => ({ ok: true, json: async () => ({ classes: ["a", "b"] }) }));
  const t = new Trainer({});
  const out = await t.setClasses(["a", "b"]);
  assert.equal(calls[0].url, "/classes");
  assert.equal(calls[0].opts.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].opts.body), { classes: ["a", "b"] });
  assert.deepEqual(out, ["a", "b"]);
});

test("setClasses surfaces a failure via onError and returns null", async () => {
  stubFetch(async () => ({ ok: false, status: 400, json: async () => ({}) }));
  const errors = [];
  const t = new Trainer({ onError: (m) => errors.push(m) });
  const out = await t.setClasses(["x"]);
  assert.equal(out, null);
  assert.equal(errors.length, 1);
});

test("pause/resume hit their endpoints and flip running", async () => {
  const calls = stubFetch(async () => ({ ok: true }));
  const t = new Trainer({});
  await t.resume();
  assert.equal(t.running, true);
  await t.pause();
  assert.equal(t.running, false);
  assert.deepEqual(
    calls.map((c) => c.url),
    ["/resume", "/pause"],
  );
});
