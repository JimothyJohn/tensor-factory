import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { localConfig, modeLine, pickMode, probeBackend } from "../js/engine.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

const backend = (sources = 0) => ({ backend: "mps", counts: {}, sources });

test("WebGPU trains in the browser, with or without a backend", () => {
  assert.equal(pickMode({ gpu: true, backend: null }), "browser");
  assert.equal(pickMode({ gpu: true, backend: backend() }), "browser");
});

test("a backend started with --source data does the training", () => {
  assert.equal(pickMode({ gpu: true, backend: backend(2) }), "backend");
});

test("no WebGPU falls back to the backend, and to nothing without one", () => {
  assert.equal(pickMode({ gpu: false, backend: backend() }), "backend");
  assert.equal(pickMode({ gpu: false, backend: null }), "none");
});

test("?trainer= overrides only when that trainer exists", () => {
  assert.equal(pickMode({ gpu: true, backend: backend(), override: "backend" }), "backend");
  assert.equal(pickMode({ gpu: true, backend: backend(3), override: "browser" }), "browser");
  assert.equal(pickMode({ gpu: true, backend: null, override: "backend" }), "browser");
  assert.equal(pickMode({ gpu: false, backend: backend(), override: "browser" }), "backend");
  assert.equal(pickMode({ gpu: true, backend: null, override: "<script>" }), "browser");
});

test("every mode explains itself", () => {
  assert.match(modeLine("browser", { gpu: true, backend: null }), /Nothing is uploaded/);
  assert.match(modeLine("browser", { gpu: true, backend: backend() }), /flashing boards/);
  assert.match(modeLine("backend", { gpu: false, backend: backend() }), /no WebGPU/);
  assert.match(modeLine("backend", { gpu: true, backend: backend(1) }), /--source/);
  assert.match(modeLine("none", { gpu: false, backend: null }), /uv run tensor-factory-studio/);
});

test("browser settings: defaults, then the backend's flags, then the page's query", () => {
  assert.deepEqual(localConfig("", null), { size: 224, width: 16, context: 4, resize: "mixed", epochs: 20 });
  const fromBackend = { config: { size: 160, width: 8, context: 2, resize: "nearest" } };
  assert.deepEqual(localConfig("", fromBackend), { size: 160, width: 8, context: 2, resize: "nearest", epochs: 20 });
  assert.deepEqual(localConfig("?size=96&epochs=5&resize=bilinear", fromBackend), {
    size: 96, width: 8, context: 2, resize: "bilinear", epochs: 5,
  });
});

test("hostile query values are ignored, not trusted", () => {
  const bad = "?size=99999&context=-1&width=1e9&epochs=0&resize=__proto__&size=abc";
  assert.deepEqual(localConfig(bad, null), { size: 224, width: 16, context: 4, resize: "mixed", epochs: 20 });
  assert.equal(localConfig("?context=9", null).context, 4);
  assert.equal(localConfig("?size=224.5", null).size, 224);
  assert.equal(localConfig("?size=%00", { config: { size: "160; rm -rf" } }).size, 224);
});

test("probeBackend: a static host answering 404 or HTML is not a backend", async () => {
  globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}) });
  assert.equal(await probeBackend(), null);
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => {
      throw new SyntaxError("Unexpected token <");
    },
  });
  assert.equal(await probeBackend(), null);
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ hello: "world" }) });
  assert.equal(await probeBackend(), null);
  globalThis.fetch = async () => {
    throw new TypeError("network");
  };
  assert.equal(await probeBackend(), null);
  globalThis.fetch = async () => ({ ok: true, json: async () => backend(1) });
  assert.equal((await probeBackend()).sources, 1);
});
