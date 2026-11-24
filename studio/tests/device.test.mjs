import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { deployLine, Device, DOWNLOADS, howToUse, portOptions } from "../js/device.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

const done = (result) => ({ state: "done", result });

test("a finished deploy says what the board measured", () => {
  const line = deployLine(
    done({ selfTest: true, netMs: 15.51, bytes: 382928, frame: { ms: 33.4, iou: 0.93 } }),
  );
  assert.equal(line.tone, "beats");
  assert.equal(
    line.text,
    "On the board · 15.5 ms per inference · 374 KB. One of your frames, file to box: 33 ms. Board and desktop agree (overlap 0.93).",
  );
});

test("a board that disagrees with the desktop is flagged", () => {
  const line = deployLine(done({ selfTest: true, netMs: 15, frame: { ms: 30, iou: 0.31 } }));
  assert.equal(line.tone, "loses");
  assert.match(line.text, /disagree \(overlap 0\.31\)/);
});

test("a failed self-test is never reported as success", () => {
  const line = deployLine(done({ selfTest: false, netMs: 15, frame: { ms: 30, iou: 1 } }));
  assert.equal(line.tone, "loses");
  assert.match(line.text, /self-test failed/);
});

test("a deploy with no frame to try still confirms the model", () => {
  const line = deployLine(done({ selfTest: true, netMs: 7.8 }));
  assert.equal(line.tone, "beats");
  assert.equal(line.text, "On the board · 7.8 ms per inference.");
});

test("a frame the board could not run is said, not hidden", () => {
  const line = deployLine(done({ selfTest: true, netMs: 7.8, frameError: "not a baseline JPEG" }));
  assert.equal(line.tone, "unsure");
  assert.match(line.text, /Could not run a frame: not a baseline JPEG/);
});

test("running and error states", () => {
  assert.equal(deployLine({ state: "running", step: "writing to the board" }).text, "Working: writing to the board…");
  assert.deepEqual(deployLine({ state: "error", error: "This chip is ESP32, not ESP32-P4" }), {
    tone: "loses",
    text: "This chip is ESP32, not ESP32-P4",
  });
  for (const s of [undefined, null, "x", {}, { state: "idle" }, { state: "done" }]) {
    assert.deepEqual(deployLine(s), { tone: "", text: "" });
  }
});

test("ports: P4-style bridges first, labels shortened, junk dropped", () => {
  const opts = portOptions(["/dev/cu.usbserial-1130", "/dev/cu.usbmodem5C94", "", null, "/dev/ttyACM0"]);
  assert.deepEqual(opts, [
    { value: "/dev/cu.usbmodem5C94", label: "usbmodem5C94" },
    { value: "/dev/ttyACM0", label: "ttyACM0" },
    { value: "/dev/cu.usbserial-1130", label: "usbserial-1130" },
  ]);
  assert.deepEqual(portOptions(undefined), []);
});

test("every download has a file name and a next step", () => {
  for (const format of Object.keys(DOWNLOADS)) {
    assert.ok(DOWNLOADS[format].name);
    assert.ok(howToUse(format).length > 10);
  }
  assert.match(howToUse("p4"), /0xB10000/);
});

test("deploy posts the port and reports a refusal with the backend's words", async () => {
  const calls = [];
  globalThis.fetch = async (url, opts) => {
    calls.push({ url, opts });
    return { ok: false, status: 503, json: async () => ({ error: "ESP32-P4 tools not found" }) };
  };
  const states = [];
  await new Device({ onState: (s) => states.push(s) }).deploy("/dev/cu.usbmodem 1&x=2");
  assert.equal(calls[0].url, "/deploy?port=%2Fdev%2Fcu.usbmodem%201%26x%3D2");
  assert.equal(calls[0].opts.method, "POST");
  assert.deepEqual(states, [{ state: "error", error: "ESP32-P4 tools not found" }]);
});

test("download throws the backend's message instead of saving an error page", async () => {
  globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({ error: "no model trained yet" }) });
  await assert.rejects(() => new Device().download("p4"), /no model trained yet/);
});

test("an unreachable backend lists no ports instead of throwing", async () => {
  globalThis.fetch = async () => {
    throw new Error("offline");
  };
  let seen = null;
  await new Device({ onDevices: (d) => (seen = d) }).refresh();
  assert.deepEqual(seen.ports, []);
});
