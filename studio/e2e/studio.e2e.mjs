// End-to-end browser tests for Tensor Factory Studio, driven with Playwright.
// Deterministic UI + ingest + error + backend-wiring flows (no GPU training in the
// assertions — that's covered by the Python integration test). Point it at a running
// backend with STUDIO_URL (default http://127.0.0.1:8089); studio/e2e/run.sh boots a
// throwaway backend and runs this for you.
//
// Use 127.0.0.1, not localhost: the backend binds IPv4 and localhost can resolve to ::1.

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const BASE = process.env.STUDIO_URL || "http://127.0.0.1:8089";
const HERE = path.dirname(fileURLToPath(import.meta.url));
const GOOD = path.join(HERE, "fixtures", "sample.webm");
const GARBAGE = path.join(os.tmpdir(), "studio-e2e-garbage.mp4");
fs.writeFileSync(GARBAGE, Buffer.from("definitely not a video".repeat(40)));

const steps = [];
async function step(name, fn) {
  process.stdout.write(`• ${name} … `);
  await fn();
  steps.push(name);
  console.log("ok");
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));
page.on("dialog", (d) => d.accept());

try {
  await step("app boots and the backend connects", async () => {
    await page.goto(`${BASE}/`, { waitUntil: "load" });
    await page.waitForFunction(
      () => /mps|cpu|cuda/.test(document.getElementById("backendTag").textContent),
      null,
      { timeout: 15000 },
    );
    assert.equal(await page.title(), "Tensor Factory Studio");
  });

  await step("clear session starts from a clean slate", async () => {
    await page.click("#clearBtn");
    await page.waitForFunction(
      () => document.getElementById("ingestStatus").textContent === "session cleared",
      null,
      { timeout: 5000 },
    );
  });

  await step("good video ingests and shows a success toast", async () => {
    await page.fill("#minDistInput", "2");
    await page.setInputFiles("#videoInput", GOOD);
    // match the ingest toast specifically (a "Backend connected" success toast may linger)
    await page.waitForSelector("#toasts .toast-success >> text=/Added \\d+ new frame/", {
      timeout: 30000,
    });
    assert.match(await page.locator("#counts").textContent(), /\d+ frames/);
    // the first frame should now be on the canvas (empty-state hidden)
    assert.equal(await page.locator("#emptyHint").isVisible(), false);
  });

  await step("garbage file shows an error toast and does not hang", async () => {
    await page.setInputFiles("#videoInput", GARBAGE);
    const toast = await page.waitForSelector("#toasts .toast-error", { timeout: 30000 });
    assert.match(await toast.textContent(), /Couldn't load|not supported|could not/i);
    assert.equal(await page.locator("#videoInput").isDisabled(), false); // input re-enabled
  });

  await step("labeling pushes approved frames to the backend", async () => {
    const bb = await page.locator("#editor").boundingBox();
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press("r");
      await page.mouse.move(bb.x + bb.width * 0.3, bb.y + bb.height * 0.3);
      await page.mouse.down();
      await page.mouse.move(bb.x + bb.width * 0.6, bb.y + bb.height * 0.6, { steps: 4 });
      await page.mouse.up();
      await page.keyboard.press("w");
      await page.waitForTimeout(60);
    }
    const status = await page.evaluate((b) => fetch(`${b}/status`).then((r) => r.json()), BASE);
    assert.ok(status.counts.positives >= 5, `backend got ${status.counts.positives} positives`);
  });

  await step("export model gives clear feedback either way", async () => {
    // Training may or may not have produced a model this fast; both outcomes are valid
    // and must surface a toast — never silence.
    await page.click("#exportModelBtn");
    await page.waitForSelector("#toasts >> text=/No trained model yet|Model exported/i", {
      timeout: 8000,
    });
  });

  assert.deepEqual(pageErrors, [], `unexpected console/page errors: ${pageErrors.join("; ")}`);
  console.log(`\n✅ all ${steps.length} e2e steps passed`);
} catch (err) {
  console.error(`\n❌ e2e failed: ${err.message}`);
  if (pageErrors.length) console.error("page errors:", pageErrors);
  process.exitCode = 1;
} finally {
  await browser.close();
  fs.rmSync(GARBAGE, { force: true });
}
