// Tensor Factory Studio — app shell wiring ingest, the canvas editor, the WASD
// keymap, and IndexedDB persistence into one labeling loop. Slice 1: label +
// dedup + persist + export. The continuous trainer / auto-label / guardrail
// (the right-hand panel placeholders) land in slice 2.

import { Store } from "./store.js";
import { CanvasEditor } from "./canvas.js";
import { installKeymap, renderKeymapHelp } from "./keymap.js";
import { extractFrames } from "./video.js";
import { exportDataset } from "./export.js";
import { connect } from "./engine.js";
import { sourceRows } from "./sources.js";
import { CALIB_FRAMES, deployLine, Device, DOWNLOADS, howToUse, portOptions } from "./device.js";
import { configLine, verdict } from "./verdict.js";
import { toast } from "./toast.js";

const PALETTE = ["#4ade80", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa"];
const DEFAULT_CLASSES = [{ name: "object", color: PALETTE[0] }];

const $ = (id) => document.getElementById(id);

const state = {
  store: null,
  frames: [], // [{id, blob, dhash, width, height, ...}]
  labels: new Map(), // frameId -> label
  classes: DEFAULT_CLASSES,
  activeClass: 0,
  index: 0,
  bitmap: null,
  lastBox: null, // last committed positive box — carried forward + used to ROI-prime GroundingDINO
};

function emptyLabel(frameId) {
  return { frameId, present: true, boxes: [], review: "pending", source: "human", flag: false };
}

function currentFrame() {
  return state.frames[state.index] || null;
}

function currentLabel() {
  const f = currentFrame();
  if (!f) return null;
  if (!state.labels.has(f.id)) state.labels.set(f.id, emptyLabel(f.id));
  return state.labels.get(f.id);
}

// --- persistence -----------------------------------------------------------
async function saveWorking(review) {
  const f = currentFrame();
  if (!f) return;
  const label = currentLabel();
  label.boxes = editor.getBoxes();
  if (review) label.review = review;
  label.present = review === "negative" ? false : label.boxes.length > 0;
  if (review === "negative") {
    label.boxes = [];
    label.review = "approved";
  }
  label.updatedAt = Date.now();
  await state.store.putLabel(label);
  renderStatus();
}

// --- navigation ------------------------------------------------------------
async function go(delta, { commit = null } = {}) {
  const fid = currentFrame()?.id;
  // Remember the committed box: carried forward to the next frame, and used to ROI-prime
  // GroundingDINO (search near where the object was).
  if (commit === "approved") {
    const boxes = editor.getBoxes();
    if (boxes.length) state.lastBox = { ...boxes[0] };
  }
  await saveWorking(commit);
  if (fid != null && (commit === "approved" || commit === "negative")) pushSample(fid);
  const n = state.frames.length;
  if (!n) return;
  state.index = (state.index + delta + n) % n;
  await loadCurrent();
}

async function loadCurrent() {
  setAuto("");
  const f = currentFrame();
  if (!f) {
    state.bitmap = null;
    editor.setFrame(null, []);
    renderStatus();
    return;
  }
  state.bitmap = await createImageBitmap(f.blob);
  const label = currentLabel();
  editor.setFrame(state.bitmap, label.boxes);
  renderStatus();
  suggest();
}

// --- class selection -------------------------------------------------------
function setActiveClass(i, reassignSelected) {
  if (i < 0 || i >= state.classes.length) return;
  state.activeClass = i;
  if (reassignSelected) editor.setSelectedClass(i);
  renderClasses();
}

// Persist the class list to IndexedDB and push it to the backend (which rewrites the COCO
// categories and retrains so the class head matches the new taxonomy). The browser is the
// source of truth for the ordered class list; indices are the contract with stored boxes.
async function persistClasses() {
  editor.setClasses(state.classes);
  renderClasses();
  await state.store.metaSet("classes", state.classes);
  if (trainer) await trainer.setClasses(state.classes.map((c) => c.name));
}

function addClass(name) {
  const n = (name || "").trim();
  if (!n) return;
  if (state.classes.some((c) => c.name.toLowerCase() === n.toLowerCase())) {
    toast.info(`Class "${n}" already exists.`);
    return;
  }
  const color = PALETTE[state.classes.length % PALETTE.length];
  state.classes = [...state.classes, { name: n, color }];
  setActiveClass(state.classes.length - 1, false);
  persistClasses();
}

async function importClassesFromCoco(file) {
  try {
    const coco = JSON.parse(await file.text());
    const cats = (coco.categories || []).slice().sort((a, b) => a.id - b.id);
    if (!cats.length) {
      toast.error(`No "categories" found in "${file.name}".`);
      return;
    }
    state.classes = cats.map((c, i) => ({ name: c.name, color: PALETTE[i % PALETTE.length] }));
    setActiveClass(0, false);
    await persistClasses();
    toast.success(`Imported ${state.classes.length} classes from "${file.name}".`);
  } catch (e) {
    toast.error(`Couldn't read classes from "${file.name}": ${e.message}`);
  }
}

// Reconcile the browser's class list with the backend on connect. A fresh browser (only the
// default "object" class) adopts a pre-populated backend's taxonomy; otherwise the browser
// wins and pushes its list down. Keeps the two from silently disagreeing on indices.
async function syncClasses(backendNames) {
  const isDefault = state.classes.length === 1 && state.classes[0].name === "object";
  if (isDefault && Array.isArray(backendNames) && backendNames.length > 1) {
    state.classes = backendNames.map((n, i) => ({ name: n, color: PALETTE[i % PALETTE.length] }));
    setActiveClass(0, false);
    editor.setClasses(state.classes);
    renderClasses();
    await state.store.metaSet("classes", state.classes);
  } else if (trainer) {
    await trainer.setClasses(state.classes.map((c) => c.name));
  }
}

// --- rendering -------------------------------------------------------------
function renderClasses() {
  const row = $("classRow");
  row.innerHTML = "";
  state.classes.forEach((c, i) => {
    const chip = document.createElement("span");
    chip.className = "class-chip" + (i === state.activeClass ? " active" : "");
    chip.style.borderColor = c.color;
    chip.innerHTML = `<i style="background:${c.color}"></i>${i + 1} ${c.name}`;
    chip.onclick = () => setActiveClass(i, true);
    row.appendChild(chip);
  });
}

function renderStatus() {
  const n = state.frames.length;
  $("frameInfo").textContent = n ? `${state.index + 1} / ${n}` : "— / —";
  const label = currentLabel();
  const badges = $("stateBadges");
  badges.innerHTML = "";
  if (label) {
    const add = (text, cls) => {
      const b = document.createElement("span");
      b.className = "badge " + cls;
      b.textContent = text;
      badges.appendChild(b);
    };
    add(label.review, label.review === "approved" ? "ok" : "warn");
    add(label.present ? `${label.boxes.length} box` : "empty", label.present ? "ok" : "muted");
    if (label.flag) add("flagged", "flag");
  }
  $("emptyHint").style.display = n ? "none" : "flex";
  renderCounts();
  renderGallery();
}

// --- gallery preview -------------------------------------------------------
// A small strip under the editor showing only annotated frames (those carrying at
// least one committed box). Click a thumbnail to jump to that frame. Object URLs are
// cached per frame and revoked on clear so a long session doesn't leak blobs.
const galleryURLs = new Map(); // frameId -> objectURL
let lastGallerySig = "";

function annotatedFrames() {
  return state.frames.filter((f) => {
    const l = state.labels.get(f.id);
    return l && l.present && l.boxes && l.boxes.length > 0;
  });
}

function galleryURL(frame) {
  let url = galleryURLs.get(frame.id);
  if (!url) {
    url = URL.createObjectURL(frame.blob);
    galleryURLs.set(frame.id, url);
  }
  return url;
}

function revokeGalleryURLs() {
  for (const url of galleryURLs.values()) URL.revokeObjectURL(url);
  galleryURLs.clear();
  lastGallerySig = "";
}

function renderGallery() {
  const strip = $("gallery");
  if (!strip) return;
  const items = annotatedFrames();
  const activeId = currentFrame()?.id;
  $("galleryCount").textContent = items.length ? `${items.length}` : "";
  // Only rebuild the thumbnails when the annotated set (or any box count) changes —
  // navigation and drag edits just re-toggle the active highlight, which stays cheap.
  const sig = items.map((f) => `${f.id}:${state.labels.get(f.id).boxes.length}`).join(",");
  if (sig !== lastGallerySig) {
    lastGallerySig = sig;
    strip.innerHTML = "";
    if (!items.length) {
      strip.classList.add("empty");
      strip.innerHTML = `<span class="gallery-empty">Annotated frames will appear here.</span>`;
    } else {
      strip.classList.remove("empty");
      for (const f of items) {
        const btn = document.createElement("button");
        btn.className = "thumb";
        btn.dataset.fid = f.id;
        btn.title = `frame ${f.id}`;
        const img = document.createElement("img");
        img.src = galleryURL(f);
        img.alt = `frame ${f.id}`;
        img.loading = "lazy";
        btn.appendChild(img);
        const n = state.labels.get(f.id).boxes.length;
        if (n > 1) {
          const tag = document.createElement("span");
          tag.className = "thumb-tag";
          tag.textContent = `${n}`;
          btn.appendChild(tag);
        }
        btn.onclick = () => jumpToFrame(f.id);
        strip.appendChild(btn);
      }
    }
  }
  for (const el of strip.querySelectorAll(".thumb")) {
    el.classList.toggle("active", Number(el.dataset.fid) === activeId);
  }
}

function renderCounts() {
  let pos = 0;
  let neg = 0;
  let pending = 0;
  for (const l of state.labels.values()) {
    if (l.review !== "approved") pending++;
    else if (l.present) pos++;
    else neg++;
  }
  $("counts").textContent = `${pos} pos · ${neg} neg · ${pending} pending · ${state.frames.length} frames`;
}

// --- ingest ----------------------------------------------------------------
async function ingest(file) {
  const fps = parseFloat($("fpsInput").value) || 1;
  const minDistance = parseInt($("minDistInput").value, 10) || 12;
  const input = $("videoInput");
  input.disabled = true;
  const prog = toast.progress(`Decoding "${file.name}"…`);
  $("ingestStatus").textContent = `decoding "${file.name}"…`;
  let kept = 0;
  let seen = 0;
  try {
    const known = await state.store.allHashes();
    for await (const ev of extractFrames(file, { fps, minDistance, knownHashes: known })) {
      seen++;
      if (ev.kept) {
        await state.store.addFrame({ ...ev.frame, addedAt: Date.now() });
        kept++;
      }
      const pct = ev.total ? Math.min(100, Math.round(((ev.index + 1) / ev.total) * 100)) : 0;
      const msg = `Sampling frames… ${pct}% · ${kept} kept, ${seen - kept} skipped`;
      prog.update(msg);
      $("ingestStatus").textContent = msg;
    }
    if (seen === 0) {
      prog.error("No frames were sampled — the video may be empty or unreadable.");
      $("ingestStatus").textContent = "no frames sampled";
      return;
    }
    await refreshFrames();
    if (kept === 0) {
      const m = `All ${seen} sampled frames matched existing ones — nothing new added. Lower "dedup" to keep more.`;
      prog.success(m);
      $("ingestStatus").textContent = m;
    } else {
      const m = `Added ${kept} new frame${kept === 1 ? "" : "s"} (skipped ${seen - kept} duplicate${seen - kept === 1 ? "" : "s"}).`;
      prog.success(m);
      $("ingestStatus").textContent = m;
      state.index = Math.max(0, state.frames.length - kept);
      await loadCurrent();
    }
  } catch (e) {
    const m = `Couldn't load "${file.name}": ${e.message}`;
    prog.error(m);
    $("ingestStatus").textContent = m;
  } finally {
    input.disabled = false;
  }
}

async function refreshFrames() {
  state.frames = await state.store.allFrames();
  const labels = await state.store.allLabels();
  state.labels = new Map(labels.map((l) => [l.frameId, l]));
  renderStatus();
}

// --- export ----------------------------------------------------------------
async function doExport() {
  await saveWorking();
  try {
    const r = await exportDataset(state.frames, state.labels, state.classes);
    if (r.positives === 0 && r.negatives === 0) {
      toast.info("Nothing to export yet — commit some labels first (approved frames only).");
      return;
    }
    const m = `Exported ${r.positives} positives (${r.annotations} boxes), ${r.negatives} negatives.`;
    $("ingestStatus").textContent = m;
    toast.success(m);
  } catch (e) {
    if (e.name === "AbortError") return; // user cancelled the picker
    toast.error(`Dataset export failed: ${e.message}`);
  }
}

// --- trainer integration ---------------------------------------------------
// Build the single-object training sample for a frame, or null if not approved.
function sampleFor(frameId) {
  const f = state.frames.find((fr) => fr.id === frameId);
  const l = state.labels.get(frameId);
  if (!f || !l || l.review !== "approved") return null;
  const present = l.present && l.boxes.length ? 1 : 0;
  const b = present ? l.boxes[0] : null;
  return {
    frameId,
    blob: f.blob,
    box: b ? [b.x1, b.y1, b.x2, b.y2] : null,
    cls: b ? (b.cls ?? 0) : 0,
    present,
  };
}

function pushSample(frameId) {
  if (!trainer) return;
  const s = sampleFor(frameId);
  if (s) trainer.upsert([s]);
}

function autoLabelEnabled() {
  return !!$("autoLabelToggle")?.checked;
}

function dinoEnabled() {
  return !!$("dinoToggle")?.checked && !!$("dinoPrompt")?.value.trim();
}

// Single suggestion entry, called on every new frame. Precedence, all opt-in and all
// only *pre-fill* a box (never commit): GroundingDINO (open-vocab, no model needed) →
// carry the previous committed box forward (instant, works from frame 1) → the trained
// model's guess. With every source off you get a fully manual, no-surprise pass.
async function suggest() {
  const f = currentFrame();
  if (!f) return;
  const label = currentLabel();
  if (label.review === "approved" || editor.getBoxes().length) return;
  if (dinoEnabled()) return dinoLabel();
  if (!autoLabelEnabled()) return;
  if (state.lastBox) {
    editor.setFrame(state.bitmap, [{ ...state.lastBox }]);
    setAuto("box carried from previous frame · drag to adjust, Space to accept, C if gone");
    return;
  }
  await autoLabel(); // nothing to carry yet — let the model suggest
}

// GroundingDINO auto-label: open-vocab, needs no trained model, primed by the previous
// frame's box (the server crops to that region). Its own opt-in toggle + text prompt.
async function dinoLabel() {
  const f = currentFrame();
  if (!f) return;
  const label = currentLabel();
  if (label.review === "approved" || editor.getBoxes().length) return;
  const fid = f.id;
  const prompt = $("dinoPrompt").value.trim();
  const params = new URLSearchParams({ prompt });
  if (state.lastBox) {
    const b = state.lastBox;
    params.set("box", [b.x1, b.y1, b.x2, b.y2].join(","));
  }
  setAuto(`GroundingDINO searching for "${prompt}"…`);
  let r;
  try {
    r = await fetch(`/autolabel?${params}`, { method: "POST", body: f.blob }).then((x) => x.json());
  } catch (e) {
    toast.error(`GroundingDINO request failed: ${e.message}`);
    setAuto("");
    return;
  }
  const cur = currentFrame();
  if (!cur || cur.id !== fid) return; // navigated away
  if (currentLabel().review === "approved" || editor.getBoxes().length) return;
  if (r.error) {
    toast.error(`GroundingDINO: ${r.error}`);
    setAuto("GroundingDINO unavailable — see the message; you can still label by hand");
    return;
  }
  if (r.present && r.box) {
    const [x1, y1, x2, y2] = r.box;
    editor.setFrame(state.bitmap, [{ x1, y1, x2, y2, cls: state.activeClass }]);
    const primed = r.primed ? " (primed by previous frame)" : "";
    setAuto(`GroundingDINO${primed} · ${r.label} ${((r.score ?? 0) * 100).toFixed(0)}% · Space to accept, drag to fix`);
  } else {
    setAuto(`GroundingDINO found no "${prompt}" · C to confirm empty`);
  }
}

// Model-prediction path (the trainer's guess); reached via suggest() as the fallback.
async function autoLabel() {
  if (!autoLabelEnabled()) return;
  const f = currentFrame();
  if (!trainer || !trainer.ready || !f) return;
  const label = currentLabel();
  if (label.review === "approved" || editor.getBoxes().length) return;
  const fid = f.id;
  const pred = await trainer.predict(fid, f.blob);
  const cur = currentFrame();
  if (!cur || cur.id !== fid) return; // navigated away
  if (currentLabel().review === "approved" || editor.getBoxes().length) return;
  if (!pred.ready) {
    setAuto("model warming up — label a few more frames");
    return;
  }
  if (pred.present) {
    const [x1, y1, x2, y2] = pred.box;
    // Adopt the model's predicted class when it has a class head; fall back to the active
    // class for single-class models. Make it the active class so the chip + Space-accept agree.
    const cls = pred.cls != null && pred.cls < state.classes.length ? pred.cls : state.activeClass;
    if (pred.cls != null) setActiveClass(cls, false);
    editor.setFrame(state.bitmap, [{ x1, y1, x2, y2, cls }]);
    const clsTxt =
      pred.clsName != null ? ` · ${pred.clsName} ${((pred.clsScore ?? 0) * 100).toFixed(0)}%` : "";
    setAuto(`auto-label · score ${pred.score.toFixed(2)}${clsTxt} · Space to accept, drag to fix`);
  } else {
    setAuto(`model sees no object · score ${pred.score.toFixed(2)} · C to confirm empty`);
  }
}

function setAuto(text) {
  $("autoBadge").textContent = text || "";
}

function fmtPx(v) {
  return v == null ? "—" : `${v.toFixed(1)} px`;
}

function renderMetrics(m) {
  $("backendTag").textContent = m.backend || "";
  const errEl = $("mErr");
  errEl.textContent = fmtPx(m.err);
  $("mBest").textContent = fmtPx(m.bestErr === Infinity ? null : m.bestErr);
  $("mBaseline").textContent = fmtPx(m.baseline);
  // does the model beat the constant-predictor floor?
  errEl.className = m.baseline == null ? "" : m.err < m.baseline ? "beats" : "loses";
  $("mPresence").textContent = m.presenceAcc == null ? "—" : `${(m.presenceAcc * 100).toFixed(0)}%`;
  $("mGain").textContent = m.gain == null ? "—" : m.gain.toFixed(2);
  $("mEpoch").textContent = `${m.epoch} · ${m.trainCount}t/${m.valCount}v`;
  const v = verdict(m);
  $("verdict").className = `verdict ${v.tone}`;
  $("verdict").textContent = v.text;
  $("configLine").textContent = configLine(m.config);
  renderSources(m.sources, m.baseline);
  renderGuardrail(m);
  drawSparkline(m.history, m.bestErr, m.baseline);
}

// One row per data source, so a source the model is bad on is visible instead of
// averaged into the headline number. Hidden until there is more than one to compare.
function renderSources(sources, baseline) {
  const table = $("sources");
  const rows = sourceRows(sources, baseline);
  table.hidden = rows.length < 2;
  table.tBodies[0].replaceChildren(
    ...rows.map((r) => {
      const tr = document.createElement("tr");
      for (const text of r.cells) {
        const td = document.createElement("td");
        td.textContent = text; // source names come from directory names: text, never HTML
        tr.appendChild(td);
      }
      if (r.tone) tr.cells[3].className = r.tone;
      return tr;
    }),
  );
}

// --- device panel -------------------------------------------------------------------
function renderDevices(d) {
  const select = $("devicePort");
  const keep = select.value;
  const opts = portOptions(d.ports);
  select.replaceChildren(
    ...opts.map((o) => {
      const el = document.createElement("option");
      el.value = o.value;
      el.textContent = o.label; // a device path: text, never HTML
      return el;
    }),
  );
  if (opts.some((o) => o.value === keep)) select.value = keep;
  $("deployBtn").disabled = !d.p4Tools || opts.length === 0;
  if (!run.backend) setDeployStatus({ tone: "", text: NEEDS_HELPER });
  else if (!d.p4Tools) setDeployStatus({ tone: "unsure", text: d.hint || "" });
  else if (opts.length === 0) setDeployStatus({ tone: "", text: "No board found. Plug one in over USB." });
}

function setDeployStatus({ tone, text }) {
  const el = $("deployStatus");
  el.className = `deploy-status ${tone}`;
  el.textContent = text;
}

function renderDeploy(state) {
  setDeployStatus(deployLine(state));
  $("deployBtn").disabled = state.state === "running";
  if (state.state === "done") toast.success("Model is on the board.");
  if (state.state === "error") toast.error(`Deploy: ${state.error}`);
}

const NEEDS_HELPER =
  "That needs the local helper: run `uv run tensor-factory-studio` in the repo and open the page it prints.";

// In browser mode the model lives in this page: hand it to the backend before asking the
// backend to convert or flash it. Returns false (and says why) when that is not possible.
async function shareModelWithBackend() {
  if (run.mode !== "browser") return true;
  if (!run.backend) {
    toast.info(NEEDS_HELPER);
    return false;
  }
  const bytes = await trainer.modelBytes();
  if (!bytes) {
    toast.info("No trained model yet — label a few frames first.");
    return false;
  }
  await device.uploadModel(bytes, calibrationFrames());
  return true;
}

// Labeled frames to send with the model: newest boxed frame first (it is the one tried on
// the board), then a spread of the rest.
function calibrationFrames() {
  const boxed = state.frames.filter((f) => {
    const l = state.labels.get(f.id);
    return l && l.review === "approved" && l.present && l.boxes.length;
  });
  if (!boxed.length) return [];
  const newest = boxed[boxed.length - 1];
  const rest = boxed.slice(0, -1);
  const step = Math.max(1, Math.floor(rest.length / (CALIB_FRAMES - 1)));
  return [newest, ...rest.filter((_, i) => i % step === 0)].slice(0, CALIB_FRAMES).map((f) => f.blob);
}

async function modelFile(format) {
  if (run.mode === "browser" && format === "onnx") {
    const bytes = await trainer.modelBytes();
    if (!bytes) throw new Error("no model trained yet");
    return { blob: new Blob([bytes], { type: "application/octet-stream" }), name: DOWNLOADS.onnx.name };
  }
  return device.download(format);
}

async function downloadModel(format) {
  $("downloadHint").textContent = format === "p4" ? "Preparing the board image…" : "";
  try {
    if (format !== "onnx" && !(await shareModelWithBackend())) {
      $("downloadHint").textContent = "";
      return;
    }
    const { blob, name } = await modelFile(format);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
    $("downloadHint").textContent = howToUse(format);
    toast.success(`Model exported: ${DOWNLOADS[format].label} (${(blob.size / 1024).toFixed(0)} KB).`);
  } catch (e) {
    $("downloadHint").textContent = "";
    if (/no model trained yet/.test(e.message)) {
      toast.info("No trained model yet — label a few frames first.");
    } else {
      toast.error(`${DOWNLOADS[format].label}: ${e.message}`);
    }
  }
}

function renderGuardrail(m) {
  const el = $("guardrail");
  if (m.regressed) {
    el.className = "guardrail alert";
    el.innerHTML =
      `⚠ regression: ${m.err.toFixed(1)}px vs best ${m.bestErr.toFixed(1)}px — serving best checkpoint. ` +
      `Likely-bad labels: `;
    m.suspects.forEach((s, i) => {
      const a = document.createElement("span");
      a.className = "suspect";
      a.textContent = `#${s.frameId}${i < m.suspects.length - 1 ? ", " : ""}`;
      a.onclick = () => jumpToFrame(s.frameId);
      el.appendChild(a);
    });
  } else {
    el.className = "guardrail";
    el.textContent = "";
  }
}

function jumpToFrame(frameId) {
  const i = state.frames.findIndex((f) => f.id === frameId);
  if (i >= 0) {
    saveWorking().then(() => {
      state.index = i;
      loadCurrent();
    });
  }
}

function drawSparkline(history, best, baseline) {
  const cv = $("sparkline");
  const ctx = cv.getContext("2d");
  // Size the buffer from the displayed size at device resolution (the CSS stretches
  // the canvas to the panel width; without this the plot renders soft on retina).
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const dw = Math.round(cv.clientWidth * dpr) || cv.width;
  const dh = Math.round(cv.clientHeight * dpr) || cv.height;
  if (cv.width !== dw || cv.height !== dh) {
    cv.width = dw;
    cv.height = dh;
  }
  const W = cv.width;
  const H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!history || history.length < 2) return;
  const vals = baseline != null ? [...history, baseline] : history;
  const max = Math.max(...vals) * 1.05;
  const min = 0;
  const y = (v) => H - 4 * dpr - ((v - min) / (max - min || 1)) * (H - 8 * dpr);
  const x = (i) => 2 * dpr + (i / (history.length - 1)) * (W - 4 * dpr);
  if (baseline != null) {
    ctx.strokeStyle = "#7b8a9e";
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath();
    ctx.moveTo(2 * dpr, y(baseline));
    ctx.lineTo(W - 2 * dpr, y(baseline));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = "#4ade80";
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  history.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();
}

// --- boot ------------------------------------------------------------------
let editor;
let trainer;
let device;
// Where training runs ("browser" | "backend" | "none") and whether the local backend answers.
const run = { mode: "none", backend: null };

async function main() {
  state.store = await Store.create();
  const savedClasses = await state.store.metaGet("classes");
  if (savedClasses) state.classes = savedClasses;

  editor = new CanvasEditor(
    $("editor"),
    () => ({ index: state.activeClass, color: state.classes[state.activeClass].color }),
    () => {
      // autosave drawing edits as pending immediately (survives reload)
      saveWorking();
    },
  );
  editor.setClasses(state.classes);

  // Resizable preview: keep the canvas drawing buffer matched to its displayed size
  // (crisp render + pixel-accurate crosshair/boxes) and remember the size across reloads.
  const wrap = document.querySelector(".canvas-wrap");
  try {
    const saved = JSON.parse(localStorage.getItem("studio.canvasSize") || "null");
    if (saved && saved.w && saved.h) {
      wrap.style.width = `${saved.w}px`;
      wrap.style.height = `${saved.h}px`;
    }
  } catch {
    /* storage unavailable or corrupt — ignore */
  }
  let sizeTimer;
  new ResizeObserver(() => {
    editor.fitToDisplay();
    clearTimeout(sizeTimer);
    sizeTimer = setTimeout(() => {
      try {
        localStorage.setItem(
          "studio.canvasSize",
          JSON.stringify({ w: Math.round(wrap.clientWidth), h: Math.round(wrap.clientHeight) }),
        );
      } catch {
        /* storage unavailable — non-fatal */
      }
    }, 300);
  }).observe(wrap);

  const choice = await connect({
    onReady: (m) => {
      $("backendTag").textContent = m.backend;
      toast.success(choice.line);
      if (run.mode === "backend") syncClasses(m.classes);
      trainer.resume();
    },
    onMetrics: renderMetrics,
    onStatus: (text) => {
      const el = $("guardrail");
      el.className = "guardrail";
      el.textContent = text;
    },
    onError: (text) => {
      // de-duped by the toast layer, so a failing poll loop shows one message, not 50
      toast.error(run.mode === "backend" ? `Backend: ${text}` : text);
      if (run.mode === "backend") $("backendTag").textContent = "offline";
    },
  });
  trainer = choice.trainer;
  run.mode = choice.mode;
  run.backend = choice.backend;
  $("modeLine").textContent = choice.line;
  if (trainer) {
    await trainer.init();
  } else {
    $("backendTag").textContent = "unavailable";
    toast.error(choice.line);
  }

  renderClasses();
  renderKeymapHelp($("keymapHelp"));

  installKeymap({
    prev: () => go(-1),
    next: () => go(+1),
    accept: () => go(+1, { commit: "approved" }),
    commit: () => go(+1, { commit: "approved" }),
    skip: () => go(+1, { commit: "pending" }),
    negative: () => go(+1, { commit: "negative" }),
    classPrev: () => setActiveClass(state.activeClass - 1, true),
    classNext: () => setActiveClass(state.activeClass + 1, true),
    classNum: (_e, key) => setActiveClass(parseInt(key, 10) - 1, true),
    undo: () => editor.undo(),
    delete: () => editor.deleteSelected(),
    clear: () => editor.clearAll(),
    flag: () => {
      const l = currentLabel();
      if (l) {
        l.flag = !l.flag;
        saveWorking();
      }
    },
    cancel: () => editor.cancelDraw(),
  });

  // auto-label opt-in: off by default, remembered across reloads
  const autoToggle = $("autoLabelToggle");
  try {
    autoToggle.checked = localStorage.getItem("studio.autoLabel") === "1";
  } catch {
    /* storage unavailable */
  }
  autoToggle.addEventListener("change", () => {
    try {
      localStorage.setItem("studio.autoLabel", autoToggle.checked ? "1" : "0");
    } catch {
      /* non-fatal */
    }
    setAuto(""); // clear any stale suggestion, then re-run the suggestion logic
    suggest(); // pre-fills the current frame if a source is enabled; no-op otherwise
  });

  // GroundingDINO opt-in: toggle + text prompt, both remembered across reloads
  const dinoToggle = $("dinoToggle");
  const dinoPrompt = $("dinoPrompt");
  try {
    dinoToggle.checked = localStorage.getItem("studio.dino") === "1";
    dinoPrompt.value = localStorage.getItem("studio.dinoPrompt") || "";
  } catch {
    /* storage unavailable */
  }
  const dinoChanged = () => {
    try {
      localStorage.setItem("studio.dino", dinoToggle.checked ? "1" : "0");
      localStorage.setItem("studio.dinoPrompt", dinoPrompt.value);
    } catch {
      /* non-fatal */
    }
    setAuto("");
    suggest();
  };
  dinoToggle.addEventListener("change", dinoChanged);
  dinoPrompt.addEventListener("change", dinoChanged);

  // class management: add by button/Enter, import names from a COCO file
  $("addClassBtn").addEventListener("click", () => {
    const input = $("newClassInput");
    addClass(input.value);
    input.value = "";
    input.focus();
  });
  $("newClassInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      $("addClassBtn").click();
    }
  });
  $("importClassesInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) importClassesFromCoco(file);
    e.target.value = "";
  });

  $("videoInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) ingest(file);
    e.target.value = "";
  });
  $("exportBtn").addEventListener("click", doExport);
  $("exportModelBtn").addEventListener("click", () => downloadModel("onnx"));
  device = new Device({ onState: renderDeploy, onDevices: renderDevices });
  // Without the local helper there is nothing to ask for ports (and no request to fail).
  const refreshDevices = () => (run.backend ? device.refresh() : renderDevices({ ports: [], p4Tools: false }));
  refreshDevices();
  $("devicePort").addEventListener("focus", refreshDevices); // a board plugged in later
  $("deployBtn").addEventListener("click", async () => {
    const port = $("devicePort").value;
    if (!port) return;
    try {
      if (await shareModelWithBackend()) device.deploy(port);
    } catch (e) {
      toast.error(`Deploy: ${e.message}`);
    }
  });
  for (const btn of document.querySelectorAll(".downloads .link")) {
    btn.addEventListener("click", () => downloadModel(btn.dataset.format));
  }
  $("pauseBtn").addEventListener("click", () => {
    if (!trainer) return;
    if (trainer.running) {
      trainer.pause();
      $("pauseBtn").textContent = "Resume training";
    } else {
      trainer.resume();
      $("pauseBtn").textContent = "Pause training";
    }
  });
  $("clearBtn").addEventListener("click", async () => {
    if (!confirm("Wipe all frames and labels from this session?")) return;
    await state.store.clearAll();
    trainer?.reset();
    revokeGalleryURLs();
    setAuto("");
    state.lastBox = null;
    state.index = 0;
    await refreshFrames();
    await loadCurrent();
    $("ingestStatus").textContent = "session cleared";
    toast.info("Session cleared — frames, labels, and trainer state wiped.");
  });

  await refreshFrames();
  // feed any already-approved labels (e.g. from a prior session) to the trainer
  const approved = [];
  for (const l of state.labels.values()) {
    if (l.review === "approved") {
      const s = sampleFor(l.frameId);
      if (s) approved.push(s);
    }
  }
  trainer?.upsert(approved);
  await loadCurrent();
}

main();
