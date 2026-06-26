// Tensor Factory Studio — app shell wiring ingest, the canvas editor, the WASD
// keymap, and IndexedDB persistence into one labeling loop. Slice 1: label +
// dedup + persist + export. The continuous trainer / auto-label / guardrail
// (the right-hand panel placeholders) land in slice 2.

import { Store } from "./store.js";
import { CanvasEditor } from "./canvas.js";
import { installKeymap, renderKeymapHelp } from "./keymap.js";
import { extractFrames } from "./video.js";
import { exportDataset } from "./export.js";
import { Trainer } from "./trainer.js";

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
  autoLabel();
}

// --- class selection -------------------------------------------------------
function setActiveClass(i, reassignSelected) {
  if (i < 0 || i >= state.classes.length) return;
  state.activeClass = i;
  if (reassignSelected) editor.setSelectedClass(i);
  renderClasses();
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
  const minDistance = parseInt($("minDistInput").value, 10) || 6;
  const known = await state.store.allHashes();
  $("ingestStatus").textContent = "ingesting…";
  let kept = 0;
  let seen = 0;
  for await (const ev of extractFrames(file, { fps, minDistance, knownHashes: known })) {
    seen++;
    if (ev.kept) {
      await state.store.addFrame({ ...ev.frame, addedAt: Date.now() });
      kept++;
    }
    $("ingestStatus").textContent = `ingesting… ${seen} sampled, ${kept} kept (deduped ${seen - kept})`;
  }
  $("ingestStatus").textContent = `done: ${kept} novel frames added, ${seen - kept} skipped as duplicates`;
  await refreshFrames();
  if (state.frames.length) {
    state.index = Math.max(0, state.frames.length - kept);
    await loadCurrent();
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
    $("ingestStatus").textContent = `exported ${r.positives} positives (${r.annotations} boxes), ${r.negatives} negatives`;
  } catch (e) {
    if (e.name === "AbortError") return; // user cancelled the picker
    $("ingestStatus").textContent = `export failed: ${e.message}`;
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
  return { frameId, blob: f.blob, box: b ? [b.x1, b.y1, b.x2, b.y2] : null, present };
}

function pushSample(frameId) {
  if (!trainer) return;
  const s = sampleFor(frameId);
  if (s) trainer.upsert([s]);
}

// Ask the trainer to pre-fill an unlabeled frame with its prediction.
async function autoLabel() {
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
    editor.setFrame(state.bitmap, [{ x1, y1, x2, y2, cls: state.activeClass }]);
    setAuto(`auto-label · score ${pred.score.toFixed(2)} · Space to accept, drag to fix`);
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
  renderGuardrail(m);
  drawSparkline(m.history, m.bestErr, m.baseline);
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
  const W = cv.width;
  const H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!history || history.length < 2) return;
  const vals = baseline != null ? [...history, baseline] : history;
  const max = Math.max(...vals) * 1.05;
  const min = 0;
  const y = (v) => H - 4 - ((v - min) / (max - min || 1)) * (H - 8);
  const x = (i) => 2 + (i / (history.length - 1)) * (W - 4);
  if (baseline != null) {
    ctx.strokeStyle = "#7b8a9e";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(2, y(baseline));
    ctx.lineTo(W - 2, y(baseline));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = "#4ade80";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  history.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();
}

// --- boot ------------------------------------------------------------------
let editor;
let trainer;

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

  trainer = new Trainer({
    onReady: (m) => {
      $("backendTag").textContent = m.backend;
      trainer.resume();
    },
    onMetrics: renderMetrics,
    onStatus: (text) => {
      const el = $("guardrail");
      el.className = "guardrail";
      el.textContent = text;
    },
    onError: (text) => {
      $("ingestStatus").textContent = "trainer error: " + text;
    },
  });
  trainer.init();

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

  $("videoInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) ingest(file);
    e.target.value = "";
  });
  $("exportBtn").addEventListener("click", doExport);
  $("exportModelBtn").addEventListener("click", async () => {
    try {
      const res = await fetch("/model");
      if (!res.ok) {
        $("ingestStatus").textContent = "no trained model yet";
        return;
      }
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "tinydetector.onnx";
      a.click();
      URL.revokeObjectURL(a.href);
      $("ingestStatus").textContent = `model exported (int8 ONNX, ${(blob.size / 1024).toFixed(0)} KB)`;
    } catch (e) {
      $("ingestStatus").textContent = "model export failed: " + e.message;
    }
  });
  $("pauseBtn").addEventListener("click", () => {
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
    trainer.reset();
    setAuto("");
    state.index = 0;
    await refreshFrames();
    await loadCurrent();
    $("ingestStatus").textContent = "session cleared";
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
  trainer.upsert(approved);
  await loadCurrent();
}

main();
