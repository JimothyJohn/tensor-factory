// Tensor Factory Studio — app shell wiring ingest, the canvas editor, the WASD
// keymap, and IndexedDB persistence into one labeling loop. Slice 1: label +
// dedup + persist + export. The continuous trainer / auto-label / guardrail
// (the right-hand panel placeholders) land in slice 2.

import { Store } from "./store.js";
import { CanvasEditor } from "./canvas.js";
import { installKeymap, renderKeymapHelp } from "./keymap.js";
import { extractFrames } from "./video.js";
import { exportDataset } from "./export.js";

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
  await saveWorking(commit);
  const n = state.frames.length;
  if (!n) return;
  state.index = (state.index + delta + n) % n;
  await loadCurrent();
}

async function loadCurrent() {
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
  $("emptyHint").style.display = n ? "none" : "block";
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

// --- boot ------------------------------------------------------------------
let editor;

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
  $("clearBtn").addEventListener("click", async () => {
    if (!confirm("Wipe all frames and labels from this session?")) return;
    await state.store.clearAll();
    state.index = 0;
    await refreshFrames();
    await loadCurrent();
    $("ingestStatus").textContent = "session cleared";
  });

  await refreshFrames();
  await loadCurrent();
}

main();
