# Tensor Factory Studio

An active-learning labeling tool for tensor-factory detectors. The **browser** handles
video ingest, the bounding-box canvas, and the video-game-style WASD labeling; a **local
Python backend** ([`tensor-factory-studio`](../packages/tensor-factory-studio)) owns the
on-disk dataset and trains the tiny detector continuously on this machine's GPU (MPS/CUDA)
as you label. See [`PROMPT.md`](PROMPT.md) for the vision and the decision history.

Label a frame → it's pushed to the backend → the detector retrains on everything labeled so
far → the next unlabeled frame comes back **pre-filled with the model's prediction**. Most
frames you just confirm. The dataset and the detector improve together, live.

## What works

- **Browser:** video ingest sampling at N fps with dHash dedup (skips frames too similar to
  ones already labeled), the bounding-box canvas editor, the WASD keymap, IndexedDB
  persistence, and File-System-Access COCO export.
- **Backend:** continuous training via `tensor_factory_train.fit` (the same path that makes
  the repo's bundled models, so it emits the canonical **int8 ONNX**), live per-epoch val
  metrics vs the constant-predictor baseline, auto-labeling from the best checkpoint, and a
  keep-best guardrail that won't let a regressed round overwrite a good model.

## Run

```sh
uv sync --extra serve          # pulls torch (heavy); training resolves cuda → mps → cpu
uv run tensor-factory-studio   # serves UI + API on http://127.0.0.1:8089
# open the printed URL in a Chromium-based browser (File System Access export needs Chromium)
```

Flags: `--port --data-dir --ui-dir --size --width --epochs --batch`. For a snappier dev
loop on CPU, a smaller model trains faster: `--size 96 --width 8 --epochs 4`.

## Workflow

1. **Load video** — frames are sampled at `fps`; any within `dedup` Hamming bits of one
   already labeled is skipped, so you spend attention on novel frames.
2. **Label** — draw a box with the mouse; the left-hand keys move you through frames and
   commit. The trainer starts automatically once ≥4 positives are labeled; from then on each
   new frame is pre-filled with the model's prediction. Glance → `Space` to accept → or drag
   a correction and `W`. The sidebar shows live val center-error vs the baseline floor,
   presence accuracy, the soft-argmax gain, and a guardrail that flags regressions (with the
   likely-bad sample ids, clickable to jump there).
3. **Export** — *Export dataset* writes the COCO dataset client-side; *Export model*
   downloads the backend's trained **int8 ONNX** (`/model`). The backend's own
   `<data-dir>/annotations.coco.json` is the same layout `tensor-factory-train` reads.

## Keys (left hand · mouse stays on the canvas)

| Key     | Action                              | Key   | Action                        |
| ------- | ----------------------------------- | ----- | ----------------------------- |
| `A`/`D` | previous / next frame               | `Z`   | undo last box                 |
| `Space` | accept auto-label, advance          | `X`   | delete selected box           |
| `W`     | commit boxes, advance               | `R`   | clear all boxes               |
| `S`     | skip frame (leave unlabeled)        | `F`   | flag frame for review         |
| `C`     | mark empty / negative               | `Esc` | cancel the box being drawn    |
| `Q`/`E` | cycle class                         | `1–5` | select class N                |

## Architecture

```
browser (studio/)                         backend (packages/tensor-factory-studio)
  ingest + dHash dedup                       POST /samples  -> writes COCO dataset, marks dirty
  canvas + WASD labeling     --- HTTP -->    GET  /metrics  -> live training metrics (polled)
  IndexedDB (UI state)                       POST /predict  -> auto-label via best checkpoint
  metrics panel + sparkline                  GET  /model    -> trained int8 ONNX
                                             (continuous torch training thread, MPS/CUDA)
```

Why a backend and not in-browser training: it reuses the trusted torch trainer (so the
browser model *is* the deployable model), runs real GPU training, and is verifiable locally
end to end. The earlier in-browser WebGPU/tfjs approach is recorded in `PROMPT.md`.

## Layout

```
index.html        app shell                js/keymap.js   WASD keymap + help table
studio.css        styling                  js/export.js   COCO writer (File System Access API)
js/codec.js       4×uint8 codec + BBox      js/trainer.js  REST client for the backend
js/dhash.js       dHash + Hamming           js/app.js      wiring
js/store.js       IndexedDB wrapper
js/video.js       fps sampling + dedup      js/canvas.js   bounding-box canvas editor
```
