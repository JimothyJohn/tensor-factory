# Tensor Factory Studio

A browser-native, active-learning labeling tool for tensor-factory detectors.
Drop in a video, label frames in a video-game rhythm (mouse draws, left hand drives),
and export straight into the `tensor-factory-train` dataset format. See
[`PROMPT.md`](PROMPT.md) for the full vision and the locked design decisions.

**What works now:** video ingest with dHash dedup, the bounding-box canvas editor, the
WASD keymap, IndexedDB persistence, COCO export — **and the active-learning loop**:
continuous in-browser training (TensorFlow.js, WebGPU → WebGL → CPU) in a Web Worker,
live validation metrics vs the constant-predictor baseline, auto-labeling of unlabeled
frames, a keep-best guardrail, and model-weight export. No build step. The only
dependency is TensorFlow.js, vendored under [`vendor/`](vendor/) (Apache-2.0) so the app
stays self-contained and offline-capable.

## Run

WebGPU and the File System Access API (used for export) need a **secure context** and a
**Chromium-based browser** — `localhost` counts as secure, `file://` does not. Serve the
directory and open it:

```sh
cd studio
python3 -m http.server 8000
# open http://localhost:8000
```

## Workflow

1. **Load video** — frames are sampled at `fps`; any frame within `dedup` Hamming bits
   of one already in the set is skipped (so you label novel frames, not 100 near-dupes).
2. **Label** — draw a box with the mouse; the left-hand keys move you through frames and
   commit. The trainer starts automatically once there are ≥3 train / ≥2 val approved
   samples; from then on each new frame is **pre-filled with the model's prediction**.
   The rhythm: glance → `Space` if the auto-label is right → otherwise drag a correction
   and `W`. The sidebar shows live val center-error vs the constant-predictor floor,
   presence accuracy, the soft-argmax gain, and a guardrail that keeps the best
   checkpoint and flags regressions (with the likely-bad sample IDs).
3. **Export dataset** — pick a folder; Studio writes `annotations.coco.json` + `images/`
   (approved positives) and `negatives/images/` (approved empty frames), ready for
   `tensor-factory-train`. **Export model** downloads the trained weights as JSON
   (`TinyDetector` architecture, keyed to match the Python model's params).

The canonical deployable **int8 ONNX** is still produced by `tensor-factory-train` from
the exported COCO (the proven path that made the bundled models); the in-browser model is
the live auto-labeler and feedback signal. Tune the browser model with URL params:
`?size=192&width=12&batch=8` (defaults shown).

## Keys (left hand · mouse stays on the canvas)

| Key     | Action                              | Key   | Action                        |
| ------- | ----------------------------------- | ----- | ----------------------------- |
| `A`/`D` | previous / next frame               | `Z`   | undo last box                 |
| `Space` | accept auto-label, advance          | `X`   | delete selected box           |
| `W`     | commit boxes, advance               | `R`   | clear all boxes               |
| `S`     | skip frame (leave unlabeled)        | `F`   | flag frame for review         |
| `C`     | mark empty / negative               | `Esc` | cancel the box being drawn    |
| `Q`/`E` | cycle class                         | `1–5` | select class N                |

## Data & persistence

Everything (frames, labels) lives in IndexedDB under `tensor-factory-studio`, so a
session survives a reload. **Clear session** wipes it. The labels round-trip through the
repo's `BBox` / 4×uint8 contract (ported in [`js/codec.js`](js/codec.js)), and export
honors the `review=approved` gate — only committed frames leave the browser.

## Layout

```
index.html        app shell
studio.css        styling
js/codec.js       4×uint8 codec + BBox helpers (port of tensor_factory.codec)
js/dhash.js       64-bit dHash + Hamming (port of extract_frames.py)
js/store.js       IndexedDB wrapper (frames / labels / meta)
js/video.js       fps sampling + dHash dedup ingest
js/canvas.js      bounding-box canvas editor
js/keymap.js      WASD keymap + help table
js/export.js      COCO + negatives writer (File System Access API)
js/trainer.worker.js  in-browser tiny detector: tfjs model, soft-argmax head,
                      masked loss, continuous train loop, val metrics, guardrail
js/trainer.js     main-thread controller wrapping the worker
js/app.js         wiring
vendor/           vendored TensorFlow.js (see vendor/README.md)
```
