# Tensor Factory Studio

A browser-native, active-learning labeling tool for tensor-factory detectors.
Drop in a video, label frames in a video-game rhythm (mouse draws, left hand drives),
and export straight into the `tensor-factory-train` dataset format. See
[`PROMPT.md`](PROMPT.md) for the full vision and the locked design decisions.

**Slice 1 (this code):** video ingest with dHash dedup, the bounding-box canvas
editor, the WASD keymap, IndexedDB persistence, and COCO export. No build step, no
dependencies. **Slice 2 (next):** continuous in-browser WebGPU training, live
validation metrics, auto-labeling of the next frame, and the keep-best guardrail.

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
   commit. The rhythm: glance → `Space` if the (future) auto-label is right → otherwise
   drag a correction and `W`.
3. **Export** — pick a folder; Studio writes `annotations.coco.json` + `images/`
   (approved positives) and `negatives/images/` (approved empty frames), ready for
   `tensor-factory-train`.

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
js/app.js         wiring
```
