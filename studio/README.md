# Tensor Factory Studio

An active-learning labeling tool for tensor-factory detectors, built on one premise:
**the human judges, the machine labors.** You don't grind through frames — you
orchestrate: nudge a box when the model's guess is off, veto a bad sample, and let the
loop do the rest. The **browser** handles video ingest, the bounding-box canvas, and the
video-game-style WASD labeling; a **local Python backend**
([`tensor-factory-studio`](../packages/tensor-factory-studio)) owns the on-disk dataset
and trains the tiny detector continuously on this machine's GPU (MPS/CUDA) as you label.
See [`PROMPT.md`](PROMPT.md) for the vision and the decision history.

The loop is reactive: label a frame → the detector retrains on everything labeled so far
→ the next frame comes back **pre-filled with the model's prediction**. Your effort
shrinks as the model grows:

- **Near-duplicates never reach you** — a dHash gate at ingest skips frames too similar
  to ones already labeled, so every frame on screen is worth your attention.
- **Mostly you confirm.** Glance → `Space` → next. When the pre-fill is off, drag the
  correction and `W` — that nudge is precisely the sample the model was missing.
- **Bad data gets caught, not collected.** Machine guesses never enter the dataset
  unconfirmed (auto-label is opt-in; you commit every box), and the keep-best guardrail
  flags regressed rounds with the likely-bad sample ids, clickable to jump there.

The dataset and the detector improve together, live.

## What works

- **Browser:** video ingest sampling at N fps with dHash dedup (skips frames too similar to
  ones already labeled), the bounding-box canvas editor, the WASD keymap, IndexedDB
  persistence, and File-System-Access COCO export.
- **Backend:** continuous training via `tensor_factory_train.fit` (the same path that makes
  the repo's bundled models, so it emits the canonical **int8 ONNX**), live per-epoch val
  metrics vs the constant-predictor baseline, auto-labeling from the best checkpoint, and a
  keep-best guardrail that won't let a regressed round overwrite a good model.
- **Device:** one button writes the current model to an ESP32-P4 on USB and reports what
  the board measured; or download the file for a P4 or for any Linux box.

## Run

```sh
uv sync --extra serve          # pulls torch (heavy); training resolves cuda → mps → cpu
uv run tensor-factory-studio   # serves UI + API on http://127.0.0.1:8089
# open the printed URL in a Chromium-based browser (File System Access export needs Chromium)
```

Flags: `--port --data-dir --ui-dir --size --width --context --resize --epochs --batch`. For
a snappier dev loop on CPU, a smaller model trains faster: `--size 96 --width 8 --epochs 4`.

### What it trains, and why

The defaults are the model that held up on a microcontroller, not the 480 px one the
bundled demo uses:

- **`--size 224`.** On a couple of hundred frames, 160 to 320 px could not be told apart
  from 480 px, and 224 px runs in 15 ms on an ESP32-P4 against 76 ms.
- **`--context 4`.** Four extra blocks so the detector sees a 159 px window instead of
  31 px. Without them it only works when the object fills a fixed view; on photos where
  the object changes size it never beat "always guess the average box".
- **`--resize mixed`.** Training images are scaled with bilinear or nearest-neighbour,
  picked per sample, so a device that can only subsample sees pixels the model knows. It
  was also the better model on the desktop.

`--size 480 --context 0 --resize bilinear` is the old behaviour.

The trainer panel opens with one sentence: whether the model is finding the object or is
no better than always guessing the average box, and whether the held-out set is big enough
to trust (under 30 held-out boxes it says so). Decoded images are cached between rounds, so
a retrain after one new label decodes one new file: 12 epochs on 212 frames took 14.9 s
before and about 5 s now.

### Train on data you already have

```bash
uv run tensor-factory-studio --source path/to/dataset --source path/to/another
```

Each `--source` is a dataset dir (`annotations.coco.json` + `images/`): an earlier Studio
session's data dir, a synthetic set, a Label Studio pull. They are read in place and train
together with the frames you label here. With a source given the trainer starts at once,
so there are predictions to correct from the first frame. The trainer panel lists every
source with its own held-out error and presence accuracy, updated each epoch, so you can
see which data the model is weak on. `--allow-unreviewed` includes labels nobody has
approved yet.

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
4. **Put it on a device** — the *Device* block:
   - **Flash ESP32-P4.** Pick the board's serial port and press the button. The machine
     running the Studio quantizes the model (about 16 s the first time for a given model),
     writes it to the board's model partition (about 11 s, no firmware rebuild), and the
     panel shows what the board reported: self-test, milliseconds per inference, and how
     well its box for one of your frames overlaps the desktop's. The board needs
     [`firmware/esp32-p4`](../firmware/esp32-p4) flashed once, and this machine needs the
     P4 tools (the panel prints the install line when they are missing).
   - **ESP32-P4 image** downloads the same file to write yourself.
   - **Linux edge.pyz** downloads one file holding the runtime, an HTTP server and the
     model: `python3 edge.pyz --host 0.0.0.0` on any CPython 3.11+, then `POST /detect`.
   - **ONNX** is the plain int8 model.

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
  device panel                               GET  /model?format=p4|pyz -> device files
                                             POST /deploy?port=  -> write it to a P4 on USB
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
js/verdict.js     "is it finding it?" line  js/device.js   device panel + deploy client
```
