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
  persistence, File-System-Access COCO export, and **training itself, on your GPU through
  WebGPU**: live per-epoch held-out error against the constant-predictor baseline,
  auto-labeling from the best checkpoint, a keep-best guardrail, and export of the
  canonical **int8 ONNX**.
- **Local helper (optional):** `uv run tensor-factory-studio` serves the same page and
  adds what a browser cannot do: one button writes the model to an ESP32-P4 on USB and
  reports what the board measured, `edge.pyz` for any Linux box, GroundingDINO auto-label,
  and training on datasets already on disk (`--source`).

## Run

Either open the hosted page (linked from the project site, under `studio/`), or in the repo:

```sh
uv run tensor-factory-studio   # http://127.0.0.1:8089, a few seconds, no torch download
```

Use a browser with WebGPU (a current Chrome or Edge; File System Access export needs
Chromium anyway). Label four frames and it starts training. Nothing is uploaded: frames
stay in the page's IndexedDB.

Where training runs is decided when the page loads, and the Trainer panel says which:

| Situation | Trains on |
|---|---|
| WebGPU available | the browser, whether or not the helper is running |
| helper started with `--source` | the helper (only it can read those folders) |
| no WebGPU, helper running with torch (`uv sync --extra serve`) | the helper (cuda → mps → cpu) |
| neither | nothing: the panel says what to install |

`?trainer=backend` or `?trainer=browser` on the page URL forces one when both exist.

Flags: `--port --data-dir --ui-dir --size --width --context --resize --epochs --batch`. The
browser trainer follows the helper's `--size --width --context --resize`, and takes
`?size=&context=&resize=&epochs=` on the page URL when there is no helper.

### The browser trainer is the same model, and that is tested

A second trainer is only worth having if its numbers mean the same thing. Each layer is
held to the Python one:

- **The math** (`js/nn/ref.js`): forward pass, loss, every gradient and three Adam steps
  equal what torch computes for `TinyDetector`, to 1e-8 or better (`tests/nn_ref.test.mjs` against
  `tests/fixtures/torch_step.json`).
- **The GPU kernels** (`js/nn/gpu.js`, plain WGSL, no library): the same checks in a real
  browser, to 3e-5 in float32 (`python3 tests/gpu/run.py`; add `--train` for a whole
  training run). Not in CI: it needs a GPU.
- **The pixels** (`js/nn/data.js`): bilinear and nearest resize are byte-identical to
  Pillow's (`test_browser_resize.py`).
- **The file** (`js/nn/onnx.js`): the exported ONNX has the same uint8 weights, bit for
  bit, as `tensor_factory_train.export_onnx` for the same parameters, and onnxruntime
  gives the same outputs from both (`test_browser_export.py`).

Measured on an M-series Mac (Metal): 49 ms a training step at 224 px, context 4, batch 16,
roughly half the speed of torch on the same GPU. 120 frames (96 for training), 14 epochs,
including decoding and resizing them: 5.7 s.

What the browser trainer does not do yet: more than one class, flip augmentation, and
per-source tables. Use the helper with torch for those.

### Train on data you already have

```bash
uv sync --extra serve
uv run tensor-factory-studio --source path/to/dataset --source path/to/another
```

Each `--source` is a dataset dir (`annotations.coco.json` + `images/`): an earlier Studio
session's data dir, a synthetic set, a Label Studio pull. They are read in place and train
together with the frames you label here, on the helper. With a source given the trainer
starts at once, so there are predictions to correct from the first frame. The trainer panel
lists every source with its own held-out error and presence accuracy, updated each epoch,
so you can see which data the model is weak on. `--allow-unreviewed` includes labels nobody
has approved yet.

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
to trust (under 30 held-out boxes it says so). Resized frames are kept between rounds, so
a retrain after one new label decodes one new file.

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
   - **Flash ESP32-P4.** Needs the local helper. Pick the board's serial port and press the
     button. A model trained in the browser is handed to the helper first, with up to 32 of
     your labeled frames to calibrate the quantizer. The machine running the helper
     quantizes the model (about 16 s the first time for a given model),
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

Training moved into the browser once the three reasons it had left were gone: it is
checked in a real headless browser with a real GPU, it uses no library (hand-written
WGSL), and its export is the same file the torch trainer writes. `PROMPT.md` has the
history.

## Layout

```
index.html        app shell                js/keymap.js   WASD keymap + help table
studio.css        styling                  js/export.js   COCO writer (File System Access API)
js/codec.js       4×uint8 codec + BBox      js/trainer.js  REST client for the backend
js/dhash.js       dHash + Hamming           js/app.js      wiring
js/store.js       IndexedDB wrapper
js/video.js       fps sampling + dedup      js/canvas.js   bounding-box canvas editor
js/engine.js      browser or backend?       js/local.js    the in-browser training loop
js/nn/ref.js      the math, on the CPU      js/nn/gpu.js   the same math in WGSL (WebGPU)
js/nn/data.js     resize, split, scoring    js/nn/onnx.js  int8 ONNX writer
models/           ONNX templates the writer fills (tools/make_templates.py)
js/verdict.js     "is it finding it?" line  js/device.js   device panel + deploy client
```
