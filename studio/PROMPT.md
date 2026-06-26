# Tensor Factory Studio — design brief

> The vision and the hard decisions for the active-learning labeling + continuous
> training loop. Same role for Studio that [`examples/helicoils/PROMPT.md`](../examples/helicoils/PROMPT.md)
> plays for the first example: this is where the *spirit* lives, ahead of the code.

## The spirit

Labeling and training are the same act, not two phases. Today the loop is: label a pile
in Label Studio → export → train somewhere else → discover the labels were mediocre →
go back. Studio collapses that into one screen. You label a frame, the model has already
retrained on everything you've labeled so far, and it auto-labels the *next* frame for
you. Most of the time you just confirm. The dataset and the detector improve together,
continuously, in front of you — a flywheel, not a pipeline.

The model is small on purpose (the same tiny soft-argmax CNN this repo already ships —
~81 KB int8). Small means it retrains in seconds on a laptop GPU, which is the whole
reason the loop can feel live. The same artifact plays **two roles**: it is the
**auto-labeler** that pre-fills your next frame, and it is the **final shipped model**.
There is no separate "labeling model" — making the auto-labeler better *is* making the
product better.

Kill Label Studio for this workflow. No server round-trips to a heavyweight Java app, no
import/export dance, no leaving the browser.

## The flywheel

```
   ┌──────────────────────────────────────────────────────────┐
   │                                                            │
   │   video frame ──► model auto-labels ──► you confirm/fix    │
   │        ▲                                      │            │
   │        │                                      ▼            │
   │   pick next DIVERSE frame            add to labeled set    │
   │   (feature-similarity                         │            │
   │    dedup vs dataset)                          ▼            │
   │        │                          background trainer       │
   │        │                          retrains continuously    │
   │        │                                      │            │
   │        └──────────── better model ◄───────────┘            │
   │                                                            │
   └──────────────────────────────────────────────────────────┘
            live metrics + guardrail watching the whole time
```

Every confirmed/corrected label feeds the set immediately. A background trainer is always
running on the current set. Metrics stream to the UI so you can *see* the model getting
better (or worse) as you label.

## Hard decisions (locked)

1. **Browser UI + local Python backend.** The browser does ingest, the canvas, and labeling;
   a local backend (`tensor-factory-studio`) owns the on-disk dataset and trains on the GPU.
   - *History:* this was first built browser-only with in-browser WebGPU/tfjs training. That
     ran (and is in the git history), but WebGPU fell back to CPU in headless verification,
     tfjs was a standing exception to the repo's Python/Rust stack, and the in-browser model
     was *not* the deployable artifact. We pivoted to the backend approach.
   - *Why the backend wins here:* it reuses the trusted `tensor_factory_train.fit` (torch on
     MPS/CUDA), so the model the UI trains **is** the canonical int8 ONNX — no parity gap. It
     trains for real on the GPU, and the whole loop is verifiable locally end to end.
   - *Cost:* no longer "just a browser" — you run `uv run tensor-factory-studio` first. Worth
     it for a correct, deployable model over a portable-but-divergent one.

2. **Continuous background retraining.** The trainer runs in a Web Worker on a loop over the
   current labeled set, not on a button. New labels are picked up on the next epoch.
   Pause/resume from the UI. You always see live "is it improving" feedback while labeling.

3. **Keep-best, flag, never silently overwrite.** Track the best validation metric seen. If a
   retrain regresses past a threshold (dataset got corrupted, a bad label slipped in,
   metrics drift down), Studio **keeps serving the best checkpoint**, raises a visible red
   flag, and surfaces *which recent samples most likely caused the regression* so you can
   fix or drop them. The good model is never clobbered by a bad epoch. This is the
   browser-native continuation of the constant-predictor / val-metric guardrail the trainer
   already carries.

## Interaction model — left hand drives, right hand draws

The mouse/trackpad is in the **right hand** and never leaves the canvas. Every workflow
action is a **left-hand key** on the WASD cluster, so labeling feels like a video game, not
a form. Defaults (rebindable):

| Key            | Action                                                              |
| -------------- | ------------------------------------------------------------------- |
| `A` / `D`      | previous / next frame                                               |
| `Space`        | **accept** the model's auto-label as-is (the one-key fast path)     |
| `W`            | commit my drawn/corrected box(es) and advance                       |
| `S`            | skip this frame (don't add to set)                                  |
| `C`            | mark frame as **empty / negative** (presence = 0) — trains the no-object head |
| `Q` / `E`      | cycle active class (multi-class)                                    |
| `1`–`5`        | select class directly                                               |
| `Z`            | undo last box                                                       |
| `X`            | delete selected box                                                 |
| `R`            | clear all boxes on this frame                                       |
| `F`            | flag frame for later review                                         |
| `Esc`          | cancel the box currently being drawn                                |

The intended rhythm: a frame appears already auto-labeled → glance → `Space` if right (most
frames) → if wrong, drag a correction with the mouse and tap `W`. Thumb on `Space`, fingers
on WASD, mouse draws. Hundreds of frames without touching the right side of the keyboard.

## Video-native, with similarity dedup

Studio ingests **video**, not folders of stills. Consecutive frames are nearly identical and
add no signal, so before a frame is ever shown for labeling Studio computes a cheap
embedding and **skips frames too similar to what's already in the dataset** — you only spend
attention on frames that are genuinely *new* to the model. This is the live, in-loop version
of the repo's existing diverse-frame extractor: instead of pre-extracting a diverse set,
diversity is enforced continuously against the growing labeled set.

The cheapest version reuses what the repo already has: `extract_frames.py` computes a 64-bit
**dHash** of the centre square and keeps a frame only if its Hamming distance to every kept
frame exceeds a threshold. Studio ports that same dHash gate to the browser and runs it
against the *labeled set* live. A sharper version swaps in the detector backbone's
penultimate features for semantic (not just pixel) similarity — but dHash is the proven,
zero-dep starting point.

## The labels

- A label is a **bounding box + a class**. Start with **one class** — effectively just "a
  box, when something is present" — so the label-class plumbing doesn't matter yet. The UI
  must already support **creating multiple classes** of boxes, ready to switch on later.
- A box round-trips through this repo's contract: four `uint8` values, normalized `xyxy`,
  via `tensor_factory.codec` / `BBox`. Studio stays inside that contract so its output is
  the same dataset the Python tooling consumes — no new format.
- **Presence is first-class.** Empty frames (`C`) are real training signal for the presence
  head, not discarded. A labeled set is positives *and* reviewed negatives, matching how the
  bundled `cam-v1` model was trained.

## Persistence & export

- The **browser** keeps frames + labels in **IndexedDB** so the UI survives a reload; the
  **backend** owns the source-of-truth dataset and the trained checkpoints on disk.
- The backend writes the exact layout the rest of tensor-factory reads —
  `annotations.coco.json` + `images/` (+ `negatives/images/`), every box stamped
  `review=approved` / `source=human`, presence carried via empty-frame samples — and the
  trained **int8 ONNX** alongside. So the dataset and the deployable model drop straight into
  `tensor-factory-train`, the `detect` CLI, and the MCP/HTTP serving surfaces with no
  conversion step. The browser's *Export dataset* also writes the same COCO layout client-side.

## Real-time feedback

The UI always shows, live: current vs best validation metric (box center-error, presence
accuracy), the metric history sparkline, labeled-sample count (pos / neg), trainer state
(running / paused / regressed), and the guardrail status. The point is that you can *watch*
the model converge and immediately feel the cost of a bad label.

## Non-goals (for now)

- Multi-user / collaborative labeling. Single operator, single machine.
- Cloud / hosted training. The backend is local-only (`127.0.0.1`); a remote GPU box would
  be a later optimization, not the architecture.
- Segmentation masks, keypoints, tracking. Boxes + class only.
- Warm-started incremental training. Each round retrains from scratch via `fit` (fast for a
  tiny model, and robust against drift); warm-start is a possible later optimization.

## Resolved in build

- **Training vehicle: `tensor_factory_train.fit` (torch), run by a backend thread.** Reused
  wholesale via a small additive `on_epoch` hook added to `fit` — so the live loop and the
  CLI share one trained, exported int8 ONNX. Continuous retrain whenever new labels arrive;
  device resolves `cuda → mps → cpu`.
- **Dataset lives server-side** at `<data-dir>/annotations.coco.json` (+ `images/` and
  `negatives/images/`), exactly the layout `fit` reads — positives stamped
  `review=approved`/`source=human`, empty frames as the presence head's negatives. The
  browser keeps an IndexedDB copy for UI state and pushes approved frames to `POST /samples`.
- **Keep-best guardrail:** a round's checkpoint is promoted to the served model only if its
  best val center-error beats the global best; otherwise the previous model stays live and
  the round is flagged a regression with the most-recently-added sample ids as suspects.
- **Auto-label** is the served best model via `POST /predict` (the real `Detector`
  inference path); the browser pre-fills the box and you accept with `Space`.
- **Serving surfaces:** `tensor-factory-studio` is a stdlib `http.server` (no web framework),
  binds `127.0.0.1`, and mirrors the shape of `tensor-factory-http`.
- **Location:** UI at repo root (`studio/`), backend as a workspace package
  (`packages/tensor-factory-studio`).
