# Tensor Factory Studio — design brief

> The vision and the hard decisions for the active-learning labeling + in-browser
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

1. **Browser-only. WebGPU.** No Python backend. Training, inference, dedup, and storage all
   run in the browser. Data lives in IndexedDB; the model trains on the WebGPU backend.
   - *Why this is even possible:* the model is tiny — a handful of conv layers + a
     soft-argmax coordinate head + a presence logit. In-browser autograd can handle it in
     seconds/epoch. This would be a non-starter for a 30 M-param transformer.
   - *The tradeoff, on record:* we give up the mature torch/MPS training path
     (`tensor-factory-train`) and re-implement forward+backward for the tiny net in the
     browser (realistically tfjs-WebGPU, or hand-rolled WGSL compute shaders to stay
     dependency-light). The Python trainer stays the reference/oracle for parity tests.
   - *Bonus already claimed:* "just needs a browser" — open the page, drop a video, label.

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

- Everything (frames, labels, model checkpoints, metric history) lives in **IndexedDB** so a
  session survives a reload with zero setup.
- **Export** produces the exact on-disk layout the rest of tensor-factory already reads —
  `annotations.coco.json` + `images/`, every box stamped `review=approved` / `source=human`
  so it's trainable, presence carried via empty-frame samples — plus the trained **int8
  ONNX** model. A Studio session drops straight into `tensor-factory-train`, the `detect`
  CLI, and the MCP/HTTP serving surfaces with no conversion step.

## Real-time feedback

The UI always shows, live: current vs best validation metric (box center-error, presence
accuracy), the metric history sparkline, labeled-sample count (pos / neg), trainer state
(running / paused / regressed), and the guardrail status. The point is that you can *watch*
the model converge and immediately feel the cost of a bad label.

## Non-goals (for now)

- Multi-user / collaborative labeling. Single operator, single machine.
- Cloud training or a hosted backend. If a GPU box is wanted later it's an *optimization*,
  not the architecture.
- Segmentation masks, keypoints, tracking. Boxes + class only.
- Replacing `tensor-factory-train` as the reference trainer — it stays the oracle for
  parity/regression tests against the in-browser implementation.

## Open questions to resolve in build

- In-browser training vehicle: tfjs-WebGPU (faster to working) vs hand-rolled WGSL
  (dependency-light, matches the repo's zero-dep-core ethos). Lean toward proving the loop
  with tfjs, then deciding whether WGSL is worth it.
- Train/val split inside the browser, and how the guardrail's "which samples caused the
  regression" attribution is computed cheaply.
- Whether Studio lives at repo root (`studio/`, as here) or graduates into a packaged form
  once shape settles.
