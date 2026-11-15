# tensor-factory

**Open, lightweight tiny-CNN object detection** — a factory for turning a prompt into a
synthetic dataset, an int8 ONNX model, and a CPU inference harness. No GPU at runtime, no
AGPL anywhere (Apache-2.0 throughout, no Ultralytics).

> 📄 Project page: [`docs/index.html`](docs/index.html) · 🔬 Live demo (runs in your browser): [`docs/demo.html`](docs/demo.html)

The pipeline: **synthesize → auto-label → train → run.**

1. **Synthesize** images from a prompt — hosted Gemini (Nano Banana) or a deterministic
   mock. Optionally synthesize negatives (no-object frames) too.
2. **Auto-label** with GroundingDINO; refine in Label Studio if you want, or train on the
   raw labels directly (`--allow-unreviewed`).
3. **Train** a tiny soft-argmax CNN and export an int8 ONNX model — optionally with a
   presence head (`--negatives`) so it can report *absent*, not just a box.
4. **Run** on CPU via onnxruntime — from a CLI or over MCP (box, plus `present`/`class_name`
   when the model has a presence head).

A detection is four `uint8` values (normalized `xyxy`, one byte each). At 480 px that's
~1.88 px per step — round-trip error under 1 px, inside the 3 px budget, and all
post-processing stays in 8-bit math.

**Reference numbers (helicoils example):** 204 fps CPU @480px · 81 KB int8 model · ~1.9 px
median localization on mock data (real photoreal data is the harder, open case).

## The Studio — supervise, don't label

[`studio/`](studio/) is the training UI, and the fastest way to go from a video to a
deployed detector. The premise: **you orchestrate, the machine does the labor.**

```sh
uv sync --extra serve && uv run tensor-factory-studio   # http://127.0.0.1:8089
```

- **Drop in a video and the busywork is already gone.** Frames are sampled at your fps
  and a perceptual-hash gate drops anything too similar to a frame you've already
  labeled — near-duplicates never reach your screen.
- **Nudge instead of drawing.** After a handful of boxes the trainer kicks in on its own
  and keeps retraining as you go; every next frame arrives **pre-filled with the
  model's current guess**. Most frames are a glance and a `Space`. When a guess is off,
  drag the box into place and commit — that nudge is exactly the training signal the
  model was missing.
- **Bad samples get caught, not collected.** Machine guesses never enter the dataset
  unconfirmed, and a guardrail watches validation error every round — when quality
  regresses it points at the likely-bad samples, clickable, so you fix or drop them and
  move on.
- **What you walk away with is the real thing.** Export is the canonical int8 ONNX from
  the same trainer that builds the repo's bundled models — the model you nudged is the
  model you ship.

The loop is reactive end to end: label → retrain → better pre-fills → less labeling.
Your attention goes only where the model is still uncertain; everything else is one
keypress. Full workflow and keymap: [`studio/README.md`](studio/README.md).

## Examples

- [`examples/helicoils`](examples/helicoils) — detecting coiled-wire threaded inserts in
  machined parts from microscope imagery. The first example, and what every package is
  currently tuned against.

## Packages

| Package | Role |
|---|---|
| [`tensor-factory`](packages/tensor-factory) | Core: BBox geometry, 4×uint8 codec, formats, ONNX inference + CLI (dependency-free, CPU-only) |
| [`tensor-factory-synth`](packages/tensor-factory-synth) | Generation (Nano Banana/Gemini) + GroundingDINO auto-label + COCO/Label Studio export |
| [`tensor-factory-train`](packages/tensor-factory-train) | Tiny soft-argmax detector → int8 ONNX |
| [`tensor-factory-mcp`](packages/tensor-factory-mcp) | FastMCP server exposing the detector (bundled demo model) |
| [`tensor-factory-label`](packages/tensor-factory-label) | Label Studio push/pull |
| [`tensor-factory-studio`](packages/tensor-factory-studio) | Backend for the [Studio](studio/) training UI: dataset, continuous trainer, auto-label |

## Develop

```bash
./Quickstart          # bootstrap (uv sync) + run the MCP detection server (try it out, no flag)
./Quickstart -c       # full gate: uv sync --locked → ruff check/format → ty check → pytest -m unit
./Quickstart -u       # unit tests only
```

Python `uv` workspace (monorepo). Generation is a hosted API call (no GPU); auto-labeling
and training are GPU-heavy and live behind extras, resolving `cuda → mps → cpu`.

Licensed under Apache-2.0.
