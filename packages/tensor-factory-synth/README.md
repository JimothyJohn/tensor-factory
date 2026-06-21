# tensor-factory-synth

Synthesize and auto-label a detection dataset. The running example is helicoils, but the
prompt and target features are parameters — point it at any object.

- **Light path (no torch):** `MockGenerator` renders deterministic, seeded coil images
  with known ground-truth boxes. The whole pipeline — generation → COCO → Label Studio
  pre-annotations — runs and tests with nothing GPU-shaped installed.
- **Real generation (`gemini` extra):** `NanoBananaGenerator` calls Nano Banana
  (`gemini-2.5-flash-image`) over the Gemini API — no GPU, no local weights, just a
  `GEMINI_API_KEY` in the environment. Emits 1:1 images downscaled to the target size.
- **Auto-labeling (`gpu` extra):** `GroundingDinoAutoLabeler` (transformers
  GroundingDINO, Apache-2.0). Resolves `cuda → mps → cpu`, so it runs on a CUDA box,
  this Mac Studio's MPS, or CPU.

```bash
# Prompt-iteration loop (mock backend, runs anywhere):
tensor-factory-synth sample --prompt "extreme macro of a helicoil in machined aluminum" --n 9 --out grid.png

# Build a labeled COCO dataset:
tensor-factory-synth dataset --prompt "..." --features helicoil --n 200 --out data/

# Real generation + auto-label (after: uv sync --extra gemini --extra gpu; needs GEMINI_API_KEY):
tensor-factory-synth --backend gemini dataset --prompt "..." --features helicoil --n 500 --out data/

# Report the review/validation state of a dataset:
tensor-factory-synth triage --data data/
```

## Review state

Every annotation is stamped with a `review` state and a `source` (see
[`tensor_factory.review`](../tensor-factory/src/tensor_factory/review.py)). GroundingDINO
auto-labels are written `pending` — they are guesses and **not trainable** until a human
validates them (via `tensor-factory-label`, which flips them to `approved`). The mock
generator's boxes are exact ground truth, so they are `approved` on creation. The
training loader enforces this gate by default, so unreviewed AI labels never enter a model.

Licensed under Apache-2.0.
