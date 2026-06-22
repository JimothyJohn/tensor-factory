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

## Scripts (`scripts/`)

Beyond the CLI, three scripts drive the helicoils example:

- `gen_samples.py` — the reusable QC-sample batch behind `SAMPLES.md` (`--reference` to
  condition on a real photo).
- `build_ds.py <out> <n> [reference]` — build a real positive dataset (Nano Banana generate
  → GroundingDINO label → COCO). The optional 3rd arg conditions every generation on a real
  part photo, so the output matches the actual application domain rather than generic
  text-to-image.
- `gen_negatives.py` — synthesize machined-part **negatives** (holes/features, no helicoil)
  for the presence head. Raw, unlabeled background images; same visual domain as the
  positives. Pair with `tensor-factory-train --negatives`.

## Review state

Every annotation is stamped with a `review` state and a `source` (see
[`tensor_factory.review`](../tensor-factory/src/tensor_factory/review.py)). GroundingDINO
auto-labels are written `pending` — guesses, not trainable until validated (via
`tensor-factory-label`, which flips them to `approved`). The mock generator's boxes are
exact ground truth, so they are `approved` on creation. The training loader enforces this
gate **by default**, so unreviewed AI labels never enter a model by accident — but
`tensor-factory-train --allow-unreviewed` opts into raw labels deliberately (the fast path).

Licensed under Apache-2.0.
