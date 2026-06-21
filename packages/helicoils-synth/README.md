# helicoils-synth

Synthesize and auto-label a helicoil detection dataset.

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
helicoils-synth sample --prompt "extreme macro of a helicoil in machined aluminum" --n 9 --out grid.png

# Build a labeled COCO dataset:
helicoils-synth dataset --prompt "..." --features helicoil --n 200 --out data/

# Real generation + auto-label (after: uv sync --extra gemini --extra gpu; needs GEMINI_API_KEY):
helicoils-synth --backend gemini dataset --prompt "..." --features helicoil --n 500 --out data/
```

Licensed under Apache-2.0.
