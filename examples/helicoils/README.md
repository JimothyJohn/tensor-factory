# Example: helicoils

The first example built on [tensor-factory](../../README.md) — detecting **helicoils**
(coiled-wire threaded inserts) in machined parts from microscope imagery. It exercises the
whole pipeline end to end: synthesize (positives **and** negatives) → auto-label →
validate → train (box + presence) → run on CPU.

- [`PROMPT.md`](PROMPT.md) — the original challenge brief and design decisions.
- [`SAMPLES.md`](SAMPLES.md) — the reusable prompt system + 25 QC-inspection samples.
- `images/` — generated samples and datasets (gitignored; indexed by `images/manifest.json`).
  Scripts under [`tensor-factory-synth/scripts/`](../../packages/tensor-factory-synth/scripts):
  `gen_samples.py` (QC samples), `build_ds.py` (real positives → COCO; optional reference
  photo as a 3rd arg for application-matched realism), `gen_negatives.py` (machined-part
  negatives, no helicoil). [`relabel.sh`](relabel.sh) `[DATA_DIR]` brings up the validation
  stack for a dataset.

## Pipeline (run from the repo root)

```bash
# 1. Synthesize + auto-label a dataset (mock backend runs anywhere; gemini/gpu extras for real).
#    GroundingDINO labels are written as review=pending -- AI guesses, not yet trainable.
uv run tensor-factory-synth --backend gemini dataset \
  --prompt "extreme macro of a helicoil in machined aluminum" \
  --features helicoil --n 500 --out data/

# 2. See what needs review.
uv run tensor-factory-synth triage --data data/

# 3. Human validation: push candidates into Label Studio, correct the boxes/classes, pull
#    them back. The pull stamps everything review=approved, source=human -- now trainable.
uv run tensor-factory-label push --data data/ --title "helicoil v1" --image-base http://localhost:8081
uv run tensor-factory-label pull --project <id> --out data/annotations.coco.json

# 3b. (optional) Synthesize negatives -- machined parts with holes but NO helicoil -- so the
#     model can report *absent* instead of always emitting a box.
uv run --with google-genai python packages/tensor-factory-synth/scripts/gen_negatives.py \
  --n 110 --out examples/helicoils/images/negatives_pool

# 4. Train a tiny int8 detector. By default ONLY approved annotations train; an all-pending
#    dataset is refused. --negatives turns on a YOLO-style objectness (presence) head and
#    trains it toward 'absent', so the model can return *no box* on a no-helicoil frame.
#    --allow-unreviewed trains straight on raw GroundingDINO labels.
uv run tensor-factory-train fit --data data/ --out model.onnx --epochs 45 --device mps \
  --negatives examples/helicoils/images/negatives_pool --allow-unreviewed

# 5. Run it on CPU. A presence-head model also reports present / score over MCP (one box or none).
uv run tensor-factory detect --model model.onnx --image some_frame.png
uv run tensor-factory bench --model model.onnx
```

**Fast path (skip hand-labeling).** Hand-correcting every GroundingDINO box is optional.
`--allow-unreviewed` trains directly on the AI labels; QC them by rendering a contact-sheet
of boxes (eyeball one image instead of clicking hundreds). Use Label Studio when you want
to fix *specific* bad labels, not as a gate.

## The human-validation gate (default-on, not mandatory)

By default, AI-labeled data is not trainable until a human has validated it. Every COCO
annotation carries `review` (`pending` / `approved` / `rejected`) and `source`
(`groundingdino` / `human` / `synthetic_gt`); see
[`tensor_factory.review`](../../packages/tensor-factory/src/tensor_factory/review.py).
GroundingDINO output is `pending`; the Label Studio pull flips it to `approved`.
`tensor-factory-train` loads only `approved` annotations **by default** — so unreviewed
labels don't leak in by accident — but `--allow-unreviewed` opts into raw AI labels
deliberately (the fast path). (Mock synthetic ground truth is exact, so it is `approved`
on creation. Negatives have no box and bypass the gate — a negative is a known absence.)

The bundled demo model in `tensor-factory-mcp` (`helicoil-mock-v1.onnx`) is the artifact
of this example, so the MCP server works with zero setup.

Licensed under Apache-2.0.
