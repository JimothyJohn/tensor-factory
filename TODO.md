# TODO

Open work for the helicoils pipeline. Roughly priority-ordered. See
[`CLAUDE.md`](CLAUDE.md) for current standing.

## Now

- [ ] **Merge the Nano Banana migration.** Branch `nano-banana-generation` (FLUX → Gemini
  API) is committed but not merged — open a PR into `main`, get CI green, merge.
- [ ] **Decide the fate of `images/` + `SAMPLES.md`.** Currently untracked. Either commit
  them (sample gallery / fixtures) or add `images/` to `.gitignore` if it's scratch.
- [x] **Helicoil fidelity in generated images.** Fixed for flush / recessed / missing /
  damaged via "wound tight, no gaps, not a loose spring" + top-down angle + photoreal
  suffix (see `SAMPLES.md`). `slightly_proud` and `cross_threaded` still render spring-like
  from text alone — those need a reference photo (next item).
- [ ] **Regenerate conditioned on a real reference photo.** `gen_samples.py --reference`
  +`NanoBananaGenerator(reference=...)` are wired and verified. Need Nick to drop a real
  macro of an installed (and ideally a proud / cross-threaded) Helicoil; then re-run the
  batch so those states stop looking like springs.

## Next — the real dataset + model

- [x] **Generate a real labeled dataset.** Done — 142 Nano Banana images + GroundingDINO
  auto-label (feature `"threaded hole"`, full-frame false-positive filtered) → COCO at
  `/tmp/real_ds` (`annotations.coco.json` copied to `images/real_ds/`).
- [x] **Train on real data.** Done — `helicoil-real-v1.onnx` (156 KB int8, width 24, 80
  epochs, loss 0.0013→0.00035). **Localizes helicoils in held-out photoreal images** where
  the mock model produced a top-left corner box. Proof: `images/RESULT_before_after.png`.
- [ ] **Persist the builder + dataset into the repo.** `/tmp/build_ds.py` and the dataset
  images are still ephemeral. Land the script under `packages/helicoils-synth/scripts/` (or
  fold its varied-prompt + auto-label flow into the `dataset` CLI) and pick a durable home
  (S3 / tracked artifacts) for images + model.
- [ ] **Skipped human review.** Labels were trusted straight from GroundingDINO (no Label
  Studio pass). For a production model, review them; spot-checks looked tight.
- [ ] **Add empty-hole negatives.** Training used insert-present states only, so the model
  will likely fire on any tapped hole (incl. `missing`). Add negatives / a confidence head.
- [ ] **Consider swapping the bundled mcp demo model** from mock to `helicoil-real-v1`
  (left as the mock for now — committed artifact, not touched autonomously).

## Housekeeping

- [x] **Move the sample-generation script into the repo.** Done — lives at
  `packages/helicoils-synth/scripts/gen_samples.py` (source of truth for `SAMPLES.md`,
  supports `--reference` for image-conditioned generation).
- [ ] **Verify the `gpu` extra is minimal.** Confirm GroundingDINO actually needs
  `accelerate`; prune it from `helicoils-synth`'s `gpu` extra if not. (`diffusers`,
  `sentencepiece`, `protobuf` were already dropped with FLUX.)
- [ ] **Rotate the HF token** that was pasted in plaintext in an earlier chat transcript.
  Generation no longer needs HF, but GroundingDINO still pulls public weights from it.
