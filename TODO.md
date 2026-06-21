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

## 2-class classifier (built; blocked on data)

- [x] **Add a class head to the detector.** Done — `TinyDetector(num_classes=…)` returns
  `(box, logits)`, `fit(classify=True, val_frac=…)`, `Detector.detect()`, CLI `--classify`.
  Trained `helicoil-2class-v1.onnx` (gitignored under `images/`).
- [x] **Stabilize training.** Added flip augmentation, box/cls loss weighting, and
  best-checkpoint export to `fit` — training no longer exports a random late epoch.
- [ ] **Make the classes learnable — still blocked on visual separability.** Tried the
  full ladder: (1) original correct-vs-incorrect = chance (labels = intended state, defects
  don't render); (2) seated-vs-empty with 284 balanced images + augmentation + best-ckpt =
  **~79% best / ~60–65% typical, low confidence (~50–78%)**. Root cause (verified by eye,
  `PREDICTIONS_seated_empty.png`): an **empty *tapped* hole looks like a helicoil** — both
  are concentric thread-rings in a bore at 480px. The classes genuinely overlap. Real
  fixes, untried: (a) train at native **1024px** (no downscale) so coil-wire vs cut-thread
  texture is resolvable — most promising; (b) use a visually-distinct negative (plain
  *drilled/unthreaded* hole or no-hole) — but that's a different question than QC; (c)
  accept ~80% as the synthetic-data ceiling.

## Housekeeping

- [x] **Move the sample-generation script into the repo.** Done — lives at
  `packages/helicoils-synth/scripts/gen_samples.py` (source of truth for `SAMPLES.md`,
  supports `--reference` for image-conditioned generation).
- [ ] **Verify the `gpu` extra is minimal.** Confirm GroundingDINO actually needs
  `accelerate`; prune it from `helicoils-synth`'s `gpu` extra if not. (`diffusers`,
  `sentencepiece`, `protobuf` were already dropped with FLUX.)
- [ ] **Rotate the HF token** that was pasted in plaintext in an earlier chat transcript.
  Generation no longer needs HF, but GroundingDINO still pulls public weights from it.
