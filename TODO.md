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

- [ ] **Generate a real labeled dataset.** `helicoils-synth --backend gemini dataset ...`
  (Nano Banana generation + GroundingDINO auto-label) at meaningful N.
- [ ] **Human-review the labels** in Label Studio, then `helicoils-label pull` → COCO.
- [ ] **Train on real data.** `helicoils-train fit` on the reviewed COCO set; compare
  localization vs the current mock-trained demo model.
- [ ] **Persist datasets + models off `/tmp`.** They're ephemeral (lost on reboot); the
  only committed model is the 81 KB demo in `helicoils-mcp`. Pick a durable home (S3 or a
  tracked artifacts dir) before the real training run.

## Housekeeping

- [x] **Move the sample-generation script into the repo.** Done — lives at
  `packages/helicoils-synth/scripts/gen_samples.py` (source of truth for `SAMPLES.md`,
  supports `--reference` for image-conditioned generation).
- [ ] **Verify the `gpu` extra is minimal.** Confirm GroundingDINO actually needs
  `accelerate`; prune it from `helicoils-synth`'s `gpu` extra if not. (`diffusers`,
  `sentencepiece`, `protobuf` were already dropped with FLUX.)
- [ ] **Rotate the HF token** that was pasted in plaintext in an earlier chat transcript.
  Generation no longer needs HF, but GroundingDINO still pulls public weights from it.
