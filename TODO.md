# TODO

Open work for the tensor-factory pipeline (helicoils example). Roughly priority-ordered.
See [`CLAUDE.md`](CLAUDE.md) for current standing.

## Now

- [ ] **Train + promote a gain-trained v4 to the bundled MCP demo.** The soft-argmax gain
  (below) measurably improves localization; the bundle currently ships the pre-gain
  `helicoil-presence-v3.onnx`. Train a v4 *with* the learnable gain on `real_ds_combined` +
  `negatives_pool`, confirm it beats v3 on a fixed held-out set, then swap it into
  `tensor-factory-mcp/src/.../models/` as the default.
- [ ] **Box localization is the quality ceiling (~25 px median on real data).** The
  soft-argmax gain helped (see below); remaining limits are (1) GroundingDINO boxes are
  loose — they bound the whole boss, not tight on the insert — and (2) tiny-model capacity.
  Further levers: tighten the loosest labels, a wider model (`--width 24/32`), native
  1024 px. (The mock-trained model hits ~1.9 px because mock geometry is exact.)

## Recently done (this session)

- [x] **Merged the Nano Banana migration** into the base branch `master` (local
  fast-forward; this repo's default branch is `master`, not `main`).
- [x] **Promoted a real model to the bundled MCP demo.** `tensor-factory-mcp` now defaults
  to `helicoil-presence-v3.onnx` (real data + presence head); the synthetic
  `helicoil-mock-v1.onnx` stays bundled for the box-only path.
- [x] **HTTP serving surface.** `tensor-factory-http` — a stdlib `http.server` endpoint
  (zero extra deps) wrapping the same `core` inference: `POST /detect` (raw image bytes) →
  the same JSON as the MCP tool, plus `/health` and `/model_info`. Lighter than MCP.
- [x] **Soft-argmax gain (learnable inverse-temperature) on the coordinate head.** A plain
  softmax of a diffuse heatmap pulls the marginal expectation toward the centre; the gain
  lets the head sharpen first. A/B on `real_ds_combined` (frozen vs learnable, same
  seed/split, two seeds) — learnable wins both: median val center-error **43.4 → 27.4 px**
  (seed 0, gain → 1.35) and **39.3 → 13.0 px** (seed 1, gain → 1.06). The gain self-learns
  above 1.0 every time. Backward-compatible (init 1.0 = plain softmax); `--freeze-gain`
  ablates. Magnitudes are noisy on the small val split, but the direction is robust.

## Recently done (this session)

- [x] **Negative-aware training + presence head.** `--negatives DIR...` adds a trailing
  `background` class, masks box loss for box-less negatives, and trains the model to report
  *absent* instead of emitting a spurious box. `gen_negatives.py` synthesizes the
  machined-part negatives (holes/features, no helicoil). Closes the old "add empty-hole
  negatives / confidence head" item.
- [x] **`present` / `class_name` in the MCP.** `export_onnx` embeds `class_names` in the
  ONNX metadata (self-describing); `Detector` reads them; `core.detect` returns
  `present` (False only when `background` fired), `class_name`, `class_id`, `class_score`.
- [x] **Reference-conditioned generation for realism.** `build_ds.py <out> <n> <reference>`
  conditions every Nano Banana generation on a real part photo — generated positives now
  match the actual cast-aluminum application, not the generic microscope look.
- [x] **More positives.** `real_ds_more` — 168 reference-conditioned, GroundingDINO-labeled
  positives. Combined with `real_ds` (110) → `real_ds_combined` (278) for training.
- [x] **Fast path past the manual-review bottleneck.** `--allow-unreviewed` trains directly
  on GroundingDINO labels; a QC contact-sheet (render N images + boxes) replaces hours of
  per-box clicking. Label Studio is now optional, for fixing specific bad labels.
- [x] **`relabel.sh` parameterized by dataset dir** — shared Label Studio, a per-dataset
  image server, so several datasets label concurrently.
- [x] **Pull `file_name` bug fixed** — the Label Studio pull stored the image URL as the
  COCO `file_name`; now inverted back to the dataset-relative path (regression-tested).

## Next — dataset + model quality

- [ ] **Validate / tighten `real_ds_more` labels (optional, for box precision).** 168
  GroundingDINO boxes, 39 corrected so far in Label Studio project 4. Only worth it if
  chasing box precision — present/absent is already fine on the raw labels.
- [ ] **Multi-dir training support.** Combining datasets is currently a manual merge into
  one dir (`real_ds_combined`). Add multi-`--data` support to `fit`/the loaders so datasets
  compose without copying images.
- [ ] **Push precision higher (optional).** More data + native 1024 px (resolve coil
  texture) and/or a wider model for the subtle, recessed helicoils.

## 2-class classifier (built; the presence head now covers the absent case)

- [x] **Class head on the detector.** `TinyDetector(num_classes=…)` → `(box, logits)`,
  `fit(classify=True)`, `Detector.detect()`, CLI `--classify`. Mean+max pooled class head
  (mean alone washed out coil-vs-smooth texture).
- [x] **Best-checkpoint export + flip augmentation + box/cls loss weighting** in `fit` —
  no more shipping a random late, overfit epoch. Val selection scores class accuracy over
  all items and box error on positives only (so a collapsed box can't hide behind accuracy).

## Housekeeping

- [x] **`examples/helicoils/images/` fate** — gitignored (scratch datasets/models; the
  reference photo `Helicoil-Application.jpg` is gitignored too, pending provenance).
- [ ] **Verify the `gpu` extra is minimal.** Confirm GroundingDINO actually needs
  `accelerate`; prune from `tensor-factory-synth`'s `gpu` extra if not.
- [ ] **Rotate the HF token** pasted in plaintext in an earlier chat transcript.
  Generation no longer needs HF, but GroundingDINO still pulls public weights from it.
