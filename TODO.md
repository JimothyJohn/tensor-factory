# TODO

Open work for the tensor-factory pipeline (helicoils example). Roughly priority-ordered.
See [`CLAUDE.md`](CLAUDE.md) for current standing.

## Now

- [ ] **First public push to GitHub.** The tree is push-ready; remaining mechanical steps:
  - [ ] `git remote add origin git@github.com:JimothyJohn/tensor-factory.git` and push `master`.
  - [ ] Decide the fate of `.claude/` (the docs-drift hook + skills): commit as shared project
    tooling, or leave local. `.claude/settings.local.json` is already gitignored.
  - [ ] Turn on branch protection for `master` (require the CI gate) once CI has run once.
  - [ ] Enable GitHub Pages from `docs/` so the landing page + live demo are hosted.
  - [ ] Enable private vulnerability reporting (Settings → Security) per `SECURITY.md`.
- [ ] **Model-in-the-loop pre-annotation (gated on v4 beating GroundingDINO).** Once the
  detector's boxes are tighter than the raw GroundingDINO auto-labels (measured against the
  human-approved subset as ground truth), close the loop: use the model itself to condition
  and pre-annotate new data — first the *synthesized* set (Nano Banana output), then *real*
  photos — replacing GroundingDINO as the label source. This bootstraps label quality off
  the model instead of the open-vocab detector that currently caps it. The bar to start is
  explicit: v4 median box error on the held-out approved set < the GroundingDINO labels'
  error on that same set. Until then, GroundingDINO stays the labeler.
- [ ] **Box localization is the quality ceiling (~20 px median on real data, down from
  ~25).** The soft-argmax gain moved it (v4 ~20 px held-out); remaining limits are (1)
  GroundingDINO boxes are loose — they bound the whole boss, not tight on the insert — and
  (2) tiny-model capacity.
  Further levers: tighten the loosest labels, a wider model (`--width 24/32`), native
  1024 px. (The mock-trained model hits ~1.9 px because mock geometry is exact.)

## Recently done (this session)

- [x] **In-browser demo (`docs/demo.html`).** Runs the bundled int8 models entirely
  client-side via onnxruntime-web (WASM, single-threaded so it works on plain static hosts):
  auto-detects a synthetic sample with the mock model on load, switch to the v4 presence
  model and upload a real photo. Linked from every doc page's nav. Contract-tested
  (`tests/test_demo.py`: demo models match the package byte-for-byte, load + detect via the
  same path the JS mirrors, class names match ONNX metadata, sample is a confirmed hit) plus
  an optional Playwright browser smoke (`tests/test_demo_browser.py`, `-m integration`).
- [x] **OSS contribution scaffolding.** `CONTRIBUTING.md` (humans + agents), `SECURITY.md`,
  `.github/` issue + PR templates, and a CI workflow mirroring `./Quickstart -c`.
- [x] **Consolidated the docs site onto `master`** and fixed v3→v4 drift in the pages.
- [x] **Merged the Nano Banana migration** into the base branch `master` (local
  fast-forward; this repo's default branch is `master`, not `main`).
- [x] **Promoted a real model to the bundled MCP demo.** `tensor-factory-mcp` now defaults
  to `helicoil-presence-v4.onnx` (real data + presence head + the soft-argmax gain); the
  synthetic `helicoil-mock-v1.onnx` stays bundled for the box-only path. (Briefly shipped
  the pre-gain v3 earlier this session before v4 was trained.)
- [x] **Trained + validated v4 (gain).** 80-epoch gain run on `real_ds_combined` +
  `negatives_pool`: best val box-median **20.7 px**, presence acc **84%**, gain learned to
  **1.39**. On v4's held-out split v4 beats the (leak-advantaged) v3 on box median **20.3 vs
  24.7 px**. A leak-free tail check (gain vs no-gain on identical held-out, both seeds)
  confirms the gain *shrinks* the outlier tail — >30 px miss rate 80%→41% and 79%→13%, with
  mean/p90/max all down — so the gain doesn't trade tail for median; it improves both.
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
