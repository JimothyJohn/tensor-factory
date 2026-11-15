# TODO

Open work for the tensor-factory pipeline (helicoils example). Roughly priority-ordered.
See [`CLAUDE.md`](CLAUDE.md) for current standing.

## Open questions for Nick

- [ ] **Promote dev to master?** GitHub Pages serves the site from `master`, and `master` is
  88 commits behind `dev` (checked 2026-11-15). Everything landed this session (zero-dependency
  inference, the Studio, multi-source training, the one-file edge bundle) is on `dev` only,
  so none of it is on the public site or the release line. Separately, the local `master` in
  the main checkout has 9 commits that were never pushed (the WATTAGE.md / margin.html
  economics study) and has diverged from `origin/master` by 8, so a plain push will be
  rejected until those are reconciled.
      options: merge dev into master now — the site and release line catch up; you reconcile the 9 local commits first | wait — master stays on the June/July state
      recommend: merge dev into master now — four months of work is invisible to anyone reading the site
      stakes: high
      result: the public site and `master` match what is tested on `dev`
      id: promote-dev-to-master
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **What should "cloud" be for this project?** The pipeline now runs data → training →
  one-file edge bundle, all on this Mac. Nothing is in AWS yet. The three candidates do
  different jobs and cost different amounts, so this is a direction call.
      options: S3 store for datasets and models — datasets stop living on one disk; the edge pulls a model by its sha256; a few cents a month | GPU training runs on EC2 — a script launches a g5/g6 spot instance, trains, uploads the model, shuts down; about $0.30–1.00 per run | hosted inference endpoint — a Lambda that serves /detect; pay per request, needs a public-access decision | nothing yet — stay local
      recommend: S3 store for datasets and models — every dataset is gitignored on a single machine today, which is the one unrecoverable loss in the pipeline, and a model store is what the edge and any later training job both need first
      stakes: high
      result: a concrete cloud piece gets built instead of guessed
      id: cloud-direction
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **Which board is the edge target, and may I deploy to it?** The one-file bundle
  (`edge.pyz`) is proven on aarch64 Linux in CI but has never run on a physical board.
  Deploying means copying one file and starting a user-level systemd service on that host.
      options: name a host — I copy the bundle, start the service, and report frames per second and the model sha256 from the device | none yet — the CI proof stands until a board is chosen
      recommend: name a host — real throughput on the real board is the number the whole "tiny CNN on CPU" claim rests on
      stakes: low
      needs: the hostname or address of a Pi-class board on the LAN, and whether anything else is running on it
      result: measured fps and a running detector on the target hardware
      id: edge-target-host
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **Which landing page wins: the restyled page or the single-file detection app?**
  `docs/index.html` on `dev` is the restyled landing page. The local branch `feat/index-app`
  (worktree `../tensor-factory-index-app`, never pushed) replaces the same file with a
  self-contained in-browser detection app (2.4 MB, live camera, YOLOX-Nano). They cannot
  both be `index.html`.
      options: restyled page stays, app moves to docs/app.html — both ship, no conflict | app becomes index.html — the demo is the front door; the restyled copy is dropped | drop the app — delete the branch and worktree
      recommend: restyled page stays, app moves to docs/app.html — keeps both pieces of work and turns a merge conflict into a new file
      stakes: low
      result: `feat/index-app` lands or is deleted instead of sitting unpushed
      id: landing-page-choice
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **Dismiss the two remaining code-scanning alerts as false positives?** Five of the
  seven alerts raised when the new code was first scanned are fixed. Two remain where no code
  change is right. Alert 11 (`studio/js/video.js`) flags `video.src = URL.createObjectURL(file)`
  as cross-site scripting; the value is a `blob:` address the browser creates from a file,
  never markup. Alert 6 (`zerort/imageio.py`) flags reading an image from a caller-supplied
  path; that is what the command-line tool and the local MCP tool are for, and the HTTP
  endpoint takes bytes, not a path.
      options: dismiss both — the security tab goes clean; reversible | leave open — they stay as a standing reminder | restrict paths in code — the MCP tool would only read under an allowed directory
      recommend: dismiss both — neither is reachable by anyone who could not already read the file themselves
      stakes: low
      id: codeql-false-positives
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **Protect master and turn on private vulnerability reporting?** Both were on the
  original launch list and neither is set (checked 2026-11-15): `master` has no branch
  protection, and private vulnerability reporting is off although `SECURITY.md` tells
  reporters to use it. These are repository settings on your release branch, so I left them.
      options: enable both — master requires the `gate` check; reporters get the private channel SECURITY.md promises | leave as is
      recommend: enable both — SECURITY.md currently points at a channel that does not exist
      stakes: low
      id: master-protection-and-pvr
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **Keep `dev` as the GitHub default branch?** GitHub's default branch is `dev`, while
  `CLAUDE.md` says the default is `master`. Dependabot and CodeQL read their configuration
  from the default branch, so today they work because the config lives on `dev`.
      options: keep dev as default, fix the docs — nothing changes in behavior | make master the default — Dependabot and CodeQL config must be promoted to master first or they go quiet
      recommend: keep dev as default, fix the docs — it is the arrangement that is already working
      stakes: low
      id: default-branch
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **Keep or drop `wip/recovered-docs-site-20260623`?** A snapshot branch another session
  made on 2026-06-23 of an orphaned docs-site edit; 4 commits that are not on `dev`. The docs
  site has since been consolidated and restyled twice.
      options: drop — delete the branch | keep — I open a PR so you can read the 4 commits against the current site
      recommend: drop — the pages it touched have been rewritten since
      stakes: low
      id: recovered-docs-site
      asked: 2026-11-15
      by: fable-2026-11-15

- [ ] **Keep or drop `wip/recovered-margin-economics-20260702`?** A second snapshot of the
  WATTAGE.md / margin.html economics study. The same work is in your unpushed local `master`
  (see the first card), so this copy is a duplicate. The worktree
  `../tensor-factory-wattage-notes-20260702` and its branch are the same story.
      options: drop all three — delete the duplicate branch, the wattage-notes branch and its worktree, once master is pushed | keep until master is pushed
      recommend: keep until master is pushed — they are the only other copy of that work until then
      stakes: low
      depends: promote-dev-to-master
      id: recovered-margin-economics
      asked: 2026-11-15
      by: fable-2026-11-15

## Now

- [x] **First public push to GitHub.** The repo is public with `dev` protected by the `gate`
  check, `.claude/` is tracked as shared tooling, and GitHub Pages serves `docs/` from
  `master`. The two settings still missing (master protection, private vulnerability
  reporting) are a question card above.
- [ ] **Run the edge bundle on a real board.** Copy `edge.pyz` to the target, start it as a
  user systemd service, record frames per second with and without onnxruntime, and confirm
  `/model_info` reports the sha256 the build printed.
      depends: edge-target-host
      est: 1h
      difficulty: low
      id: edge-on-real-board
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

## Model 2 — advanced manifestation detection (v6, ⚠ BLOCKED on labels)

**Two models, deliberately separate.** Model 1 = **basic object detection** (v5, box+presence,
shipping). Model 2 = **this**: damage + insertion manifestations. **Model 2 is blocked — no
dataset has damage/seating labels** (all current data is box+presence, state "ok"). Item 3 is
the blocker; items 4–6 wait on it. Item 1 (architecture) is done and unit-tested but
**untrained**. Real-data training goes via Gemini/Nano Banana (`GEMINI_API_KEY`) + a labeling
pass — there is no synthetic shortcut that produces meaningful damage/seating labels.

Capacity/roadmap analysis: [`MODEL_SIZE.md`](MODEL_SIZE.md). v5 (box + YOLO-style objectness)
is the tiny base and doubles as the **stage-1** region proposer; v6 adds a **stage-2 crop
head** that reports state on the high-res crop. Insertion is a *measurement* (regress a signed
seating depth → over/correct/under), damage is a *classification*
(`ok/deformed/cross_threaded/broken_tang/burr`). A PC teacher then trains + pre-annotates and
distils down to the deployment student. Spend resolution only on the crop.

**Target platform (decided 2026-07-02): a tiny ARM PC** — a Linux-class aarch64 SBC
(Raspberry-Pi tier), because the workload will grow past simple detection. Both stages run
on the SBC under onnxruntime (zerort as the floor); the MCU tier below is **deferred** and
MCU constraints (no-FPU math, 4-byte wire frames, width-8/128px coarsening) are no longer
design drivers. The 4×uint8 box codec stays as the canonical compact form but is not
load-bearing for the target.

- [x] **1. Stage-2 crop head (architecture).** `Stage2Head` on the predicted-box crop: damage
  softmax + signed `seating` regressor, plus a `crop_to_box` stage-1→2 bridge and int8 ONNX
  export (`damage`/`seating` outputs). Self-contained in
  [`tensor_factory_train.stage2`](packages/tensor-factory-train/src/tensor_factory_train/stage2.py),
  fully unit-tested (`tests/test_stage2.py`). Purely additive — the v5 detector is untouched.
  **Architecture only: the heads are untrained until item 3 lands the data.**
- [ ] **2. Label schema.** Damage classes + insertion target (seating scalar now; keypoints/
  mask later for a true geometric measurement). Extend the dataset format + loaders (folds
  into the multi-`--data` item below).
- [ ] **3. Per-failure-mode data (⚠ THE BLOCKER — the long pole).** Synthesize damaged/over/
  under variants with Nano Banana (reuse `build_ds.py` reference conditioning, `GEMINI_API_KEY`);
  backfill real photos under controlled lighting; label damage class + seating. Hundreds per
  mode. **Nothing below trains until this lands — model 2 is blocked here, not on capacity.**
- [ ] **4. PC teacher.** Larger multi-task backbone @1024px; use it to pre-annotate (bootstrap
  label quality past GroundingDINO). Ties into the model-in-the-loop item under "Now".
- [ ] **5. SBC two-stage student**, distilled from the teacher (soft targets + seg distill).
  This is now the deployment tier (ARM SBC), not an intermediate.
- [ ] ~~**6. Distill + coarsen to MCU**~~ — **deferred 2026-07-02** (target platform is an
  ARM SBC; revisit only if a genuinely MCU-bound deployment appears).

## Recently done (this session)

- [x] **zerort: zero-dependency inference floor — roadmap items 1–5 all landed.**
  Pure-stdlib runtime under `tensor_factory.zerort` (protobuf reader + flat `.tfm`
  container, int8 op set with bigint-SIMD ConvInteger, PNG/baseline-JPEG/PPM decode,
  triangle resize), zero-dep HTTP serving, and a CI portability job on a bare
  `python:3.12-slim`. Parity-tested against onnxruntime (exact integer kernels,
  <=5e-3 full-model, import-blocked subprocess proofs). Status lives in
  [`ROADMAP.md`](ROADMAP.md); the only open roadmap item (GroundingDINO exit) is
  gated on model quality above.

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
- [x] **Multi-dir training support.** `fit` / `--data` take several dataset dirs, read in
  place: classes merge by name, each source is held out and scored separately (stdout at
  the best checkpoint, `sources` in `on_epoch`). The Studio takes `--source DIR` and shows
  the per-source table live. `real_ds_combined` no longer needs to exist.
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
