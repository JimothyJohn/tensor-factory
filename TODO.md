# TODO

Open work for tensor-factory (helicoils is the first example). Questions first, then tasks
in rough priority order. Finished work is not kept here: `git log` and
[`CLAUDE.md`](CLAUDE.md) ("Current standing") record it. Platform ownership is tracked
separately in [`ROADMAP.md`](ROADMAP.md).

## Open questions for Nick

- [ ] **Promote dev to master?** GitHub Pages serves the site from `master`, and `master` is
  88 commits behind `dev` (checked 2026-11-15). Everything landed this session (zero-dependency
  inference, the Studio, multi-source training, the one-file edge bundle) is on `dev` only,
  so none of it is on the public site or the release line. Separately, the local `master` in
  the main checkout has 9 commits that were never pushed (the WATTAGE.md / margin.html
  economics study) and has diverged from `origin/master` by 8, so a plain push will be
  rejected until those are reconciled.
  **2026-11-23:** draft PR #72 (`dev` → `master`) is open and conflict-free: `master` was
  merged into `dev` in #70, so it is a fast-forward. Merge it with a merge commit, not a
  squash. The 9 local commits are not part of it. `master` is already the GitHub default
  branch, so CodeQL and the current automerge workflow stay off until this lands.
      options: merge dev into master now — the site and release line catch up; you reconcile the 9 local commits first | wait — master stays on the June/July state
      recommend: merge dev into master now — four months of work is invisible to anyone reading the site
      stakes: high
      result: the public site and `master` match what is tested on `dev`
      id: promote-dev-to-master
      asked: 2026-11-15
      by: fable-2026-11-15
      answer: merge dev into master now
      answered: 2026-11-15

- [x] **What should "cloud" be for this project?** The pipeline now runs data → training →
  one-file edge bundle, all on this Mac. Nothing is in AWS yet. The three candidates do
  different jobs and cost different amounts, so this is a direction call.
      options: S3 store for datasets and models — datasets stop living on one disk; the edge pulls a model by its sha256; a few cents a month | GPU training runs on EC2 — a script launches a g5/g6 spot instance, trains, uploads the model, shuts down; about $0.30–1.00 per run | hosted inference endpoint — a Lambda that serves /detect; pay per request, needs a public-access decision | nothing yet — stay local
      recommend: S3 store for datasets and models — every dataset is gitignored on a single machine today, which is the one unrecoverable loss in the pipeline, and a model store is what the edge and any later training job both need first
      stakes: high
      result: a concrete cloud piece gets built instead of guessed
      id: cloud-direction
      asked: 2026-11-15
      by: fable-2026-11-15
      answer: S3 store for datasets and models
      answered: 2026-11-15

- [ ] **Which board is the edge target, and may I deploy to it?** The one-file bundle
  (`edge.pyz`) is proven on aarch64 Linux in CI but has never run on a physical board.
  Deploying means copying one file and starting a user-level systemd service on that host.
  **2026-11-23:** the answer picked "name a host" but no hostname came with it, and
  `~/.ssh/config` lists no Pi-class board. Put the hostname or address on the `answer:` line.
      options: name a host — I copy the bundle, start the service, and report frames per second and the model sha256 from the device | none yet — the CI proof stands until a board is chosen
      recommend: name a host — real throughput on the real board is the number the whole "tiny CNN on CPU" claim rests on
      stakes: low
      needs: the hostname or address of a Pi-class board on the LAN, and whether anything else is running on it
      result: measured fps and a running detector on the target hardware
      id: edge-target-host
      asked: 2026-11-15
      by: fable-2026-11-15
      answer: name a host
      answered: 2026-11-15

- [x] **Which landing page wins: the restyled page or the single-file detection app?**
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
      answer: restyled page stays, app moves to docs/app.html
      answered: 2026-11-17

- [x] **Dismiss the two remaining code-scanning alerts as false positives?** Five of the
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
      answer: dismiss both
      answered: 2026-11-15

- [x] **Protect master and turn on private vulnerability reporting?** Both were on the
  original launch list and neither is set (checked 2026-11-15): `master` has no branch
  protection, and private vulnerability reporting is off although `SECURITY.md` tells
  reporters to use it. These are repository settings on your release branch, so I left them.
      options: enable both — master requires the `gate` check; reporters get the private channel SECURITY.md promises | leave as is
      recommend: enable both — SECURITY.md currently points at a channel that does not exist
      stakes: low
      id: master-protection-and-pvr
      asked: 2026-11-15
      by: fable-2026-11-15
      answer: enable both
      answered: 2026-11-15

- [x] **Keep `dev` as the GitHub default branch?** GitHub's default branch is `dev`, while
  `CLAUDE.md` says the default is `master`. Dependabot and CodeQL read their configuration
  from the default branch, so today they work because the config lives on `dev`.
      options: keep dev as default, fix the docs — nothing changes in behavior | make master the default — Dependabot and CodeQL config must be promoted to master first or they go quiet
      recommend: keep dev as default, fix the docs — it is the arrangement that is already working
      stakes: low
      id: default-branch
      asked: 2026-11-15
      by: fable-2026-11-15
      answer: make master the default
      answered: 2026-11-15

- [x] **Keep or drop `wip/recovered-docs-site-20260623`?** A snapshot branch another session
  made on 2026-06-23 of an orphaned docs-site edit; 4 commits that are not on `dev`. The docs
  site has since been consolidated and restyled twice.
      options: drop — delete the branch | keep — I open a PR so you can read the 4 commits against the current site
      recommend: drop — the pages it touched have been rewritten since
      stakes: low
      id: recovered-docs-site
      asked: 2026-11-15
      by: fable-2026-11-15
      answer: drop
      answered: 2026-11-15

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

- [ ] **Run the edge bundle on a real board.** Copy `edge.pyz` to the target, start it as a
  user systemd service, record frames per second with and without onnxruntime, and confirm
  `/model_info` reports the sha256 the build printed.
      depends: edge-target-host
      est: 1h
      difficulty: low
      id: edge-on-real-board

- [ ] **Give the edge its own read-only AWS key.** The S3 store works from this Mac with
  your own credentials. A board should not carry those: it needs an IAM user or role that
  can only `s3:GetObject` and `s3:ListBucket` on the store bucket.
      needs: a named edge board (see `edge-target-host`)
      result: the board pulls a model by sha256 with a key that cannot write or delete
      depends: edge-target-host
      est: 30m
      difficulty: low
      id: edge-readonly-key

- [ ] **Feed the ESP32-P4 detector from the camera.** The firmware under
  `firmware/esp32-p4/` runs the detector on frames embedded in the image. Next is the real
  path: MIPI-CSI camera into the ISP, the PPA scaling to the model's input size, inference
  on the second core. Step 7 of the P4 brief.
      needs: which camera module is on the board, and a part in front of it
      result: live boxes from the camera, with frames per second measured end to end
      difficulty: high
      id: p4-camera-pipeline

- [ ] **Pick the ESP32-P4 input size with more held-out data.** On the board 480 / 320 /
  224 / 160 px run at 13 / 32 / 65 / 128 frames a second. On about 26 held-out frames the
  four sizes cannot be told apart for accuracy (see `firmware/esp32-p4/README.md`). Repeat
  the retraining over several seeds with a larger held-out set before choosing.
      result: one input size chosen for the P4, with its accuracy cost stated
      est: 3h
      difficulty: med
      id: p4-input-size

## Model quality

- [ ] **Box localization is the quality ceiling.** Center error is about 34 px on real
  microscope camera footage (the bundled `cam-v1`) and about 25 px on synthetic photoreal
  data, against 1.9 px on mock data where geometry is exact. Known limits: GroundingDINO
  boxes are loose (they bound the whole boss, not the insert) and the model is tiny. Levers
  not yet tried: tighten the loosest labels, a wider model (`--width 24/32`), native
  1024 px input, more real frames. Multi-source training now reports each dataset
  separately, so start by finding which source carries the error.
      difficulty: high
      id: box-localization

- [ ] **Find out why `real_ds` scores 59% presence accuracy in a mixed run.** In a 30-epoch
  run over `real`, `real_ds`, `real_ds_more` and both negatives pools (2026-11-15), every
  other source was at 86% or better. That run did not beat the constant-predictor floor, so
  first repeat it at the epoch count `cam-v1` was trained with before reading anything into it.
      est: 2h
      difficulty: med
      id: real-ds-presence

- [ ] **Model-in-the-loop pre-annotation.** Use the detector itself to pre-annotate new
  data (synthesized first, then real photos), replacing GroundingDINO as the label source.
  The bar to start is explicit: the detector's median box error on the held-out
  human-approved set must be lower than the GroundingDINO labels' error on that same set.
  Until then GroundingDINO stays the labeler. This is also `ROADMAP.md` item 6.
      depends: box-localization
      difficulty: high
      id: model-in-the-loop

- [ ] **Tighten the `real_ds_more` labels (optional).** 168 GroundingDINO boxes, 39
  corrected so far in Label Studio project 4. Only worth it for box precision;
  present/absent is already fine on the raw labels.
      difficulty: low
      id: tighten-real-ds-more

- [ ] **Check the bundled sample against the bundled model.** `docs/sample-helicoil.png`
  scores 0.07 (absent) on `cam-v1`. It is a synthetic image and `cam-v1` was trained on
  real camera frames, so this may be expected, but the demo's default pairing should be a hit.
      est: 1h
      difficulty: low
      id: sample-vs-cam-v1

## Model 2: damage and seating detection (blocked on labels)

Two models, deliberately separate. Model 1 is the box + presence detector that ships today.
Model 2 adds a second stage on the high-resolution crop of the detected box: damage is a
classification (`ok / deformed / cross_threaded / broken_tang / burr`) and insertion is a
measurement (a signed seating depth, read as over / correct / under). The stage-2 head
exists and is unit-tested (`tensor_factory_train.stage2`) but is **untrained**: no dataset
has damage or seating labels. Both stages run on the ARM single-board computer under
onnxruntime; the microcontroller tier was deferred on 2026-07-02.

- [ ] **Label schema for damage and seating.** Damage classes plus the insertion target
  (a seating scalar now; keypoints or a mask later for a true geometric measurement).
  Extend the dataset format and loaders.
      difficulty: med
      id: m2-label-schema

- [ ] **Per-failure-mode data.** The long pole. Synthesize damaged, over- and under-seated
  variants with Nano Banana (`build_ds.py` reference conditioning, `GEMINI_API_KEY`),
  backfill real photos under controlled lighting, and label damage class + seating.
  Hundreds per mode. Nothing below trains until this lands.
      depends: m2-label-schema
      difficulty: high
      id: m2-failure-mode-data

- [ ] **PC teacher.** A larger multi-task backbone at 1024 px, used to pre-annotate and
  lift label quality past GroundingDINO.
      depends: m2-failure-mode-data
      difficulty: high
      id: m2-teacher

- [ ] **Two-stage student for the ARM board**, distilled from the teacher. This is the
  deployment model.
      depends: m2-teacher
      difficulty: high
      id: m2-student

## Housekeeping

- [ ] **Rotate the Hugging Face token.** It was pasted in plaintext in an earlier chat
  transcript. Generation no longer needs it, but GroundingDINO still pulls public weights
  with it. Only Nick can do this.
      est: 10m
      difficulty: low
      id: rotate-hf-token

- [ ] **Verify the `gpu` extra is minimal.** Confirm GroundingDINO actually needs
  `accelerate`; drop it from `tensor-factory-synth`'s `gpu` extra if not.
      est: 30m
      difficulty: low
      id: gpu-extra-minimal
