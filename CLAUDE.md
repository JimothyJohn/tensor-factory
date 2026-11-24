# tensor-factory

Open, lightweight **tiny-CNN object detection** — a factory for synthetic-data → int8
detectors that run on CPU. **Helicoil detection** (coiled-wire threaded inserts in
machined parts, from microscope imagery) is the first example, under
[`examples/helicoils`](examples/helicoils). Python `uv` workspace (monorepo), one language.

The full vision and decisions for the first example are in
[`examples/helicoils/PROMPT.md`](examples/helicoils/PROMPT.md). Shape: synthesize a
labeled dataset (Nano Banana / Gemini generation + GroundingDINO auto-label), train a
tiny int8 CNN, run it on CPU via onnxruntime, drive it from a CLI. License: Apache-2.0
throughout (zero AGPL exposure — no Ultralytics).

## Current standing

Five-package workspace, full pipeline working end to end: **204 fps CPU @480px, 81 KB int8
model**. Architecture/throughput numbers hold across models; the ~1.9 px localization
figure is on **mock** data (exact geometry). On synthetic photoreal data box localization
is ~25 px; on **real microscope camera footage** (the bundled `cam-v1`) it's ~34 px — the
open quality ceiling (see [`TODO.md`](TODO.md)).

- **The bundled MCP model is now `helicoil-presence-cam-v1.onnx`** — the first detector
  trained on *actual microscope camera frames* (extracted from `helicoils.mp4`: 131
  human-validated positives + 81 reviewed-empty real negatives), carrying the YOLO-style
  objectness/presence head + the soft-argmax gain. Held-out (80/20): **~34 px box
  center-error, 98% presence accuracy**, 81 KB int8, ~190 fps CPU @480px. The synthetic
  `helicoil-mock-v1.onnx` is still bundled alongside it (box-only, selectable via
  `model_path`); earlier synthetic/photoreal models (v1–v5) live gitignored under
  `examples/helicoils/images/`.
- **Zero-dependency inference floor (`tensor_factory.zerort`).** A pure-stdlib runtime —
  protobuf wire-format reader (plus the flat `.tfm` container, `tensor-factory pack`),
  the int8 op set (bigint-SIMD ConvInteger), PNG/baseline-JPEG/PPM decoding, triangle
  resize — so image → detection (and HTTP serving via `tensor-factory-http`) runs on
  bare CPython with **no third-party packages**. `Detector(engine="auto")` uses
  onnxruntime when the `infer` extra is installed (~215 fps) and falls back to zerort
  otherwise (~0.9 fps); parity vs onnxruntime is pinned by tests (op-level exact for
  integer kernels; full-model ≤5e-3; import-blocked subprocess proofs) and CI runs the
  whole stack in a bare `python:3.12-slim` container
  (`scripts/portability_check.py`). Plan + status: [`ROADMAP.md`](ROADMAP.md).
- **Two serving surfaces.** `tensor-factory-mcp` (stdio MCP) and `tensor-factory-http` (a
  stdlib `http.server` endpoint, zero extra deps) both wrap the same `core` inference:
  `POST /detect` takes raw image bytes → the same JSON as the MCP `detect` tool. HTTP binds
  `127.0.0.1` by default.
- **Edge = one file.** `tensor-factory-bundle --model M --out edge.pyz` packs
  `tensor_factory` (zerort included), the stdlib HTTP server and the model into a
  reproducible zipapp (~100 KB); `python3 edge.pyz` on a bare CPython ≥ 3.11 serves
  `/detect`. The model is unpacked to `~/.cache/tensor-factory/` (sha256-verified each
  start). `scripts/portability_check.py` step 5 builds and runs one with `-S -I`, and the
  CI `portability` job runs it on x86-64 and aarch64 Linux.
- **S3 store (`tensor_factory.store`, `tensor-factory store`).** Models and datasets are
  pushed to and pulled from one private bucket, content-addressed by sha256
  (`blobs/sha256/<hex>`, `models/<name>.json`, `datasets/<name>/manifest.json`). Pure
  stdlib (SigV4 over `urllib`), so it ships inside `edge.pyz` and a bare-Python device can
  pull a model by sha256; a pull is verified, then `os.replace`d, which hot reload picks
  up. `scripts/store_bucket.sh` creates or re-asserts the bucket (public access blocked,
  TLS-only, versioned, lifecycle rules). Config: `TENSOR_FACTORY_STORE=s3://…` plus the
  usual `AWS_*` variables. The real-bucket tests are `-m integration` and skip without it.
- **Model hot reload.** `core` ties each cached detector to its file's mtime + size: replace
  the file (atomically: write beside, `mv` over) and the next request loads it. A
  replacement that fails to load keeps the old model serving; `/model_info` reports
  `sha256` of what is loaded and `reload_error` when a swap failed.
- **Presence head / negatives.** The detector returns one box or none, YOLO-style. A single
  objectness logit (`TinyDetector(presence=True)`, ONNX output `presence`) is trained by
  `--negatives DIR`: positives → objectness 1, no-object negatives → objectness 0 with box
  loss masked. `detect` thresholds `sigmoid(presence)` and returns `present` (bool) + `score`;
  when absent the box fields come back `null`. No class label, no `background` class.
  `gen_negatives.py` synthesizes machined-part negatives.
- **Soft-argmax gain (localization).** The coordinate head carries a learnable inverse-
  softmax-temperature (`TinyDetector.log_gain`) so it can sharpen a diffuse heatmap before
  the marginal expectation, countering the centre-bias that dominates real-data box error.
  Init gain 1.0 = plain softmax (backward-compatible); `--freeze-gain` ablates it. Whether
  it actually moves the ~25 px ceiling is being validated — see [`TODO.md`](TODO.md).
- **Generation: FLUX.1-schnell → Nano Banana** (Gemini `gemini-2.5-flash-image`) — a hosted
  API call, no local GPU, reads `GEMINI_API_KEY`. `build_ds.py <out> <n> <reference>` can
  condition generation on a real part photo for application-matched realism. **Merged to
  the base branch `master`** (this repo's default branch is `master`, not `main`).
- **Datasets** (all gitignored under `examples/helicoils/images/`): `real` (131
  real-camera positives from `helicoils.mp4` frames + `real_neg`, its 81 reviewed-empty
  real negatives — trains the bundled `cam-v1`), `real_ds` (110 human-validated synthetic
  positives), `real_ds_more` (168 reference-conditioned, GroundingDINO-labeled),
  `negatives_pool` (110 synthetic negatives), `real_ds_combined` (278, used to train v3).
- **Several data sources train as one set.** `tensor-factory-train fit --data A B C` (and
  `fit([A, B, C], ...)`) reads each dataset dir in place, merges classes by name, holds
  `--val-frac` out of *each* source and reports each separately; negatives dirs count as
  sources. The Studio takes `--source DIR` (repeatable) to train prior datasets alongside
  the frames labeled in the browser — with a source given it trains a first model before
  anything is labeled — and its trainer panel shows the per-source table every epoch.
- **Training tests need torch, which the `gate` job does not install.** The `train` CI job
  (macOS runner, `--extra train --extra serve`) runs them; locally:
  `uv run --all-packages --extra train --extra serve python -m pytest packages/tensor-factory-train packages/tensor-factory-studio -m "unit or integration"`.
- **Human-review gate is optional.** Training enforces `review=approved` by default, but
  `--allow-unreviewed` trains straight on GroundingDINO labels — the fast path (QC a
  contact-sheet instead of hand-correcting every box). The prompt system + QC samples are
  in [`SAMPLES.md`](examples/helicoils/SAMPLES.md) (`manifest.json` maps each).

## Detection contract

A detection is **four `uint8` values** (normalized `xyxy`, one byte each). At 480 px the
quantization step is ~1.88 px, so round-trip error stays under 1 px — inside the 3 px
budget — and post-processing stays in 8-bit math. This lives in `tensor_factory.codec`; the
canonical box is `tensor_factory.geometry.BBox` (normalized `xyxy`, top-left origin).

## Compute

Core (`tensor-factory`) is dependency-free and CPU-only — geometry, codec, formats,
inference. Image **generation** is now a hosted Gemini API call (no GPU) behind
`tensor-factory-synth`'s `gemini` extra. **Auto-labeling** (GroundingDINO) and **training**
are GPU-heavy, live in sibling packages behind extras, and resolve the device
**cuda → mps → cpu** — so the dev loop runs on this Mac Studio's MPS and scales out to a
CUDA box or AWS g5/g6 for bulk runs.

## Layout

```
pyproject.toml              # workspace root: members, dev deps, ruff + pytest config
uv.lock                     # single lockfile for the whole workspace
Quickstart                  # install -> lint+format -> types -> unit (and -p to publish)
docs/index.html             # tensor-factory landing page (helicoils as example #1)
packages/
  tensor-factory/           # core library: BBox geometry, 4xuint8 codec, formats, onnxruntime inference + CLI
  tensor-factory-synth/     # generation (Nano Banana/Gemini) + GroundingDINO auto-label + COCO/Label Studio export
  tensor-factory-train/     # tiny soft-argmax detector -> int8 ONNX
  tensor-factory-mcp/       # FastMCP server exposing the detector (bundled demo model)
  tensor-factory-label/     # Label Studio push/pull
examples/
  helicoils/                # first example: PROMPT.md brief, SAMPLES.md, generated images/
```

Each package uses src layout, ships `py.typed`, and carries pytest markers
(`unit`/`integration`).

New libraries go under `packages/<name>/` with their own `pyproject.toml`; `[tool.uv.workspace] members = ["packages/*"]` picks them up automatically. Shared dev tooling and lint/test config live at the root, not per-package.

## Commands

- `./Quickstart` — no flag: bootstrap (`uv sync --locked --all-packages`) then run the `tensor-factory-mcp` server (stdio, bundled demo model) so the project is runnable with zero setup. Stdio means it exits on EOF when no MCP client is attached — clients spawn it on demand via `.mcp.json`.
- `./Quickstart -c` — full local gate: `uv sync --locked` → `ruff check --fix` → `ruff format` → `ty check` → `pytest -m unit`. Run this before pushing.
- `./Quickstart -u` — unit tests only.
- `./Quickstart -p` — full gate, then `uv build` + `uv publish` (needs `UV_PUBLISH_TOKEN`).
- Direct: `uv run python -m pytest`, `uv run ruff check`, `uv run ty check`. Always go through `uv run` — `ruff`/`ty` are dev deps, not on PATH globally. Use `python -m pytest` (not bare `uv run pytest`) so a pytest earlier on PATH can't shadow the locked one.

## Conventions

- **Tooling:** `uv` (env + build), `ruff` (lint + format, line length 100), `ty` (types), `pytest`. Don't introduce other tools without flagging first.
- **Lockfile is part of the test surface.** CI uses `uv sync --locked`; if you change deps, commit the updated `uv.lock` in the same change. Unexpected drift is a signal, not noise.
- **Tests carry markers.** `@pytest.mark.unit` for fast/isolated, `@pytest.mark.integration` for anything touching real external resources (no mocked services — use the real thing or cached fixtures). Add markers to new tests so the gate can select them.
- **`py.typed` is shipped** — the library is typed; keep `ty check` clean.
- Bug fix → regression test first (it fails before the fix, stays forever after).

## Parallel work — one task, one worktree

Multiple agents/sessions run against this repo concurrently. The failure mode is **sharing
one checkout**: two sessions in the same directory share one `git status`, one HEAD, one
index, one `.venv`, and one dev-server port, so their commits land on each other's branch
and their edits collide in one tree. A branch name is *not* isolation — `git checkout -b`
only moves the single shared HEAD. Isolation comes from a separate **git worktree**.

- **Before starting concurrent work, give the task its own worktree:** `scripts/worktree <branch> [base]`
  creates `../tensor-factory-<slug>` on that branch, backed by the shared `.git`. The main
  checkout stays the primary/interactive session; everything parallel gets its own dir.
  (`scripts/worktree --list` / `--rm <branch>` to manage them. The `Agent` tool's
  `isolation: "worktree"` does the same for spawned agents.)
- **Each worktree is its own checkout** — run `uv sync --locked --all-packages` once inside
  it before working; its `.venv` is separate.
- **Never `git stash` to clear a tree that has someone else's uncommitted work.** Commit it
  (a `wip:` message is fine) so it's durable, or leave it and use a different worktree.
  Stashing hides it from `git status` and from the author.
- **Never `git checkout`/`switch` to another branch with a dirty tree you didn't create** —
  git silently drags those changes across, polluting both branches. Commit or worktree first.
- **Before adding to an existing branch/PR, check who else was there:** `git log --since=2h <branch>`
  and `gh pr view <n> --json comments,commits`. Add on top of another agent's commits; don't
  rewrite over them.
