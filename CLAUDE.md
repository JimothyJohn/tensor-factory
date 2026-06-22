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

Five-package workspace, end-to-end pipeline working on synthetic data: **204 fps CPU
@480px, 81 KB int8 model, ~1.9 px median localization**. The only committed model is the
demo bundled in `tensor-factory-mcp`, trained on the **mock** generator — not photoreal yet.

- **Generation migrated FLUX.1-schnell → Nano Banana** (Gemini `gemini-2.5-flash-image`)
  on branch `nano-banana-generation` — a hosted API call, no local GPU. Reads
  `GEMINI_API_KEY` from the env. Not yet merged to `main`.
- The reusable prompt system + 25 QC-inspection samples live in
  [`examples/helicoils/SAMPLES.md`](examples/helicoils/SAMPLES.md) and
  `examples/helicoils/images/` (`manifest.json` maps each to its prompt + seed + QC state).
- Open work is tracked in [`TODO.md`](TODO.md). The next real milestone: train the
  detector on a Nano-Banana-generated, GroundingDINO-labeled dataset.

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
