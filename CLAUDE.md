# helicoils

Open, lightweight **helicoil detection** — detect coiled-wire threaded inserts in
machined parts from microscope imagery. Python `uv` workspace (monorepo), one language.

The full vision and decisions are in [`PROMPT.md`](PROMPT.md). Shape: synthesize a
labeled dataset (FLUX.1-schnell generation + GroundingDINO auto-label), train a tiny
int8 CNN, run it on CPU via onnxruntime, drive it from a CLI. License: Apache-2.0
throughout (zero AGPL exposure — no Ultralytics).

## Detection contract

A detection is **four `uint8` values** (normalized `xyxy`, one byte each). At 480 px the
quantization step is ~1.88 px, so round-trip error stays under 1 px — inside the 3 px
budget — and post-processing stays in 8-bit math. This lives in `helicoils.codec`; the
canonical box is `helicoils.geometry.BBox` (normalized `xyxy`, top-left origin).

## Compute

Core (`helicoils`) is dependency-free and CPU-only — geometry, codec, formats, inference.
Generation and training are GPU-heavy and live in sibling packages behind extras; they
resolve the device **cuda → mps → cpu**, so the dev loop runs on this Mac Studio's MPS and
scales out to a CUDA box or AWS g5/g6 for bulk runs.

## Layout

```
pyproject.toml              # workspace root: members, dev deps, ruff + pytest config
uv.lock                     # single lockfile for the whole workspace
Quickstart                  # install -> lint+format -> types -> unit (and -p to publish)
packages/
  helicoils/                # the library (only member today)
    pyproject.toml          # package metadata + hatchling build
    src/helicoils/          # src layout; py.typed ships type info
    tests/                  # pytest, markered (unit/integration)
```

New libraries go under `packages/<name>/` with their own `pyproject.toml`; `[tool.uv.workspace] members = ["packages/*"]` picks them up automatically. Shared dev tooling and lint/test config live at the root, not per-package.

## Commands

- `./Quickstart` — full local gate: `uv sync --locked` → `ruff check --fix` → `ruff format` → `ty check` → `pytest -m unit`. Run this before pushing.
- `./Quickstart -u` — unit tests only.
- `./Quickstart -p` — full gate, then `uv build` + `uv publish` (needs `UV_PUBLISH_TOKEN`).
- Direct: `uv run pytest`, `uv run ruff check`, `uv run ty check`. Always go through `uv run` — `ruff`/`ty` are dev deps, not on PATH globally.

## Conventions

- **Tooling:** `uv` (env + build), `ruff` (lint + format, line length 100), `ty` (types), `pytest`. Don't introduce other tools without flagging first.
- **Lockfile is part of the test surface.** CI uses `uv sync --locked`; if you change deps, commit the updated `uv.lock` in the same change. Unexpected drift is a signal, not noise.
- **Tests carry markers.** `@pytest.mark.unit` for fast/isolated, `@pytest.mark.integration` for anything touching real external resources (no mocked services — use the real thing or cached fixtures). Add markers to new tests so the gate can select them.
- **`py.typed` is shipped** — the library is typed; keep `ty check` clean.
- Bug fix → regression test first (it fails before the fix, stays forever after).
