# helicoils

Python library, distributed as a `uv` workspace (monorepo). One language: Python only.

> **TODO:** Replace this line and the package `description` fields with what `helicoils` actually does — the scaffold left them as placeholders.

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
