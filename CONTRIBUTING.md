# Contributing to tensor-factory

Thanks for helping build an open, lightweight path from a prompt to a tiny int8 detector.
This guide is for **humans and coding agents alike** — the conventions below are mechanical
on purpose so either can follow them.

## TL;DR

```bash
git clone https://github.com/JimothyJohn/tensor-factory
cd tensor-factory
./Quickstart -c        # sync (locked) → ruff check+format → ty → pytest -m unit
```

If `./Quickstart -c` is green, your change is ready to open as a PR. That one command is the
whole local gate, and CI runs the same thing.

## Ground rules

- **One language: Python.** The whole repo is a [`uv`](https://docs.astral.sh/uv/) workspace
  (monorepo). Don't introduce JS/TS/Go tooling. The only JavaScript in the tree is the static
  docs site under `docs/` (no build step, no npm).
- **Tooling is fixed:** `uv` (env + build), `ruff` (lint + format, line length 100), `ty`
  (types), `pytest`. Don't swap them out without opening an issue first.
- **Apache-2.0 throughout.** No AGPL dependencies, ever (that constraint is the reason this
  project exists). New deps get justified in the PR — see *Dependencies* below.
- **CPU-only core.** `tensor-factory` (the core package) stays dependency-light and CPU-only.
  Anything GPU-heavy (training, auto-labeling, generation) lives in a sibling package behind
  an optional extra.

## Layout

```
packages/
  tensor-factory/        # core: geometry, the 4×uint8 codec, formats, onnxruntime inference + CLI
  tensor-factory-synth/  # generation (Gemini) + GroundingDINO auto-label + COCO/Label Studio export
  tensor-factory-train/  # the tiny soft-argmax detector → int8 ONNX
  tensor-factory-mcp/    # MCP + HTTP servers exposing the detector (bundled models)
  tensor-factory-label/  # Label Studio push/pull
docs/                    # static docs site + the in-browser demo (demo.html)
tests/                   # cross-cutting tests (e.g. the demo contract tests)
examples/helicoils/      # the first worked example
```

A new library goes under `packages/<name>/` with its own `pyproject.toml`; the workspace
picks it up automatically (`[tool.uv.workspace] members = ["packages/*"]`). Shared lint/test
config lives at the root, not per-package.

## Tests — the bar

We take testing seriously (see the full philosophy in [`CLAUDE.md`](CLAUDE.md)). The short
version:

- **Every test carries a marker:** `@pytest.mark.unit` (fast, isolated) or
  `@pytest.mark.integration` (touches a real external resource). The gate runs `-m unit`.
- **Test the contract, not the implementation.** A refactor that preserves behavior should
  not break tests.
- **No mocked services in integration tests** — use the real thing or a cached fixture.
- **Bug fix → regression test first.** It fails before your fix and stays forever after.
- **Determinism is non-negotiable** — seed randomness, sort before asserting. Flaky tests get
  fixed, never retried.

Run a subset directly: `uv run --all-packages python -m pytest tests/ -m unit`.

## Dependencies

Adding a dependency is a supply-chain decision. Before `uv add`:

1. Is there a stdlib or existing-dep answer? Reach for it first.
2. Is the package actively maintained, and what's its transitive footprint (`uv tree`)?
3. **Say so in the PR** — what it's for, the alternatives, the footprint. A new entry in a
   `pyproject.toml` without that context isn't ready to merge.

The lockfile is part of the test surface: if you change deps, commit the updated `uv.lock` in
the same change. CI uses `uv sync --locked`, so drift fails the build.

## Pull requests

- Branch from `master`. **One task, one branch, one direction** — don't mix unrelated changes.
- Keep commits focused with clear messages (imperative subject, a body explaining *why*).
- Make sure `./Quickstart -c` is green before you open the PR; CI will re-run it.
- Fill in the PR template. If you touched deps, docs, or the detection contract, call it out.

## For coding agents

This repo is designed to be agent-navigable. If you're an automated contributor:

- Read [`CLAUDE.md`](CLAUDE.md) (root) first — it's the canonical project contract: tooling,
  the testing bar, the git/worktree conventions, and the "Current standing" snapshot.
- **Work in a `git worktree`, not the shared checkout.** Multiple agents in one working
  directory share a single `HEAD` and index; a fresh branch name is *not* isolation. Use
  `git worktree add` (or the harness's worktree isolation) so your commits stay disjoint.
- Don't commit local session state. `examples/helicoils/images/` (scratch datasets/models)
  is gitignored, and `.claude/settings.local.json` is local-only. Check `git status` before
  staging — never blindly `git add -A`.
- The gate is the source of truth: if `./Quickstart -c` is green and your change has tests,
  it's mergeable.

By contributing you agree your work is licensed under Apache-2.0.
