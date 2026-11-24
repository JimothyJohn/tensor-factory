# Studio end-to-end tests

Browser-level tests (Playwright) for the full Studio loop: boot, video ingest (good →
success toast; undecodable → error toast, no hang), labeling pushing frames to the
backend, and graceful empty-model export. The deterministic flows live here; GPU training
correctness is covered by the Python integration test (`packages/tensor-factory-studio`).

## Run

```sh
studio/e2e/run.sh          # boots a throwaway backend, runs the spec, tears it down
```

Or against an already-running backend:

```sh
STUDIO_URL=http://127.0.0.1:8089 node studio/e2e/studio.e2e.mjs
```

## Requirements (why this isn't in `./Quickstart -c`)

These need a **browser binary** and a **trainer** (torch), so they're a manual/optional
suite, deliberately kept out of the unit gate:

- Node + Playwright's chromium: `npx playwright install chromium` (Playwright is **not** a
  committed project dependency — install it ad hoc to run these).
- The `serve` extra synced (`uv sync --extra serve`) for the backend `run.sh` starts.

The fast, zero-dependency tests that *do* gate every change:

- `studio/tests/*.test.mjs` — Studio JS logic, via `node --test` (run by `./Quickstart -c`).
- `packages/tensor-factory-studio/tests/` — backend HTTP + dataset + trainer, via pytest.
