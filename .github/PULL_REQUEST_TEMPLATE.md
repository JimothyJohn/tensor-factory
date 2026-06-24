<!-- Keep it short. The gate is the source of truth. -->

## What & why

<!-- One or two sentences: what this changes and the reason. -->

## Checklist

- [ ] `./Quickstart -c` is green (sync-locked, ruff, ty, `pytest -m unit`)
- [ ] New behavior has tests; bug fixes have a regression test that fails without the fix
- [ ] Tests carry a marker (`@pytest.mark.unit` / `@pytest.mark.integration`)
- [ ] If deps changed: `uv.lock` is committed and the PR explains the new dependency
- [ ] If the detection contract, CLI flags, or public API changed: docs under `docs/` updated

## Notes

<!-- Anything reviewers should know: trade-offs, follow-ups, screenshots for docs/demo. -->
