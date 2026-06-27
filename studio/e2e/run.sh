#!/usr/bin/env bash
# Self-contained Studio e2e: boot a throwaway backend (small/fast model), run the
# Playwright spec against it, tear the backend down. Needs Node + Playwright's chromium
# (`npx playwright install chromium`) and the 'serve' extra (torch). NOT part of the
# './Quickstart -c' gate -- it needs a browser binary and a GPU/CPU trainer, so it's a
# manual/optional suite. Run from the repo root: studio/e2e/run.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

PORT="${STUDIO_E2E_PORT:-8189}"
DATA="$(mktemp -d)"
LOG="$(mktemp)"

cleanup() {
  [[ -n "${SRV_PID:-}" ]] && kill "$SRV_PID" 2>/dev/null || true
  rm -rf "$DATA" "$LOG"
}
trap cleanup EXIT

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "port $PORT is busy; set STUDIO_E2E_PORT to a free port" >&2
  exit 1
fi

echo "starting throwaway backend on :$PORT (data: $DATA)"
uv run --package tensor-factory-studio --extra serve tensor-factory-studio \
  --port "$PORT" --data-dir "$DATA" --size 96 --width 8 --epochs 4 >"$LOG" 2>&1 &
SRV_PID=$!

for _ in $(seq 1 60); do
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1 && break
  sleep 1
done
if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  echo "backend failed to start:" >&2
  cat "$LOG" >&2
  exit 1
fi

STUDIO_URL="http://127.0.0.1:$PORT" node studio/e2e/studio.e2e.mjs
