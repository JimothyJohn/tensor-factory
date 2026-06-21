#!/usr/bin/env bash
# relabel.sh — bring up the human-validation stack for the real_ds dataset and push the
# GroundingDINO candidates into Label Studio for review.
#
# Pipeline (see examples/helicoils/README.md): the 142 GroundingDINO boxes in real_ds are
# review=pending (AI guesses, NOT trainable). You correct them in Label Studio, then pull
# them back — the pull stamps review=approved, source=human, which is what makes them
# trainable.
#
# Everything that needs to keep running (Label Studio :8080, the image server :8081) is
# launched inside tmux via scripts/run-bg, so an SSH disconnect won't kill it. Reattach
# any time with `tmux attach -t tf-labelstudio` / `tf-imgserver`.
#
# Usage:
#   examples/helicoils/relabel.sh            # start servers + push the dataset
#   examples/helicoils/relabel.sh --stop     # tear the stack down
#
# Then: open the printed URL, correct boxes, and run the printed `pull` command.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="examples/helicoils/images/real_ds"
PROJECT_TITLE="helicoil real_ds v1"
LS_SESSION="tf-labelstudio"
IMG_SESSION="tf-imgserver"
LS_PORT=8080
IMG_PORT=8081
RUN_BG="scripts/run-bg"

# --- load .env (LABEL_STUDIO_*; GEMINI_API_KEY stays in the shell env) ---
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${1:-}" == "--stop" ]]; then
  tmux kill-session -t "$LS_SESSION" 2>/dev/null && echo "stopped $LS_SESSION" || true
  tmux kill-session -t "$IMG_SESSION" 2>/dev/null && echo "stopped $IMG_SESSION" || true
  echo "stack down. (The LS SQLite DB persists in ~/.local/share/label-studio.)"
  exit 0
fi

: "${LABEL_STUDIO_URL:?set LABEL_STUDIO_URL in .env}"
: "${LABEL_STUDIO_API_KEY:?set LABEL_STUDIO_API_KEY in .env}"

if [[ ! -f "$DATA_DIR/annotations.coco.json" ]]; then
  echo "no dataset at $DATA_DIR — run build_ds.py first." >&2
  exit 1
fi
if ! command -v label-studio >/dev/null 2>&1; then
  echo "label-studio not found — install with: uv tool install label-studio" >&2
  exit 1
fi

# --- 1. triage: show what needs review ---
# `uv run --package X` guarantees X's entrypoint is present regardless of which workspace
# member the venv was last synced to (members aren't all installed at once).
echo "== triage =="
uv run --package tensor-factory-synth tensor-factory-synth triage --data "$DATA_DIR" || true
echo

# --- 2. image server (serves real_ds/images/* to Label Studio, with CORS) ---
# CORS matters: LS draws each image onto a crossorigin canvas, which the browser blocks
# unless the server sends Access-Control-Allow-Origin. Plain http.server doesn't.
if tmux has-session -t "$IMG_SESSION" 2>/dev/null; then
  echo "image server already running ($IMG_SESSION)"
else
  "$RUN_BG" "$IMG_SESSION" \
    python3 "$REPO_ROOT/scripts/cors_server.py" "$IMG_PORT" "$REPO_ROOT/$DATA_DIR"
fi

# --- 3. Label Studio ---
if tmux has-session -t "$LS_SESSION" 2>/dev/null; then
  echo "label studio already running ($LS_SESSION)"
else
  "$RUN_BG" "$LS_SESSION" \
    label-studio start \
    --port "$LS_PORT" \
    --username "${LABEL_STUDIO_USER:-demo@tensor-factory.local}" \
    --password "${LABEL_STUDIO_PASSWORD:-tensorfactory12345}" \
    --user-token "$LABEL_STUDIO_API_KEY" \
    --enable-legacy-api-token \
    --agree-fix-sqlite
fi

# --- 4. wait for Label Studio to answer, then push ---
echo -n "waiting for Label Studio on :$LS_PORT "
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:$LS_PORT/health" >/dev/null 2>&1; then
    echo " up"
    break
  fi
  echo -n "."
  sleep 2
done

echo "== push candidates =="
uv run --package tensor-factory-label tensor-factory-label push \
  --data "$DATA_DIR" \
  --title "$PROJECT_TITLE" \
  --image-base "http://localhost:$IMG_PORT"

cat <<EOF

== next ==
1. Open the project URL printed above and correct the 142 boxes.
2. When done, pull the corrected labels back (this is what makes them trainable):

     uv run --package tensor-factory-label tensor-factory-label pull \\
         --project <ID> --out $DATA_DIR/annotations.coco.json

   (<ID> is the project number in the URL above.)
3. Re-check progress any time:
     uv run --package tensor-factory-synth tensor-factory-synth triage --data $DATA_DIR
4. Tear the stack down when finished:  examples/helicoils/relabel.sh --stop
EOF
