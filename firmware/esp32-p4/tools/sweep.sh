#!/usr/bin/env bash
# Export the detector at several input sizes (random weights: latency does not depend on
# them) into main/models/, and write models.inc for the firmware.
#   tools/sweep.sh "96 128 160 224 320 480" [width]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"; repo="$(cd "$here/../.." && pwd)"
sizes="${1:-96 128 160 224 320 480}"; width="${2:-16}"
py="${PPQ_PYTHON:-$HOME/esp/ppq-venv/bin/python}"
mkdir -p "$here/main/models"; rm -f "$here"/main/models/*.espdl "$here/main/models/models.inc"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
for s in $sizes; do
  PYTHONPATH="$repo/packages/tensor-factory-train/src:$repo/packages/tensor-factory/src" \
    "$py" "$here/tools/export_espdl.py" --size "$s" --width "$width" --out "$tmp/w${width}_${s}.espdl" | tail -1
  cp "$tmp/w${width}_${s}.espdl" "$here/main/models/"
  echo "MODEL(w${width}_${s})" >> "$here/main/models/models.inc"
done
