#!/usr/bin/env bash
# Run every test JPEG through the board in batches that fit the app partition.
#   tools/run_pets.sh MODEL.onnx SIZE DATASET_TEST_DIR PORT OUT_PREFIX [CALIB_DIR]
# Needs: . ~/esp/esp-idf/export.sh, the ESP-PPQ venv (see README), a built project.
set -euo pipefail
onnx="$1"; size="$2"; data="$3"; port="$4"; prefix="$5"; calib="${6:-$3/images}"
here="$(cd "$(dirname "$0")/.." && pwd)"; repo="$(cd "$here/../.." && pwd)"
py="${PPQ_PYTHON:-$HOME/esp/ppq-venv/bin/python}"
total="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['annotations']))" "$data/annotations.coco.json")"
skip=0; batch=0
cd "$here"
while [ "$skip" -lt "$total" ]; do
  find main/models -type f -delete
  tmp="$(mktemp -d)"
  ( cd "$tmp" && PYTHONPATH="$repo/packages/tensor-factory-train/src:$repo/packages/tensor-factory/src" \
      "$py" "$here/tools/export_espdl.py" --size "$size" --onnx "$onnx" --calib "$calib" \
      --calib-steps 100 --eval "$data" --jpegs 1000 --jpegs-skip "$skip" \
      --jpegs-out "$here/main/models" --out model.espdl > export.log 2>&1 ) || { tail -5 "$tmp/export.log"; exit 1; }
  cp "$tmp/model.espdl" main/models/pets.espdl
  [ "$batch" -eq 0 ] && cp "$tmp/model.report.json" "$prefix-desktop.json"
  rm -rf "$tmp"
  n="$(grep -c '^JPG' main/models/jpegs.inc)"
  echo "MODEL(pets)" > main/models/detect.inc; : > main/models/models.inc; : > main/models/frames.inc
  idf.py reconfigure > /dev/null 2>&1 && idf.py build > /dev/null 2>&1
  idf.py -p "$port" -b 921600 flash > /dev/null 2>&1
  python tools/capture.py "$port" --until JPEG_DONE --timeout 600 --grep "" | grep -aE '^(JPEG_|DET )' > "$prefix-batch$batch.txt"
  echo "batch $batch: files $skip..$((skip + n - 1)), $(grep -c '^DET' "$prefix-batch$batch.txt") DET lines"
  skip=$((skip + n)); batch=$((batch + 1))
done
