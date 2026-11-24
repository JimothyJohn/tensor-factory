# tensor-factory on the ESP32-P4

Runs the tensor-factory detector on an ESP32-P4 with [ESP-DL](https://github.com/espressif/esp-dl),
as a static int8 `.espdl` model. This directory is an ESP-IDF project (C++), outside the
Python workspace; nothing in `packages/` depends on it.

What runs on the chip is the convolutional part of the detector: image in, 4-channel edge
heatmap and presence logit out. The soft-argmax that turns the heatmap into a box is a few
lines of firmware (`decode_box` in `main/app_main.cpp`).

## Needs

- ESP-IDF **v5.5.5 or newer** (chip revision v3.2 needs it); built and measured with v5.5.6.
- ESP-DL 3.3.15 (fetched by the component manager, pinned in `main/idf_component.yml`).
- ESP-PPQ in its own Python environment, for quantization:
  `uv venv ~/esp/ppq-venv --python 3.11 && VIRTUAL_ENV=~/esp/ppq-venv uv pip install esp-ppq torch pillow`

## Use

```sh
. ~/esp/esp-idf/export.sh
cd firmware/esp32-p4

# 1. latency / memory sweep of the architecture at several input sizes (random weights)
tools/sweep.sh "96 128 160 224 320 480"

# 2. the real model + real frames for the soak (writes main/models/)
PYTHONPATH=../../packages/tensor-factory-train/src:../../packages/tensor-factory/src \
  ~/esp/ppq-venv/bin/python tools/export_espdl.py --size 480 \
  --onnx ../../packages/tensor-factory-mcp/src/tensor_factory_mcp/models/helicoil-presence-cam-v1.onnx \
  --calib DATASET/images --eval DATASET --negatives NEGATIVES \
  --frames 8 --frames-out main/models --out main/models/camv1_480.espdl
echo "MODEL(camv1_480)" > main/models/detect.inc

idf.py set-target esp32p4 && idf.py build
idf.py -p PORT flash
python tools/capture.py PORT --until BENCH_DONE        # resets the board, prints its output
```

`main/models/` is generated and gitignored. `models.inc` lists the models to sweep,
`detect.inc` the one model to soak, `frames.inc` the frames and the answers the desktop
int8 graph gave for them. An empty file skips that part.

## What the firmware prints

- `BENCH {...}`: per model, with activations in PSRAM and then in internal RAM first:
  single- and dual-core latency, memory taken, and whether ESP-DL's self-test passed (the
  output matches the test values ESP-PPQ embedded at export).
- `FRAME {...}`: first pass over each soak frame: the box and presence the board computed
  next to the desktop's.
- `SOAK {...}`: running totals. `desktop_mismatch` counts frames that differ from the
  desktop answer; `self_mismatch` counts frames whose raw output bytes differ from the
  board's own first answer for that frame. Both must stay 0.

Measured numbers are in `results/`.
