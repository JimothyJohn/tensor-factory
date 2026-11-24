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

After adding, removing or renaming a file in `main/models/`, run `idf.py reconfigure`
before `idf.py build`; the file list is read at configure time.

## Measured (2026-11-23/24)

Board: ESP32-P4 revision v3.2, 16 MB flash, 32 MB PSRAM at 250 MHz, about 375 KB of internal
RAM free at boot. ESP-IDF v5.5.6, ESP-DL 3.3.15, `-O2`, dual core unless stated. Raw output
is in `results/`.

**Reference point.** Espressif's unmodified `mobilenetv2_cls` example: 164 ms single core,
126 ms dual core.

**The detector, end to end** (uint8 RGB in memory -> int8 input -> network -> box), real
trained weights, eight real camera frames each:

| Input | Network | Whole frame | Frames/s | Activations | Matches desktop int8 |
|---|---|---|---|---|---|
| 480 px (bundled `cam-v1`) | 57.2 ms | 76.0 ms | 13 | 1.6 MB, PSRAM | 8 of 8, exact |
| 320 px | 23.7 ms | 31.5 ms | 32 | 722 KB, mostly PSRAM | 8 of 8, exact |
| 224 px | 11.5 ms | 15.5 ms | 65 | 356 KB | 8 of 8, exact |
| 160 px | 5.5 ms | 7.8 ms | 128 | 184 KB, internal RAM | 8 of 8, exact |

Single core is about 1.7x slower (`results/sweep-w16-random-20261123.txt`). The time outside
the network is almost all the uint8 -> int8 input conversion of frames read from flash; a
camera pipeline would hand over a buffer in RAM.

**Soak.** `cam-v1` at 480 px for 30 minutes: 23,408 frames, 0 differing from the desktop
answer, 0 differing from the board's own first answer, worst frame 76.4 ms against a
76.0 ms mean, memory low-water marks unchanged from the first round.

**Internal RAM vs PSRAM for activations.** Keeping them in internal RAM helped by 0 to 10%
(largest at 224 px), not the large factor we expected. At 480 px nothing fits and it makes
no difference.

**Float vs int8** (desktop, all 212 real frames: 131 with a helicoil, 81 without; error in
480-px units; most of these frames were also trained on, so read the difference, not the
level):

| Model | Float center error | Int8 center error | Presence, float and int8 |
|---|---|---|---|
| bundled `cam-v1`, 480 px | 36.0 px | 36.8 px | 99.5% both |
| retrained 480 px | 33.5 px | 36.1 px | 99.1% both |
| retrained 320 px | 39.2 px | 39.6 px | 99.1% both |
| retrained 224 px | 30.0 px | 29.7 px | 98.6% both |
| retrained 160 px | 31.4 px | 31.9 px | 98.6% both |

On the 20% held out during retraining (about 26 frames with a helicoil, one seed), center
error in 480-px units was 41.7 / 31.5 / 43.5 / 45.0 px at 480 / 320 / 224 / 160 px and
presence 98 / 98 / 93 / 95%. That is too few frames to rank the sizes.

**Image files as bytes** (2026-11-24). All 432 held-out Oxford-IIIT Pet test photos,
original JPEG files embedded untouched (mean 100 KB, 0.17 megapixel), 224 px model. The
board decodes, resizes and detects each one itself:

| Stage | Mean time |
|---|---|
| JPEG decode (software) | 56.3 ms (about 330 ms per megapixel) |
| Resize + quantize into the model input | 2.4 ms |
| Network | 11.4 ms |
| Box decode | 0.5 ms |
| **File to box** | **70.6 ms mean, 75 ms median, 96 ms at the 95th percentile, 375 ms worst** (a 1600x1200 photo) |

- 430 of 432 files decoded. The two that did not are progressive JPEGs, which ESP-DL's
  software decoder does not take.
- The chip's hardware JPEG decoder took only 26 of 432: it refuses any picture whose width
  or height is not a multiple of 8. On files it accepts it is about 9x faster. Twenty of the
  same photos re-encoded at 480x320: 3.8 ms hardware against 33.6 ms software, 18 ms file
  to box (`results/jpeg-decoders-480x320-20261124.txt`).
- Accuracy on this dataset is **not a result**: the tiny detector does not learn pet heads.
  Two trainings (width 16 for 40 epochs, width 32 for 100) both ended no better than always
  predicting the average box on held-out images (26.7 and 27.1 px against 24.8 px at 224 px),
  on the desktop in float, before the board is involved. What the run does show is that the
  board tracks the desktop: mean IoU against the labels 0.283 on the board and 0.289 on the
  desktop over the same 430 files.
- Per file the board's box and the desktop's overlap by 0.67 IoU on average, not 1.0 as
  with raw frames. The decoders differ and ESP-DL resizes by nearest neighbour where
  training uses bilinear. With a model this unsure of itself that is an upper bound on the
  preprocessing effect, not a measurement of it.

For scale, Espressif's stock YOLO11n COCO detector (640 px, multi-object) on the same board:
62 ms to decode its 405x540 test JPEG, 1,462 ms per detection pass, same four boxes as
Espressif's reference output.

Reproduce: `tools/pets_dataset.py` builds the dataset, `tools/run_pets.sh` runs it through
the board in batches, `tools/score.py` scores the output.

**Operators.** All 15 converted and quantized; none failed, none substituted. The one
design change is deliberate: the soft-argmax (Softmax, Range, Reshape, ReduceSum) is not in
the on-device graph; it is done in firmware.

**Not done.** Images are embedded in the firmware image, not sent to the board at run time. No
quantization-aware training. Only one of the two boards was flashed.
