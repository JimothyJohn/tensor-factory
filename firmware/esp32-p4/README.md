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
  `uv venv ~/esp/ppq-venv --python 3.11 && VIRTUAL_ENV=~/esp/ppq-venv uv pip install esp-ppq torch pillow esptool`
  (`esptool` is for `tools/deploy.py`; it brings `pyserial`).

## Put a model on the board (no rebuild)

Flash this firmware once (`idf.py set-target esp32p4 && idf.py build && idf.py -p PORT flash`
with an empty `main/models/`). After that a model is one write to the `model` flash
partition, about 11 s, and the Studio's **Flash ESP32-P4** button does all of it:

```sh
PYTHONPATH=../../packages/tensor-factory-train/src:../../packages/tensor-factory/src \
  ~/esp/ppq-venv/bin/python tools/export_espdl.py --size 224 --onnx MODEL.onnx \
  --calib DATASET/images --out m.espdl                      # about 16 s
~/esp/ppq-venv/bin/python tools/deploy.py pack --espdl m.espdl --report m.report.json --out m.p4.bin
~/esp/ppq-venv/bin/python tools/deploy.py flash --port PORT --image photo.jpg m.p4.bin
~/esp/ppq-venv/bin/python tools/deploy.py detect --port PORT photo.jpg
```

`pack` puts a 64-byte header in front of the `.espdl` (name, input size, soft-argmax gain,
CRC32). On boot the firmware checks the CRC, runs ESP-DL's self-test, times the network and
prints one `MODEL {json}` line, then `READY`. From there each `IMG <base64 JPEG>` line on
the console gets a `DET {json}` line back (box, presence logit, per-stage times), and
`PING <token>` gets `READY <token>`. JPEGs must be baseline. `flash` fails unless the board
reports the CRC that was written and a passing self-test; a board that is not a P4 is
refused by `esptool`.

Measured 2026-11-24 on board 1: the pets model (224 px, context 4, 383 KB) written and
confirmed in 11 s; a 240x320 JPEG sent over the console, file to box, 33.5 ms on the board
(15.7 decode, 1.7 resize, 15.6 network, 0.5 box). The console runs at 115200 baud, so
sending that 14 KB file (19 KB as base64) takes well over a second; that is the transport, not the detector.

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

The app partition is 11 MB (it was 15 MB before the `model` partition existed), so
`tools/run_pets.sh` embeds 8.5 MB of JPEGs per batch.

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

**Image files as bytes, on an open detection dataset** (2026-11-24). Oxford-IIIT Pet head
boxes: trained on 3,238 photos, tested on 432 the model never saw. The test photos are
embedded as their original JPEG files (mean 100 KB, 0.17 megapixel); the board decodes,
resizes and detects each one itself. Model: 224 px, width 16, `--context 4`.
Raw output: `results/pets-224-c4-*` (bilinear) and `results/pets-224-c4-mixed-*`.

| Stage | Mean time |
|---|---|
| JPEG decode (software) | 56.3 ms (about 330 ms per megapixel) |
| Resize + quantize into the model input | 2.4 ms |
| Network | 15.5 ms |
| Box decode | 0.5 ms |
| **File to box** | **74.7 ms mean, 379 ms worst** (a 1600x1200 photo) |

| On the 430 files that decoded | Mean IoU | Boxes with IoU >= 0.5 | Median center error |
|---|---|---|---|
| Trained on bilinear resize: desktop, int8 | 0.593 | 71.9% | 5.2% of the image |
| Trained on bilinear resize: board | 0.546 | 63.7% | 5.4% of the image |
| **Trained with `--resize mixed`: desktop, int8** | **0.629** | **78.1%** | **4.7% of the image** |
| **Trained with `--resize mixed`: board, from the JPEG file** | **0.618** | **73.5%** | **4.8% of the image** |

- ESP-DL resizes by nearest neighbour; the detector used to be trained on bilinear only,
  and that mismatch cost the board 0.047 IoU (the desktop lost the same when made to resize
  the same way: 0.552). `--resize mixed` trains on both, picked per sample. It closes the
  gap to 0.011 and is also the better model on the desktop. Timings above are unchanged.
  What remains is the JPEG decoder and the exact sampling positions, which differ between
  PIL and ESP-DL. Float and int8 agree to within 0.4 px.
- 430 of 432 files decoded. The two that did not are progressive JPEGs, which ESP-DL's
  software decoder does not take.
- The chip's hardware JPEG decoder took only 26 of 432: it refuses any picture whose width
  or height is not a multiple of 8. On files it accepts it is about 9x faster. Twenty of the
  same photos re-encoded at 480x320: 3.8 ms hardware against 33.6 ms software
  (`results/jpeg-decoders-480x320-20261124.txt`).
- Depth is what made this work. The original four-layer detector (a 31 px window) never
  beat the average box on held-out pet photos (26.7 px against 24.8 px, and 27.1 px at
  double width). Four context blocks widen the window to 159 px: 10.7 px held out, for
  4 ms more network time. `results/pets-224-w16-*` is the failed model's run, kept for
  comparison.

For scale, Espressif's stock YOLO11n COCO detector (640 px, multi-object) on the same board:
62 ms to decode its 405x540 test JPEG, 1,462 ms per detection pass, same four boxes as
Espressif's reference output.

Reproduce: `tools/pets_dataset.py` builds the dataset, `tools/run_pets.sh` runs it through
the board in batches, `tools/score.py` scores the output.

**Operators.** All 15 converted and quantized; none failed, none substituted. The one
design change is deliberate: the soft-argmax (Softmax, Range, Reshape, ReduceSum) is not in
the on-device graph; it is done in firmware.

**Not done.** The only way to hand the board an image at run time is the 115200 baud
console (there is no camera and no network code). No quantization-aware training. Only one
of the two boards was flashed. A model with a class head deploys as box + presence only.
