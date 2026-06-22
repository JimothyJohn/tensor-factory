# tensor-factory

**Open, lightweight tiny-CNN object detection** — a factory for turning a prompt into a
synthetic dataset, an int8 ONNX model, and a CPU inference harness. No GPU at runtime, no
AGPL anywhere (Apache-2.0 throughout, no Ultralytics).

> 📄 Project page: [`docs/index.html`](docs/index.html)

The pipeline: **synthesize → auto-label → train → run.**

1. **Synthesize** images from a prompt — hosted Gemini (Nano Banana) or a deterministic mock.
2. **Auto-label** with GroundingDINO; refine in Label Studio if you want.
3. **Train** a tiny soft-argmax CNN and export an int8 ONNX model.
4. **Run** on CPU via onnxruntime — from a CLI or over MCP.

A detection is four `uint8` values (normalized `xyxy`, one byte each). At 480 px that's
~1.88 px per step — round-trip error under 1 px, inside the 3 px budget, and all
post-processing stays in 8-bit math.

**Reference numbers (helicoils example):** 204 fps CPU @480px · 81 KB int8 model · ~1.9 px
median localization.

## Examples

- [`examples/helicoils`](examples/helicoils) — detecting coiled-wire threaded inserts in
  machined parts from microscope imagery. The first example, and what every package is
  currently tuned against.

## Packages

| Package | Role |
|---|---|
| [`tensor-factory`](packages/tensor-factory) | Core: BBox geometry, 4×uint8 codec, formats, ONNX inference + CLI (dependency-free, CPU-only) |
| [`tensor-factory-synth`](packages/tensor-factory-synth) | Generation (Nano Banana/Gemini) + GroundingDINO auto-label + COCO/Label Studio export |
| [`tensor-factory-train`](packages/tensor-factory-train) | Tiny soft-argmax detector → int8 ONNX |
| [`tensor-factory-mcp`](packages/tensor-factory-mcp) | FastMCP server exposing the detector (bundled demo model) |
| [`tensor-factory-label`](packages/tensor-factory-label) | Label Studio push/pull |

## Develop

```bash
./Quickstart          # bootstrap (uv sync) + run the MCP detection server (try it out, no flag)
./Quickstart -c       # full gate: uv sync --locked → ruff check/format → ty check → pytest -m unit
./Quickstart -u       # unit tests only
```

Python `uv` workspace (monorepo). Generation is a hosted API call (no GPU); auto-labeling
and training are GPU-heavy and live behind extras, resolving `cuda → mps → cpu`.

Licensed under Apache-2.0.
