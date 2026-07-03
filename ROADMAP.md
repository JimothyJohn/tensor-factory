# Roadmap — own the stack, stay light

The north star: **the detector runs anywhere CPython runs, like Doom runs anywhere a
framebuffer exists.** Every third-party package is an optional *accelerator*, never a
requirement. The floor is the standard library; the ceiling is whatever the platform
offers. This file tracks what is self-implemented, what deliberately is not, and the
order of attack for the rest. Sibling docs: [`TODO.md`](TODO.md) tracks model quality;
this file tracks *platform ownership*.

## Where the floor stands today

`tensor_factory.zerort` (landed 2026-07) runs the full pipeline — PNG/JPEG/PPM bytes →
model parse (ONNX protobuf or the flat `.tfm` container) → int8 graph execution → box +
presence — **on the stdlib alone**, and the stdlib HTTP endpoint serves it. Proof lives
at three levels: parity tests against onnxruntime, import-blocked subprocess tests
(`tests/test_zerort_parity.py`, the HTTP equivalent in `tensor-factory-mcp`), and a CI
job running `scripts/portability_check.py` in a bare `python:3.12-slim` container with
no wheels installed.

| Engine | Deps | 480 px throughput (Mac Studio, CPU) | Role |
|---|---|---|---|
| onnxruntime (`infer` extra) | onnxruntime, numpy, PIL | ~215 fps | fast path |
| zerort | **none** | ~0.9 fps | floor, oracle, last resort |

The trick that makes pure Python viable: **bigint SIMD** — image planes packed into
single arbitrary-precision integers so each ConvInteger tap is one C-speed bigint
multiply-add (~205M MACs/frame in ~1.15 s). Parity: ≤2e-3 normalized worst observed
(one quantization lane at an f32/f64 rounding boundary), 5e-3 gate — under the 1/255
wire-quantization step of the detection contract.

## Dependency ledger

| Dependency | Where | Status |
|---|---|---|
| onnxruntime + numpy + PIL | inference (`infer` extra) | **optional** — zerort is the fallback (was: required) |
| protobuf/onnx (model loading) | inference | **eliminated** — `zerort.pb` reads the wire format |
| PIL (image decode) | inference | **optional** — stdlib PNG, baseline JPEG, and PPM decoders (progressive JPEG still needs PIL) |
| torch | training | required, **stays** — see non-goals |
| GroundingDINO | auto-labeling | scheduled exit: model-in-the-loop labeling (`TODO.md` "Now") |
| Gemini API | image generation | inherently hosted — mitigate with dataset durability, not reimplementation |
| onnxruntime-web | docs demo | acceptable — browser WASM is the platform there |

## Next, in order

Items 1–5 below landed 2026-07 (kept with their outcomes for the record); item 6 is
gated on model quality, not code — the detector must first beat GroundingDINO's boxes.

1. **Baseline JPEG decoder (stdlib) — DONE (2026-07).** `zerort.jpeg`: baseline
   sequential Huffman, gray + YCbCr, 4:4:4/4:2:2/4:2:0, restart markers; two-level
   Huffman LUT gets 480 px decodes to ~0.4 s. Pinned against libjpeg (mean <1.5 code
   values on photographic content); progressive/arithmetic/CMYK rejected clearly;
   truncation/hostile-header/fuzz tests like the PNG decoder's.
2. **Zero-dep HTTP serving — DONE (2026-07).** `core.detect/detect_bytes` fall back
   to the stdlib decoder when PIL is absent; every result is one forward pass
   (`Detector.detect_outputs`); `tensor-factory-http --engine`; pinned by an
   import-blocked subprocess test doing a real socket round-trip.
3. **Continuous portability proof — DONE (2026-07).** The CI `portability` job runs
   `scripts/portability_check.py` in `python:3.12-slim` with no wheels: PNG detect,
   `.tfm` pack + output-identical re-detect, PPM bytes, HTTP round-trip.
4. **zerort speed pass (bounded) — DONE (2026-07).** Weight prep hoisted into
   `Runtime.__init__`, planes kept as `bytes` rows (C-speed slice/join packing),
   elementwise ops dispatched to C functions instead of per-element lambdas, lazy
   clipping in DynamicQuantizeLinear. Measured: 1.44 s → 1.15 s/frame (~1.25×).
   The remainder is the floor: ~0.6 s is CPython's C bigint machinery itself, and
   ~0.2 s is spec-exact half-to-even `round()` that parity with onnxruntime requires.
   Pure Python will not approach onnxruntime and should not try — the floor's job is
   to exist everywhere, not to race.
5. **Own the model container (`.tfm`) — DONE (2026-07, pulled forward).**
   `zerort.tfm`: flat little-endian versioned container, spec = the module docstring;
   `tensor-factory pack` converts (comes out ~3% smaller than the ONNX); `Runtime`
   loads either format by magic sniff. Round-trip bar is exact output equality on the
   bundled models. *(2026-07-02: the target platform is now an ARM SBC and the MCU tier
   is deferred — `.tfm`'s remaining value is the protobuf-free loader for zerort, not an
   MCU story.)*
6. **GroundingDINO exit** (tracked in `TODO.md`): once v4+ beats the auto-labeler on the
   human-approved set, the model labels its own data and the heaviest dependency in the
   repo leaves the loop.

## Non-goals — where self-implementation is negative-value

- **Autograd/training.** Reimplementing torch is a research career, not a dependency
  win. Training is an offline, GPU-shaped step behind an extra; it never ships to the
  edge. The exported int8 graph is the contract, and that we own fully.
- **zlib/deflate.** The stdlib ships it; CPython itself is the dependency floor.
- **A Rust runtime.** Tempting (and would beat 0.7 fps handily), but this is a
  one-language repo by policy. If a deployment ever needs fast inference where
  onnxruntime wheels don't exist, that becomes a *separate* decision — flag it, don't
  drift into it.
- **GPU inference.** The whole point of the tiny-CNN factory is that CPU is enough.
