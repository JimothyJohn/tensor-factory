# Roadmap — own the stack, stay light

The north star: **the detector runs anywhere CPython runs, like Doom runs anywhere a
framebuffer exists.** Every third-party package is an optional *accelerator*, never a
requirement. The floor is the standard library; the ceiling is whatever the platform
offers. This file tracks what is self-implemented, what deliberately is not, and the
order of attack for the rest. Sibling docs: [`TODO.md`](TODO.md) tracks model quality;
this file tracks *platform ownership*.

## Where the floor stands today

`tensor_factory.zerort` (landed 2026-07) runs the full pipeline — PNG/PPM bytes →
protobuf model parse → int8 graph execution → box + presence — **on the stdlib alone**,
parity-tested against onnxruntime (`tests/test_zerort_parity.py` proves it in a
subprocess that *blocks* numpy/PIL/onnxruntime/onnx imports).

| Engine | Deps | 480 px throughput (Mac Studio, CPU) | Role |
|---|---|---|---|
| onnxruntime (`infer` extra) | onnxruntime, numpy, PIL | ~215 fps | fast path |
| zerort | **none** | ~0.7 fps | floor, oracle, last resort |

The trick that makes pure Python viable: **bigint SIMD** — image planes packed into
single arbitrary-precision integers so each ConvInteger tap is one C-speed bigint
multiply-add (~205M MACs/frame in ~1.4 s). Parity: ≤2e-3 normalized worst observed
(one quantization lane at an f32/f64 rounding boundary), 5e-3 gate — under the 1/255
wire-quantization step of the detection contract.

## Dependency ledger

| Dependency | Where | Status |
|---|---|---|
| onnxruntime + numpy + PIL | inference (`infer` extra) | **optional** — zerort is the fallback (was: required) |
| protobuf/onnx (model loading) | inference | **eliminated** — `zerort.pb` reads the wire format |
| PIL (image decode) | inference | **optional** — stdlib PNG/PPM decoder (JPEG still needs PIL) |
| torch | training | required, **stays** — see non-goals |
| GroundingDINO | auto-labeling | scheduled exit: model-in-the-loop labeling (`TODO.md` "Now") |
| Gemini API | image generation | inherently hosted — mitigate with dataset durability, not reimplementation |
| onnxruntime-web | docs demo | acceptable — browser WASM is the platform there |

## Next, in order

1. **Baseline JPEG decoder (stdlib).** Cameras emit JPEG; the zero-dep story currently
   stops at PNG/PPM. Baseline sequential DCT (no progressive) is ~400 lines and closes
   the gap for real capture pipelines. Adversarial tests like the PNG decoder's
   (truncation, hostile headers, fuzz).
2. **Zero-dep HTTP serving.** The HTTP endpoint (`tensor_factory_mcp.http_server`) is
   already stdlib `http.server`, but `core.py` PIL-opens the posted bytes — route
   PNG/PPM bytes through the stdlib decoder instead so a bare CPython can *serve*
   detections. Add the import-blocked test to pin it, and a `--engine` flag.
3. **Continuous portability proof.** A CI leg that runs the pipeline in a bare
   `python:3.12-slim` container with **no wheels installed** — the import-blocker unit
   test, but with the honesty of a real empty environment.
4. **zerort speed pass (bounded) — DONE (2026-07).** Weight prep hoisted into
   `Runtime.__init__`, planes kept as `bytes` rows (C-speed slice/join packing),
   elementwise ops dispatched to C functions instead of per-element lambdas, lazy
   clipping in DynamicQuantizeLinear. Measured: 1.44 s → 1.15 s/frame (~1.25×).
   The remainder is the floor: ~0.6 s is CPython's C bigint machinery itself, and
   ~0.2 s is spec-exact half-to-even `round()` that parity with onnxruntime requires.
   Pure Python will not approach onnxruntime and should not try — the floor's job is
   to exist everywhere, not to race.
5. **Own the model container (`.tfm`).** ONNX-protobuf stays the interchange format, but
   the MCU distillation target (`TODO.md` model 2, item 6) wants a flat, versioned,
   ~200-line-spec container: header + tensor table + raw little-endian data. Generated
   from the ONNX export; parseable in any language in an afternoon. Do this *when the
   MCU work starts*, not before.
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
