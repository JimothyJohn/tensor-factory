# Datatypes: why int8/uint8, and not FP8

A debate prompted by a fair question: *"FP8 might be a better idea than int8 — its
fine precision near 0 beats int8's uniform spacing."* The instinct is sound in the
world it comes from (LLM inference on datacenter GPUs). It is the wrong call **here**,
and this note says why, with numbers.

## TL;DR

- **Verdict: keep uint8** for both the output codec and the deployed weights.
- This project has **two distinct 8-bit decisions**, and people conflate them:
  1. the **output codec** — how a box is encoded on the wire, and
  2. the **model quantization** — how weights are stored for CPU inference.
- For the **codec**, FP8 is *quantitatively worse*: box edges are ~uniform over
  `[0,1]`, and uniform quantization (uint8) is optimal for a uniform distribution.
  FP8's non-uniform spacing wastes precision near 0 and goes coarse (up to ~30 px
  error) exactly where most coordinates live.
- For the **weights**, FP8 buys nothing on our target (CPU, onnxruntime, edge): x86/ARM
  CPUs have **no native FP8 arithmetic**, onnxruntime's CPU kernels are int8 (VNNI /
  ARM dotprod), and an FP8 model is the same 8 bits — no smaller than the 81 KB int8
  one, just slower (emulated as fp32).

## First, correct the premise

Two facts reframe the question:

1. **There is no signed `-128..127` in the hot path.** The codec is `uint8`,
   `0..255` (`packages/tensor-factory/src/tensor_factory/codec.py`, `_MAX = 255`).
   The weights are quantized **unsigned** too —
   `quantize_dynamic(..., weight_type=QuantType.QUInt8)`
   (`packages/tensor-factory-train/src/tensor_factory_train/train.py:123`).

2. **FP8 is not a `[0,1]` format.** The two IEEE-ish FP8 variants are:
   | format | bits (S/E/M) | max magnitude | smallest normal |
   |--------|--------------|---------------|-----------------|
   | E4M3   | 1 / 4 / 3    | ±448          | 2⁻⁶ ≈ 0.0156    |
   | E5M2   | 1 / 5 / 2    | ±57344        | 2⁻¹⁴ ≈ 6.1e-5   |

   FP8 is a *signed float*. Its real characteristic isn't a `0–1` range — it's
   **non-uniform spacing**: many representable values bunched near 0, few far away.
   That is the whole debate.

## Decision 1 — the output codec (uint8 wins, decisively)

A detection is four normalized coordinates in `[0,1]`. The question is how to discretize
each into 8 bits. The distribution of a box edge over an image is **roughly uniform** —
an edge is about as likely at x=0.1 as at x=0.9. For a uniform source, the
minimum-mean-squared-error quantizer is the **uniform** quantizer. uint8 *is* that:

- step = `480 / 255 ≈ 1.882 px`, flat across the whole range
- worst-case round-trip = `image_size / 255 / 2 ≈ 0.94 px` (`codec.max_error_px`)

FP8 (E4M3) over the same `[0,1]`, by binade (step within `[2ᵉ, 2ᵉ⁺¹)` is `2ᵉ⁻³`):

| coordinate range | FP8 step | FP8 error @480px | uint8 error @480px |
|------------------|----------|------------------|--------------------|
| 0.50 – 1.00      | 0.0625   | **30.0 px**      | 1.88 px            |
| 0.25 – 0.50      | 0.03125  | 15.0 px          | 1.88 px            |
| 0.125 – 0.25     | 0.015625 | 7.5 px           | 1.88 px            |
| 0.0625 – 0.125   | 0.0078   | 3.75 px          | 1.88 px            |
| 0.031 – 0.0625   | 0.0039   | 1.88 px          | 1.88 px            |
| 0.016 – 0.031    | 0.00195  | 0.94 px          | 1.88 px            |
| < 0.016 (subnormal) | ↓     | < 0.5 px         | 1.88 px            |

Read the top row: for the **entire upper half of the coordinate range** — half of all
box edges — FP8 is **~16× worse** than uint8 (30 px vs 1.88 px). FP8 only ties or beats
uint8 below ~0.06, i.e. ~6% of the range, hugging the top-left corner. With a 3 px
localization budget, FP8 blows it for the majority of boxes; uint8 stays under 1 px
everywhere. This isn't close.

*Why the intuition misfires:* "fine precision near 0" is exactly the wrong gift when the
signal is uniform — you spend your codes where the data isn't dense and starve the rest.
FP8 shines when the signal itself is heavy-tailed / log-distributed (activations,
gradients). Coordinates are not.

## Decision 2 — model weight quantization (uint8 wins on hardware, ties on accuracy)

Here the case is about the deployment target, which is explicit in the repo: **CPU-only,
dependency-free, onnxruntime, edge** (`CLAUDE.md` "Compute"; inference resolves to
`CPUExecutionProvider`). Against that target:

**Hardware support — the decider.**
- **int8**: first-class on CPU. x86 AVX2 / AVX-512-VNNI and ARM NEON `SDOT/UDOT` do
  8-bit integer dot-products in hardware; onnxruntime ships mature `ConvInteger` /
  `MatMulInteger` / QLinear kernels. This is *why* the int8 model runs at 204 fps.
- **FP8**: no native FP8 ALU on any mainstream x86 or ARM CPU. ONNX added FP8 *types*
  (opset 19, `Float8E4M3FN`/`E5M2`) and CUDA can use them on Hopper/Ada tensor cores,
  but the **CPU execution provider has no FP8 compute** — it would up-convert to fp32
  and emulate. Result: same 8-bit storage, *fp32 speed or worse*. The entire perf thesis
  of this project evaporates.

**Size.** Both are 8 bits/weight → both ~4× smaller than fp32. FP8 would **not** shrink
the 81 KB model. No win.

**Accuracy.** The tiny CNN's weights are well-behaved (bounded, ~Gaussian, batchnorm'd),
no fat outliers — precisely the regime where uniform int8 with per-channel scales is
near-lossless. FP8's dynamic-range advantage addresses a problem this model doesn't have.
And the binding accuracy constraint is the **codec** (~1.9 px median localization),
not weight precision — so even a "better" weight format can't move the headline metric.

## Where FP8 genuinely wins (and why none of it applies)

To be fair to the idea — FP8 is winning real ground, just not ours:

- **LLM / transformer inference on H100 / Ada / MI300.** Activations have large dynamic
  range and outlier channels; uniform int8 clips or needs per-group scaling, while FP8's
  range absorbs it. We have no transformer and no such tensors.
- **fp8 mixed-precision *training*** on datacenter GPUs. Our training is an offline PyTorch
  step; the *deployment* artifact is what we're quantizing, and it targets CPU.
- **Tensors spanning many orders of magnitude** (log-distributed). Coordinates and CNN
  weights are not.

Every FP8 advantage is conditioned on hardware we don't deploy to or distributions we
don't have.

## When to revisit

Reopen this only if the project's premises change:

- The deployment target becomes an **FP8-capable accelerator** (Hopper/Ada/Blackwell,
  MI300) *and* throughput matters more than the dependency-free CPU story.
- The architecture grows into something with **outlier-heavy activations** (e.g. a
  transformer backbone), where int8 calibration starts costing accuracy.
- The output stops being uniform-over-`[0,1]` coordinates and becomes a quantity that is
  genuinely **log/heavy-tailed**, where non-uniform spacing pays off.

Until then: uint8 for the codec (optimal for uniform coordinates), uint8 dynamic
quantization for the weights (the only 8-bit format with CPU hardware behind it).

## Sources in this repo

- Codec: `packages/tensor-factory/src/tensor_factory/codec.py` (`_MAX = 255`,
  `max_error_px = size / 255 / 2`).
- Weight quant: `packages/tensor-factory-train/src/tensor_factory_train/train.py:120-123`
  (`quantize_dynamic`, `QuantType.QUInt8`).
- Inference / providers: `packages/tensor-factory/src/tensor_factory/inference.py`;
  observed `providers: ["CPUExecutionProvider"]` from `tensor_factory_model_info`.
- Compute thesis: `CLAUDE.md` ("Compute", "Detection contract").
