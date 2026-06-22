# tensor-factory-train

Train the tiny single-object detector and export an **int8 ONNX** model.

```bash
# install torch for the run (a few hundred MB; not needed for the rest of the repo):
uv pip install torch

# train on a synth dataset and export a quantized model:
tensor-factory-train fit --data data/ --out model.onnx --epochs 30 --device mps

# add a presence head: train an 'absent' (background) class from no-object images, so the
# model can say "no helicoil here" instead of emitting a box (box loss is masked for them):
tensor-factory-train fit --data data/ --out model.onnx --device mps \
  --negatives negatives_pool/ --allow-unreviewed

# then run it on CPU via the core inference harness:
tensor-factory bench --model model.onnx
tensor-factory detect --model model.onnx --image some_frame.png
```

The model: four stride-2 conv blocks → a 1×1 conv heatmap per box edge → spatial softmax →
**soft-argmax** of each marginal yields the edge (sub-pixel, no size regression). Tiny
enough to int8-quantize and run well above 10 fps on CPU. Output is normalized `xyxy`,
matching `tensor_factory.inference`'s contract. With `--classify`/`--negatives` a mean+max
pooled class head rides alongside (`forward` returns `(box, logits)`), and the class names
are embedded in the exported ONNX metadata so the runtime is self-describing. Device
resolves `cuda → mps → cpu`; ONNX export runs on CPU.

**Validation gate.** Only human-`approved` annotations train by default (see
[`tensor_factory.review`](../tensor-factory/src/tensor_factory/review.py)); a dataset of
un-reviewed AI labels is refused with guidance to triage it first. Pass
`--allow-unreviewed` to deliberately train on raw labels (the fast path). Negatives have no
box and bypass the gate — a negative is a known absence.

Licensed under Apache-2.0.
