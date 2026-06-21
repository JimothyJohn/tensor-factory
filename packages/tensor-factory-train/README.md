# tensor-factory-train

Train the tiny single-object detector and export an **int8 ONNX** model.

```bash
# install torch for the run (a few hundred MB; not needed for the rest of the repo):
uv pip install torch

# train on a synth dataset and export a quantized model:
tensor-factory-train fit --data data/ --out model.onnx --epochs 30 --device mps

# then run it on CPU via the core inference harness:
tensor-factory bench --model model.onnx
tensor-factory detect --model model.onnx --image some_frame.png
```

The model: five stride-2 conv blocks → global pool → linear → sigmoid `xyxy`. Tiny
enough to int8-quantize and run well above 10 fps on CPU. Output is normalized `xyxy`,
matching `tensor_factory.inference`'s contract. Device resolves `cuda → mps → cpu`; ONNX
export runs on CPU.

**Validation gate.** Only human-`approved` annotations train by default (see
[`tensor_factory.review`](../tensor-factory/src/tensor_factory/review.py)); a dataset of
un-reviewed AI labels is refused with guidance to triage it first. Pass
`--allow-unreviewed` to deliberately train on raw labels.

Licensed under Apache-2.0.
