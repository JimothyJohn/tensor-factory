# tensor-factory-studio

Local backend for [Tensor Factory Studio](../../studio/) — serves the browser labeling UI
and runs **continuous training** on the frames you label, on this machine's GPU (MPS/CUDA)
via `tensor-factory-train`.

The browser handles ingest, the canvas, and the WASD labeling; this backend owns the
on-disk COCO dataset and the training loop. Each labeled frame is pushed to `POST /samples`,
which marks the dataset dirty; a background thread retrains the tiny detector (reusing
`tensor_factory_train.fit`, so it emits the same int8 ONNX the rest of the repo uses),
streams per-epoch val metrics to `GET /metrics`, and serves the best checkpoint for
auto-labeling via `POST /predict`. A keep-best guardrail only promotes a round's model if it
beats the global-best val center-error; otherwise the previous model stays live and the
round is flagged as a regression.

## Run

```sh
uv sync --extra serve          # pulls torch
uv run tensor-factory-studio   # http://127.0.0.1:8089
# open the printed URL in a Chromium-based browser
```

Flags: `--host --port --data-dir --ui-dir --size --width --epochs --batch`.
Training resolves `cuda → mps → cpu` automatically.

## Layout it writes

```
<data-dir>/annotations.coco.json       positives (review=approved, source=human)
<data-dir>/images/frame_NNNNN.png
<data-dir>/negatives/images/frame_NNNNN.png   empty frames (presence-head 0s)
<data-dir>/models/served-vN.onnx        best checkpoint, served for auto-label
```

This is exactly the layout `tensor-factory-train` reads, so the dataset and the deployable
int8 ONNX both drop straight into the existing tooling.
