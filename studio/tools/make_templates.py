"""Write the ONNX templates the browser exporter fills in (studio/js/nn/onnx.js).

One real export per context count, at width 1 so each file is a few kilobytes. The graph
does not depend on width or input size; only the weight tensors do, and the browser
replaces those. Regenerate when ``TinyDetector`` or ``export_onnx`` changes:

    uv run --all-packages --extra train python studio/tools/make_templates.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch  # ty: ignore[import-not-found]

from tensor_factory_train.model import TinyDetector
from tensor_factory_train.train import export_onnx

OUT = Path(__file__).resolve().parents[1] / "models"
MAX_CONTEXT = 8


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for context in range(MAX_CONTEXT + 1):
        torch.manual_seed(0)
        model = TinyDetector(1, presence=True, context=context)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_onnx(model, Path(tmp) / "t.onnx", size=64)
            data = path.read_bytes()
        (OUT / f"template-c{context}.onnx").write_bytes(data)
        print(f"template-c{context}.onnx  {len(data)} B")


if __name__ == "__main__":
    main()
