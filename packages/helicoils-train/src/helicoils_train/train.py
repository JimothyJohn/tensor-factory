"""Train the tiny detector on a COCO dataset and export an int8 ONNX model.

Device resolves cuda -> mps -> cpu for the training loop; export always runs on CPU
(ONNX export from an MPS graph is unreliable), then onnxruntime dynamic quantization
produces the uint8-weight model the edge runtime loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # ty: ignore[unresolved-import]
from PIL import Image
from torch import nn  # ty: ignore[unresolved-import]
from torch.utils.data import DataLoader, Dataset  # ty: ignore[unresolved-import]

from helicoils.geometry import BBox

from .data import load_coco_boxes
from .model import TinyDetector


def resolve_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _BoxDataset(Dataset):
    def __init__(self, items: list[tuple[Path, BBox]], size: int) -> None:
        self.items = items
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):  # noqa: ANN201
        path, box = self.items[index]
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((self.size, self.size), Image.Resampling.BILINEAR)
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        y = torch.tensor([box.x1, box.y1, box.x2, box.y2], dtype=torch.float32)
        return x, y


def export_onnx(
    model: nn.Module, out_path: str | Path, *, size: int, quantize: bool = True
) -> Path:
    """Export to ONNX on CPU, then dynamic-quantize weights to uint8."""
    out = Path(out_path)
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, size, size)
    fp32 = out.with_suffix(".fp32.onnx") if quantize else out
    torch.onnx.export(
        model,
        dummy,
        str(fp32),
        input_names=["image"],
        output_names=["box"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "box": {0: "batch"}},
    )
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QUInt8)
    return out


def fit(
    data_dir: str | Path,
    out_path: str | Path,
    *,
    epochs: int = 10,
    batch: int = 16,
    lr: float = 1e-3,
    size: int = 480,
    width: int = 16,
    device: str | None = None,
) -> Path:
    """Train on ``<data_dir>/annotations.coco.json`` + images and export int8 ONNX."""
    data_dir = Path(data_dir)
    items = load_coco_boxes(data_dir / "annotations.coco.json", data_dir)
    if not items:
        raise ValueError(f"no annotations found under {data_dir}")

    dev = resolve_device(device)
    loader = DataLoader(_BoxDataset(items, size), batch_size=batch, shuffle=True)
    model = TinyDetector(width).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        print(f"epoch {epoch + 1}/{epochs}  loss {total / len(items):.5f}")

    return export_onnx(model, out_path, size=size)
