"""Train the tiny detector on a COCO dataset and export an int8 ONNX model.

Device resolves cuda -> mps -> cpu for the training loop; export always runs on CPU
(ONNX export from an MPS graph is unreliable), then onnxruntime dynamic quantization
produces the uint8-weight model the edge runtime loads.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

import numpy as np
import torch  # ty: ignore[unresolved-import]
from PIL import Image
from torch import nn  # ty: ignore[unresolved-import]
from torch.utils.data import DataLoader, Dataset  # ty: ignore[unresolved-import]

from .data import load_coco_boxes, load_coco_labeled
from .model import TinyDetector


def resolve_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_chw(path: Path, size: int):  # noqa: ANN202
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class _BoxDataset(Dataset):
    """Yields ``(image, box)`` for box-only training, or ``(image, box, label)`` when
    ``items`` carry a class id (3-tuples). One dataset, both heads."""

    def __init__(self, items: list, size: int, *, labeled: bool = False) -> None:
        self.items = items
        self.size = size
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):  # noqa: ANN201
        if self.labeled:
            path, box, label = self.items[index]
        else:
            path, box = self.items[index]
        x = _load_chw(path, self.size)
        y = torch.tensor([box.x1, box.y1, box.x2, box.y2], dtype=torch.float32)
        if self.labeled:
            return x, y, torch.tensor(label, dtype=torch.long)
        return x, y


def export_onnx(
    model: nn.Module, out_path: str | Path, *, size: int, quantize: bool = True
) -> Path:
    """Export to ONNX on CPU, then dynamic-quantize weights to uint8."""
    out = Path(out_path)
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, size, size)
    fp32 = out.with_suffix(".fp32.onnx") if quantize else out

    # A class head makes forward return (box, logits): export both, name them so the
    # runtime can read each by name regardless of graph order.
    with torch.no_grad():
        multi = isinstance(model(dummy), tuple)
    output_names = ["box", "logits"] if multi else ["box"]
    dynamic_axes = {name: {0: "batch"} for name in ["image", *output_names]}
    torch.onnx.export(
        model,
        dummy,
        str(fp32),
        input_names=["image"],
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
        # Legacy TorchScript exporter: a static tiny CNN needs nothing fancier, and
        # it avoids the onnxscript dependency the dynamo exporter requires.
        dynamo=False,
    )
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QUInt8)
    return out


def _split(items: list, val_frac: float, seed: int) -> tuple[list, list]:
    if val_frac <= 0:
        return items, []
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(items) * val_frac))
    return [items[i] for i in idx[n_val:]], [items[i] for i in idx[:n_val]]


def _report_val(model, val_items, size, dev, *, classify, names) -> None:  # noqa: ANN001
    """Print held-out median center error (px) and, when classifying, class accuracy."""
    model.eval()
    errs: list[float] = []
    correct = 0
    with torch.no_grad():
        for item in val_items:
            path, box = item[0], item[1]
            out = model(_load_chw(path, size).unsqueeze(0).to(dev))
            pb = (out[0] if isinstance(out, tuple) else out)[0].cpu().numpy()
            pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            gcx, gcy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
            errs.append(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5 * size)
            if classify:
                correct += int(out[1][0].argmax().item() == item[2])
    msg = f"VAL ({len(val_items)}): median center err {statistics.median(errs):.1f}px"
    if classify:
        msg += f"  |  class acc {correct / len(val_items):.1%}  classes={names}"
    print(msg)


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
    classify: bool = False,
    val_frac: float = 0.0,
    seed: int = 0,
) -> Path:
    """Train on ``<data_dir>/annotations.coco.json`` + images and export int8 ONNX.

    ``classify=True`` adds the class head (categories read from the COCO file) and trains
    box regression + cross-entropy jointly. ``val_frac`` holds out a split for end-of-run
    metrics (median center error, and class accuracy when classifying).
    """
    data_dir = Path(data_dir)
    coco = data_dir / "annotations.coco.json"
    if classify:
        items, names = load_coco_labeled(coco, data_dir)
    else:
        items, names = load_coco_boxes(coco, data_dir), []
    if not items:
        raise ValueError(f"no annotations found under {data_dir}")
    num_classes = len(names) if classify else 0

    train_items, val_items = _split(items, val_frac, seed)
    dev = resolve_device(device)
    loader = DataLoader(
        _BoxDataset(train_items, size, labeled=classify), batch_size=batch, shuffle=True
    )
    model = TinyDetector(width, num_classes).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    box_loss = nn.SmoothL1Loss()
    cls_loss = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for data in loader:
            optimizer.zero_grad()
            if classify:
                x, yb, yc = data[0].to(dev), data[1].to(dev), data[2].to(dev)
                pb, logits = model(x)
                loss = box_loss(pb, yb) + cls_loss(logits, yc)
            else:
                x, yb = data[0].to(dev), data[1].to(dev)
                loss = box_loss(model(x), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        print(f"epoch {epoch + 1}/{epochs}  loss {total / len(train_items):.5f}")

    if val_items:
        _report_val(model, val_items, size, dev, classify=classify, names=names)
    return export_onnx(model, out_path, size=size)
