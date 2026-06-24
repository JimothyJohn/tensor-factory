"""Train the tiny detector on a COCO dataset and export an int8 ONNX model.

Device resolves cuda -> mps -> cpu for the training loop; export always runs on CPU
(ONNX export from an MPS graph is unreliable), then onnxruntime dynamic quantization
produces the uint8-weight model the edge runtime loads.
"""

from __future__ import annotations

import copy
import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch  # ty: ignore[unresolved-import]
from PIL import Image
from torch import nn  # ty: ignore[unresolved-import]
from torch.utils.data import DataLoader, Dataset  # ty: ignore[unresolved-import]

from tensor_factory.review import review_summary

from .data import load_coco_boxes, load_coco_labeled, load_negatives
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


def _flip_box(
    coords: tuple[float, float, float, float], *, horizontal: bool
) -> tuple[float, float, float, float]:
    """Mirror a normalized ``xyxy`` box. Horizontal swaps/reflects x, vertical reflects y."""
    x1, y1, x2, y2 = coords
    if horizontal:
        return (1.0 - x2, y1, 1.0 - x1, y2)
    return (x1, 1.0 - y2, x2, 1.0 - y1)


class _BoxDataset(Dataset):
    """Yields ``(image, box)`` for box-only training, or ``(image, box, label, has_box)``
    when ``items`` carry a class id (3-tuples ``(path, box | None, label)``). One dataset,
    both heads.

    A negative (background) item has ``box is None``: the image still trains the class
    head, but ``has_box == 0`` masks it out of the box-regression loss -- a no-object frame
    has no box to fit. ``augment`` adds random horizontal/vertical flips (image and box
    transformed together; a box-less image just flips the pixels).
    """

    def __init__(
        self, items: list, size: int, *, labeled: bool = False, augment: bool = False
    ) -> None:
        self.items = items
        self.size = size
        self.labeled = labeled
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):  # noqa: ANN201
        if self.labeled:
            path, box, label = self.items[index]
        else:
            path, box = self.items[index]
        x = _load_chw(path, self.size)
        has_box = box is not None
        coords = (box.x1, box.y1, box.x2, box.y2) if has_box else (0.0, 0.0, 0.0, 0.0)
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, [2])  # width
                if has_box:
                    coords = _flip_box(coords, horizontal=True)
            if random.random() < 0.5:
                x = torch.flip(x, [1])  # height
                if has_box:
                    coords = _flip_box(coords, horizontal=False)
        y = torch.tensor(coords, dtype=torch.float32)
        if self.labeled:
            return (
                x,
                y,
                torch.tensor(label, dtype=torch.long),
                torch.tensor(float(has_box), dtype=torch.float32),
            )
        return x, y


def _embed_class_names(path: Path, names: list[str]) -> None:
    """Record class names in the ONNX file's metadata so the runtime is self-describing.

    Lets inference map a class id to a name -- and tell *present* from *background* --
    without hard-coding the training-time class ordering on the edge.
    """
    import onnx

    model = onnx.load(str(path))
    model.metadata_props.add(key="class_names", value=json.dumps(names))
    onnx.save(model, str(path))


def export_onnx(
    model: nn.Module,
    out_path: str | Path,
    *,
    size: int,
    quantize: bool = True,
    class_names: list[str] | None = None,
) -> Path:
    """Export to ONNX on CPU, then dynamic-quantize weights to uint8.

    ``class_names`` (when the model has a class head) is embedded in the final file's
    metadata so the runtime can resolve class ids to names.
    """
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
    if class_names:
        _embed_class_names(out, class_names)
    return out


def _split(items: list, val_frac: float, seed: int) -> tuple[list, list]:
    if val_frac <= 0:
        return items, []
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(items) * val_frac))
    return [items[i] for i in idx[n_val:]], [items[i] for i in idx[:n_val]]


def _val_metrics(model, val_items, size, dev, *, classify) -> tuple[float, float | None]:  # noqa: ANN001
    """Held-out (median center error px, class accuracy or None) on ``val_items``."""
    model.eval()
    errs: list[float] = []
    correct = 0
    with torch.no_grad():
        for item in val_items:
            box = item[1]
            out = model(_load_chw(item[0], size).unsqueeze(0).to(dev))
            # Box error only where there is a box -- negatives (background) have none.
            if box is not None:
                pb = (out[0] if isinstance(out, tuple) else out)[0].cpu().numpy()
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                gcx, gcy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
                errs.append(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5 * size)
            if classify:
                correct += int(out[1][0].argmax().item() == item[2])
    acc = correct / len(val_items) if classify else None
    model.train()
    # No positive in the val split -> no box error to report (0 penalty; acc still leads).
    return (statistics.median(errs) if errs else 0.0), acc


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
    box_weight: float = 1.0,
    cls_weight: float = 1.0,
    augment: bool = False,
    weight_decay: float = 0.0,
    require_review: bool = True,
    negatives: Sequence[str | Path] | None = None,
    learn_gain: bool = True,
) -> Path:
    """Train on ``<data_dir>/annotations.coco.json`` + images and export int8 ONNX.

    ``classify=True`` adds the class head (categories read from the COCO file) and trains
    box regression + cross-entropy jointly; ``box_weight``/``cls_weight`` balance the two
    (the box loss is far smaller, so weight it up to keep the class gradient from drowning
    it). ``augment`` adds flip augmentation. With ``val_frac`` a split is held out and the
    **best-scoring** checkpoint (class acc, then center error) is what gets exported -- not
    whatever the last, possibly-overfit epoch produced.

    With ``require_review`` (default) only human-validated annotations train; an
    all-pending dataset is refused with guidance to triage it first. ``require_review=False``
    trains on everything regardless of review state.

    ``negatives`` is one or more directories of raw no-object images (e.g. machined parts
    with holes but no helicoil). They force the class head on, add a trailing ``background``
    class, and train it to report *absent*; box loss is masked out for them. This is what
    lets the detector say "nothing here" instead of emitting a spurious box.
    """
    data_dir = Path(data_dir)
    coco = data_dir / "annotations.coco.json"

    summary = review_summary(json.loads(coco.read_text(encoding="utf-8")))
    a = summary["annotations"]
    print(
        f"triage: {a['total']} annotations -- {a['approved']} approved, "
        f"{a['pending']} pending, {a['rejected']} rejected"
        + ("" if require_review else "  (review gate OFF: training on all)")
    )

    # Negatives add a background class, so the class head is mandatory when they are used.
    classify = classify or bool(negatives)
    if classify:
        items, names = load_coco_labeled(coco, data_dir, require_review=require_review)
        if negatives:
            bg_label = len(names)
            names = [*names, "background"]
            neg_paths = [p for d in negatives for p in load_negatives(d)]
            items = [*items, *[(p, None, bg_label) for p in neg_paths]]
            print(f"negatives: +{len(neg_paths)} background images -> class {bg_label}")
    else:
        items, names = load_coco_boxes(coco, data_dir, require_review=require_review), []
    if not items:
        if require_review and a["pending"]:
            raise ValueError(
                f"no approved annotations under {data_dir}: {a['pending']} pending human "
                "review. Validate them via tensor-factory-label (push -> correct -> pull), "
                "or pass require_review=False / --allow-unreviewed to train on unvalidated "
                "labels deliberately."
            )
        raise ValueError(f"no annotations found under {data_dir}")
    num_classes = len(names) if classify else 0

    train_items, val_items = _split(items, val_frac, seed)
    # Seed torch so weight init is reproducible run-to-run -- the split is already seeded, so
    # with this the only variable between two runs is the hyperparameters (e.g. the gain).
    torch.manual_seed(seed)
    dev = resolve_device(device)
    loader = DataLoader(
        _BoxDataset(train_items, size, labeled=classify, augment=augment),
        batch_size=batch,
        shuffle=True,
    )
    model = TinyDetector(width, num_classes, learn_gain=learn_gain).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    box_loss = nn.SmoothL1Loss()
    cls_loss = nn.CrossEntropyLoss()

    best_score = float("-inf")
    best_state: dict | None = None
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for data in loader:
            optimizer.zero_grad()
            if classify:
                x, yb, yc, m = (d.to(dev) for d in (data[0], data[1], data[2], data[3]))
                pb, logits = model(x)
                # Box loss only over box-bearing items; negatives (mask 0) contribute none.
                mb = m.bool()
                box_term = box_loss(pb[mb], yb[mb]) if bool(mb.any()) else pb.sum() * 0.0
                loss = box_weight * box_term + cls_weight * cls_loss(logits, yc)
            else:
                x, yb = data[0].to(dev), data[1].to(dev)
                loss = box_weight * box_loss(model(x), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)

        line = f"epoch {epoch + 1}/{epochs}  loss {total / len(train_items):.5f}"
        if val_items:
            err, acc = _val_metrics(model, val_items, size, dev, classify=classify)
            # Higher is better. ``err`` is the median center error in *pixels*; normalize by
            # image size so localization trades against accuracy on a comparable [0,1] scale.
            # The old ``err / 1000`` made a 15px miss worth 0.015 -- swamped by a single
            # accuracy step -- so selection shipped the best-classifying epoch with no regard
            # for box quality. Accuracy still leads (one val step >> a few px), but boxes now
            # matter ~50x more, and box-only runs (acc is None) still rank by lowest error.
            score = (acc if acc is not None else 0.0) - err / size
            line += f"  | val err {err:.1f}px" + (f" acc {acc:.0%}" if acc is not None else "")
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                line += "  *best"
        print(line)

    if best_state is not None:
        model.load_state_dict(best_state)
        if classify:
            err, acc = _val_metrics(model, val_items, size, dev, classify=classify)
            print(f"BEST checkpoint: val err {err:.1f}px  class acc {acc:.0%}  classes={names}")
    print(f"soft-argmax gain: {float(model.log_gain.exp()):.2f} (1.0 = plain softmax)")
    return export_onnx(model, out_path, size=size, class_names=names if classify else None)
