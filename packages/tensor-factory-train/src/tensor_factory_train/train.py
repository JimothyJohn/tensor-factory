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
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch  # ty: ignore[unresolved-import]
from PIL import Image
from torch import nn  # ty: ignore[unresolved-import]
from torch.utils.data import DataLoader, Dataset  # ty: ignore[unresolved-import]

from tensor_factory.review import review_summary

from .data import baseline_center_err, load_coco_labeled, load_negatives
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
    """Yields ``(image, box)`` for box-only training, ``(image, box, has_box)`` with
    ``presence``, or ``(image, box, has_box, label)`` with ``multiclass``. Items are
    ``(path, box | None)`` or ``(path, box | None, label_int)``. One dataset, all heads.

    A negative (no-object) item has ``box is None``: the image still trains the presence
    head toward *absent* (objectness target 0), but ``has_box == 0`` doubles as the mask
    that keeps it out of the box-regression *and* class-loss -- a no-object frame has no box
    to fit and no class to name. For a positive, ``has_box == 1`` is both those masks and the
    objectness target, and ``label`` is its class id. ``augment`` adds random
    horizontal/vertical flips (image and box transformed together; a box-less image just
    flips the pixels -- the class label is flip-invariant).
    """

    def __init__(
        self,
        items: list,
        size: int,
        *,
        presence: bool = False,
        augment: bool = False,
        multiclass: bool = False,
    ) -> None:
        self.items = items
        self.size = size
        self.presence = presence
        self.augment = augment
        self.multiclass = multiclass

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):  # noqa: ANN201
        item = self.items[index]
        path, box = item[0], item[1]
        label = item[2] if len(item) > 2 else 0
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
        if self.multiclass:
            # has_box masks both box and class loss; label is the class id for positives
            # (CrossEntropyLoss wants a long target), arbitrary-but-masked for negatives.
            return (
                x,
                y,
                torch.tensor(float(has_box), dtype=torch.float32),
                torch.tensor(int(label), dtype=torch.long),
            )
        if self.presence:
            # has_box is both the objectness target and the box-loss mask.
            return x, y, torch.tensor(float(has_box), dtype=torch.float32)
        return x, y


def export_onnx(
    model: nn.Module,
    out_path: str | Path,
    *,
    size: int,
    quantize: bool = True,
) -> Path:
    """Export to ONNX on CPU, then dynamic-quantize weights to uint8.

    A presence-head model makes ``forward`` return ``(box, presence)`` and a multi-class
    model appends ``logits``; every output is exported and named so the runtime reads each by
    name regardless of graph order. The output names are the whole contract -- ``presence``
    tells the runtime the model can say *absent*, ``logits`` that it can name a class.
    """
    out = Path(out_path)
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, size, size)
    fp32 = out.with_suffix(".fp32.onnx") if quantize else out

    # The model declares its own output contract (box / presence / logits, in graph order);
    # fall back to inferring from the forward shape for any plain module without it.
    output_names = getattr(model, "output_names", None)
    if output_names is None:
        with torch.no_grad():
            multi = isinstance(model(dummy), tuple)
        output_names = ["box", "presence"] if multi else ["box"]
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


def _val_metrics(  # noqa: ANN001
    model, val_items, size, dev, *, presence, multiclass=False
) -> tuple[float, float | None, float | None]:
    """Held-out (median center error px, presence accuracy, class accuracy) on ``val_items``.

    ``presence``/``class`` accuracies are ``None`` when the model has no such head. Class
    accuracy is computed over *positives only* (a negative has no class to get right).
    """
    model.eval()
    errs: list[float] = []
    correct = 0
    cls_correct = 0
    cls_total = 0
    with torch.no_grad():
        for item in val_items:
            box = item[1]
            raw = model(_load_chw(item[0], size).unsqueeze(0).to(dev))
            out: dict[str, Any] = (
                dict(zip(model.output_names, raw, strict=True))
                if isinstance(raw, tuple)
                else {"box": raw}
            )
            # Box error only where there is a box -- negatives have none.
            if box is not None:
                pb = out["box"][0].cpu().numpy()
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                gcx, gcy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
                errs.append(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5 * size)
            if presence:
                # objectness logit >= 0 <=> sigmoid >= 0.5 <=> predicted present.
                pred_present = float(out["presence"][0].item()) >= 0.0
                correct += int(pred_present == (box is not None))
            if multiclass and box is not None:
                pred_cls = int(out["logits"][0].argmax().item())
                cls_correct += int(pred_cls == int(item[2]))
                cls_total += 1
    acc = correct / len(val_items) if presence else None
    cls_acc = (cls_correct / cls_total) if (multiclass and cls_total) else None
    model.train()
    # No positive in the val split -> no box error to report (0 penalty; acc still leads).
    return (statistics.median(errs) if errs else 0.0), acc, cls_acc


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
    presence: bool = False,
    val_frac: float = 0.0,
    seed: int = 0,
    box_weight: float = 1.0,
    presence_weight: float = 1.0,
    class_weight: float = 1.0,
    augment: bool = False,
    weight_decay: float = 0.0,
    require_review: bool = True,
    negatives: Sequence[str | Path] | None = None,
    learn_gain: bool = True,
    on_epoch: Callable[[dict], None] | None = None,
) -> Path:
    """Train on ``<data_dir>/annotations.coco.json`` + images and export int8 ONNX.

    ``presence=True`` adds a single objectness head (YOLO-style) and trains box regression
    + binary presence jointly; ``box_weight``/``presence_weight`` balance the two (the box
    loss is far smaller, so weight it up to keep the presence gradient from drowning it).
    ``augment`` adds flip augmentation. With ``val_frac`` a split is held out and the
    **best-scoring** checkpoint (presence acc, then center error) is what gets exported --
    not whatever the last, possibly-overfit epoch produced.

    Multi-class is automatic: a dataset whose ``annotations.coco.json`` declares more than
    one category trains a classification head (one logit per class for the single detected
    object, exported as ``logits``), weighted by ``class_weight``. A one-category dataset
    takes the box-only path unchanged. The detector stays single-object -- the class head
    names *which* class the one object is, it does not find multiple objects.

    With ``require_review`` (default) only human-validated annotations train; an
    all-pending dataset is refused with guidance to triage it first. ``require_review=False``
    trains on everything regardless of review state.

    ``negatives`` is one or more directories of raw no-object images (e.g. machined parts
    with holes but no helicoil). They turn the presence head on and train it toward *absent*
    (objectness 0); box loss is masked out for them. This is what lets the detector return
    no box at all instead of emitting a spurious one. There is no class label and no
    "background" class -- absence is just the low tail of the one objectness score.

    ``on_epoch``, if given, is called once per epoch with a metrics dict (epoch, loss,
    val_err, presence_acc, class_acc, num_classes, baseline, best_err, is_best, gain,
    train_count, val_count) -- a live hook for callers like the Studio backend; it does not
    affect training.
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

    # Negatives only make sense with a presence head to point them at, so they enable it.
    presence = presence or bool(negatives)
    labeled, names = load_coco_labeled(coco, data_dir, require_review=require_review)
    # A dataset with >1 category trains a classification head; one category is the box-only
    # path (label always 0, no head, identical to before). Items stay (path, box, label).
    num_classes = len(names)
    multiclass = num_classes > 1
    items: list = [(p, b, lbl) for p, b, lbl in labeled]
    if multiclass:
        print(f"classes: {num_classes} -- {', '.join(names)}")
    if negatives:
        neg_paths = [p for d in negatives for p in load_negatives(d)]
        # Negatives carry a placeholder class (0); the box/class loss mask discards it.
        items = [*items, *[(p, None, 0) for p in neg_paths]]
        print(f"negatives: +{len(neg_paths)} no-object images (objectness target 0)")
    if not items:
        if require_review and a["pending"]:
            raise ValueError(
                f"no approved annotations under {data_dir}: {a['pending']} pending human "
                "review. Validate them via tensor-factory-label (push -> correct -> pull), "
                "or pass require_review=False / --allow-unreviewed to train on unvalidated "
                "labels deliberately."
            )
        raise ValueError(f"no annotations found under {data_dir}")

    train_items, val_items = _split(items, val_frac, seed)
    # Constant-predictor floor: median val center-error of "always emit the mean train box".
    # A model that can't beat this isn't localizing -- it's riding center-bias. Printed up
    # front and again as a verdict so a good-looking loss curve can't disguise a non-localizer.
    baseline_err = baseline_center_err(train_items, val_items, size)
    if baseline_err is not None:
        print(f"baseline: constant-predictor val err {baseline_err:.1f}px (the floor to beat)")
    # Seed torch so weight init is reproducible run-to-run -- the split is already seeded, so
    # with this the only variable between two runs is the hyperparameters (e.g. the gain).
    torch.manual_seed(seed)
    dev = resolve_device(device)
    loader = DataLoader(
        _BoxDataset(train_items, size, presence=presence, augment=augment, multiclass=multiclass),
        batch_size=batch,
        shuffle=True,
    )
    model = TinyDetector(
        width, presence=presence, learn_gain=learn_gain, num_classes=num_classes
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    box_loss = nn.SmoothL1Loss()
    presence_loss = nn.BCEWithLogitsLoss()
    class_loss = nn.CrossEntropyLoss()

    best_score = float("-inf")
    best_err: float | None = None
    best_state: dict | None = None
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for data in loader:
            optimizer.zero_grad()
            if presence:
                x, yb, m = (d.to(dev) for d in (data[0], data[1], data[2]))
                raw = model(x)
                out_t: dict[str, Any] = dict(zip(model.output_names, raw, strict=True))
                pb = out_t["box"]
                # Box loss only over box-bearing items; negatives (mask 0) contribute none.
                # The same mask m is the objectness target: 1 for positives, 0 for negatives.
                mb = m.bool()
                box_term = box_loss(pb[mb], yb[mb]) if bool(mb.any()) else pb.sum() * 0.0
                loss = box_weight * box_term + presence_weight * presence_loss(
                    out_t["presence"].reshape(-1), m
                )
                if multiclass:
                    # Class loss on positives only -- a negative has no class to name.
                    labels = data[3].to(dev)
                    if bool(mb.any()):
                        loss = loss + class_weight * class_loss(out_t["logits"][mb], labels[mb])
            else:
                x, yb = data[0].to(dev), data[1].to(dev)
                loss = box_weight * box_loss(model(x), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)

        err: float | None = None
        acc: float | None = None
        cls_acc: float | None = None
        is_best = False
        line = f"epoch {epoch + 1}/{epochs}  loss {total / len(train_items):.5f}"
        if val_items:
            err, acc, cls_acc = _val_metrics(
                model, val_items, size, dev, presence=presence, multiclass=multiclass
            )
            # Higher is better. ``err`` is the median center error in *pixels*; normalize by
            # image size so localization trades against accuracy on a comparable [0,1] scale.
            # The old ``err / 1000`` made a 15px miss worth 0.015 -- swamped by a single
            # accuracy step -- so selection shipped the best-classifying epoch with no regard
            # for box quality. Accuracy still leads (one val step >> a few px), but boxes now
            # matter ~50x more, and box-only runs (acc is None) still rank by lowest error.
            # Class accuracy joins on the same footing as presence when there's a class head.
            score = (acc or 0.0) + (cls_acc or 0.0) - err / size
            line += f"  | val err {err:.1f}px" + (f" acc {acc:.0%}" if acc is not None else "")
            if cls_acc is not None:
                line += f" cls {cls_acc:.0%}"
            if score > best_score:
                best_score = score
                best_err = err
                best_state = copy.deepcopy(model.state_dict())
                is_best = True
                line += "  *best"
        print(line)
        # Optional live hook (e.g. the Studio backend streaming metrics to a browser).
        # Default None keeps fit() behaviour byte-for-byte unchanged.
        if on_epoch is not None:
            on_epoch(
                {
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": total / len(train_items),
                    "val_err": err,
                    "presence_acc": acc,
                    "class_acc": cls_acc,
                    "num_classes": num_classes,
                    "baseline": baseline_err,
                    "best_err": best_err,
                    "is_best": is_best,
                    "gain": float(model.log_gain.detach().exp()),
                    "train_count": len(train_items),
                    "val_count": len(val_items),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        if presence:
            err, acc, cls_acc = _val_metrics(
                model, val_items, size, dev, presence=presence, multiclass=multiclass
            )
            cls = f"  class acc {cls_acc:.0%}" if cls_acc is not None else ""
            print(f"BEST checkpoint: val err {err:.1f}px  presence acc {acc:.0%}{cls}")
    # Verdict: does the model actually localize, or just match the do-nothing constant?
    if baseline_err is not None and best_err is not None:
        if best_err < baseline_err:
            verdict = f"{(1 - best_err / baseline_err) * 100:.0f}% better -- localizing"
        else:
            verdict = (
                f"NOT better ({best_err / baseline_err - 1:+.0%}) -- center-bias, not localizing"
            )
        print(
            f"localization: model {best_err:.1f}px vs constant-predictor "
            f"{baseline_err:.1f}px  ->  {verdict}"
        )
    print(f"soft-argmax gain: {float(model.log_gain.exp()):.2f} (1.0 = plain softmax)")
    return export_onnx(model, out_path, size=size)
