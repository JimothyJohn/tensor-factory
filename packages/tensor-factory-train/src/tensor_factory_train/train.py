"""Train the tiny detector on a COCO dataset and export an int8 ONNX model.

Device resolves cuda -> mps -> cpu for the training loop; export always runs on CPU
(ONNX export from an MPS graph is unreliable), then onnxruntime dynamic quantization
produces the uint8-weight model the edge runtime loads.
"""

from __future__ import annotations

import copy
import json
import os
import random
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch  # ty: ignore[import-not-found]
from PIL import Image
from torch import nn  # ty: ignore[import-not-found]
from torch.utils.data import DataLoader, Dataset  # ty: ignore[import-not-found]

from tensor_factory.review import review_summary

from .data import baseline_center_err, load_negatives, load_sources, source_names
from .model import TinyDetector

_DEVICES = ("cuda", "mps", "cpu")


def resolve_device(prefer: str | None = None) -> str:
    """The training device: ``prefer``, else ``$TENSOR_FACTORY_DEVICE``, else the best of
    cuda -> mps -> cpu.

    The environment variable pins a machine without touching any call site -- a CI runner
    whose virtualized GPU reports itself available but cannot allocate, or a shared box
    where the GPU belongs to someone else. An explicit ``prefer`` still wins over it.
    """
    env = os.environ.get("TENSOR_FACTORY_DEVICE", "").strip().lower()
    if not prefer and env:
        if env not in _DEVICES:
            raise ValueError(f"TENSOR_FACTORY_DEVICE={env!r}: expected one of {_DEVICES}")
        return env
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


RESIZE_MODES = ("bilinear", "nearest", "mixed")


def _resample(resize: str, *, train: bool) -> Image.Resampling:
    """The PIL filter for a ``resize`` mode.

    ``mixed`` picks bilinear or nearest per training sample, so one model tolerates both a
    desktop runtime (bilinear) and a runtime that can only subsample (ESP-DL on a
    microcontroller); it validates on bilinear so runs stay deterministic and comparable.
    """
    if resize not in RESIZE_MODES:
        raise ValueError(f"resize must be one of {RESIZE_MODES}, got {resize!r}")
    if resize == "nearest" or (resize == "mixed" and train and random.random() < 0.5):
        return Image.Resampling.NEAREST
    return Image.Resampling.BILINEAR


class ImageCache:
    """Decoded, resized training images, kept between epochs and between ``fit`` calls.

    Decoding and resizing every file again each epoch is most of a training run's time on
    small models. A cache entry is the ``uint8`` pixels for one (file, size, filter); a file
    that is rewritten (a frame re-labeled in the Studio) is decoded again because its
    modification time changed. Nothing is evicted: once ``max_bytes`` is reached, further
    images are simply decoded each time, as without a cache.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3) -> None:
        self.max_bytes = max_bytes
        self.bytes = 0
        self._items: dict[tuple[str, int, int], tuple[int, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
        self.bytes = 0

    def pixels(self, path: Path, size: int, resample: Image.Resampling) -> np.ndarray:
        """``uint8`` HWC pixels of ``path`` at ``size`` x ``size``."""
        key = (str(path), size, int(resample))
        stamp = path.stat().st_mtime_ns
        hit = self._items.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        arr = _pixels(path, size, resample)
        if hit is not None:
            self.bytes -= hit[1].nbytes
            del self._items[key]
        if self.bytes + arr.nbytes <= self.max_bytes:
            self._items[key] = (stamp, arr)
            self.bytes += arr.nbytes
        return arr


def _pixels(path: Path, size: int, resample: Image.Resampling) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), resample), dtype=np.uint8)


def _load_chw(  # noqa: ANN202
    path: Path,
    size: int,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
    cache: ImageCache | None = None,
):
    px = cache.pixels(path, size, resample) if cache is not None else _pixels(path, size, resample)
    arr = px.astype(np.float32) / 255.0
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
    """Yields ``(image, box, has_box, label)`` for every item, whatever heads are training.
    Items are ``(path, box | None)``, optionally followed by a class label and a source
    index (ignored here). One dataset, all heads, one tuple shape -- callers ignore the
    elements their mode doesn't use.

    A negative (no-object) item has ``box is None``: the image still trains the presence
    head toward *absent* (objectness target 0), but ``has_box == 0`` doubles as the mask
    that keeps it out of the box-regression *and* class-loss -- a no-object frame has no box
    to fit and no class to name. For a positive, ``has_box == 1`` is both those masks and the
    objectness target, and ``label`` is its class id (0 when the item carries none).
    ``augment`` adds random horizontal/vertical flips (image and box transformed together; a
    box-less image just flips the pixels -- the class label is flip-invariant).
    """

    def __init__(
        self,
        items: list,
        size: int,
        *,
        presence: bool = False,
        augment: bool = False,
        multiclass: bool = False,
        resize: str = "bilinear",
        cache: ImageCache | None = None,
    ) -> None:
        _resample(resize, train=True)  # reject a bad mode up front, not on the first batch
        self.resize = resize
        self.cache = cache
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
        x = _load_chw(path, self.size, _resample(self.resize, train=True), self.cache)
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
        # One shape for every mode: has_box is the objectness target and the box/class-loss
        # mask; label is the class id for positives (CrossEntropyLoss wants a long target),
        # arbitrary-but-masked for negatives.
        return (
            x,
            y,
            torch.tensor(float(has_box), dtype=torch.float32),
            torch.tensor(int(label), dtype=torch.long),
        )


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


def _split_by_source(items: list, val_frac: float, seed: int) -> tuple[list, list]:
    """Hold out ``val_frac`` of *each* source, so every source is scored on its own data.

    One global shuffle lets a small source land entirely in train (or entirely in val) and
    its error then never shows up. Items carry their source index last. A source with a
    single item stays in train: holding it out would leave nothing to learn it from.
    """
    groups: dict[int, list] = {}
    for it in items:
        groups.setdefault(it[3], []).append(it)
    train: list = []
    val: list = []
    for src in sorted(groups):
        group = groups[src]
        if len(group) < 2:
            train.extend(group)
            continue
        t, v = _split(group, val_frac, seed)
        train.extend(t)
        val.extend(v)
    return train, val


def _summarize(
    records: Sequence[tuple[float | None, bool | None, bool | None]],
) -> tuple[float, float | None, float | None]:
    """(median center error px, presence accuracy, class accuracy) over per-item records.

    A record is ``(center_err | None, presence_correct | None, class_correct | None)``;
    ``None`` means "not applicable to this item" (a negative has no box and no class; a
    model without the head has no verdict). No box-bearing record -> error 0.0.
    """
    errs = [r[0] for r in records if r[0] is not None]
    pres = [r[1] for r in records if r[1] is not None]
    cls = [r[2] for r in records if r[2] is not None]
    return (
        statistics.median(errs) if errs else 0.0,
        (sum(pres) / len(pres)) if pres else None,
        (sum(cls) / len(cls)) if cls else None,
    )


def _val_records(  # noqa: ANN001
    model, val_items, size, dev, *, presence, multiclass=False, resize="bilinear", cache=None
) -> list[tuple[float | None, bool | None, bool | None]]:
    """One ``(center_err px, presence_correct, class_correct)`` record per held-out item,
    in ``val_items`` order -- the raw material for overall and per-source metrics alike."""
    model.eval()
    records: list[tuple[float | None, bool | None, bool | None]] = []
    with torch.no_grad():
        for item in val_items:
            box = item[1]
            x = _load_chw(item[0], size, _resample(resize, train=False), cache)
            raw = model(x.unsqueeze(0).to(dev))
            out: dict[str, Any] = (
                dict(zip(model.output_names, raw, strict=True))
                if isinstance(raw, tuple)
                else {"box": raw}
            )
            err: float | None = None
            pres_ok: bool | None = None
            cls_ok: bool | None = None
            # Box error only where there is a box -- negatives have none.
            if box is not None:
                pb = out["box"][0].cpu().numpy()
                pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                gcx, gcy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
                err = float(((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5 * size)
            if presence:
                # objectness logit >= 0 <=> sigmoid >= 0.5 <=> predicted present.
                pred_present = float(out["presence"][0].item()) >= 0.0
                pres_ok = pred_present == (box is not None)
            if multiclass and box is not None:
                cls_ok = int(out["logits"][0].argmax().item()) == int(item[2])
            records.append((err, pres_ok, cls_ok))
    model.train()
    return records


def _val_metrics(  # noqa: ANN001
    model, val_items, size, dev, *, presence, multiclass=False, resize="bilinear"
) -> tuple[float, float | None, float | None]:
    """Held-out (median center error px, presence accuracy, class accuracy) on ``val_items``.

    ``presence``/``class`` accuracies are ``None`` when the model has no such head. Class
    accuracy is computed over *positives only* (a negative has no class to get right). With
    no positive in the split there is no box error to report (0 penalty; accuracy leads).
    """
    return _summarize(
        _val_records(
            model, val_items, size, dev, presence=presence, multiclass=multiclass, resize=resize
        )
    )


def _per_source(
    names: Sequence[str],
    train_items: Sequence[tuple],
    val_items: Sequence[tuple],
    records: Sequence[tuple[float | None, bool | None, bool | None]],
) -> list[dict[str, Any]]:
    """Per-source breakdown of one validation pass: counts plus that source's own metrics.

    ``val_err`` is ``None`` for a source with no held-out box (a negatives pool, or a source
    too small to hold anything out) -- not 0, which would read as a perfect score.
    """
    out: list[dict[str, Any]] = []
    for src, name in enumerate(names):
        recs = [r for it, r in zip(val_items, records, strict=True) if it[3] == src]
        err, acc, cls_acc = _summarize(recs)
        out.append(
            {
                "name": name,
                "train": sum(1 for it in train_items if it[3] == src),
                "val": len(recs),
                "val_err": err if any(r[0] is not None for r in recs) else None,
                "presence_acc": acc,
                "class_acc": cls_acc,
            }
        )
    return out


def _source_line(s: dict[str, Any]) -> str:
    parts = [f"  {s['name']}: {s['train']} train / {s['val']} val"]
    if s["val_err"] is not None:
        parts.append(f"err {s['val_err']:.1f}px")
    if s["presence_acc"] is not None:
        parts.append(f"presence {s['presence_acc']:.0%}")
    if s["class_acc"] is not None:
        parts.append(f"class {s['class_acc']:.0%}")
    return "  ".join(parts)


def fit(
    data_dir: str | Path | Sequence[str | Path],
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
    context: int = 0,
    resize: str = "bilinear",
    on_epoch: Callable[[dict], None] | None = None,
    cache: ImageCache | None = None,
) -> Path:
    """Train on ``<data_dir>/annotations.coco.json`` + images and export int8 ONNX.

    ``data_dir`` may be a list of dataset dirs: they train as one set, read in place (no
    merge-and-copy step). Classes are merged by name, ``val_frac`` is held out of *each*
    source so none goes unscored, and every source is reported separately -- at the best
    checkpoint on stdout and per epoch in ``on_epoch``'s ``sources`` -- so a source that
    drags the model down is visible instead of averaged away. Negatives dirs count as
    sources too.

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
    train_count, val_count, val_boxes, sources) -- a live hook for callers like the Studio
    backend; it does not affect training. ``val_boxes`` is how many held-out items carry a
    box (the sample the error is measured on). ``sources`` is one dict per source (name,
    train, val, val_err, presence_acc, class_acc), data dirs first, then negatives dirs.

    ``cache``, an :class:`ImageCache`, keeps decoded images between epochs; pass the same
    one to later calls and only new or rewritten files are decoded again. The pixels are
    the same either way, so results do not change.
    """
    data_dirs = (
        [Path(data_dir)] if isinstance(data_dir, (str, Path)) else [Path(d) for d in data_dir]
    )
    if not data_dirs:
        raise ValueError("no dataset dir given")
    neg_dirs = [Path(d) for d in negatives or ()]
    src_names = source_names([*data_dirs, *neg_dirs])
    where = ", ".join(str(d) for d in data_dirs)

    pending = 0
    for name, d in zip(src_names[: len(data_dirs)], data_dirs, strict=True):
        coco = d / "annotations.coco.json"
        a = review_summary(json.loads(coco.read_text(encoding="utf-8")))["annotations"]
        pending += a["pending"]
        print(
            f"triage [{name}]: {a['total']} annotations -- {a['approved']} approved, "
            f"{a['pending']} pending, {a['rejected']} rejected"
            + ("" if require_review else "  (review gate OFF: training on all)")
        )

    # Negatives only make sense with a presence head to point them at, so they enable it.
    presence = presence or bool(neg_dirs)
    # Items are (path, box | None, label, source). More than one class name across the
    # sources trains a classification head; one is the box-only path (label always 0).
    labeled, names = load_sources(data_dirs, require_review=require_review)
    num_classes = len(names)
    multiclass = num_classes > 1
    items: list = list(labeled)
    if multiclass:
        print(f"classes: {num_classes} -- {', '.join(names)}")
    for offset, d in enumerate(neg_dirs):
        neg_paths = load_negatives(d)
        # Negatives carry a placeholder class (0); the box/class loss mask discards it.
        items.extend((p, None, 0, len(data_dirs) + offset) for p in neg_paths)
        print(
            f"negatives [{src_names[len(data_dirs) + offset]}]: +{len(neg_paths)} no-object "
            "images (objectness target 0)"
        )
    if not items:
        if require_review and pending:
            raise ValueError(
                f"no approved annotations under {where}: {pending} pending human "
                "review. Validate them via tensor-factory-label (push -> correct -> pull), "
                "or pass require_review=False / --allow-unreviewed to train on unvalidated "
                "labels deliberately."
            )
        raise ValueError(f"no annotations found under {where}")

    # One dataset keeps the original single shuffle (same split as before for a given
    # seed); several are each held out separately so every source gets its own score.
    if len(data_dirs) > 1:
        train_items, val_items = _split_by_source(items, val_frac, seed)
    else:
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
        _BoxDataset(
            train_items,
            size,
            presence=presence,
            augment=augment,
            multiclass=multiclass,
            resize=resize,
            cache=cache,
        ),
        batch_size=batch,
        shuffle=True,
    )
    model = TinyDetector(
        width,
        presence=presence,
        learn_gain=learn_gain,
        num_classes=num_classes,
        context=context,
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
            # _BoxDataset always yields (image, box, has_box, label); the heads that are off
            # ignore their element.
            x, yb, m, labels = (d.to(dev) for d in data)
            if presence:
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
                # Class loss on positives only -- a negative has no class to name.
                if multiclass and bool(mb.any()):
                    loss = loss + class_weight * class_loss(out_t["logits"][mb], labels[mb])
            else:
                loss = box_weight * box_loss(model(x), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)

        err: float | None = None
        acc: float | None = None
        cls_acc: float | None = None
        is_best = False
        records: list[tuple[float | None, bool | None, bool | None]] = []
        line = f"epoch {epoch + 1}/{epochs}  loss {total / len(train_items):.5f}"
        if val_items:
            records = _val_records(
                model,
                val_items,
                size,
                dev,
                presence=presence,
                multiclass=multiclass,
                resize=resize,
                cache=cache,
            )
            err, acc, cls_acc = _summarize(records)
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
                    "val_boxes": sum(1 for item in val_items if item[1] is not None),
                    "sources": _per_source(src_names, train_items, val_items, records),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        records = _val_records(
            model,
            val_items,
            size,
            dev,
            presence=presence,
            multiclass=multiclass,
            resize=resize,
            cache=cache,
        )
        err, acc, cls_acc = _summarize(records)
        if presence:
            cls = f"  class acc {cls_acc:.0%}" if cls_acc is not None else ""
            print(f"BEST checkpoint: val err {err:.1f}px  presence acc {acc:.0%}{cls}")
        if len(src_names) > 1:
            print("by source (best checkpoint):")
            for source in _per_source(src_names, train_items, val_items, records):
                print(_source_line(source))
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
