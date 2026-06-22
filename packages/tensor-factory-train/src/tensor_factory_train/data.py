"""Read a COCO detection dataset into (image path, box) pairs -- torch-free.

By default both loaders enforce the review gate: only annotations a human has validated
(``review == approved``; see :mod:`tensor_factory.review`) are returned. AI-labeled
candidates that have not been triaged never reach the training set. Pass
``require_review=False`` to deliberately load everything (e.g. a mock-only dataset where
the boxes are exact synthetic ground truth already marked approved -- or to inspect raw,
unvalidated labels).
"""

from __future__ import annotations

import json
from pathlib import Path

from tensor_factory.formats import from_coco_bbox
from tensor_factory.geometry import BBox
from tensor_factory.review import is_trainable


def load_coco_boxes(
    coco_json: str | Path,
    images_root: str | Path,
    *,
    require_review: bool = True,
) -> list[tuple[Path, BBox]]:
    """Load one box per annotation as ``(absolute_image_path, BBox)``.

    The tiny detector is single-object, so the first/only annotation per image is the
    target; images without an annotation are skipped. With ``require_review`` (default),
    annotations that are not human-validated are skipped too.
    """
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    root = Path(images_root)

    items: list[tuple[Path, BBox]] = []
    for ann in data["annotations"]:
        if require_review and not is_trainable(ann.get("review")):
            continue
        im = images[ann["image_id"]]
        box = from_coco_bbox(ann["bbox"], width=im["width"], height=im["height"])
        items.append((root / im["file_name"], box))
    return items


def load_coco_labeled(
    coco_json: str | Path,
    images_root: str | Path,
    *,
    require_review: bool = True,
) -> tuple[list[tuple[Path, BBox, int]], list[str]]:
    """Load ``(image_path, BBox, label)`` plus the ordered category names.

    ``label`` is a 0-indexed class id into the returned names (categories sorted by their
    COCO id), the form the classification head trains against. Used for the multi-class
    detector; :func:`load_coco_boxes` stays the box-only path. With ``require_review``
    (default), only human-validated annotations are returned.
    """
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    root = Path(images_root)

    cats = sorted(data["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    label_of = {c["id"]: i for i, c in enumerate(cats)}

    items: list[tuple[Path, BBox, int]] = []
    for ann in data["annotations"]:
        if require_review and not is_trainable(ann.get("review")):
            continue
        im = images[ann["image_id"]]
        box = from_coco_bbox(ann["bbox"], width=im["width"], height=im["height"])
        items.append((root / im["file_name"], box, label_of[ann["category_id"]]))
    return items, names


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_negatives(images_dir: str | Path) -> list[Path]:
    """List raw negative (no-object) image paths under ``images_dir``.

    Negatives carry no annotation or box; they supply the *background* class that teaches
    the detector to report *absent* instead of emitting a spurious box. Looks inside an
    ``images/`` subdirectory when present (the pool layout ``<dir>/images/*.png``),
    otherwise scans ``images_dir`` directly. There is no review gate -- a negative is
    trainable by construction (it is, definitionally, a known absence).
    """
    d = Path(images_dir)
    root = d / "images" if (d / "images").is_dir() else d
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
