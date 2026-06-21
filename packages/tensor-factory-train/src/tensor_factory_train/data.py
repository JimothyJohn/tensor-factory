"""Read a COCO detection dataset into (image path, box) pairs -- torch-free."""

from __future__ import annotations

import json
from pathlib import Path

from tensor_factory.formats import from_coco_bbox
from tensor_factory.geometry import BBox


def load_coco_boxes(coco_json: str | Path, images_root: str | Path) -> list[tuple[Path, BBox]]:
    """Load one box per annotation as ``(absolute_image_path, BBox)``.

    The tiny detector is single-object, so the first/only annotation per image is the
    target; images without an annotation are skipped.
    """
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    root = Path(images_root)

    items: list[tuple[Path, BBox]] = []
    for ann in data["annotations"]:
        im = images[ann["image_id"]]
        box = from_coco_bbox(ann["bbox"], width=im["width"], height=im["height"])
        items.append((root / im["file_name"], box))
    return items


def load_coco_labeled(
    coco_json: str | Path, images_root: str | Path
) -> tuple[list[tuple[Path, BBox, int]], list[str]]:
    """Load ``(image_path, BBox, label)`` plus the ordered category names.

    ``label`` is a 0-indexed class id into the returned names (categories sorted by their
    COCO id), the form the classification head trains against. Used for the multi-class
    detector; :func:`load_coco_boxes` stays the box-only path.
    """
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data["images"]}
    root = Path(images_root)

    cats = sorted(data["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    label_of = {c["id"]: i for i, c in enumerate(cats)}

    items: list[tuple[Path, BBox, int]] = []
    for ann in data["annotations"]:
        im = images[ann["image_id"]]
        box = from_coco_bbox(ann["bbox"], width=im["width"], height=im["height"])
        items.append((root / im["file_name"], box, label_of[ann["category_id"]]))
    return items, names
