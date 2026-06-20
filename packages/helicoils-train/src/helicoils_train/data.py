"""Read a COCO detection dataset into (image path, box) pairs -- torch-free."""

from __future__ import annotations

import json
from pathlib import Path

from helicoils.formats import from_coco_bbox
from helicoils.geometry import BBox


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
