"""Convert the canonical :class:`~tensor_factory.geometry.BBox` to and from the common
annotation formats: COCO, YOLO, and Pascal VOC.

Canonical form is normalized ``xyxy``; every exporter here has a matching importer
so a box survives a round-trip through any supported format (within the format's own
precision -- COCO/YOLO keep float coords and are lossless, VOC rounds to integer
pixels). Dataset-level file IO is layered on top of these in tensor_factory-data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .geometry import BBox


def to_coco_bbox(box: BBox, *, width: int, height: int) -> list[float]:
    """COCO ``[x, y, w, h]`` in absolute pixels, top-left origin."""
    return [
        box.x1 * width,
        box.y1 * height,
        box.width * width,
        box.height * height,
    ]


def from_coco_bbox(bbox: Sequence[float], *, width: int, height: int) -> BBox:
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return BBox.from_pixels(x, y, x + w, y + h, width=width, height=height)


def to_yolo(box: BBox) -> tuple[float, float, float, float]:
    """YOLO ``(cx, cy, w, h)`` -- all normalized to ``[0, 1]``."""
    return box.to_cxcywh()


def from_yolo(cx: float, cy: float, w: float, h: float) -> BBox:
    return BBox.from_cxcywh(cx, cy, w, h)


def to_voc_box(box: BBox, *, width: int, height: int) -> dict[str, int]:
    """Pascal VOC ``xmin/ymin/xmax/ymax`` in absolute integer pixels."""
    x1, y1, x2, y2 = box.to_pixels(width=width, height=height)
    return {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}


def from_voc_box(voc: Mapping[str, float], *, width: int, height: int) -> BBox:
    return BBox.from_pixels(
        voc["xmin"], voc["ymin"], voc["xmax"], voc["ymax"], width=width, height=height
    )
