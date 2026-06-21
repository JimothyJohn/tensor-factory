"""Export detections to COCO (canonical) and Label Studio pre-annotations.

COCO is the interchange format the rest of the toolchain reads/writes; YOLO and VOC
exports layer on top of tensor_factory.formats in tensor_factory-data. Label Studio predictions
let a human correct the auto-labels in the same UI used for real microscope images.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from tensor_factory import review
from tensor_factory.formats import to_coco_bbox

from .autolabel import Detection

# (file_name, width, height, detections)
Record = tuple[str, int, int, list[Detection]]


def build_coco(records: Sequence[Record], categories: Sequence[str]) -> dict:
    """Build a COCO detection dict from records and an ordered category list."""
    cat_id = {name: i + 1 for i, name in enumerate(categories)}
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    for image_id, (file_name, width, height, dets) in enumerate(records, start=1):
        # An image is APPROVED only when it has detections and all are approved; anything
        # else (auto-labeled, or empty and so possibly a missed feature) needs triage.
        img_review = (
            review.APPROVED
            if dets and all(d.review == review.APPROVED for d in dets)
            else review.PENDING
        )
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "review": img_review,
            }
        )
        for det in dets:
            x, y, w, h = to_coco_bbox(det.box, width=width, height=height)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id.get(det.label, 1),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": det.score,
                    "review": det.review,
                    "source": det.source,
                }
            )
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in cat_id.items()],
    }


def build_label_studio_predictions(records: Sequence[Record]) -> list[dict]:
    """Build Label Studio import tasks with detections as model pre-annotations.

    Coordinates are percentages of image size, per the RectangleLabels spec.
    """
    tasks: list[dict] = []
    for file_name, width, height, dets in records:
        results = []
        for det in dets:
            b = det.box
            results.append(
                {
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "value": {
                        "x": b.x1 * 100.0,
                        "y": b.y1 * 100.0,
                        "width": b.width * 100.0,
                        "height": b.height * 100.0,
                        "rotation": 0,
                        "rectanglelabels": [det.label],
                    },
                    "score": det.score,
                }
            )
        mean_score = sum(d.score for d in dets) / len(dets) if dets else 0.0
        tasks.append(
            {
                "data": {"image": file_name},
                "predictions": [
                    {
                        "model_version": "groundingdino",
                        "result": results,
                        "score": mean_score,
                    }
                ],
            }
        )
    return tasks


def write_json(path: str | Path, obj: object) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
