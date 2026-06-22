"""Convert between our canonical COCO and Label Studio tasks/annotations.

``coco_to_tasks`` turns a COCO dataset (+ GroundingDINO scores) into Label Studio import
tasks with *predictions*, so a human starts from the auto-label candidates and corrects
them. ``ls_export_to_coco`` does the inverse on a Label Studio JSON export -- corrected
annotations back to COCO for training. That round-trip is what closes the labeling loop.

Label Studio boxes are percentages of image size; the canonical box is normalized
``[0, 1]`` -- the factor of 100 is the only difference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from tensor_factory import review
from tensor_factory.formats import from_coco_bbox, to_coco_bbox
from tensor_factory.geometry import BBox

FROM_NAME = "label"
TO_NAME = "image"


def http_image_url(base_url: str) -> Callable[[str], str]:
    """URL factory for images served over HTTP (e.g. a static file server)."""
    base = base_url.rstrip("/")
    return lambda file_name: f"{base}/{file_name.lstrip('/')}"


def local_storage_url(prefix: str = "/data/local-files/?d=") -> Callable[[str], str]:
    """URL factory for Label Studio Local Storage references."""
    return lambda file_name: f"{prefix}{file_name}"


def _file_name_from_ref(ref: str) -> str:
    """Recover the dataset-relative file name from a Label Studio image reference.

    The push wraps each file name in an image URL -- ``http_image_url`` gives
    ``http://host/images/foo.png``; ``local_storage_url`` gives ``/data/local-files/?d=foo``.
    The pull must invert that so COCO ``file_name`` matches the on-disk layout the
    trainer loads (``images/foo.png``), rather than storing the raw URL.
    """
    if not ref:
        return ref
    parsed = urlparse(ref)
    # Label Studio local storage: the file name is carried in the ?d= query param.
    if parsed.query:
        d = parse_qs(parsed.query).get("d")
        if d:
            return unquote(d[0])
    # http(s) image server: keep the URL path, drop the leading slash.
    if parsed.scheme:
        return unquote(parsed.path).lstrip("/")
    # Already a plain relative path.
    return ref


def _rect_result(box: BBox, label: str, width: int, height: int) -> dict[str, Any]:
    return {
        "type": "rectanglelabels",
        "from_name": FROM_NAME,
        "to_name": TO_NAME,
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "x": box.x1 * 100.0,
            "y": box.y1 * 100.0,
            "width": box.width * 100.0,
            "height": box.height * 100.0,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


def coco_to_tasks(coco: dict[str, Any], image_url: Callable[[str], str]) -> list[dict[str, Any]]:
    """Build Label Studio import tasks (with predictions) from a COCO dataset."""
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)

    tasks: list[dict[str, Any]] = []
    for image in coco["images"]:
        width, height = image["width"], image["height"]
        anns = by_image.get(image["id"], [])
        results = [
            _rect_result(
                from_coco_bbox(a["bbox"], width=width, height=height),
                categories.get(a["category_id"], "helicoil"),
                width,
                height,
            )
            for a in anns
        ]
        task: dict[str, Any] = {"data": {"image": image_url(image["file_name"])}}
        if results:
            scores = [a["score"] for a in anns if a.get("score") is not None]
            task["predictions"] = [
                {
                    "model_version": "groundingdino",
                    "result": results,
                    "score": sum(scores) / len(scores) if scores else 0.0,
                }
            ]
        tasks.append(task)
    return tasks


def _box_from_value(value: dict[str, Any]) -> BBox:
    x = value["x"] / 100.0
    y = value["y"] / 100.0
    w = value["width"] / 100.0
    h = value["height"] / 100.0
    return BBox.clamped(x, y, x + w, y + h)


def ls_export_to_coco(
    export: Sequence[dict[str, Any]], *, image_field: str = "image"
) -> dict[str, Any]:
    """Convert a Label Studio JSON export back to a COCO detection dict.

    Tasks with no (uncancelled) rectangle annotations are skipped. Image size is read
    from each result's ``original_width``/``original_height``. Everything that survives the
    round-trip was looked at by a human, so each annotation and image is stamped
    ``review=APPROVED``, ``source=HUMAN`` -- this is the step that makes a label trainable.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    cat_id: dict[str, int] = {}
    ann_id = 1

    for image_id, task in enumerate(export, start=1):
        results: list[dict[str, Any]] = []
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            results.extend(ann.get("result", []))
        rects = [r for r in results if r.get("type") == "rectanglelabels"]
        if not rects:
            continue

        width = rects[0]["original_width"]
        height = rects[0]["original_height"]
        file_name = _file_name_from_ref((task.get("data") or {}).get(image_field, ""))
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "review": review.APPROVED,
            }
        )
        for r in rects:
            box = _box_from_value(r["value"])
            label = (r["value"].get("rectanglelabels") or ["helicoil"])[0]
            cid = cat_id.setdefault(label, len(cat_id) + 1)
            x, y, w, h = to_coco_bbox(box, width=width, height=height)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cid,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "review": review.APPROVED,
                    "source": review.HUMAN,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in cat_id.items()],
    }
