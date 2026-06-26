#!/usr/bin/env python3
"""Convert Oxford-IIIT Pet head-bbox annotations -> tensor-factory COCO format.

A *separate validation experiment*: the Pet set is a large, clean, single-object
localization benchmark used to prove the soft-argmax TinyDetector can localize a
small, off-center feature given enough data. If it nails px-error here, the helicoil
ceiling is a data problem, not an architecture problem.

Source: PASCAL VOC XMLs under data/annotations/xmls/ (one head box per image, absolute
pixels). Target: data/annotations.coco.json with bbox=[x, y, w, h] absolute pixels,
review="approved" (these are curated ground-truth boxes), file_name relative to data/.

Usage:
    uv run python examples/pets-validation/convert.py [--limit N]

Stdlib only — no deps, no image copying (COCO file_name points at the extracted images/).
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path(__file__).parent / "data"
XMLS = DATA / "annotations" / "xmls"
IMAGES = DATA / "images"
OUT = DATA / "annotations.coco.json"

CATEGORY = "pet"  # single class — the detector is class-agnostic (one box or none)


def _text(node: ET.Element, tag: str) -> str:
    el = node.find(tag)
    if el is None or el.text is None:
        raise ValueError(f"missing <{tag}>")
    return el.text


def parse_xml(path: Path) -> tuple[str, int, int, list[tuple[int, int, int, int]]]:
    """Return (filename, width, height, [(xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(path).getroot()
    filename = _text(root, "filename")
    size = root.find("size")
    if size is None:
        raise ValueError("missing <size>")
    width, height = int(_text(size, "width")), int(_text(size, "height"))
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        boxes.append(
            (
                int(_text(b, "xmin")),
                int(_text(b, "ymin")),
                int(_text(b, "xmax")),
                int(_text(b, "ymax")),
            )
        )
    return filename, width, height, boxes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="cap images (smoke test)")
    args = ap.parse_args()

    xmls = sorted(XMLS.glob("*.xml"))
    if args.limit:
        xmls = xmls[: args.limit]

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    skipped_no_image = 0
    skipped_bad_box = 0
    multi_box = 0

    for image_id, xml in enumerate(xmls, start=1):
        try:
            filename, width, height, boxes = parse_xml(xml)
        except ValueError:
            skipped_bad_box += 1
            continue
        if not (IMAGES / filename).exists():
            skipped_no_image += 1
            continue
        if len(boxes) > 1:
            multi_box += 1  # Pet head set is single-object; flag if not

        images.append(
            {
                "id": image_id,
                "file_name": f"images/{filename}",
                "width": width,
                "height": height,
                "review": "approved",
            }
        )
        for xmin, ymin, xmax, ymax in boxes:
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0:
                skipped_bad_box += 1
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "review": "approved",
                    "source": "human",
                }
            )
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": CATEGORY}],
    }
    OUT.write_text(json.dumps(coco))

    print(f"wrote {OUT}")
    print(f"  images:      {len(images)}")
    print(f"  annotations: {len(annotations)}")
    print(f"  skipped (no image file): {skipped_no_image}")
    print(f"  skipped (bad/empty box): {skipped_bad_box}")
    print(f"  images with >1 box:      {multi_box}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
