"""Build tensor-factory datasets from Oxford-IIIT Pet (head boxes, one per image).

Inputs are the two upstream artifacts, read in place and never executed or unpacked to disk:

- ``annotations.tar.gz`` from https://www.robots.ox.ac.uk/~vgg/data/pets/ (the head-box XMLs)
- the trainval images as the ``timm/oxford-iiit-pet`` train parquet on Hugging Face
  (same files, keyed by the original file stem in ``image_id``)

Output: ``<out>/train`` and ``<out>/test``, each ``annotations.coco.json`` + ``images/``
holding the original JPEG bytes untouched. The split is a hash of the file stem, so it is
stable. Dataset licence: CC BY-SA 4.0 (Parkhi, Vedaldi, Zisserman, Jawahar, 2012).

    uv run --with pyarrow python tools/pets_dataset.py annotations.tar.gz train.parquet OUT
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq  # ty: ignore[import-not-found]

_STEM = re.compile(r"[A-Za-z0-9_]+")
_TAGS = ("width", "height", "xmin", "ymin", "xmax", "ymax")


def head_boxes(tar_path: str) -> dict[str, tuple[int, int, int, int, int, int]]:
    """file stem -> (width, height, xmin, ymin, xmax, ymax) from annotations/xmls/*.xml."""
    boxes = {}
    with tarfile.open(tar_path) as tar:
        for member in tar:
            name = Path(member.name)
            if not member.isfile() or name.suffix != ".xml" or name.parent.name != "xmls":
                continue
            text = tar.extractfile(member).read().decode("utf-8", "replace")  # ty: ignore[possibly-missing-attribute]
            # The files are tiny, flat VOC records; a regex per tag avoids an XML parser on
            # downloaded data. Records with more than one object are skipped.
            if text.count("<object>") != 1:
                continue
            vals = [re.search(rf"<{t}>\s*(\d+)\s*</{t}>", text) for t in _TAGS]
            if all(vals) and _STEM.fullmatch(name.stem):
                boxes[name.stem] = tuple(int(v.group(1)) for v in vals)  # ty: ignore[possibly-missing-attribute]
    return boxes


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    tar_path, parquet_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    boxes = head_boxes(tar_path)
    sets = {name: {"images": [], "annotations": []} for name in ("train", "test")}
    for name in sets:
        (out / name / "images").mkdir(parents=True, exist_ok=True)

    table = pq.ParquetFile(parquet_path)
    for batch in table.iter_batches(batch_size=64, columns=["image", "image_id"]):
        for image, stem in zip(
            batch.column("image").to_pylist(), batch.column("image_id").to_pylist(), strict=True
        ):
            data = image["bytes"]
            if stem not in boxes or not _STEM.fullmatch(stem) or data[:2] != b"\xff\xd8":
                continue
            w, h, x1, y1, x2, y2 = boxes[stem]
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                continue
            split = (
                "test" if int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) % 9 == 0 else "train"
            )
            (out / split / "images" / f"{stem}.jpg").write_bytes(data)
            ds = sets[split]
            n = len(ds["images"]) + 1
            ds["images"].append(
                {
                    "id": n,
                    "file_name": f"images/{stem}.jpg",
                    "width": w,
                    "height": h,
                    "review": "approved",
                }
            )
            ds["annotations"].append(
                {
                    "id": n,
                    "image_id": n,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "review": "approved",  # the dataset's own human annotation
                    "source": "oxford-iiit-pet",
                }
            )
    for name, ds in sets.items():
        ds["categories"] = [{"id": 1, "name": "pet_head"}]
        (out / name / "annotations.coco.json").write_text(json.dumps(ds))
        print(f"{name}: {len(ds['images'])} images")


if __name__ == "__main__":
    main()
