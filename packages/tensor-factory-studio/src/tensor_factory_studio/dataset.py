"""On-disk dataset the browser pushes labels into and the trainer reads from.

Layout matches what ``tensor_factory_train.fit`` expects with no conversion step:

    <root>/annotations.coco.json     positives, review=approved, source=human
    <root>/images/frame_NNNNN.png
    <root>/negatives/images/frame_NNNNN.png   empty frames (the presence head's 0s)

Frames are keyed by the browser's frame id, so re-labeling a frame overwrites in place
(and flipping present<->empty moves the image between images/ and negatives/).
"""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

from PIL import Image


class Dataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.neg_dir = self.root / "negatives" / "images"
        self.coco_path = self.root / "annotations.coco.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.neg_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # fid -> {"present": bool, "box": [x1,y1,x2,y2] | None, "w": int, "h": int}
        self.samples: dict[int, dict] = {}
        self.order: list[int] = []  # upsert order, for cheap regression attribution
        self._load()

    def _name(self, fid: int) -> str:
        return f"frame_{fid:05d}.png"

    def _load(self) -> None:
        if self.coco_path.exists():
            coco = json.loads(self.coco_path.read_text(encoding="utf-8"))
            ann = {a["image_id"]: a for a in coco.get("annotations", [])}
            for im in coco.get("images", []):
                a = ann.get(im["id"])
                if not a:
                    continue
                w, h = im["width"], im["height"]
                x, y, bw, bh = a["bbox"]
                self.samples[im["id"]] = {
                    "present": True,
                    "box": [x / w, y / h, (x + bw) / w, (y + bh) / h],
                    "w": w,
                    "h": h,
                }
        for p in sorted(self.neg_dir.glob("frame_*.png")):
            fid = int(p.stem.split("_")[1])
            self.samples.setdefault(fid, {"present": False, "box": None, "w": 0, "h": 0})
        self.order = sorted(self.samples)

    def upsert(self, fid: int, present: bool, box: list[float] | None, png: bytes) -> None:
        img = Image.open(io.BytesIO(png)).convert("RGB")
        w, h = img.size
        with self._lock:
            # a frame lives in exactly one of images/ or negatives/ -- clear both first
            (self.images_dir / self._name(fid)).unlink(missing_ok=True)
            (self.neg_dir / self._name(fid)).unlink(missing_ok=True)
            if present and box:
                img.save(self.images_dir / self._name(fid))
                self.samples[fid] = {"present": True, "box": list(box), "w": w, "h": h}
            else:
                img.save(self.neg_dir / self._name(fid))
                self.samples[fid] = {"present": False, "box": None, "w": w, "h": h}
            if fid not in self.order:
                self.order.append(fid)
            self._write_coco()

    def _write_coco(self) -> None:
        images, annotations, aid = [], [], 0
        for fid in sorted(self.samples):
            s = self.samples[fid]
            if not s["present"]:
                continue
            w, h = s["w"], s["h"]
            x1, y1, x2, y2 = s["box"]
            bbox = [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h]
            aid += 1
            images.append(
                {
                    "id": fid,
                    "file_name": f"images/{self._name(fid)}",
                    "width": w,
                    "height": h,
                    "review": "approved",
                }
            )
            annotations.append(
                {
                    "id": aid,
                    "image_id": fid,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "review": "approved",
                    "source": "human",
                }
            )
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "object"}],
        }
        self.coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    def counts(self) -> dict[str, int]:
        pos = sum(1 for s in self.samples.values() if s["present"])
        return {"positives": pos, "negatives": len(self.samples) - pos, "total": len(self.samples)}

    def ids(self) -> set[int]:
        return set(self.samples)

    def recent(self, since: set[int], limit: int = 5) -> list[int]:
        """Ids upserted after a snapshot (most-recent first) — regression suspects."""
        added = [fid for fid in reversed(self.order) if fid not in since]
        return added[:limit]

    def clear(self) -> None:
        with self._lock:
            for p in self.images_dir.glob("frame_*.png"):
                p.unlink()
            for p in self.neg_dir.glob("frame_*.png"):
                p.unlink()
            self.coco_path.unlink(missing_ok=True)
            self.samples.clear()
            self.order.clear()
