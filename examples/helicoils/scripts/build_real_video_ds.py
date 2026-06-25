"""Build the real-camera helicoil dataset from extracted video frames.

Source frames are 1920x1080 microscope captures in ``examples/helicoils/images/real/``
(produced by ffmpeg from helicoils.mp4). This script:

  1. Center-crops each frame to 1080x1080 (the subject stays centred across the whole
     clip) and resizes to 480x480 -- the model's input size and the convention every
     other dataset here uses.
  2. Labels each crop with GroundingDINO (feature "threaded hole"), keeping the single
     best qualifying box -- identical thresholds and pick() logic to build_ds.py.
  3. Writes a COCO dataset:  real/images/*.png + real/annotations.coco.json
     (review=pending, source=groundingdino -- not trainable until reviewed in Label Studio).

Unlike build_ds.py these are *real photos*, so there's no Gemini step. The 80/20
train/test split is deferred to training time via ``tensor-factory-train fit --val-frac
0.2 --seed <S>`` (one dataset, seeded split) -- no physical split is materialised here.

Usage:
    uv run python examples/helicoils/scripts/build_real_video_ds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from tensor_factory_synth.autolabel import Detection, GroundingDinoAutoLabeler
from tensor_factory_synth.export import build_coco, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "examples" / "helicoils" / "images" / "real"
OUT = RAW_DIR  # the dataset lives in-place: real/images/ + real/annotations.coco.json
SIZE = 480
FEATURE = "threaded hole"


def center_square_crop(img: Image.Image) -> Image.Image:
    """Crop the largest centred square, then resize to SIZE x SIZE (bilinear)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    sq = img.crop((left, top, left + side, top + side))
    return sq.resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def pick(dets: list[Detection]) -> Detection | None:
    # Identical to build_ds.py: best-scoring box within sane area/aspect bounds.
    good = [
        d for d in dets if 0.03 < d.box.area < 0.85 and d.box.width < 0.95 and d.box.height < 0.95
    ]
    return max(good, key=lambda d: d.score) if good else None


def main() -> int:
    frames = sorted(RAW_DIR.glob("frame_*.jpg"))
    if not frames:
        print(f"no frame_*.jpg in {RAW_DIR}", file=sys.stderr)
        return 1
    print(f"cropping {len(frames)} frames -> {SIZE}x{SIZE} PNG")

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    crops: list[tuple[str, Path]] = []  # (file_name relative to OUT, abs png path)
    for f in frames:
        name = f"{f.stem}.png"
        dst = img_dir / name
        center_square_crop(Image.open(f).convert("RGB")).save(dst)
        crops.append((f"images/{name}", dst))

    print(f"labeling {len(crops)} crops with GroundingDINO (feature: {FEATURE!r})")
    labeler = GroundingDinoAutoLabeler(box_threshold=0.25, text_threshold=0.2)

    records = []
    labeled = 0
    for i, (file_name, png) in enumerate(crops, start=1):
        img = Image.open(png).convert("RGB")
        d = pick(labeler.label(img, [FEATURE]))
        dets = [Detection(label="helicoil", box=d.box, score=d.score)] if d else []
        records.append((file_name, SIZE, SIZE, dets))
        labeled += bool(d)
        if i % 25 == 0:
            print(f"  {i}/{len(crops)}  ({labeled} with a box)")

    write_json(OUT / "annotations.coco.json", build_coco(records, ["helicoil"]))
    # Drop the now-superseded raw frames so OUT is a clean COCO dataset dir.
    for f in frames:
        f.unlink()
    print(
        f"\nDATASET: {len(crops)} images, {labeled} with a qualifying box "
        f"({len(crops) - labeled} empty / no box) -> {OUT}"
    )
    print("  next: review in Label Studio, then train with --val-frac 0.2 for the 80/20 split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
