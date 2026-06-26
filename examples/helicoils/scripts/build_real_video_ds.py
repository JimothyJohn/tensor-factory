"""Build the real-camera helicoil dataset from extracted video frames.

Source frames are microscope captures named ``frame_*.jpg`` in ``--dir`` (default
``examples/helicoils/images/real/``) -- typically the deduped output of
``extract_frames.py`` (or a raw ffmpeg dump). This script:

  1. Center-crops each frame to its largest centred square and resizes to 480x480 --
     the model's input size and the convention every other dataset here uses.
  2. Labels each crop with GroundingDINO (feature "threaded hole"), keeping the single
     best qualifying box -- identical thresholds and pick() logic to build_ds.py.
  3. Writes a COCO dataset:  <dir>/images/*.png + <dir>/annotations.coco.json
     (review=pending, source=groundingdino -- not trainable until reviewed in Label Studio).

Unlike build_ds.py these are *real photos*, so there's no Gemini step. The 80/20
train/test split is deferred to training time via ``tensor-factory-train fit --val-frac
0.2 --seed <S>`` (one dataset, seeded split) -- no physical split is materialised here.

Usage:
    # one or more clips -> deduped frames -> labeled COCO dataset
    uv run python examples/helicoils/scripts/extract_frames.py CLIP.mp4 --out images/real_v2
    uv run python examples/helicoils/scripts/build_real_video_ds.py --dir images/real_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from tensor_factory_synth.autolabel import Detection, GroundingDinoAutoLabeler
from tensor_factory_synth.export import build_coco, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = REPO_ROOT / "examples" / "helicoils" / "images" / "real"
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help="dataset dir holding frame_*.jpg (extract_frames.py --out); built in-place. "
        "Default: images/real. Point at a new dir to build a fresh dataset without "
        "clobbering an existing one.",
    )
    out = ap.parse_args(argv).dir

    frames = sorted(out.glob("frame_*.jpg"))
    if not frames:
        print(f"no frame_*.jpg in {out}", file=sys.stderr)
        return 1
    print(f"cropping {len(frames)} frames -> {SIZE}x{SIZE} PNG")

    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    crops: list[tuple[str, Path]] = []  # (file_name relative to out, abs png path)
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

    write_json(out / "annotations.coco.json", build_coco(records, ["helicoil"]))
    # Drop the now-superseded raw frames so the dir is a clean COCO dataset dir.
    for f in frames:
        f.unlink()
    print(
        f"\nDATASET: {len(crops)} images, {labeled} with a qualifying box "
        f"({len(crops) - labeled} empty / no box) -> {out}"
    )
    print("  next: review in Label Studio, then train with --val-frac 0.2 for the 80/20 split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
