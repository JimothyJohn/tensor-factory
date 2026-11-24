"""Extract *diverse* frames from one or more microscope videos for dataset building.

The front-end the real-camera pipeline was missing. ``build_real_video_ds.py`` labels
every ``frame_*.jpg`` it finds -- but consecutive video frames are near-duplicates, so a
naive ffmpeg dump floods Label Studio review with redundant shots and feeds the model
redundant data. That redundancy is what caps *diversity* and keeps real-data localization
stuck (the model already beats its 69px constant-predictor baseline at 34px; the lever now
is more genuinely-different frames, not more copies of the same one).

This script:
  1. Samples frames from each video at ``--fps`` (default 1/s) via the ffmpeg CLI -- the
     same tool that produced the original frames, so no new dependency.
  2. Drops near-duplicates with a perceptual hash (dHash on the centre square -- the region
     the builder crops to). A frame is kept only if it differs from every already-kept
     frame by more than ``--min-distance`` bits (Hamming, out of 64). Dedup is *global*
     across all input videos, so the same shot appearing in two clips is kept once.
  3. Writes ``frame_NNNNN.jpg`` into ``--out`` (continuing the numbering if frames already
     exist there, so multiple runs/clips merge), ready for ``build_real_video_ds.py``.
     A ``frames_manifest.json`` records each frame's source video + timestamp for provenance.

Only Pillow (already a dep) + the ffmpeg CLI. No OpenCV, no decode library.

Usage:
    uv run python examples/helicoils/scripts/extract_frames.py CLIP.mp4 [CLIP2.mp4 ...] \\
        --out examples/helicoils/images/real_v2 [--fps 1.0] [--min-distance 6]
    # then: point build_real_video_ds.py at --out and review in Label Studio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HASH_SIDE = 8  # dHash grid: resize to (HASH_SIDE+1, HASH_SIDE) -> 64-bit hash


def center_square(img: Image.Image) -> Image.Image:
    """Largest centred square -- the same region build_real_video_ds.py trains on."""
    w, h = img.size
    side = min(w, h)
    return img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))


def dhash(img: Image.Image) -> int:
    """64-bit difference hash of the centre square (grayscale, horizontal gradient).

    Set bit per (row, x): is this pixel darker than its right neighbour? Robust to scale,
    brightness, and JPEG noise -- two frames within a few set bits are visually the same shot.
    """
    w = HASH_SIDE + 1
    small = center_square(img).convert("L").resize((w, HASH_SIDE), Image.Resampling.BILINEAR)
    data = small.tobytes()  # row-major, one int byte per pixel (mode "L")
    bits = 0
    for y in range(HASH_SIDE):
        row = y * w
        for x in range(HASH_SIDE):
            bits = (bits << 1) | int(data[row + x] < data[row + x + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ffmpeg_sample(video: Path, dst_dir: Path, fps: float) -> list[Path]:
    """Sample ``video`` at ``fps`` frames/sec into ``dst_dir`` as zero-padded JPEGs."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(dst_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(dst_dir.glob("*.jpg"))


def next_index(out_dir: Path) -> int:
    """One past the highest existing frame_NNNNN.jpg, so reruns/clips append cleanly."""
    existing = [int(p.stem.split("_")[1]) for p in out_dir.glob("frame_*.jpg")]
    return (max(existing) + 1) if existing else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("videos", nargs="+", type=Path, help="source video file(s)")
    ap.add_argument("--out", required=True, type=Path, help="output dir for frame_*.jpg")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second (default 1)")
    ap.add_argument(
        "--min-distance",
        type=int,
        default=6,
        help="keep a frame only if it differs from every kept frame by >N hash bits "
        "(0-64; higher = more aggressive dedup, fewer/more-distinct frames). Default 6.",
    )
    args = ap.parse_args(argv)

    missing = [v for v in args.videos if not v.is_file()]
    if missing:
        print(f"no such video(s): {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "frames_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []

    # Seed dedup state from frames already in --out, so a later run (a new clip) dedups
    # against what's there -- not just within this invocation. This is what makes appending
    # multiple videos into one growing, genuinely-diverse dataset work.
    existing = sorted(args.out.glob("frame_*.jpg"))
    kept_hashes: list[int] = [dhash(Image.open(p).convert("RGB")) for p in existing]
    if existing:
        print(f"seeded dedup from {len(existing)} frame(s) already in {args.out}")
    idx = next_index(args.out)
    start_idx = idx
    sampled_total = 0

    for video in args.videos:
        with tempfile.TemporaryDirectory() as tmp:
            frames = ffmpeg_sample(video, Path(tmp), args.fps)
            sampled_total += len(frames)
            print(f"{video.name}: sampled {len(frames)} frames @ {args.fps}/s")
            for src_i, frame in enumerate(frames):
                img = Image.open(frame).convert("RGB")
                h = dhash(img)
                if any(hamming(h, k) <= args.min_distance for k in kept_hashes):
                    continue  # near-duplicate of a frame we already kept
                kept_hashes.append(h)
                name = f"frame_{idx:05d}.jpg"
                img.save(args.out / name, quality=95)
                manifest.append(
                    {"frame": name, "source": video.name, "src_index": src_i, "fps": args.fps}
                )
                idx += 1

    manifest_path.write_text(json.dumps(manifest, indent=2))
    kept = idx - start_idx
    print(
        f"\nKEPT {kept} diverse frames "
        f"({sampled_total - kept} dropped as near-duplicates) -> {args.out}"
    )
    print(f"  manifest: {manifest_path}")
    print(f"  next: build_real_video_ds.py on {args.out}, then review in Label Studio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
