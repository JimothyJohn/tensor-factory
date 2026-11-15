"""GroundingDINO auto-labeling for Studio, primed by the previous frame's box.

GroundingDINO is open-vocabulary and *text*-prompted -- it can't take a box as input.
So we "prime it from the previous frame" spatially: crop the image to the previous box
(expanded by a margin), run open-vocab detection on that region, and map the result back
to full-frame coordinates. In video the object barely moves frame-to-frame, so the ROI
almost always contains it -- which both focuses the model and rejects far-away false
positives. With no previous box (first frame, or the object was lost) we fall back to the
whole frame.

The heavy model (transformers GroundingDINO) is reused from ``tensor_factory_synth`` and
imported lazily, so this module loads without the ``autolabel`` extra; the endpoint that
calls it surfaces a clear "install the extra" error instead of crashing.
"""

from __future__ import annotations

import functools
import io

from PIL import Image

from tensor_factory.geometry import BBox

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def roi_from_box(
    box: list[float], margin: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Pixel crop rect (left, top, right, bottom) around a normalized xyxy box, expanded by
    ``margin`` (a fraction of each side) and clamped to the image. Always at least 1px."""
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    left = round(max(0.0, x1 - mx) * width)
    top = round(max(0.0, y1 - my) * height)
    right = round(min(1.0, x2 + mx) * width)
    bottom = round(min(1.0, y2 + my) * height)
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, right, bottom


def map_box_to_full(
    local: list[float], roi: tuple[int, int, int, int], width: int, height: int
) -> list[float]:
    """Map a box given in ROI-crop normalized coords back to full-frame normalized coords."""
    left, top, right, bottom = roi
    cw = right - left
    ch = bottom - top
    lx1, ly1, lx2, ly2 = local
    return [
        (left + lx1 * cw) / width,
        (top + ly1 * ch) / height,
        (left + lx2 * cw) / width,
        (top + ly2 * ch) / height,
    ]


@functools.lru_cache(maxsize=1)
def _labeler(model: str, box_threshold: float, text_threshold: float):
    # Lazy: pulls torch+transformers (the 'autolabel' extra) and downloads the model once.
    from tensor_factory_synth.autolabel import GroundingDinoAutoLabeler

    return GroundingDinoAutoLabeler(
        model=model, box_threshold=box_threshold, text_threshold=text_threshold
    )


def _pick(detections: list, width: int, height: int):
    """Best detection by score, ignoring degenerate full-frame boxes (open-vocab models
    sometimes return the whole image)."""
    best = None
    for d in detections:
        if d.box.area >= 0.95:  # near whole-frame -> not a real localization
            continue
        if best is None or d.score > best.score:
            best = d
    return best


def autolabel(
    png: bytes,
    prompt: str,
    *,
    prev_box: list[float] | None = None,
    margin: float = 0.4,
    model: str = DEFAULT_MODEL,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
    labeler=None,
) -> dict:
    """Detect ``prompt`` in ``png``. If ``prev_box`` is given, search its (expanded) region
    first; fall back to the whole frame if that region turns up nothing. ``labeler`` is for
    tests -- any object with ``.label(image, features) -> [Detection]``."""
    lb = labeler if labeler is not None else _labeler(model, box_threshold, text_threshold)
    features = [prompt]
    with Image.open(io.BytesIO(png)) as im:
        img = im.convert("RGB")
        width, height = img.size

        if prev_box:
            roi = roi_from_box(prev_box, margin, width, height)
            crop = img.crop(roi)
            best = _pick(lb.label(crop, features), *crop.size)
            if best is not None:
                full = map_box_to_full(
                    [best.box.x1, best.box.y1, best.box.x2, best.box.y2], roi, width, height
                )
                b = BBox.clamped(*full)  # ROI mapping can land a hair outside [0,1]
                return {
                    "present": True,
                    "box": [b.x1, b.y1, b.x2, b.y2],
                    "score": float(best.score),
                    "label": best.label,
                    "primed": True,
                }

        best = _pick(lb.label(img, features), width, height)
    if best is None:
        return {"present": False, "box": None, "score": 0.0, "label": None, "primed": False}
    return {
        "present": True,
        "box": [best.box.x1, best.box.y1, best.box.x2, best.box.y2],
        "score": float(best.score),
        "label": best.label,
        "primed": False,
    }
