"""Quantize a normalized box to four 8-bit values -- the detection contract.

The model output is deliberately tiny: a helicoil box is four ``uint8`` values, one
per normalized coordinate. At 480 px the quantization step is ``480 / 255 ~= 1.88``
px, so the worst-case round-trip error is under one pixel -- comfortably inside the
3 px budget -- while keeping the model head and post-processing in 8-bit math.
"""

from __future__ import annotations

from collections.abc import Sequence

from .geometry import BBox

_MAX = 255  # 256 levels, 0..255


def _q(v: float) -> int:
    q = round(v * _MAX)
    return 0 if q < 0 else _MAX if q > _MAX else q


def encode_uint8(box: BBox) -> tuple[int, int, int, int]:
    """Encode a normalized box to four ``uint8`` values ``(x1, y1, x2, y2)``."""
    return (_q(box.x1), _q(box.y1), _q(box.x2), _q(box.y2))


def decode_uint8(values: Sequence[int]) -> BBox:
    """Decode four ``uint8`` values back to a normalized :class:`BBox`.

    Order is repaired and coordinates clamped, so a model that emits ``x2 < x1`` for
    a near-degenerate box still yields a valid box rather than raising.
    """
    if len(values) != 4:
        raise ValueError(f"expected 4 values, got {len(values)}")
    for v in values:
        if not 0 <= v <= _MAX:
            raise ValueError(f"value {v!r} outside uint8 range [0, {_MAX}]")
    x1, y1, x2, y2 = (v / _MAX for v in values)
    return BBox.clamped(x1, y1, x2, y2)


def max_error_px(image_size: int) -> float:
    """Worst-case per-coordinate round-trip error, in pixels, at ``image_size``."""
    return image_size / _MAX / 2.0
