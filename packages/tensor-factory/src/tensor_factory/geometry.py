"""Geometry primitives for helicoil detection.

The canonical bounding box is normalized ``xyxy`` in ``[0, 1]`` with the origin at
the top-left of the image. Keeping boxes resolution-independent lets one annotation
flow unchanged through 480x480 training and whatever the microscope (or, later, the
robot) feeds at inference time.
"""

from __future__ import annotations

_EPS = 1e-6


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


class BBox:
    """Axis-aligned box in normalized ``xyxy`` coordinates (top-left origin).

    Immutable. Construct directly only with values you know are valid and ordered;
    use :meth:`clamped` for anything derived from a model, a file, or arithmetic.
    """

    __slots__ = ("x1", "y1", "x2", "y2")

    x1: float
    y1: float
    x2: float
    y2: float

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        for name, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            if not -_EPS <= v <= 1.0 + _EPS:
                raise ValueError(f"{name}={v!r} outside normalized range [0, 1]")
        if x2 + _EPS < x1 or y2 + _EPS < y1:
            raise ValueError(f"degenerate box: ({x1}, {y1}, {x2}, {y2})")
        object.__setattr__(self, "x1", _clamp01(x1))
        object.__setattr__(self, "y1", _clamp01(y1))
        object.__setattr__(self, "x2", _clamp01(x2))
        object.__setattr__(self, "y2", _clamp01(y2))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("BBox is immutable")

    def __repr__(self) -> str:
        return f"BBox({self.x1:.6g}, {self.y1:.6g}, {self.x2:.6g}, {self.y2:.6g})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BBox):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (
            other.x1,
            other.y1,
            other.x2,
            other.y2,
        )

    def __hash__(self) -> int:
        return hash((self.x1, self.y1, self.x2, self.y2))

    @classmethod
    def clamped(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        """Build a box, clamping into ``[0, 1]`` and fixing coordinate order."""
        lo_x, hi_x = sorted((_clamp01(x1), _clamp01(x2)))
        lo_y, hi_y = sorted((_clamp01(y1), _clamp01(y2)))
        return cls(lo_x, lo_y, hi_x, hi_y)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Build from center-x, center-y, width, height (all normalized)."""
        return cls.clamped(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @classmethod
    def from_pixels(
        cls, x1: float, y1: float, x2: float, y2: float, *, width: int, height: int
    ) -> BBox:
        """Build from absolute pixel ``xyxy`` given the image ``width``/``height``."""
        return cls.clamped(x1 / width, y1 / height, x2 / width, y2 / height)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        cx, cy = self.center
        return (cx, cy, self.width, self.height)

    def to_pixels(self, *, width: int, height: int) -> tuple[int, int, int, int]:
        """Absolute integer pixel ``xyxy`` for an image of ``width`` x ``height``."""
        return (
            round(self.x1 * width),
            round(self.y1 * height),
            round(self.x2 * width),
            round(self.y2 * height),
        )

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with ``other`` in ``[0, 1]``."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0.0 else 0.0
