"""tensor_factory -- open, lightweight tiny-CNN object detection.

This core package is dependency-free and CPU-only: geometry, the 8-bit detection
codec, and annotation-format conversions. Heavy image generation (tensor-factory-synth)
and training (tensor-factory-train) live in sibling workspace packages behind GPU extras.
Helicoil detection is the first example built on it (see examples/helicoils).
"""

from .codec import decode_uint8, encode_uint8, max_error_px
from .geometry import BBox

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "__version__",
    "decode_uint8",
    "encode_uint8",
    "max_error_px",
]
