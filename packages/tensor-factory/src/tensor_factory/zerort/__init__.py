"""zerort -- a zero-dependency ONNX runtime for tiny int8 detectors.

The portability floor of tensor-factory: everything needed to go from an image file to
a detection -- protobuf model parsing, the quantized-graph operator set, PNG/PPM
decoding, resizing -- implemented on the CPython standard library alone. No numpy, no
onnxruntime, no PIL, no compiled extension: if a machine runs Python 3.11, it runs the
detector. onnxruntime (the ``infer`` extra) remains the fast path (~200 fps vs ~1 fps);
zerort is the fallback, the correctness oracle, and the door to platforms where wheels
don't exist.

The speed trick that makes pure Python viable at all is bigint SIMD: image planes are
packed into single arbitrary-precision integers so each convolution tap is one C-speed
bigint multiply-add. See :mod:`tensor_factory.zerort.ops`.
"""

from .engine import Runtime
from .imageio import ImageError, decode, decode_png, decode_ppm, preprocess, resize_rgb
from .ops import UnsupportedOp
from .pb import WireError, parse_model
from .tensor import Tensor
from .tfm import TfmError

__all__ = [
    "ImageError",
    "Runtime",
    "Tensor",
    "TfmError",
    "UnsupportedOp",
    "WireError",
    "decode",
    "decode_png",
    "decode_ppm",
    "parse_model",
    "preprocess",
    "resize_rgb",
]
