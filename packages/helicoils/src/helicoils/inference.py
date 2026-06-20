"""CPU inference for the tiny helicoil detector via onnxruntime.

The model contract is fixed: input ``image`` is ``(1, 3, S, S)`` float32 in ``[0, 1]``
(CHW, RGB), output ``box`` is ``(1, 4)`` float32 normalized ``xyxy`` in ``[0, 1]``. That
is the whole interface between training and the edge -- decode to a :class:`BBox` (or the
four ``uint8`` values) with the helpers here. onnxruntime + numpy are an ``infer`` extra,
lazily imported, so the pure-Python core stays dependency-free.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .codec import encode_uint8
from .geometry import BBox

if TYPE_CHECKING:
    from PIL import Image


class Detector:
    """Run the ONNX detector on CPU and decode its output to a box."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 480,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "inference needs the 'infer' extra: uv pip install 'helicoils[infer]'"
            ) from exc

        self._np = np
        self.input_size = input_size
        self.session = ort.InferenceSession(
            str(model_path),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, image: Image.Image):
        from PIL import Image as PILImage

        np = self._np
        img = image.convert("RGB").resize(
            (self.input_size, self.input_size), PILImage.Resampling.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
        arr = arr.transpose(2, 0, 1)[None, ...]  # 1,C,H,W
        return np.ascontiguousarray(arr)

    def detect_box(self, image: Image.Image) -> BBox:
        x = self.preprocess(image)
        out = self.session.run(None, {self.input_name: x})[0]
        coords = self._np.asarray(out).reshape(-1)[:4]
        return BBox.clamped(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))

    def detect_uint8(self, image: Image.Image) -> tuple[int, int, int, int]:
        """Detection as four ``uint8`` -- the on-the-wire 8-bit contract."""
        return encode_uint8(self.detect_box(image))


def benchmark(detector: Detector, image: Image.Image, *, n: int = 100, warmup: int = 5) -> float:
    """Return mean throughput (frames/sec) of ``detect_box`` over ``n`` runs."""
    for _ in range(warmup):
        detector.detect_box(image)
    start = time.perf_counter()
    for _ in range(n):
        detector.detect_box(image)
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed > 0 else float("inf")
