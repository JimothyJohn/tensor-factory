"""CPU inference for the tiny helicoil detector -- onnxruntime fast path, zerort floor.

The model contract is fixed: input ``image`` is ``(1, 3, S, S)`` float32 in ``[0, 1]``
(CHW, RGB), output ``box`` is ``(1, 4)`` float32 normalized ``xyxy`` in ``[0, 1]``. A model
trained with a presence head emits a second output ``presence`` of shape ``(1, 1)`` -- a
raw objectness logit whose sigmoid is P(target present); the caller thresholds it to decide
between *one box* and *no box*. A multi-class model adds a third output ``logits`` of shape
``(1, num_classes)`` -- softmax-argmax names which class the one detected object is. That is
the whole interface between training and the edge --
decode to a :class:`BBox` (or the four ``uint8`` values) with the helpers here.

Two interchangeable engines sit behind :class:`Detector`:

- ``ort`` -- onnxruntime + numpy (+ PIL for image decoding), the ``infer`` extra.
  ~200 fps on a laptop CPU at 480 px.
- ``zerort`` -- :mod:`tensor_factory.zerort`, pure CPython standard library.
  ~1 fps, zero third-party packages: the portability floor and correctness oracle.

``engine="auto"`` (the default) uses onnxruntime when the extra is installed and falls
back to zerort otherwise, so ``detect`` works on a bare Python install. Images may be
passed as PIL images (PIL path) or as a path / raw bytes of a PNG or binary PPM
(decoded by the stdlib decoder, no PIL needed).
"""

from __future__ import annotations

import math
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
        engine: str = "auto",
    ) -> None:
        if engine not in ("auto", "ort", "zerort"):
            raise ValueError(f"engine must be 'auto', 'ort' or 'zerort', got {engine!r}")
        self.input_size = input_size
        self._np = None
        self._session = None
        self._rt = None
        if engine in ("auto", "ort"):
            try:
                import numpy as np
                import onnxruntime as ort
            except ModuleNotFoundError as exc:
                if engine == "ort":
                    raise RuntimeError(
                        "engine='ort' needs the 'infer' extra: "
                        "uv pip install 'tensor-factory[infer]'"
                    ) from exc
            else:
                self._np = np
                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=list(providers) if providers else ["CPUExecutionProvider"],
                )
                self.engine = "ort"
                self.input_name = self._session.get_inputs()[0].name
                self.output_names = [o.name for o in self._session.get_outputs()]
        if self._session is None:
            from . import zerort

            self._rt = zerort.Runtime(model_path)
            self.engine = "zerort"
            self.input_name = self._rt.input_names[0]
            self.output_names = list(self._rt.output_names)
        # A model trained with a presence head also emits a "presence" output (one
        # objectness logit). Box-only models have just "box".
        self.has_presence = "presence" in self.output_names
        # A multi-class model also emits "logits" (one per class for the detected object).
        self.has_class = "logits" in self.output_names

    @property
    def session(self):
        """The underlying onnxruntime session (``None`` on the zerort engine)."""
        return self._session

    def _chw_floats(self, image: Image.Image) -> list[float]:
        """PIL image -> flat CHW float list in [0, 1] (no numpy)."""
        from PIL import Image as PILImage

        img = image.convert("RGB").resize(
            (self.input_size, self.input_size), PILImage.Resampling.BILINEAR
        )
        raw = img.tobytes()
        n = self.input_size * self.input_size
        chw = [0.0] * (3 * n)
        for c in range(3):
            chw[c * n : (c + 1) * n] = [raw[i * 3 + c] / 255.0 for i in range(n)]
        return chw

    def preprocess(self, image: Image.Image | str | Path | bytes):
        """Image -> the model input: a numpy array (ort) or zerort ``Tensor``.

        Accepts a PIL image, or a path / raw bytes of a PNG or binary PPM which are
        decoded by the stdlib decoder -- the latter works without PIL installed.
        """
        shape = (1, 3, self.input_size, self.input_size)
        if isinstance(image, str | Path | bytes):
            from .zerort import imageio

            t = imageio.preprocess(image, self.input_size)
            if self.engine == "ort":
                np = self._np
                assert np is not None
                return np.asarray(t.data, dtype=np.float32).reshape(shape)
            return t
        if self.engine == "ort":
            # PIL + numpy: the all-C fast path (~200 fps end to end).
            from PIL import Image as PILImage

            np = self._np
            assert np is not None
            img = image.convert("RGB").resize(
                (self.input_size, self.input_size), PILImage.Resampling.BILINEAR
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
            arr = arr.transpose(2, 0, 1)[None, ...]  # 1,C,H,W
            return np.ascontiguousarray(arr)
        from .zerort.tensor import Tensor

        return Tensor(shape, 1, self._chw_floats(image))

    def _run(self, image: Image.Image | str | Path | bytes) -> dict[str, list[float]]:
        x = self.preprocess(image)
        if self._session is not None:
            outs = self._session.run(None, {self.input_name: x})
            np = self._np
            assert np is not None
            return {
                name: [float(v) for v in np.asarray(o, dtype=np.float64).reshape(-1)]
                for name, o in zip(self.output_names, outs, strict=True)
            }
        assert self._rt is not None
        outs = self._rt.run({self.input_name: x})
        return {name: [float(v) for v in t.data] for name, t in outs.items()}

    def _box_from(self, values: list[float]) -> BBox:
        return BBox.clamped(values[0], values[1], values[2], values[3])

    def detect_box(self, image: Image.Image | str | Path | bytes) -> BBox:
        outs = self._run(image)
        return self._box_from(outs.get("box", next(iter(outs.values()))))

    def detect_uint8(self, image: Image.Image | str | Path | bytes) -> tuple[int, int, int, int]:
        """Detection as four ``uint8`` -- the on-the-wire 8-bit contract."""
        return encode_uint8(self.detect_box(image))

    def detect_presence(self, image: Image.Image | str | Path | bytes) -> float:
        """Probability the target is present: ``sigmoid(presence_logit)`` in ``[0, 1]``.

        The caller thresholds this to decide between returning one box and returning none.
        Raises if the model has no ``presence`` output -- use :meth:`detect_box` for the
        box-only models.
        """
        if not self.has_presence:
            raise RuntimeError("model has no presence head; use detect_box()")
        outs = self._run(image)
        logit = outs["presence"][0]
        return 1.0 / (1.0 + math.exp(-logit))

    def detect_class(self, image: Image.Image | str | Path | bytes) -> tuple[int, float]:
        """The detected object's ``(class_index, softmax_confidence)``.

        ``class_index`` is a 0-based index into the dataset's ordered category names (the
        same order the multi-class head was trained on); the caller maps it to a name.
        Raises if the model has no ``logits`` output -- single-class/box-only models have no
        class to name.
        """
        if not self.has_class:
            raise RuntimeError("model has no class head; this is a single-class detector")
        logits = self._run(image)["logits"]
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        idx = logits.index(m)
        return idx, exps[idx] / sum(exps)


def benchmark(
    detector: Detector,
    image: Image.Image | str | Path | bytes,
    *,
    n: int = 100,
    warmup: int = 5,
) -> float:
    """Return mean throughput (frames/sec) of ``detect_box`` over ``n`` runs."""
    for _ in range(warmup):
        detector.detect_box(image)
    start = time.perf_counter()
    for _ in range(n):
        detector.detect_box(image)
    elapsed = time.perf_counter() - start
    return n / elapsed if elapsed > 0 else float("inf")
