"""Detection logic behind the MCP tools -- framework-free and unit-testable.

Wraps :class:`tensor_factory.inference.Detector`. The default model is the int8 ONNX bundled
with this package (a real-data detector with a presence head). Detectors are cached by
(model, input_size) so repeated calls reuse the ONNX session.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from tensor_factory.inference import Detector
from tensor_factory.inference import benchmark as _benchmark

DEFAULT_MODEL_NAME = "helicoil-presence-cam-v1.onnx"
DEFAULT_INPUT_SIZE = 480
# A presence-head model returns a box only when sigmoid(objectness) clears this. Below it,
# the detection is "nothing here" -- no box at all, the YOLO-style no-object case.
PRESENCE_THRESHOLD = 0.5


def default_model_path() -> Path:
    return Path(__file__).parent / "models" / DEFAULT_MODEL_NAME


def resolve_model(model_path: str | None) -> str:
    path = Path(model_path) if model_path else default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"model not found: {path}. Pass an explicit model_path, or train one with "
            "`tensor-factory-train fit` and point at the resulting ONNX file."
        )
    return str(path)


@functools.lru_cache(maxsize=4)
def _detector(model_path: str, input_size: int) -> Detector:
    return Detector(model_path, input_size=input_size)


def _box_fields(detector: Detector, img: Any, width: int, height: int) -> dict[str, Any]:
    """The box portion of the result: normalized, pixel, and 4xuint8 forms."""
    box = detector.detect_box(img)
    px = box.to_pixels(width=width, height=height)
    return {
        "box_norm": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
        "box_pixels": {"x1": px[0], "y1": px[1], "x2": px[2], "y2": px[3]},
        "uint8": list(detector.detect_uint8(img)),
    }


def _detect_image(img: Any, resolved: str, detector: Detector) -> dict[str, Any]:
    """Build the detection result dict from an already-opened RGB ``PIL.Image``.

    Shared by :func:`detect` (path in) and :func:`detect_bytes` (HTTP body in) so the
    two entry points return byte-identical JSON.

    A presence-head model returns one box or none: ``present`` (bool) and ``score`` (the
    objectness probability) are always set, and the box fields are populated only when the
    target is present -- otherwise ``box_norm`` / ``box_pixels`` / ``uint8`` are ``null``.
    A box-only model has no presence head, so it always returns a box and omits
    ``present`` / ``score``.
    """
    width, height = img.size
    result: dict[str, Any] = {
        "image_size": {"width": width, "height": height},
        "model": resolved,
    }
    if detector.has_presence:
        score = detector.detect_presence(img)
        present = score >= PRESENCE_THRESHOLD
        result["present"] = present
        result["score"] = score
        box = (
            _box_fields(detector, img, width, height)
            if present
            else {
                "box_norm": None,
                "box_pixels": None,
                "uint8": None,
            }
        )
        result.update(box)
    else:
        result.update(_box_fields(detector, img, width, height))
    return result


def detect(
    image_path: str,
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> dict[str, Any]:
    """Detect the helicoil in one image file; return normalized, pixel, and uint8 boxes."""
    from PIL import Image

    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size)
    with Image.open(image_path) as img:
        return _detect_image(img.convert("RGB"), resolved, detector)


def detect_bytes(
    data: bytes,
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> dict[str, Any]:
    """Detect in raw image bytes (e.g. an HTTP request body); same JSON as :func:`detect`.

    Raises whatever ``PIL.Image.open`` raises on undecodable bytes -- the HTTP layer maps
    that to a 400 rather than letting it 500.
    """
    import io

    from PIL import Image

    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size)
    with Image.open(io.BytesIO(data)) as img:
        return _detect_image(img.convert("RGB"), resolved, detector)


def model_info(
    model_path: str | None = None, input_size: int = DEFAULT_INPUT_SIZE
) -> dict[str, Any]:
    """Report the resolved model path, input size, IO name, and ORT providers."""
    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size)
    return {
        "model": resolved,
        "input_size": input_size,
        "input_name": detector.input_name,
        "providers": list(detector.session.get_providers()),
    }


def benchmark(
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    n: int = 100,
) -> dict[str, Any]:
    """Measure CPU throughput (fps) on a synthetic gray frame."""
    from PIL import Image

    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size)
    image = Image.new("RGB", (input_size, input_size), (128, 128, 128))
    fps = _benchmark(detector, image, n=n)
    return {
        "model": resolved,
        "fps": round(fps, 1),
        "iterations": n,
        "input_size": input_size,
    }
