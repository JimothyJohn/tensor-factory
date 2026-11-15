"""Detection logic behind the MCP tools -- framework-free and unit-testable.

Wraps :class:`tensor_factory.inference.Detector`. The default model is the int8 ONNX bundled
with this package (a real-data detector with a presence head). Detectors are cached by
(model, input_size, engine) so repeated calls reuse the session.

Zero-dependency serving: when PIL is not installed, image paths and bytes route through
the stdlib decoder (PNG / binary PPM) and inference falls back to the pure-Python zerort
engine -- a bare CPython can serve detections. Each result is built from a *single*
forward pass (:meth:`Detector.detect_outputs`); on zerort an extra pass costs ~1.5 s.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tensor_factory import BBox, encode_uint8
from tensor_factory.inference import Detector
from tensor_factory.inference import benchmark as _benchmark

DEFAULT_MODEL_NAME = "helicoil-presence-cam-v1.onnx"
DEFAULT_INPUT_SIZE = 480
# A presence-head model returns a box only when sigmoid(objectness) clears this. Below it,
# the detection is "nothing here" -- no box at all, the YOLO-style no-object case.
PRESENCE_THRESHOLD = 0.5

_detectors: dict[tuple[str, int, str], Detector] = {}


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


def _detector(model_path: str, input_size: int, engine: str = "auto") -> Detector:
    key = (model_path, input_size, engine)
    if key not in _detectors:
        if len(_detectors) >= 4:  # bounded like the old lru_cache(maxsize=4)
            _detectors.pop(next(iter(_detectors)))
        _detectors[key] = Detector(model_path, input_size=input_size, engine=engine)
    return _detectors[key]


def _box_fields(box: BBox, width: int, height: int) -> dict[str, Any]:
    """The box portion of the result: normalized, pixel, and 4xuint8 forms."""
    px = box.to_pixels(width=width, height=height)
    return {
        "box_norm": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
        "box_pixels": {"x1": px[0], "y1": px[1], "x2": px[2], "y2": px[3]},
        "uint8": list(encode_uint8(box)),
    }


def _detect_source(
    src: Any, width: int, height: int, resolved: str, detector: Detector
) -> dict[str, Any]:
    """Build the detection result dict from anything :class:`Detector` accepts.

    ``src`` is a PIL image (PIL installed) or raw PNG/PPM bytes (stdlib path); both
    entry points return byte-identical JSON. One forward pass produces every field.

    A presence-head model returns one box or none: ``present`` (bool) and ``score`` (the
    objectness probability) are always set, and the box fields are populated only when the
    target is present -- otherwise ``box_norm`` / ``box_pixels`` / ``uint8`` are ``null``.
    A box-only model has no presence head, so it always returns a box and omits
    ``present`` / ``score``.
    """
    outs = detector.detect_outputs(src)
    box_vals = outs.get("box", next(iter(outs.values())))
    box = BBox.clamped(box_vals[0], box_vals[1], box_vals[2], box_vals[3])
    result: dict[str, Any] = {
        "image_size": {"width": width, "height": height},
        "model": resolved,
    }
    if detector.has_presence:
        score = 1.0 / (1.0 + math.exp(-outs["presence"][0]))
        present = score >= PRESENCE_THRESHOLD
        result["present"] = present
        result["score"] = score
        result.update(
            _box_fields(box, width, height)
            if present
            else {"box_norm": None, "box_pixels": None, "uint8": None}
        )
    else:
        result.update(_box_fields(box, width, height))
    return result


def _dimensions(data: bytes) -> tuple[int, int]:
    """Width and height of raw image bytes via the stdlib decoder."""
    from tensor_factory.zerort import imageio

    width, height, _rgb = imageio.decode(data)
    return width, height


def detect(
    image_path: str,
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    engine: str = "auto",
) -> dict[str, Any]:
    """Detect the helicoil in one image file; return normalized, pixel, and uint8 boxes."""
    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size, engine)
    try:
        from PIL import Image
    except ModuleNotFoundError:
        data = Path(image_path).read_bytes()
        width, height = _dimensions(data)
        return _detect_source(data, width, height, resolved, detector)
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        return _detect_source(rgb, rgb.size[0], rgb.size[1], resolved, detector)


def detect_bytes(
    data: bytes,
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    engine: str = "auto",
) -> dict[str, Any]:
    """Detect in raw image bytes (e.g. an HTTP request body); same JSON as :func:`detect`.

    Raises whatever the image decoder raises on undecodable bytes (``PIL.Image.open``
    when PIL is installed, :class:`tensor_factory.zerort.ImageError` otherwise) -- the
    HTTP layer maps that to a 400 rather than letting it 500.
    """
    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size, engine)
    try:
        from PIL import Image
    except ModuleNotFoundError:
        width, height = _dimensions(data)
        return _detect_source(data, width, height, resolved, detector)
    import io

    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")
        return _detect_source(rgb, rgb.size[0], rgb.size[1], resolved, detector)


def model_info(
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    engine: str = "auto",
) -> dict[str, Any]:
    """Report the resolved model path, input size, IO name, engine, and providers."""
    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size, engine)
    return {
        "model": resolved,
        "input_size": input_size,
        "input_name": detector.input_name,
        "engine": detector.engine,
        "providers": list(detector.session.get_providers()) if detector.session else ["zerort"],
    }


def benchmark(
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
    n: int = 100,
    engine: str = "auto",
) -> dict[str, Any]:
    """Measure CPU throughput (fps) on a synthetic gray frame."""
    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size, engine)
    try:
        from PIL import Image
    except ModuleNotFoundError:
        image: Any = b"P6\n%d %d\n255\n" % (input_size, input_size) + bytes([128]) * (
            input_size * input_size * 3
        )
    else:
        image = Image.new("RGB", (input_size, input_size), (128, 128, 128))
    fps = _benchmark(detector, image, n=n)
    return {
        "model": resolved,
        "fps": round(fps, 1),
        "iterations": n,
        "input_size": input_size,
    }
