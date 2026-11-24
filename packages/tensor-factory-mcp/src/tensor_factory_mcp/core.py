"""Detection logic behind the MCP tools -- framework-free and unit-testable.

Wraps :class:`tensor_factory.inference.Detector`. The default model is the int8 ONNX bundled
with this package (a real-data detector with a presence head). Detectors are cached by
(model, input_size, engine) so repeated calls reuse the session.

Zero-dependency serving: when PIL is not installed, image paths and bytes route through
the stdlib decoder (PNG / binary PPM) and inference falls back to the pure-Python zerort
engine -- a bare CPython can serve detections. Each result is built from a *single*
forward pass (:meth:`Detector.detect_outputs`); on zerort an extra pass costs ~1.5 s.

Hot reload: a cached detector is tied to its model file's identity (mtime + size). Replace
the file and the next call loads the new model -- a running server takes a new model with
no restart. If the replacement does not load (half-copied, corrupt), the model that was
working keeps serving and :func:`model_info` reports why under ``reload_error``.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from dataclasses import dataclass
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


@dataclass
class _Loaded:
    """A live detector plus the identity of the file it was loaded from."""

    detector: Detector
    stamp: tuple[int, int]  # (mtime_ns, size) of the model file at load time
    sha256: str
    size: int
    # Set when a newer file at the same path failed to load: its stamp (so it is not
    # retried on every request) and the reason, surfaced through model_info.
    failed_stamp: tuple[int, int] | None = None
    reload_error: str | None = None


_detectors: dict[tuple[str, int, str], _Loaded] = {}
_lock = threading.Lock()


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


def _stamp(model_path: str) -> tuple[int, int]:
    st = os.stat(model_path)
    return st.st_mtime_ns, st.st_size


def _load(model_path: str, input_size: int, engine: str) -> _Loaded:
    # Read once and hand the engine the same bytes that were hashed, so the reported
    # sha256 is exactly the model that is running even if the file changes mid-load.
    stamp = _stamp(model_path)
    data = Path(model_path).read_bytes()
    detector = Detector(model_path, input_size=input_size, engine=engine, model_bytes=data)
    return _Loaded(detector, stamp, hashlib.sha256(data).hexdigest(), len(data))


def _loaded(model_path: str, input_size: int, engine: str = "auto") -> _Loaded:
    key = (model_path, input_size, engine)
    with _lock:
        cached = _detectors.get(key)
        try:
            stamp = _stamp(model_path)
        except OSError:
            if cached is None:
                raise
            # The file vanished mid-replace (rm then cp): keep serving what is loaded.
            cached.reload_error = "model file is missing; serving the last loaded model"
            return cached
        if cached is not None and stamp in (cached.stamp, cached.failed_stamp):
            return cached
        try:
            fresh = _load(model_path, input_size, engine)
        except Exception as exc:  # noqa: BLE001 -- any load failure keeps the old model
            if cached is None:
                raise
            cached.failed_stamp = stamp
            cached.reload_error = f"{type(exc).__name__}: {exc}"
            return cached
        if cached is None and len(_detectors) >= 4:  # bounded like the old lru_cache(4)
            _detectors.pop(next(iter(_detectors)))
        _detectors[key] = fresh
        return fresh


def _detector(model_path: str, input_size: int, engine: str = "auto") -> Detector:
    return _loaded(model_path, input_size, engine).detector


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
    """Report the resolved model path, input size, IO name, engine, and providers.

    ``sha256`` and ``bytes`` identify the model that is actually loaded, so a deploy can
    confirm a device is running exactly the file that was pushed. ``reload_error`` is
    present only when a newer file at that path failed to load and the previous model is
    still the one serving.
    """
    resolved = resolve_model(model_path)
    loaded = _loaded(resolved, input_size, engine)
    detector = loaded.detector
    info: dict[str, Any] = {
        "model": resolved,
        "sha256": loaded.sha256,
        "bytes": loaded.size,
        "input_size": input_size,
        "input_name": detector.input_name,
        "engine": detector.engine,
        "providers": list(detector.session.get_providers()) if detector.session else ["zerort"],
    }
    if loaded.reload_error:
        info["reload_error"] = loaded.reload_error
    return info


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
