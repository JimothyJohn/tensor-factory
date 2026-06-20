"""Detection logic behind the MCP tools -- framework-free and unit-testable.

Wraps :class:`helicoils.inference.Detector`. The default model is the int8 ONNX bundled
with this package (trained on synthetic data -- a working demo, not a production model).
Detectors are cached by (model, input_size) so repeated calls reuse the ONNX session.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from helicoils.inference import Detector
from helicoils.inference import benchmark as _benchmark

DEFAULT_MODEL_NAME = "helicoil-mock-v1.onnx"
DEFAULT_INPUT_SIZE = 480


def default_model_path() -> Path:
    return Path(__file__).parent / "models" / DEFAULT_MODEL_NAME


def resolve_model(model_path: str | None) -> str:
    path = Path(model_path) if model_path else default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"model not found: {path}. Pass an explicit model_path, or train one with "
            "`helicoils-train fit` and point at the resulting ONNX file."
        )
    return str(path)


@functools.lru_cache(maxsize=4)
def _detector(model_path: str, input_size: int) -> Detector:
    return Detector(model_path, input_size=input_size)


def detect(
    image_path: str,
    model_path: str | None = None,
    input_size: int = DEFAULT_INPUT_SIZE,
) -> dict[str, Any]:
    """Detect the helicoil in one image; return normalized, pixel, and uint8 boxes."""
    from PIL import Image

    resolved = resolve_model(model_path)
    detector = _detector(resolved, input_size)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        box = detector.detect_box(img)
        packed = detector.detect_uint8(img)
    px = box.to_pixels(width=width, height=height)
    return {
        "box_norm": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
        "box_pixels": {"x1": px[0], "y1": px[1], "x2": px[2], "y2": px[3]},
        "uint8": list(packed),
        "image_size": {"width": width, "height": height},
        "model": resolved,
    }


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
