"""The browser's image resize against PIL, which is what the Python trainer uses.

A model only works on pixels like the ones it trained on, so the browser trainer has to
scale frames the way ``tensor_factory_train`` does. Runs ``studio/js/nn/data.js`` in node.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CLI = REPO / "studio" / "tools" / "resize_cli.mjs"
NODE = shutil.which("node") or ""
pytestmark = pytest.mark.skipif(not NODE, reason="node is not installed")


def _browser(mode: str, rgb: np.ndarray, size: int) -> np.ndarray:
    h, w, _ = rgb.shape
    proc = subprocess.run(  # noqa: S603
        [NODE, str(CLI), mode, str(w), str(h), str(size)],
        input=rgb.tobytes(),
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(size, size, 3)


def _image(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    smooth = rng.integers(0, 255, (h // 8 + 2, w // 8 + 2, 3), dtype=np.uint8)
    big = np.asarray(Image.fromarray(smooth).resize((w, h), Image.Resampling.BICUBIC))
    noise = rng.integers(-20, 20, (h, w, 3))
    return np.clip(big.astype(np.int64) + noise, 0, 255).astype(np.uint8)


SHAPES = [
    (640, 480, 224),
    (1920, 1080, 224),
    (100, 300, 160),  # width grows, height shrinks: Pillow runs the vertical pass first
    (97, 301, 64),
    (301, 97, 64),
    (64, 64, 96),
    (37, 53, 16),
    (5, 3, 32),
    (224, 224, 224),
]


@pytest.mark.unit
@pytest.mark.parametrize(("w", "h", "size"), SHAPES)
def test_nearest_is_exactly_pil(w, h, size):
    rgb = _image(w, h, seed=w + h)
    want = np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST))
    np.testing.assert_array_equal(_browser("nearest", rgb, size), want)


@pytest.mark.unit
@pytest.mark.parametrize(("w", "h", "size"), SHAPES)
def test_bilinear_is_exactly_pil(w, h, size):
    rgb = _image(w, h, seed=w * h)
    want = np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(_browser("bilinear", rgb, size), want)


@pytest.mark.unit
def test_wrong_byte_count_is_refused():
    proc = subprocess.run(  # noqa: S603
        [NODE, str(CLI), "bilinear", "10", "10", "8"],
        input=b"\x00" * 7,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode != 0 and b"not a 10x10x3 image" in proc.stderr
