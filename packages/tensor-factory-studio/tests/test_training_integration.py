"""A real training round through the Trainer — no mocks, real torch + ONNX export.

Marked integration and skipped when torch isn't installed (the dev gate has no torch).
Run it with the serve extra synced:  uv run --extra serve python -m pytest -m integration
"""

import io
import time

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")  # skip the whole module without the serve extra

from tensor_factory_studio.dataset import Dataset  # noqa: E402
from tensor_factory_studio.trainer import Trainer  # noqa: E402


def _positive(i: int) -> tuple[bytes, list[float]]:
    """A frame with a white box at a position that varies with i, on textured noise."""
    import random

    rng = random.Random(i)
    img = Image.new("RGB", (96, 96))
    img.putdata(
        [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(96 * 96)]
    )
    x = 20 + (i % 4) * 10
    y = 30
    ImageDraw.Draw(img).rectangle([x, y, x + 30, y + 30], fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    box = [x / 96, y / 96, (x + 30) / 96, (y + 30) / 96]
    return buf.getvalue(), box


def _negative(i: int) -> bytes:
    import random

    rng = random.Random(1000 + i)
    img = Image.new("RGB", (96, 96))
    img.putdata(
        [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(96 * 96)]
    )
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.integration
def test_continuous_round_produces_served_model_and_metrics(tmp_path):
    ds = Dataset(tmp_path / "data")
    for i in range(8):
        png, box = _positive(i)
        ds.upsert(i, True, box, png)
    for i in range(2):
        ds.upsert(100 + i, False, None, _negative(i))

    tr = Trainer(
        ds,
        tmp_path / "data" / "models",
        size=96,
        width=8,
        epochs=4,
        batch=4,
        min_positives=4,
        device="cpu",
    )
    tr.start()
    tr.mark_dirty()

    deadline = time.time() + 120
    while time.time() < deadline and tr.served is None:
        time.sleep(0.5)
    tr.stop()

    assert tr.served is not None, "trainer never promoted a served model"
    assert tr.served.is_file()
    m = tr.metrics()
    assert m["backend"] == "cpu"
    assert m["bestErr"] is not None and m["bestErr"] >= 0
    assert m["epoch"] == 4  # all epochs ran
    assert m["valCount"] >= 1

    # the served ONNX must load + run through the real inference path
    from tensor_factory.inference import Detector

    det = Detector(str(tr.served), input_size=96)
    assert det.has_presence
    png, _ = _positive(0)
    with Image.open(io.BytesIO(png)) as im:
        score = det.detect_presence(im.convert("RGB"))
    assert 0.0 <= score <= 1.0
