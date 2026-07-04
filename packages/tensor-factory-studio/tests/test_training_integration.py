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


def _class_positive(i: int, cls: int) -> tuple[bytes, list[float]]:
    """A box on noise whose *colour* encodes the class -- a tint the class head can learn
    while the box stays in a class-independent position so localization isn't a giveaway."""
    import random

    rng = random.Random(i)
    img = Image.new("RGB", (96, 96))
    img.putdata(
        [(rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80)) for _ in range(96 * 96)]
    )
    # Vary the box position so the constant-predictor baseline is non-degenerate.
    x = 20 + (i % 4) * 10
    y = 30
    # class 0 -> red box, class 1 -> green box: a separable, learnable cue.
    fill = (240, 20, 20) if cls == 0 else (20, 240, 20)
    ImageDraw.Draw(img).rectangle([x, y, x + 30, y + 30], fill=fill)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    box = [x / 96, y / 96, (x + 30) / 96, (y + 30) / 96]
    return buf.getvalue(), box


@pytest.mark.integration
def test_multiclass_round_learns_to_classify(tmp_path):
    # End-to-end: a 2-category dataset must train a class head, export a model with a
    # "logits" output, and actually separate the two classes through the real inference path.
    ds = Dataset(tmp_path / "data")
    ds.set_classes(["red", "green"])
    for i in range(6):
        png, box = _class_positive(i, cls=0)
        ds.upsert(i, True, box, png, cls=0)
    for i in range(6):
        png, box = _class_positive(100 + i, cls=1)
        ds.upsert(100 + i, True, box, png, cls=1)
    ds.upsert(900, False, None, _negative(0))

    tr = Trainer(
        ds,
        tmp_path / "data" / "models",
        size=96,
        width=8,
        epochs=40,
        batch=4,
        min_positives=4,
        device="cpu",
    )
    tr.start()
    tr.mark_dirty()
    deadline = time.time() + 180
    while time.time() < deadline and tr.served is None:
        time.sleep(0.5)
    tr.stop()
    assert tr.served is not None, "multi-class trainer never promoted a model"

    # metrics carry the class signal through the studio hook
    m = tr.metrics()
    assert m["numClasses"] == 2
    assert m["classAcc"] is None or 0.0 <= m["classAcc"] <= 1.0

    from tensor_factory.inference import Detector

    det = Detector(str(tr.served), input_size=96)
    assert det.has_class
    # The trained head names each colour correctly (separable cue, 40 epochs).
    red_png, _ = _class_positive(0, cls=0)
    green_png, _ = _class_positive(100, cls=1)
    with Image.open(io.BytesIO(red_png)) as im:
        red_idx, _ = det.detect_class(im.convert("RGB"))
    with Image.open(io.BytesIO(green_png)) as im:
        green_idx, _ = det.detect_class(im.convert("RGB"))
    assert red_idx == 0 and green_idx == 1


@pytest.mark.integration
def test_trainer_idles_without_new_labels(tmp_path):
    # Regression: the loop must only train when newly dirty. A bare wait()+train
    # retrained every second forever (version kept climbing), saturating the server.
    ds = Dataset(tmp_path / "data")
    for i in range(6):
        png, box = _positive(i)
        ds.upsert(i, True, box, png)
    ds.upsert(100, False, None, _negative(0))

    tr = Trainer(
        ds,
        tmp_path / "data" / "models",
        size=96,
        width=8,
        epochs=2,
        batch=4,
        min_positives=4,
        device="cpu",
    )
    tr.start()
    tr.mark_dirty()
    deadline = time.time() + 120
    while time.time() < deadline and tr.version == 0:
        time.sleep(0.5)
    assert tr.version == 1, "first dirty should trigger exactly one round"

    # no new labels -> no new rounds
    time.sleep(4)
    tr.stop()
    assert tr.version == 1, f"trainer retrained without new labels (version={tr.version})"
