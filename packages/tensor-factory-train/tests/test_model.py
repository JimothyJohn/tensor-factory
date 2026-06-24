"""Model + export tests. Skipped unless torch is installed (the 'train' extra)."""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tensor_factory.inference import Detector  # noqa: E402
from tensor_factory_train.model import TinyDetector, soft_argmax_xyxy  # noqa: E402
from tensor_factory_train.train import _flip_box, export_onnx, fit  # noqa: E402


@pytest.mark.unit
def test_flip_box_is_a_mirror_involution():
    box = (0.1, 0.2, 0.6, 0.9)  # asymmetric so reflections are observable
    assert _flip_box(box, horizontal=True) == pytest.approx((0.4, 0.2, 0.9, 0.9))
    assert _flip_box(box, horizontal=False) == pytest.approx((0.1, 0.1, 0.6, 0.8))
    # flipping twice on the same axis restores the original
    once = _flip_box(box, horizontal=True)
    assert _flip_box(once, horizontal=True) == pytest.approx(box)


@pytest.mark.unit
def test_gain_sharpens_soft_argmax_toward_peak():
    # An off-centre peak (cell 25 of 30, i.e. ~0.86) on every edge channel. A plain softmax
    # (gain 1) over a near-flat logit field is pulled toward the 0.5 centre; raising the gain
    # sharpens it so the marginal expectation climbs toward the true peak. This is exactly
    # the centre-bias the gain exists to fix.
    h = w = 30
    heat = torch.zeros(1, 4, h, w)
    heat[:, :, 25, 25] = 1.0  # modest peak -> diffuse at gain 1, sharp at high gain
    low = soft_argmax_xyxy(heat, gain=1.0)
    high = soft_argmax_xyxy(heat, gain=20.0)
    peak = 25 / (h - 1)
    # Every coordinate moves strictly closer to the peak as the gain rises.
    assert bool((torch.abs(high - peak) < torch.abs(low - peak)).all())
    # At high gain it lands near the peak; at gain 1 it sits well short (centre-pulled).
    assert bool((torch.abs(high - peak) < 0.05).all())
    assert bool((low < high).all())


@pytest.mark.unit
def test_learnable_gain_starts_at_unity():
    # log_gain initialized to 0 -> exp -> 1.0, so a fresh model reproduces a plain softmax
    # and any previously-trained behaviour is unchanged at init.
    model = TinyDetector(width=8)
    assert float(model.log_gain.exp()) == pytest.approx(1.0)


@pytest.mark.unit
def test_forward_shape_and_finite():
    model = TinyDetector(width=8)
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 4)
    # Raw xyxy can sit just outside [0, 1] pre-clamp; inference clamps via BBox.
    assert bool(torch.isfinite(out).all())


@pytest.mark.unit
def test_export_int8_onnx_is_loadable(tmp_path):
    model = TinyDetector(width=8)
    out = export_onnx(model, tmp_path / "m.onnx", size=64)
    assert out.exists()
    # The exported model must satisfy the inference contract end to end.
    det = Detector(out, input_size=64)
    box = det.detect_box(Image.new("RGB", (64, 64), (128, 128, 128)))
    assert 0.0 <= box.x1 <= 1.0
    assert box.x1 <= box.x2
    assert not det.has_classes  # box-only model exposes no class head


@pytest.mark.unit
def test_fit_refuses_all_pending_dataset(tmp_path):
    import json

    # A fresh auto-labeled dataset: every annotation pending human review.
    coco = {
        "images": [{"id": 1, "file_name": "images/a.png", "width": 64, "height": 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 16, 16], "review": "pending"}
        ],
        "categories": [{"id": 1, "name": "helicoil"}],
    }
    (tmp_path / "annotations.coco.json").write_text(json.dumps(coco))
    # The gate refuses to train -- and says why -- before touching any images.
    with pytest.raises(ValueError, match="pending human review"):
        fit(tmp_path, tmp_path / "m.onnx", epochs=1, size=64)


@pytest.mark.unit
def test_class_head_forward_shapes():
    model = TinyDetector(width=8, num_classes=2)
    box, logits = model(torch.zeros(3, 3, 64, 64))
    assert box.shape == (3, 4)
    assert logits.shape == (3, 2)


@pytest.mark.unit
def test_export_and_detect_class_roundtrip(tmp_path):
    model = TinyDetector(width=8, num_classes=2)
    out = export_onnx(model, tmp_path / "m.onnx", size=64)
    det = Detector(out, input_size=64)
    assert det.has_classes
    box, cls, score = det.detect(Image.new("RGB", (64, 64), (128, 128, 128)))
    assert box.x1 <= box.x2
    assert cls in (0, 1)
    assert 0.0 <= score <= 1.0
    # box-only API still works on a class-head model
    assert det.detect_box(Image.new("RGB", (64, 64), (10, 10, 10))).x1 <= 1.0
