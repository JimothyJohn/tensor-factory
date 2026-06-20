"""Model + export tests. Skipped unless torch is installed (the 'train' extra)."""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from helicoils.inference import Detector  # noqa: E402
from helicoils_train.model import TinyDetector  # noqa: E402
from helicoils_train.train import export_onnx  # noqa: E402


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
