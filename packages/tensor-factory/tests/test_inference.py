"""Inference plumbing tests using a hand-built constant ONNX model (no torch)."""

import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402
from PIL import Image  # noqa: E402

from tensor_factory import BBox, encode_uint8  # noqa: E402
from tensor_factory.inference import Detector, benchmark  # noqa: E402


def _make_const_model(path, coords, size=8):
    """An ONNX model that ignores its input and emits a fixed (1, 4) box."""
    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, size, size])
    box = helper.make_tensor_value_info("box", TensorProto.FLOAT, [1, 4])
    const = helper.make_node(
        "Constant",
        [],
        ["box"],
        value=helper.make_tensor("c", TensorProto.FLOAT, [1, 4], list(coords)),
    )
    graph = helper.make_graph([const], "const_detector", [image], [box])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.mark.unit
def test_detect_box_decodes_model_output(tmp_path):
    _make_const_model(tmp_path / "m.onnx", (0.1, 0.2, 0.6, 0.8), size=8)
    det = Detector(tmp_path / "m.onnx", input_size=8)
    img = Image.new("RGB", (20, 20), (128, 128, 128))
    box = det.detect_box(img)
    expected = BBox.clamped(0.1, 0.2, 0.6, 0.8)
    assert box.x1 == pytest.approx(expected.x1, abs=1e-5)
    assert box.y1 == pytest.approx(expected.y1, abs=1e-5)
    assert box.x2 == pytest.approx(expected.x2, abs=1e-5)
    assert box.y2 == pytest.approx(expected.y2, abs=1e-5)


@pytest.mark.unit
def test_detect_uint8_matches_codec(tmp_path):
    _make_const_model(tmp_path / "m.onnx", (0.1, 0.2, 0.6, 0.8), size=8)
    det = Detector(tmp_path / "m.onnx", input_size=8)
    img = Image.new("RGB", (32, 16), (10, 20, 30))
    assert det.detect_uint8(img) == encode_uint8(BBox.clamped(0.1, 0.2, 0.6, 0.8))


@pytest.mark.unit
def test_preprocess_shape(tmp_path):
    _make_const_model(tmp_path / "m.onnx", (0.0, 0.0, 1.0, 1.0), size=8)
    det = Detector(tmp_path / "m.onnx", input_size=8)
    x = det.preprocess(Image.new("RGB", (50, 70), (0, 0, 0)))
    assert x.shape == (1, 3, 8, 8)
    assert x.dtype.name == "float32"


@pytest.mark.unit
def test_benchmark_returns_positive_fps(tmp_path):
    _make_const_model(tmp_path / "m.onnx", (0.0, 0.0, 1.0, 1.0), size=8)
    det = Detector(tmp_path / "m.onnx", input_size=8)
    fps = benchmark(det, Image.new("RGB", (8, 8), (0, 0, 0)), n=10, warmup=2)
    assert fps > 0
