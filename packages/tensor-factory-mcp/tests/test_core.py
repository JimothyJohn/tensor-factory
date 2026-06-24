import pytest
from PIL import Image

from tensor_factory_mcp import core


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("RGB", (200, 150), (128, 128, 128)).save(p)
    return str(p)


@pytest.fixture
def mock_model():
    # The box-only demo model is still bundled alongside the default presence model, so
    # tests can exercise the no-class-head path explicitly.
    return str(core.default_model_path().parent / "helicoil-mock-v1.onnx")


@pytest.mark.unit
def test_default_model_exists():
    assert core.default_model_path().exists()


@pytest.mark.unit
def test_detect_box_only_model_always_has_a_box(image, mock_model):
    # A box-only model (no presence head): no present/score keys, box always populated.
    out = core.detect(image, model_path=mock_model)
    assert "present" not in out and "score" not in out
    assert out["box_norm"] is not None and len(out["uint8"]) == 4


@pytest.mark.unit
def test_detect_presence_model_reports_present_and_box_or_none(image):
    # The default model carries a presence head -> present/score appear, and the box is
    # either a full box (present) or null (absent) -- one box or no box, never partial.
    out = core.detect(image)
    assert isinstance(out["present"], bool)
    assert 0.0 <= out["score"] <= 1.0
    if out["present"]:
        assert out["box_norm"] is not None and len(out["uint8"]) == 4
    else:
        assert out["box_norm"] is None and out["box_pixels"] is None and out["uint8"] is None


@pytest.mark.unit
def test_detect_result_is_json_serializable_with_native_types(image):
    # Regression: detect_presence returned a numpy float64, so `present` was numpy.bool_ --
    # which json.dumps (the MCP/HTTP serving layers) cannot encode. present/score must be
    # native python types and the whole dict must round-trip through JSON.
    import json

    out = core.detect(image)
    assert type(out["present"]) is bool
    assert type(out["score"]) is float
    json.dumps(out)  # must not raise


@pytest.mark.unit
def test_detect_shape_and_ranges(image, mock_model):
    # Use the box-only model so a box is guaranteed; exercises the box/image-size contract.
    out = core.detect(image, model_path=mock_model)
    assert {"box_norm", "box_pixels", "uint8", "image_size", "model"} <= set(out)
    assert out["image_size"] == {"width": 200, "height": 150}
    n = out["box_norm"]
    assert 0.0 <= n["x1"] <= 1.0 and n["x1"] <= n["x2"]
    assert 0.0 <= n["y1"] <= 1.0 and n["y1"] <= n["y2"]
    assert len(out["uint8"]) == 4
    assert all(0 <= v <= 255 for v in out["uint8"])
    # Pixel box must respect the actual (non-square) image dimensions.
    assert 0 <= out["box_pixels"]["x2"] <= 200
    assert 0 <= out["box_pixels"]["y2"] <= 150


@pytest.mark.unit
def test_model_info(image):
    info = core.model_info()
    assert info["input_size"] == 480
    assert "CPUExecutionProvider" in info["providers"]
    assert info["input_name"]


@pytest.mark.unit
def test_benchmark_positive(image):
    out = core.benchmark(n=5)
    assert out["fps"] > 0
    assert out["iterations"] == 5


@pytest.mark.unit
def test_resolve_model_missing_path_raises():
    with pytest.raises(FileNotFoundError, match="model not found"):
        core.resolve_model("/no/such/model.onnx")
