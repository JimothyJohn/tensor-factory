import pytest
from PIL import Image

from tensor_factory_mcp import core


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "frame.png"
    Image.new("RGB", (200, 150), (128, 128, 128)).save(p)
    return str(p)


@pytest.mark.unit
def test_default_model_exists():
    assert core.default_model_path().exists()


@pytest.mark.unit
def test_presence_maps_class_to_name_and_flag():
    names = ["helicoil", "background"]
    assert core._presence(0, names) == ("helicoil", True)
    assert core._presence(1, names) == ("background", False)
    # No embedded names, or an out-of-range id -> undecidable.
    assert core._presence(0, None) == (None, None)
    assert core._presence(5, names) == (None, None)


@pytest.mark.unit
def test_detect_box_only_model_has_no_presence_fields(image):
    # The bundled demo is box-only (no class head): no class_*/present keys leak in.
    out = core.detect(image)
    assert "present" not in out and "class_id" not in out


@pytest.mark.unit
def test_detect_shape_and_ranges(image):
    out = core.detect(image)
    assert set(out) == {"box_norm", "box_pixels", "uint8", "image_size", "model"}
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
