import pytest

from tensor_factory.geometry import BBox
from tensor_factory_synth.generator import GeneratedSample, MockGenerator, NanoBananaGenerator


@pytest.mark.unit
def test_mock_is_deterministic():
    g = MockGenerator()
    a = g.generate("p", 7, size=64)
    b = g.generate("p", 7, size=64)
    assert a.image.tobytes() == b.image.tobytes()
    assert a.box == b.box


@pytest.mark.unit
def test_mock_differs_by_seed():
    g = MockGenerator()
    a = g.generate("p", 1, size=64)
    b = g.generate("p", 2, size=64)
    assert a.image.tobytes() != b.image.tobytes()


@pytest.mark.unit
def test_mock_sample_shape_and_box():
    g = MockGenerator()
    s = g.generate("p", 0, size=480)
    assert isinstance(s, GeneratedSample)
    assert s.image.size == (480, 480)
    assert isinstance(s.box, BBox)
    # Coil fills a large but varying fraction of the frame.
    assert s.box.area > 0.2


@pytest.mark.unit
def test_nano_banana_requires_gemini_extra():
    # google-genai is not in the default (locked) env, so construction must fail loudly.
    import importlib.util

    if importlib.util.find_spec("google.genai") is not None:
        pytest.skip("google-genai installed; cannot exercise the missing-extra path")
    with pytest.raises(RuntimeError, match="gemini"):
        NanoBananaGenerator()
