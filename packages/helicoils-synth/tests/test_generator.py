import pytest

from helicoils.geometry import BBox
from helicoils_synth.generator import FluxGenerator, GeneratedSample, MockGenerator


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
    # Coil should fill a large fraction of the frame (~80% per the brief).
    assert s.box.area > 0.45


@pytest.mark.unit
def test_flux_requires_gpu_extra():
    # torch/diffusers are not in the default env, so construction must fail loudly.
    with pytest.raises(RuntimeError, match="gpu"):
        FluxGenerator()
