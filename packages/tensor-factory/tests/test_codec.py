import pytest
from hypothesis import given
from hypothesis import strategies as st

from tensor_factory import BBox, decode_uint8, encode_uint8, max_error_px


@st.composite
def boxes(draw):
    xs = sorted([draw(st.floats(0, 1)), draw(st.floats(0, 1))])
    ys = sorted([draw(st.floats(0, 1)), draw(st.floats(0, 1))])
    return BBox(xs[0], ys[0], xs[1], ys[1])


@pytest.mark.unit
def test_max_error_within_three_pixel_budget():
    # The whole premise: 8-bit coords land within 3 px on a 480 image.
    assert max_error_px(480) < 3.0
    assert max_error_px(480) == pytest.approx(480 / 255 / 2)


@pytest.mark.unit
def test_encode_corners():
    assert encode_uint8(BBox(0.0, 0.0, 1.0, 1.0)) == (0, 0, 255, 255)


@pytest.mark.unit
@given(boxes())
def test_roundtrip_within_half_step_at_480(b):
    decoded = decode_uint8(encode_uint8(b))
    budget = max_error_px(480) + 1e-6
    assert abs(decoded.x1 - b.x1) * 480 <= budget
    assert abs(decoded.y1 - b.y1) * 480 <= budget
    assert abs(decoded.x2 - b.x2) * 480 <= budget
    assert abs(decoded.y2 - b.y2) * 480 <= budget


@pytest.mark.unit
@given(
    st.integers(0, 255),
    st.integers(0, 255),
    st.integers(0, 255),
    st.integers(0, 255),
)
def test_decode_accepts_any_uint8_and_repairs_order(a, b, c, d):
    box = decode_uint8((a, b, c, d))
    assert box.x1 <= box.x2
    assert box.y1 <= box.y2


@pytest.mark.unit
@pytest.mark.parametrize("bad", [(0, 0, 0), (0, 0, 0, 0, 0)])
def test_decode_rejects_wrong_length(bad):
    with pytest.raises(ValueError):
        decode_uint8(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [(-1, 0, 0, 0), (0, 256, 0, 0)])
def test_decode_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        decode_uint8(bad)
