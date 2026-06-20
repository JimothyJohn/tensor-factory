import pytest
from hypothesis import given
from hypothesis import strategies as st

from helicoils.geometry import BBox


@st.composite
def boxes(draw):
    xs = sorted([draw(st.floats(0, 1)), draw(st.floats(0, 1))])
    ys = sorted([draw(st.floats(0, 1)), draw(st.floats(0, 1))])
    return BBox(xs[0], ys[0], xs[1], ys[1])


@pytest.mark.unit
def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        BBox(-0.1, 0.0, 0.5, 0.5)


@pytest.mark.unit
def test_rejects_inverted_coords():
    with pytest.raises(ValueError):
        BBox(0.6, 0.1, 0.2, 0.5)


@pytest.mark.unit
def test_immutable():
    b = BBox(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(AttributeError):
        b.x1 = 0.5  # type: ignore[misc]


@pytest.mark.unit
def test_clamped_fixes_range_and_order():
    b = BBox.clamped(1.2, 0.8, -0.3, 0.2)
    assert (b.x1, b.y1, b.x2, b.y2) == (0.0, 0.2, 1.0, 0.8)


@pytest.mark.unit
def test_area_and_center():
    b = BBox(0.0, 0.0, 0.5, 0.4)
    assert b.area == pytest.approx(0.2)
    assert b.center == pytest.approx((0.25, 0.2))


@pytest.mark.unit
def test_iou_identical_is_one():
    b = BBox(0.1, 0.1, 0.9, 0.9)
    assert b.iou(b) == pytest.approx(1.0)


@pytest.mark.unit
def test_iou_disjoint_is_zero():
    a = BBox(0.0, 0.0, 0.2, 0.2)
    b = BBox(0.5, 0.5, 0.7, 0.7)
    assert a.iou(b) == 0.0


@pytest.mark.unit
def test_iou_known_value():
    a = BBox(0.0, 0.0, 0.5, 0.5)  # area 0.25
    b = BBox(0.25, 0.25, 0.75, 0.75)  # area 0.25, overlap 0.25 x 0.25
    inter = 0.25 * 0.25
    union = 0.25 + 0.25 - inter
    assert a.iou(b) == pytest.approx(inter / union)


@pytest.mark.unit
@given(boxes())
def test_cxcywh_roundtrip(b):
    cx, cy, w, h = b.to_cxcywh()
    b2 = BBox.from_cxcywh(cx, cy, w, h)
    assert b2.x1 == pytest.approx(b.x1, abs=1e-9)
    assert b2.y1 == pytest.approx(b.y1, abs=1e-9)
    assert b2.x2 == pytest.approx(b.x2, abs=1e-9)
    assert b2.y2 == pytest.approx(b.y2, abs=1e-9)


@pytest.mark.unit
@given(boxes(), st.integers(16, 4096), st.integers(16, 4096))
def test_pixels_roundtrip_within_one_pixel(b, width, height):
    px = b.to_pixels(width=width, height=height)
    b2 = BBox.from_pixels(*px, width=width, height=height)
    assert abs(b2.x1 - b.x1) * width <= 1.0
    assert abs(b2.y1 - b.y1) * height <= 1.0
    assert abs(b2.x2 - b.x2) * width <= 1.0
    assert abs(b2.y2 - b.y2) * height <= 1.0
