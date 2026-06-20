import pytest
from hypothesis import given
from hypothesis import strategies as st

from helicoils.formats import (
    from_coco_bbox,
    from_voc_box,
    from_yolo,
    to_coco_bbox,
    to_voc_box,
    to_yolo,
)
from helicoils.geometry import BBox


@st.composite
def boxes(draw):
    xs = sorted([draw(st.floats(0, 1)), draw(st.floats(0, 1))])
    ys = sorted([draw(st.floats(0, 1)), draw(st.floats(0, 1))])
    return BBox(xs[0], ys[0], xs[1], ys[1])


sizes = st.integers(16, 4096)


@pytest.mark.unit
def test_coco_known_value():
    b = BBox(0.1, 0.2, 0.6, 0.8)
    assert to_coco_bbox(b, width=100, height=200) == pytest.approx([10.0, 40.0, 50.0, 120.0])


@pytest.mark.unit
@given(boxes(), sizes, sizes)
def test_coco_roundtrip_lossless(b, width, height):
    coco = to_coco_bbox(b, width=width, height=height)
    b2 = from_coco_bbox(coco, width=width, height=height)
    assert b2.x1 == pytest.approx(b.x1, abs=1e-6)
    assert b2.y1 == pytest.approx(b.y1, abs=1e-6)
    assert b2.x2 == pytest.approx(b.x2, abs=1e-6)
    assert b2.y2 == pytest.approx(b.y2, abs=1e-6)


@pytest.mark.unit
@given(boxes())
def test_yolo_roundtrip_lossless(b):
    cx, cy, w, h = to_yolo(b)
    b2 = from_yolo(cx, cy, w, h)
    assert b2.x1 == pytest.approx(b.x1, abs=1e-9)
    assert b2.y1 == pytest.approx(b.y1, abs=1e-9)
    assert b2.x2 == pytest.approx(b.x2, abs=1e-9)
    assert b2.y2 == pytest.approx(b.y2, abs=1e-9)


@pytest.mark.unit
@given(boxes(), sizes, sizes)
def test_voc_roundtrip_within_one_pixel(b, width, height):
    voc = to_voc_box(b, width=width, height=height)
    b2 = from_voc_box(voc, width=width, height=height)
    assert abs(b2.x1 - b.x1) * width <= 1.0
    assert abs(b2.y1 - b.y1) * height <= 1.0
    assert abs(b2.x2 - b.x2) * width <= 1.0
    assert abs(b2.y2 - b.y2) * height <= 1.0
