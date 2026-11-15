import json
from pathlib import Path

import pytest

from tensor_factory.geometry import BBox
from tensor_factory_train.data import (
    baseline_center_err,
    load_coco_boxes,
    load_coco_labeled,
    load_negatives,
)


def _box_at(cx: float, cy: float, half: float = 0.05) -> BBox:
    """A small box centered at (cx, cy) in normalized coords."""
    return BBox(cx - half, cy - half, cx + half, cy + half)


@pytest.mark.unit
def test_load_coco_boxes_maps_paths_and_boxes(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "images/a.png", "width": 480, "height": 480},
            {"id": 2, "file_name": "images/b.png", "width": 200, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [48.0, 96.0, 240.0, 192.0],
                "review": "approved",
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [20.0, 10.0, 100.0, 50.0],
                "review": "approved",
            },
        ],
        "categories": [{"id": 1, "name": "helicoil"}],
    }
    p = tmp_path / "annotations.coco.json"
    p.write_text(json.dumps(coco))

    items = load_coco_boxes(p, tmp_path)
    assert len(items) == 2
    path, box = items[0]
    assert path == tmp_path / "images/a.png"
    assert box == BBox.from_pixels(48, 96, 288, 288, width=480, height=480)


@pytest.mark.unit
def test_load_coco_boxes_empty(tmp_path):
    coco = {"images": [], "annotations": [], "categories": []}
    p = tmp_path / "annotations.coco.json"
    p.write_text(json.dumps(coco))
    assert load_coco_boxes(p, tmp_path) == []


@pytest.mark.unit
def test_load_coco_labeled_maps_class_ids(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "images/a.png", "width": 480, "height": 480},
            {"id": 2, "file_name": "images/b.png", "width": 480, "height": 480},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [48.0, 96.0, 240.0, 192.0],
                "review": "approved",
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [0.0, 0.0, 240.0, 240.0],
                "review": "approved",
            },
        ],
        # category ids deliberately not 0-indexed; loader must map them to 0,1 by id order.
        "categories": [{"id": 1, "name": "helicoil"}, {"id": 2, "name": "incorrect"}],
    }
    p = tmp_path / "annotations.coco.json"
    p.write_text(json.dumps(coco))

    items, names = load_coco_labeled(p, tmp_path)
    assert names == ["helicoil", "incorrect"]
    assert [label for _, _, label in items] == [0, 1]
    path, box, label = items[0]
    assert path == tmp_path / "images/a.png"
    assert box == BBox.from_pixels(48, 96, 288, 288, width=480, height=480)
    assert label == 0


def _mixed_review_coco():
    """Three annotations: approved, pending, and missing-review (untrusted by default)."""
    return {
        "images": [
            {"id": i, "file_name": f"images/{i}.png", "width": 480, "height": 480}
            for i in (1, 2, 3)
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "review": "approved",
            },
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "review": "pending"},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 10, 10]},  # no review key
        ],
        "categories": [{"id": 1, "name": "helicoil"}],
    }


@pytest.mark.unit
def test_load_negatives_finds_images_in_pool_layout(tmp_path):
    # Pool layout <dir>/images/*.png (what gen_negatives.py writes).
    imgs = tmp_path / "images"
    imgs.mkdir()
    for name in ("neg_00001.png", "neg_00000.png", "b.JPG", "notes.txt"):
        (imgs / name).write_bytes(b"x")
    found = load_negatives(tmp_path)
    # Only images, sorted, with absolute paths -- no .txt.
    assert [p.name for p in found] == ["b.JPG", "neg_00000.png", "neg_00001.png"]
    assert all(p.is_absolute() for p in found)


@pytest.mark.unit
def test_load_negatives_flat_dir(tmp_path):
    # No images/ subdir -> scan the dir directly.
    (tmp_path / "a.png").write_bytes(b"x")
    assert [p.name for p in load_negatives(tmp_path)] == ["a.png"]


@pytest.mark.unit
def test_review_gate_loads_only_approved_by_default(tmp_path):
    p = tmp_path / "annotations.coco.json"
    p.write_text(json.dumps(_mixed_review_coco()))
    # The safety property: pending and unmarked annotations never train by default.
    items = load_coco_boxes(p, tmp_path)
    assert len(items) == 1
    assert items[0][0] == tmp_path / "images/1.png"


@pytest.mark.unit
def test_review_gate_can_be_disabled(tmp_path):
    p = tmp_path / "annotations.coco.json"
    p.write_text(json.dumps(_mixed_review_coco()))
    assert len(load_coco_boxes(p, tmp_path, require_review=False)) == 3
    items, _ = load_coco_labeled(p, tmp_path, require_review=False)
    assert len(items) == 3


@pytest.mark.unit
def test_review_gate_applies_to_labeled_loader(tmp_path):
    p = tmp_path / "annotations.coco.json"
    p.write_text(json.dumps(_mixed_review_coco()))
    items, _ = load_coco_labeled(p, tmp_path)
    assert len(items) == 1


@pytest.mark.unit
def test_baseline_center_err_fits_on_train_scores_on_val():
    # Constant is the *train* mean center (0.5, 0.5); val centers sit 0/0.1/0.2 away in x.
    # Distances at size=480: 0, 48, 96 px -> median 48. Proves it's median, not mean (= 48).
    train = [(Path("t.png"), _box_at(0.5, 0.5)) for _ in range(4)]
    val = [
        (Path("v0.png"), _box_at(0.5, 0.5)),
        (Path("v1.png"), _box_at(0.6, 0.5)),
        (Path("v2.png"), _box_at(0.7, 0.5)),
    ]
    assert baseline_center_err(train, val, size=480) == pytest.approx(48.0)


@pytest.mark.unit
def test_baseline_center_err_uses_train_mean_not_val_mean():
    # Train centers average to x=0.3; a val box exactly at 0.3 must score ~0, proving the
    # constant comes from train (a val-fit constant would instead center on the val mean).
    train = [(Path("a.png"), _box_at(0.2, 0.5)), (Path("b.png"), _box_at(0.4, 0.5))]
    val = [(Path("v.png"), _box_at(0.3, 0.5))]
    assert baseline_center_err(train, val, size=480) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_baseline_center_err_skips_none_boxes_and_handles_empty():
    # Negatives (None box) are skipped, not crashed on.
    train = [(Path("p.png"), _box_at(0.5, 0.5)), (Path("neg.png"), None)]
    val = [(Path("v.png"), _box_at(0.6, 0.5)), (Path("vneg.png"), None)]
    assert baseline_center_err(train, val, size=480) == pytest.approx(48.0)
    # No box-bearing item in a split -> no comparison possible.
    assert baseline_center_err(train, [(Path("x.png"), None)], size=480) is None
    assert baseline_center_err([(Path("x.png"), None)], val, size=480) is None
