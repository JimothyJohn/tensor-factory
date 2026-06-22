import json

import pytest

from tensor_factory.geometry import BBox
from tensor_factory_train.data import load_coco_boxes, load_coco_labeled, load_negatives


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
