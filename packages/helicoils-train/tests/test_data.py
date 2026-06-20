import json

import pytest

from helicoils.geometry import BBox
from helicoils_train.data import load_coco_boxes


@pytest.mark.unit
def test_load_coco_boxes_maps_paths_and_boxes(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "images/a.png", "width": 480, "height": 480},
            {"id": 2, "file_name": "images/b.png", "width": 200, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [48.0, 96.0, 240.0, 192.0]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20.0, 10.0, 100.0, 50.0]},
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
