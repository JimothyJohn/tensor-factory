import pytest

from tensor_factory.formats import from_coco_bbox
from tensor_factory.geometry import BBox
from tensor_factory_synth.autolabel import Detection
from tensor_factory_synth.export import (
    build_coco,
    build_label_studio_predictions,
    write_json,
)


def _records():
    box = BBox(0.1, 0.2, 0.6, 0.8)
    return [("images/a.png", 480, 480, [Detection("helicoil", box, 0.9)])], box


@pytest.mark.unit
def test_coco_structure_and_bbox_roundtrip():
    records, box = _records()
    coco = build_coco(records, ["helicoil"])
    assert [c["name"] for c in coco["categories"]] == ["helicoil"]
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["category_id"] == 1
    assert ann["image_id"] == 1
    back = from_coco_bbox(ann["bbox"], width=480, height=480)
    assert back.x1 == pytest.approx(box.x1, abs=1e-6)
    assert back.x2 == pytest.approx(box.x2, abs=1e-6)


@pytest.mark.unit
def test_coco_carries_review_and_source_metadata():
    from tensor_factory import review

    box = BBox(0.1, 0.2, 0.6, 0.8)
    # An auto-labeler Detection defaults to pending/groundingdino -> untrainable.
    ai = [("images/a.png", 480, 480, [Detection("helicoil", box, 0.9)])]
    coco = build_coco(ai, ["helicoil"])
    assert coco["annotations"][0]["review"] == review.PENDING
    assert coco["annotations"][0]["source"] == review.GROUNDINGDINO
    assert coco["images"][0]["review"] == review.PENDING

    # An approved Detection (e.g. mock synthetic GT) rolls the image up to approved.
    approved = [
        (
            "images/b.png",
            480,
            480,
            [Detection("helicoil", box, 1.0, review=review.APPROVED, source=review.SYNTHETIC_GT)],
        )
    ]
    coco2 = build_coco(approved, ["helicoil"])
    assert coco2["annotations"][0]["review"] == review.APPROVED
    assert coco2["images"][0]["review"] == review.APPROVED


@pytest.mark.unit
def test_empty_image_is_pending_for_triage():
    from tensor_factory import review

    coco = build_coco([("images/a.png", 480, 480, [])], ["helicoil"])
    # No detection could be a missed feature -- it must be reviewed, not silently empty.
    assert coco["images"][0]["review"] == review.PENDING


@pytest.mark.unit
def test_unknown_label_falls_back_to_first_category():
    records = [("images/a.png", 480, 480, [Detection("mystery", BBox(0, 0, 1, 1), 0.5)])]
    coco = build_coco(records, ["helicoil"])
    assert coco["annotations"][0]["category_id"] == 1


@pytest.mark.unit
def test_label_studio_predictions_percent_coords():
    records, box = _records()
    tasks = build_label_studio_predictions(records)
    assert tasks[0]["data"]["image"] == "images/a.png"
    value = tasks[0]["predictions"][0]["result"][0]["value"]
    assert value["x"] == pytest.approx(10.0)
    assert value["width"] == pytest.approx(50.0)
    assert value["rectanglelabels"] == ["helicoil"]
    assert 0.0 <= value["x"] <= 100.0


@pytest.mark.unit
def test_empty_detections_score_zero():
    records = [("images/a.png", 480, 480, [])]
    tasks = build_label_studio_predictions(records)
    assert tasks[0]["predictions"][0]["score"] == 0.0


@pytest.mark.unit
def test_write_json_roundtrip(tmp_path):
    import json

    p = tmp_path / "out.json"
    write_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
