from xml.dom import minidom

import pytest

from tensor_factory_label import (
    bbox_config,
    coco_to_tasks,
    http_image_url,
    local_storage_url,
    ls_export_to_coco,
)


def _coco():
    return {
        "images": [{"id": 1, "file_name": "images/a.png", "width": 480, "height": 480}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [48.0, 96.0, 240.0, 192.0],
                "score": 0.8,
            }
        ],
        "categories": [{"id": 1, "name": "helicoil"}],
    }


@pytest.mark.unit
def test_coco_to_tasks_percentages_and_url():
    tasks = coco_to_tasks(_coco(), http_image_url("http://host/"))
    assert tasks[0]["data"]["image"] == "http://host/images/a.png"
    result = tasks[0]["predictions"][0]["result"][0]
    assert result["from_name"] == "label" and result["to_name"] == "image"
    v = result["value"]
    assert v["x"] == pytest.approx(10.0)  # 48/480
    assert v["y"] == pytest.approx(20.0)  # 96/480
    assert v["width"] == pytest.approx(50.0)  # 240/480
    assert v["rectanglelabels"] == ["helicoil"]
    assert tasks[0]["predictions"][0]["score"] == pytest.approx(0.8)


@pytest.mark.unit
def test_roundtrip_coco_to_ls_to_coco():
    coco = _coco()
    tasks = coco_to_tasks(coco, http_image_url("http://host/"))
    # A reviewer accepting the predictions == an export whose annotations == the results.
    export = [
        {"data": t["data"], "annotations": [{"result": t["predictions"][0]["result"]}]}
        for t in tasks
    ]
    back = ls_export_to_coco(export)
    assert len(back["annotations"]) == 1
    got = back["annotations"][0]["bbox"]
    for a, b in zip(got, coco["annotations"][0]["bbox"], strict=True):
        assert a == pytest.approx(b, abs=0.5)
    assert back["categories"][0]["name"] == "helicoil"
    # The pull is the human-validation step: everything it returns is trainable.
    from tensor_factory import review

    assert back["annotations"][0]["review"] == review.APPROVED
    assert back["annotations"][0]["source"] == review.HUMAN
    assert back["images"][0]["review"] == review.APPROVED
    # file_name must round-trip to the dataset-relative path the trainer loads,
    # not the wrapped image URL the push handed Label Studio.
    assert back["images"][0]["file_name"] == "images/a.png"


@pytest.mark.unit
def test_pull_recovers_dataset_relative_file_name():
    """Regression: the pull must invert the push's image-URL wrapping so COCO
    file_name matches the on-disk layout (e.g. ``images/sample_02.png``) instead of
    storing the raw URL -- otherwise the trainer can't find the images."""
    rect = {
        "type": "rectanglelabels",
        "original_width": 480,
        "original_height": 480,
        "value": {
            "x": 10.0,
            "y": 20.0,
            "width": 50.0,
            "height": 40.0,
            "rectanglelabels": ["helicoil"],
        },
    }
    # http image server (what relabel.sh uses)
    http_export = [
        {
            "data": {"image": "http://localhost:8081/images/sample_02.png"},
            "annotations": [{"result": [rect]}],
        }
    ]
    assert ls_export_to_coco(http_export)["images"][0]["file_name"] == "images/sample_02.png"
    # Label Studio local storage reference
    ls_export = [
        {
            "data": {"image": "/data/local-files/?d=images/sample_02.png"},
            "annotations": [{"result": [rect]}],
        }
    ]
    assert ls_export_to_coco(ls_export)["images"][0]["file_name"] == "images/sample_02.png"


@pytest.mark.unit
def test_ls_export_skips_cancelled_and_empty():
    export = [
        {"data": {"image": "a"}, "annotations": [{"was_cancelled": True, "result": []}]},
        {"data": {"image": "b"}, "annotations": []},
    ]
    back = ls_export_to_coco(export)
    assert back["images"] == [] and back["annotations"] == []


@pytest.mark.unit
def test_local_storage_url():
    url = local_storage_url()
    assert url("images/a.png") == "/data/local-files/?d=images/a.png"


@pytest.mark.unit
def test_bbox_config_is_valid_xml():
    xml = bbox_config(["helicoil", "thread"])
    minidom.parseString(xml)  # raises on malformed XML
    assert 'value="helicoil"' in xml and 'value="thread"' in xml
    assert 'name="label"' in xml and 'value="$image"' in xml
