import io
import json

import pytest
from PIL import Image

from tensor_factory_studio.dataset import Dataset


def _png(color=(120, 130, 140), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.unit
def test_positive_writes_coco_with_review_gate(tmp_path):
    ds = Dataset(tmp_path)
    ds.upsert(1, True, [0.25, 0.5, 0.75, 1.0], _png(size=(80, 40)))
    coco = json.loads((tmp_path / "annotations.coco.json").read_text())

    assert len(coco["images"]) == 1 and len(coco["annotations"]) == 1
    img, ann = coco["images"][0], coco["annotations"][0]
    assert img["file_name"] == "images/frame_00001.png"
    assert img["review"] == "approved"
    assert ann["review"] == "approved" and ann["source"] == "human"
    # normalized [0.25,0.5,0.75,1.0] on an 80x40 image -> COCO [x,y,w,h] pixels
    assert ann["bbox"] == [20.0, 20.0, 40.0, 20.0]
    assert (tmp_path / "images" / "frame_00001.png").is_file()


@pytest.mark.unit
def test_negative_goes_to_negatives_not_coco(tmp_path):
    ds = Dataset(tmp_path)
    ds.upsert(2, False, None, _png())
    coco = json.loads((tmp_path / "annotations.coco.json").read_text())
    assert coco["images"] == [] and coco["annotations"] == []
    assert (tmp_path / "negatives" / "images" / "frame_00002.png").is_file()
    assert ds.counts() == {"positives": 0, "negatives": 1, "total": 1}


@pytest.mark.unit
def test_reupsert_flips_present_and_moves_file(tmp_path):
    ds = Dataset(tmp_path)
    ds.upsert(3, True, [0.1, 0.1, 0.2, 0.2], _png())
    assert (tmp_path / "images" / "frame_00003.png").is_file()
    # relabel the same frame as empty -> image moves to negatives/, COCO drops it
    ds.upsert(3, False, None, _png())
    assert not (tmp_path / "images" / "frame_00003.png").exists()
    assert (tmp_path / "negatives" / "images" / "frame_00003.png").is_file()
    assert ds.counts()["positives"] == 0


@pytest.mark.unit
def test_load_reconstructs_state_from_disk(tmp_path):
    ds = Dataset(tmp_path)
    ds.upsert(1, True, [0.0, 0.0, 0.5, 0.5], _png())
    ds.upsert(5, False, None, _png())
    # a fresh Dataset over the same dir must see the same samples
    again = Dataset(tmp_path)
    assert again.counts() == {"positives": 1, "negatives": 1, "total": 2}
    assert again.samples[1]["present"] is True
    assert again.samples[5]["present"] is False


@pytest.mark.unit
def test_recent_returns_ids_added_after_snapshot(tmp_path):
    ds = Dataset(tmp_path)
    for i in (1, 2, 3):
        ds.upsert(i, True, [0.1, 0.1, 0.2, 0.2], _png())
    snapshot = {1, 2}
    assert ds.recent(snapshot) == [3]


@pytest.mark.unit
def test_clear_wipes_everything(tmp_path):
    ds = Dataset(tmp_path)
    ds.upsert(1, True, [0.1, 0.1, 0.2, 0.2], _png())
    ds.upsert(2, False, None, _png())
    ds.clear()
    assert ds.counts()["total"] == 0
    assert not (tmp_path / "annotations.coco.json").exists()
    assert list((tmp_path / "images").glob("*.png")) == []
