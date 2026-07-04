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
def test_concurrent_upserts_no_lost_writes(tmp_path):
    # The real thread-safety invariant the server relies on: N threads upserting distinct
    # frames at once must all land (Dataset serializes writes with a lock). Tested directly,
    # not through the HTTP layer, so connection limits can't make it flaky.
    import concurrent.futures

    ds = Dataset(tmp_path)
    n = 40
    png = _png()

    def put(i):
        ds.upsert(i, True, [0.1, 0.1, 0.2, 0.2], png)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(put, range(n)))

    assert ds.counts() == {"positives": n, "negatives": 0, "total": n}
    # every image file and the reconstructed view agree
    assert len(list((tmp_path / "images").glob("frame_*.png"))) == n
    again = Dataset(tmp_path)
    assert again.counts()["total"] == n


@pytest.mark.unit
def test_default_is_single_class(tmp_path):
    # A fresh dataset is single-class so the box-only training path is unchanged: one
    # "object" category, and a positive upserted with no cls lands in it.
    ds = Dataset(tmp_path)
    assert ds.classes == ["object"]
    ds.upsert(1, True, [0.1, 0.1, 0.2, 0.2], _png())
    coco = json.loads((tmp_path / "annotations.coco.json").read_text())
    assert coco["categories"] == [{"id": 1, "name": "object"}]
    assert coco["annotations"][0]["category_id"] == 1


@pytest.mark.unit
def test_class_index_written_as_coco_category(tmp_path):
    ds = Dataset(tmp_path)
    ds.set_classes(["helicoil", "bolt", "washer"])
    ds.upsert(1, True, [0.1, 0.1, 0.2, 0.2], _png(), cls=2)  # "washer" -> category_id 3
    coco = json.loads((tmp_path / "annotations.coco.json").read_text())
    assert coco["categories"] == [
        {"id": 1, "name": "helicoil"},
        {"id": 2, "name": "bolt"},
        {"id": 3, "name": "washer"},
    ]
    assert coco["annotations"][0]["category_id"] == 3  # 0-based cls 2 -> 1-based id 3


@pytest.mark.unit
def test_out_of_range_class_falls_back_to_first(tmp_path):
    ds = Dataset(tmp_path)
    ds.set_classes(["a", "b"])
    ds.upsert(1, True, [0.1, 0.1, 0.2, 0.2], _png(), cls=9)  # no such class
    assert ds.samples[1]["cls"] == 0
    coco = json.loads((tmp_path / "annotations.coco.json").read_text())
    assert coco["annotations"][0]["category_id"] == 1


@pytest.mark.unit
def test_set_classes_drops_blanks_and_never_empties(tmp_path):
    ds = Dataset(tmp_path)
    ds.set_classes(["  cat ", "", "  ", "dog"])
    assert ds.classes == ["cat", "dog"]  # trimmed, blanks removed
    ds.set_classes([])  # empty must not leave the dataset class-less
    assert ds.classes == ["object"]


@pytest.mark.unit
def test_load_reconstructs_classes_and_per_sample_class(tmp_path):
    ds = Dataset(tmp_path)
    ds.set_classes(["helicoil", "bolt"])
    ds.upsert(1, True, [0.1, 0.1, 0.2, 0.2], _png(), cls=1)
    ds.upsert(2, True, [0.3, 0.3, 0.4, 0.4], _png(), cls=0)
    # a fresh Dataset over the same dir recovers names AND each box's class index
    again = Dataset(tmp_path)
    assert again.classes == ["helicoil", "bolt"]
    assert again.samples[1]["cls"] == 1
    assert again.samples[2]["cls"] == 0


@pytest.mark.unit
def test_non_contiguous_category_ids_map_to_dense_indices(tmp_path):
    # A COCO imported from elsewhere may use sparse ids (1, 5, 9). They must collapse to
    # dense 0-based indices in load order so the trained class head lines up.
    coco = {
        "images": [
            {"id": 1, "file_name": "images/frame_00001.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 5, "bbox": [6, 6, 12, 12]},
        ],
        "categories": [
            {"id": 1, "name": "a"},
            {"id": 5, "name": "b"},
            {"id": 9, "name": "c"},
        ],
    }
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(tmp_path / "images" / "frame_00001.png")
    (tmp_path / "annotations.coco.json").write_text(json.dumps(coco))
    ds = Dataset(tmp_path)
    assert ds.classes == ["a", "b", "c"]
    assert ds.samples[1]["cls"] == 1  # category_id 5 is the 2nd category -> index 1


@pytest.mark.unit
def test_clear_wipes_everything(tmp_path):
    ds = Dataset(tmp_path)
    ds.upsert(1, True, [0.1, 0.1, 0.2, 0.2], _png())
    ds.upsert(2, False, None, _png())
    ds.clear()
    assert ds.counts()["total"] == 0
    assert not (tmp_path / "annotations.coco.json").exists()
    assert list((tmp_path / "images").glob("*.png")) == []
