"""Multi-source training: several dataset dirs train as one set and are scored apart.

The loader/split/summary tests are torch-free. The end-to-end ``fit`` tests run real
training on tiny images and are skipped without the 'train' extra.
"""

import json
import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tensor_factory_train.data import load_sources, source_names


def _write_dataset(
    root: Path,
    n: int,
    *,
    category: tuple[int, str] = (1, "helicoil"),
    review: str = "approved",
    seed: int = 0,
    x0: int = 8,
) -> Path:
    """A COCO dataset of ``n`` 64px noise frames, each with one white square to find."""
    rng = random.Random(seed)
    (root / "images").mkdir(parents=True)
    images, annotations = [], []
    for i in range(n):
        img = Image.new("RGB", (64, 64))
        img.putdata([(rng.randrange(256),) * 3 for _ in range(64 * 64)])
        x = x0 + (i % 4) * 6
        ImageDraw.Draw(img).rectangle([x, 20, x + 20, 40], fill=(255, 255, 255))
        img.save(root / "images" / f"{i}.png")
        images.append({"id": i, "file_name": f"images/{i}.png", "width": 64, "height": 64})
        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": category[0],
                "bbox": [x, 20, 20, 20],
                "review": review,
            }
        )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category[0], "name": category[1]}],
    }
    (root / "annotations.coco.json").write_text(json.dumps(coco))
    return root


def _write_negatives(root: Path, n: int) -> Path:
    rng = random.Random(99)
    (root / "images").mkdir(parents=True)
    for i in range(n):
        img = Image.new("RGB", (64, 64))
        img.putdata([(rng.randrange(256),) * 3 for _ in range(64 * 64)])
        img.save(root / "images" / f"n{i}.png")
    return root


# --- naming -------------------------------------------------------------------------


@pytest.mark.unit
def test_source_names_are_basenames():
    assert source_names(["/data/real", "/data/synthetic"]) == ["real", "synthetic"]


@pytest.mark.unit
def test_source_names_disambiguate_shared_basenames():
    # Two different datasets must never merge into one metrics row.
    names = source_names(["/a/frames", "/b/frames", "/c/frames/"])
    assert names == ["frames", "frames-2", "frames-3"]
    assert len(set(names)) == 3


@pytest.mark.unit
def test_source_names_survive_a_basename_that_looks_like_a_suffix():
    # "frames-2" already exists when the second "frames" needs a name.
    names = source_names(["/a/frames", "/x/frames-2", "/b/frames"])
    assert len(set(names)) == 3


# --- loading ------------------------------------------------------------------------


@pytest.mark.unit
def test_load_sources_tags_each_item_with_its_source(tmp_path):
    a = _write_dataset(tmp_path / "a", 3)
    b = _write_dataset(tmp_path / "b", 5)
    items, names = load_sources([a, b])
    assert names == ["helicoil"]
    assert [it[3] for it in items] == [0, 0, 0, 1, 1, 1, 1, 1]
    # read in place: paths still point into each source dir, nothing was copied
    assert all(it[0].parent == (a if it[3] == 0 else b) / "images" for it in items)


@pytest.mark.unit
def test_load_sources_merges_classes_by_name_not_by_id(tmp_path):
    # Same name under different COCO ids -> one class. Same id, different name -> two.
    a = _write_dataset(tmp_path / "a", 2, category=(1, "helicoil"))
    b = _write_dataset(tmp_path / "b", 2, category=(7, "helicoil"))
    c = _write_dataset(tmp_path / "c", 2, category=(1, "burr"))
    items, names = load_sources([a, b, c])
    assert names == ["helicoil", "burr"]
    labels = {src: {it[2] for it in items if it[3] == src} for src in (0, 1, 2)}
    assert labels == {0: {0}, 1: {0}, 2: {1}}


@pytest.mark.unit
def test_load_sources_applies_the_review_gate_per_source(tmp_path):
    a = _write_dataset(tmp_path / "a", 2)
    b = _write_dataset(tmp_path / "b", 3, review="pending")
    gated, _ = load_sources([a, b])
    assert {it[3] for it in gated} == {0}
    everything, _ = load_sources([a, b], require_review=False)
    assert len(everything) == 5


@pytest.mark.unit
def test_load_sources_reports_a_missing_dataset(tmp_path):
    a = _write_dataset(tmp_path / "a", 2)
    with pytest.raises(FileNotFoundError):
        load_sources([a, tmp_path / "nope"])


# --- torch-backed: split, summaries, fit ---------------------------------------------

torch = pytest.importorskip("torch")

from tensor_factory.inference import Detector  # noqa: E402
from tensor_factory_train.cli import _build_parser  # noqa: E402
from tensor_factory_train.train import (  # noqa: E402
    _per_source,
    _split,
    _split_by_source,
    _summarize,
    fit,
    resolve_device,
)


def _items(counts: list[int]) -> list[tuple]:
    return [
        (Path(f"s{src}/{i}.png"), None, 0, src) for src, n in enumerate(counts) for i in range(n)
    ]


@pytest.mark.unit
def test_split_by_source_holds_out_part_of_every_source():
    items = _items([40, 5, 10])
    train, val = _split_by_source(items, 0.2, seed=0)
    assert sorted(train + val) == sorted(items)  # a partition: nothing lost, nothing doubled
    assert not set(train) & set(val)
    for src, n in enumerate([40, 5, 10]):
        held = sum(1 for it in val if it[3] == src)
        assert 1 <= held < n, f"source {src}: {held} of {n} held out"


@pytest.mark.unit
def test_split_by_source_keeps_a_single_item_source_in_train():
    # One global shuffle could put a small source's only item in val, leaving none to learn.
    train, val = _split_by_source(_items([10, 1]), 0.2, seed=0)
    assert [it for it in train if it[3] == 1] == _items([0, 1])
    assert all(it[3] == 0 for it in val)


@pytest.mark.unit
def test_split_by_source_is_deterministic_and_seeded():
    items = _items([20, 20])
    assert _split_by_source(items, 0.25, seed=3) == _split_by_source(items, 0.25, seed=3)
    assert _split_by_source(items, 0.25, seed=3) != _split_by_source(items, 0.25, seed=4)


@pytest.mark.unit
def test_split_by_source_with_no_val_frac_trains_on_everything():
    items = _items([3, 4])
    train, val = _split_by_source(items, 0.0, seed=0)
    assert sorted(train) == sorted(items)
    assert val == []


@pytest.mark.unit
def test_summarize_ignores_not_applicable_records():
    records = [(10.0, True, None), (30.0, False, None), (None, True, None)]
    err, acc, cls_acc = _summarize(records)
    assert err == 20.0  # median over the two box-bearing records only
    assert acc == pytest.approx(2 / 3)
    assert cls_acc is None


@pytest.mark.unit
def test_summarize_of_nothing():
    assert _summarize([]) == (0.0, None, None)


@pytest.mark.unit
def test_per_source_reports_none_not_zero_without_a_held_out_box():
    train = _items([4, 4])
    val = [(Path("a.png"), "box", 0, 0), (Path("n.png"), None, 0, 1)]
    records = [(12.0, True, None), (None, False, None)]
    data, neg = _per_source(["real", "negatives"], train, val, records)
    assert data == {
        "name": "real",
        "train": 4,
        "val": 1,
        "val_err": 12.0,
        "presence_acc": 1.0,
        "class_acc": None,
    }
    assert neg["val_err"] is None  # a negatives pool has no box: no error, not a perfect one
    assert neg["presence_acc"] == 0.0


@pytest.mark.unit
def test_cli_data_takes_several_dirs():
    args = _build_parser().parse_args(["fit", "--data", "a", "b", "--epochs", "1"])
    assert args.data == ["a", "b"]
    assert _build_parser().parse_args(["fit", "--data", "a"]).data == ["a"]


@pytest.mark.unit
def test_fit_trains_on_several_sources_and_scores_each(tmp_path, capsys):
    a = _write_dataset(tmp_path / "real", 8, seed=1)
    b = _write_dataset(tmp_path / "synthetic", 6, seed=2, x0=30)
    neg = _write_negatives(tmp_path / "empty", 4)
    seen: list[dict] = []
    out = fit(
        [a, b],
        tmp_path / "m.onnx",
        epochs=2,
        size=64,
        width=8,
        batch=4,
        device="cpu",
        val_frac=0.25,
        negatives=[neg],
        on_epoch=seen.append,
    )

    assert len(seen) == 2
    sources = seen[-1]["sources"]
    assert [s["name"] for s in sources] == ["real", "synthetic", "empty"]
    # every image is accounted for exactly once, in the source it came from
    assert [s["train"] + s["val"] for s in sources] == [8, 6, 4]
    assert sum(s["train"] for s in sources) == seen[-1]["train_count"]
    assert sum(s["val"] for s in sources) == seen[-1]["val_count"]
    # each source was held out of and scored on its own data
    assert all(s["val"] >= 1 for s in sources)
    assert sources[0]["val_err"] is not None and sources[1]["val_err"] is not None
    assert sources[2]["val_err"] is None
    assert all(s["presence_acc"] is not None for s in sources)

    printed = capsys.readouterr().out
    assert "triage [real]" in printed and "triage [synthetic]" in printed
    assert "by source (best checkpoint):" in printed

    det = Detector(out, input_size=64)  # the usual int8 model falls out
    assert det.has_presence


@pytest.mark.unit
def test_fit_on_one_dir_keeps_its_original_split(tmp_path):
    # A single dataset must split exactly as it did before sources existed.
    a = _write_dataset(tmp_path / "only", 10)
    seen: list[dict] = []
    fit(
        a, tmp_path / "m.onnx", epochs=1, size=64, width=8, device="cpu",
        val_frac=0.3, seed=5, on_epoch=seen.append,
    )  # fmt: skip
    items, _ = load_sources([a])
    train, val = _split(list(items), 0.3, 5)
    assert (seen[0]["train_count"], seen[0]["val_count"]) == (len(train), len(val)) == (7, 3)
    assert [s["name"] for s in seen[0]["sources"]] == ["only"]


@pytest.mark.unit
def test_fit_merges_differently_named_sources_into_a_class_head(tmp_path):
    a = _write_dataset(tmp_path / "a", 6, category=(1, "helicoil"))
    b = _write_dataset(tmp_path / "b", 6, category=(1, "burr"), x0=30)
    seen: list[dict] = []
    out = fit(
        [a, b], tmp_path / "m.onnx", epochs=1, size=64, width=8, device="cpu",
        presence=True, val_frac=0.34, on_epoch=seen.append,
    )  # fmt: skip
    assert seen[0]["num_classes"] == 2
    assert Detector(out, input_size=64).has_class


@pytest.mark.unit
def test_fit_refuses_when_every_source_is_pending(tmp_path):
    a = _write_dataset(tmp_path / "a", 2, review="pending")
    b = _write_dataset(tmp_path / "b", 3, review="pending")
    with pytest.raises(ValueError, match="5 pending human review"):
        fit([a, b], tmp_path / "m.onnx", epochs=1, size=64)


@pytest.mark.unit
def test_fit_refuses_an_empty_source_list(tmp_path):
    with pytest.raises(ValueError, match="no dataset dir"):
        fit([], tmp_path / "m.onnx", epochs=1, size=64)


# --- device pinning --------------------------------------------------------------------


@pytest.mark.unit
def test_device_env_var_pins_the_machine(monkeypatch):
    monkeypatch.setenv("TENSOR_FACTORY_DEVICE", "cpu")
    assert resolve_device() == "cpu"
    assert resolve_device(None) == "cpu"


@pytest.mark.unit
def test_explicit_device_beats_the_env_var(monkeypatch):
    monkeypatch.setenv("TENSOR_FACTORY_DEVICE", "cpu")
    assert resolve_device("mps") == "mps"


@pytest.mark.unit
def test_device_env_var_is_validated(monkeypatch):
    monkeypatch.setenv("TENSOR_FACTORY_DEVICE", "gpu0; rm -rf")
    with pytest.raises(ValueError, match="TENSOR_FACTORY_DEVICE"):
        resolve_device()


@pytest.mark.unit
def test_blank_device_env_var_falls_through_to_autodetect(monkeypatch):
    monkeypatch.setenv("TENSOR_FACTORY_DEVICE", "  ")
    assert resolve_device() in ("cuda", "mps", "cpu")
