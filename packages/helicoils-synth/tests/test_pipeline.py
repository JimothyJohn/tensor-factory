import json

import pytest

from helicoils_synth.cli import main
from helicoils_synth.generator import MockGenerator
from helicoils_synth.pipeline import synth_dataset


@pytest.mark.unit
def test_synth_dataset_writes_full_bundle(tmp_path):
    records = synth_dataset(
        MockGenerator(),
        "macro helicoil",
        ["helicoil"],
        n=3,
        out_dir=tmp_path,
        size=64,
    )
    assert len(records) == 3
    # Mock provides ground truth, so every image is labeled.
    assert all(len(dets) == 1 for *_, dets in records)

    images = list((tmp_path / "images").glob("*.png"))
    assert len(images) == 3

    coco = json.loads((tmp_path / "annotations.coco.json").read_text())
    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3
    assert (tmp_path / "label_studio.json").exists()


@pytest.mark.unit
def test_synth_dataset_requires_features(tmp_path):
    with pytest.raises(ValueError, match="feature"):
        synth_dataset(MockGenerator(), "p", [], n=1, out_dir=tmp_path, size=64)


@pytest.mark.unit
def test_cli_dataset_smoke(tmp_path):
    out = tmp_path / "ds"
    rc = main(
        [
            "--size",
            "64",
            "dataset",
            "--prompt",
            "macro helicoil",
            "--features",
            "helicoil",
            "--n",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "annotations.coco.json").exists()
    assert len(list((out / "images").glob("*.png"))) == 2


@pytest.mark.unit
def test_cli_sample_smoke(tmp_path):
    out = tmp_path / "grid.png"
    rc = main(
        ["--size", "64", "sample", "--prompt", "macro helicoil", "--n", "4", "--out", str(out)]
    )
    assert rc == 0
    assert out.exists()
