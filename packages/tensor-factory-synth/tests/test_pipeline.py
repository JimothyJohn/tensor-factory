import json

import pytest

from tensor_factory_synth.cli import main
from tensor_factory_synth.generator import MockGenerator
from tensor_factory_synth.pipeline import synth_dataset


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

    # Mock boxes are exact synthetic GT: approved on creation, so they are trainable
    # without a human review pass.
    from tensor_factory import review

    assert all(a["review"] == review.APPROVED for a in coco["annotations"])
    assert all(a["source"] == review.SYNTHETIC_GT for a in coco["annotations"])
    assert review.review_summary(coco)["annotations"]["trainable"] == 3


@pytest.mark.unit
def test_synth_dataset_requires_features(tmp_path):
    with pytest.raises(ValueError, match="feature"):
        synth_dataset(MockGenerator(), "p", [], n=1, out_dir=tmp_path, size=64)


class _FlakyGenerator:
    """Mock generator that raises on odd seeds -- stands in for Gemini refusals/limits."""

    def __init__(self) -> None:
        self._mock = MockGenerator()

    def generate(self, prompt, seed, *, size=64):  # noqa: ANN001, ANN201
        if seed % 2 == 1:
            raise RuntimeError(f"simulated API failure on seed {seed}")
        return self._mock.generate(prompt, seed, size=size)


@pytest.mark.unit
def test_skip_errors_keeps_the_batch_alive(tmp_path):
    seen: list[tuple[int, int]] = []
    records = synth_dataset(
        _FlakyGenerator(),
        "macro helicoil",
        ["helicoil"],
        n=4,
        out_dir=tmp_path,
        size=64,
        skip_errors=True,
        progress=lambda done, total: seen.append((done, total)),
    )
    # Seeds 0 and 2 succeed; 1 and 3 are skipped. No orphan images for the failures.
    assert len(records) == 2
    assert len(list((tmp_path / "images").glob("*.png"))) == 2
    assert seen[-1] == (4, 4)  # progress still ticks for skipped images
    coco = json.loads((tmp_path / "annotations.coco.json").read_text())
    assert len(coco["images"]) == 2


@pytest.mark.unit
def test_without_skip_errors_a_failure_aborts(tmp_path):
    with pytest.raises(RuntimeError, match="simulated API failure"):
        synth_dataset(_FlakyGenerator(), "p", ["helicoil"], n=4, out_dir=tmp_path, size=64)


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
