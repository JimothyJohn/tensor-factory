"""Regression tests for the metrics path that produced silent empty responses.

The trainer is never started here, so no torch import — these are unit-fast.
"""

import json

import numpy as np
import pytest

from tensor_factory_studio.dataset import Dataset
from tensor_factory_studio.trainer import Trainer


def _trainer(tmp_path) -> Trainer:
    return Trainer(Dataset(tmp_path / "d"), tmp_path / "d" / "models")


@pytest.mark.unit
def test_shape_coerces_numpy_scalars_to_json(tmp_path):
    # _val_metrics / baseline_center_err hand back numpy scalars; json.dumps can't
    # serialize np.float64, which previously crashed /metrics into an empty response.
    tr = _trainer(tmp_path)
    raw = {
        "epoch": np.int64(3),
        "epochs": 4,
        "loss": np.float64(0.12),
        "val_err": np.float64(12.5),
        "baseline": np.float32(11.5),
        "presence_acc": np.float64(1.0),
        "gain": 1.0,
        "is_best": True,
        "best_err": np.float64(12.5),
        "train_count": 10,
        "val_count": 2,
    }
    shaped = tr._shape(raw)
    json.dumps(shaped)  # must not raise
    assert isinstance(shaped["err"], float)
    assert isinstance(shaped["epoch"], int)
    assert shaped["err"] == 12.5


@pytest.mark.unit
def test_metrics_is_json_serializable_when_idle(tmp_path):
    tr = _trainer(tmp_path)
    json.dumps(tr.metrics())  # warming-up state must serialize too


@pytest.mark.unit
def test_shape_passes_per_source_metrics_through_as_json(tmp_path):
    tr = _trainer(tmp_path)
    raw = {
        "epoch": 1,
        "epochs": 2,
        "loss": 0.5,
        "val_err": np.float64(9.0),
        "baseline": None,
        "presence_acc": None,
        "gain": 1.0,
        "train_count": 6,
        "val_count": 2,
        "sources": [
            {
                "name": "studio",
                "train": np.int64(4),
                "val": 1,
                "val_err": np.float64(9.0),
                "presence_acc": np.float64(1.0),
                "class_acc": None,
            },
            # a negatives pool: no held-out box, so no error -- must stay null, not 0
            {
                "name": "negatives",
                "train": 2,
                "val": 1,
                "val_err": None,
                "presence_acc": 0.5,
                "class_acc": None,
            },
        ],
    }
    shaped = json.loads(json.dumps(tr._shape(raw)))
    assert shaped["sources"] == [
        {"name": "studio", "train": 4, "val": 1, "err": 9.0, "presenceAcc": 1.0, "classAcc": None},
        {
            "name": "negatives",
            "train": 2,
            "val": 1,
            "err": None,
            "presenceAcc": 0.5,
            "classAcc": None,
        },
    ]


@pytest.mark.unit
def test_shape_without_sources_is_an_empty_list(tmp_path):
    # An older fit() (or a caller's hand-built dict) has no "sources" key.
    raw = {
        "epoch": 1, "epochs": 1, "loss": 0.1, "val_err": None, "baseline": None,
        "presence_acc": None, "gain": 1.0, "train_count": 1, "val_count": 0,
    }  # fmt: skip
    assert _trainer(tmp_path)._shape(raw)["sources"] == []


@pytest.mark.unit
def test_extra_sources_train_before_anything_is_labeled(tmp_path):
    src = tmp_path / "prior"
    src.mkdir()
    ds = Dataset(tmp_path / "d")
    tr = Trainer(ds, tmp_path / "d" / "models", sources=[src])
    # dirty from the start: the data is already on disk, no label has to arrive first
    assert tr._dirty.is_set()
    # the empty browser set has no annotations file yet, so it is left out of the round
    assert tr._data_dirs() == [src]


@pytest.mark.unit
def test_browser_labels_join_the_extra_sources_once_they_exist(tmp_path):
    import io

    from PIL import Image

    src = tmp_path / "prior"
    src.mkdir()
    ds = Dataset(tmp_path / "d")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG")
    ds.upsert(1, True, [0.1, 0.1, 0.5, 0.5], buf.getvalue())
    tr = Trainer(ds, tmp_path / "d" / "models", sources=[src])
    assert tr._data_dirs() == [ds.root, src]


@pytest.mark.unit
def test_no_sources_waits_for_labels_as_before(tmp_path):
    tr = _trainer(tmp_path)
    assert not tr._dirty.is_set()
    assert tr._data_dirs() == []
