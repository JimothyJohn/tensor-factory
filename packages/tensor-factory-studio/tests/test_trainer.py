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
