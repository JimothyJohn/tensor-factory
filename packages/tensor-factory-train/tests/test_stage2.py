"""Stage-2 crop-head tests. Skipped unless torch is installed (the 'train' extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensor_factory_train.stage2 import (  # noqa: E402
    DAMAGE_CLASSES,
    Stage2Head,
    crop_to_box,
    export_stage2_onnx,
    insertion_state,
)


@pytest.mark.unit
def test_forward_shapes_and_finite():
    model = Stage2Head(width=8)
    damage, seating = model(torch.zeros(2, 3, 128, 128))
    assert damage.shape == (2, len(DAMAGE_CLASSES))
    assert seating.shape == (2, 1)
    assert bool(torch.isfinite(damage).all()) and bool(torch.isfinite(seating).all())


@pytest.mark.unit
def test_damage_class_count_is_configurable():
    model = Stage2Head(width=8, num_damage_classes=3)
    damage, _ = model(torch.zeros(1, 3, 64, 64))
    assert damage.shape == (1, 3)


@pytest.mark.unit
def test_crop_to_box_selects_the_named_region():
    # Black everywhere except a white square in the bottom-right quadrant.
    img = torch.zeros(3, 100, 100)
    img[:, 50:100, 50:100] = 1.0
    bright = crop_to_box(img, (0.5, 0.5, 1.0, 1.0), out_size=16, context=0.0)
    assert bright.shape == (3, 16, 16)
    assert float(bright.mean()) > 0.9  # the crop is (almost) all white
    # The opposite quadrant is dark -- proves crop_to_box indexes the box, not the image centre.
    dark = crop_to_box(img, (0.0, 0.0, 0.5, 0.5), out_size=16, context=0.0)
    assert float(dark.mean()) < 0.1


@pytest.mark.unit
def test_crop_clamps_out_of_bounds_box():
    img = torch.zeros(3, 32, 48)  # non-square to catch H/W transposition
    # Box runs past the right/bottom edges (and context pushes it further): must clamp, keep
    # out_size, not crash.
    crop = crop_to_box(img, (0.8, 0.8, 1.5, 1.5), out_size=24)
    assert crop.shape == (3, 24, 24)
    assert bool(torch.isfinite(crop).all())


@pytest.mark.unit
def test_crop_degenerate_box_is_safe():
    # A zero-area box (a collapsed stage-1 prediction) must still yield a valid crop.
    img = torch.zeros(3, 20, 20)
    crop = crop_to_box(img, (0.5, 0.5, 0.5, 0.5), out_size=8, context=0.0)
    assert crop.shape == (3, 8, 8)
    assert bool(torch.isfinite(crop).all())


@pytest.mark.unit
def test_crop_normalizes_xyxy_ordering():
    # A reversed box (x2<x1, y2<y1) describes the same region; crop_to_box must canonicalize.
    img = torch.zeros(3, 100, 100)
    img[:, 50:100, 50:100] = 1.0
    forward = crop_to_box(img, (0.5, 0.5, 1.0, 1.0), out_size=16, context=0.0)
    reversed_ = crop_to_box(img, (1.0, 1.0, 0.5, 0.5), out_size=16, context=0.0)
    assert torch.allclose(forward, reversed_)


@pytest.mark.unit
def test_insertion_state_thresholds():
    assert insertion_state(0.5) == "under"  # proud of the chamfer
    assert insertion_state(-0.5) == "over"  # recessed below it
    assert insertion_state(0.0) == "correct"
    # exactly +/-tol is still correct (strict comparison); just past it flips.
    assert insertion_state(0.1, tol=0.1) == "correct"
    assert insertion_state(0.11, tol=0.1) == "under"
    assert insertion_state(-0.11, tol=0.1) == "over"


@pytest.mark.unit
def test_insertion_state_is_monotonic_in_seating():
    assert [insertion_state(s) for s in (-1.0, -0.05, 1.0)] == ["over", "correct", "under"]


@pytest.mark.unit
def test_export_int8_onnx_is_loadable(tmp_path):
    import onnxruntime as ort

    model = Stage2Head(width=8)
    out = export_stage2_onnx(model, tmp_path / "s2.onnx", size=64)
    assert out.exists()
    sess = ort.InferenceSession(str(out))
    assert {o.name for o in sess.get_outputs()} == {"damage", "seating"}
    outputs = sess.run(None, {"crop": np.zeros((1, 3, 64, 64), dtype=np.float32)})
    by_name = dict(zip([o.name for o in sess.get_outputs()], outputs, strict=True))
    # np.asarray is a no-op on the ndarray outputs but gives the type checker a concrete
    # shape (ort.run's return type is an un-narrowable union).
    assert np.asarray(by_name["damage"]).shape == (1, len(DAMAGE_CLASSES))
    assert np.asarray(by_name["seating"]).shape == (1, 1)
