"""Model + export tests. Skipped unless torch is installed (the 'train' extra)."""

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402

from tensor_factory.geometry import BBox  # noqa: E402
from tensor_factory.inference import Detector  # noqa: E402
from tensor_factory_train.model import TinyDetector, soft_argmax_xyxy  # noqa: E402
from tensor_factory_train.train import (  # noqa: E402
    ImageCache,
    _BoxDataset,
    _flip_box,
    _load_chw,
    _resample,
    export_onnx,
    fit,
)


@pytest.mark.unit
def test_flip_box_is_a_mirror_involution():
    box = (0.1, 0.2, 0.6, 0.9)  # asymmetric so reflections are observable
    assert _flip_box(box, horizontal=True) == pytest.approx((0.4, 0.2, 0.9, 0.9))
    assert _flip_box(box, horizontal=False) == pytest.approx((0.1, 0.1, 0.6, 0.8))
    # flipping twice on the same axis restores the original
    once = _flip_box(box, horizontal=True)
    assert _flip_box(once, horizontal=True) == pytest.approx(box)


@pytest.mark.unit
def test_gain_sharpens_soft_argmax_toward_peak():
    # An off-centre peak (cell 25 of 30, i.e. ~0.86) on every edge channel. A plain softmax
    # (gain 1) over a near-flat logit field is pulled toward the 0.5 centre; raising the gain
    # sharpens it so the marginal expectation climbs toward the true peak. This is exactly
    # the centre-bias the gain exists to fix.
    h = w = 30
    heat = torch.zeros(1, 4, h, w)
    heat[:, :, 25, 25] = 1.0  # modest peak -> diffuse at gain 1, sharp at high gain
    low = soft_argmax_xyxy(heat, gain=1.0)
    high = soft_argmax_xyxy(heat, gain=20.0)
    peak = 25 / (h - 1)
    # Every coordinate moves strictly closer to the peak as the gain rises.
    assert bool((torch.abs(high - peak) < torch.abs(low - peak)).all())
    # At high gain it lands near the peak; at gain 1 it sits well short (centre-pulled).
    assert bool((torch.abs(high - peak) < 0.05).all())
    assert bool((low < high).all())


@pytest.mark.unit
def test_learnable_gain_starts_at_unity():
    # log_gain initialized to 0 -> exp -> 1.0, so a fresh model reproduces a plain softmax
    # and any previously-trained behaviour is unchanged at init.
    model = TinyDetector(width=8)
    assert float(model.log_gain.exp()) == pytest.approx(1.0)


@pytest.mark.unit
def test_forward_shape_and_finite():
    model = TinyDetector(width=8)
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 4)
    # Raw xyxy can sit just outside [0, 1] pre-clamp; inference clamps via BBox.
    assert bool(torch.isfinite(out).all())


@pytest.mark.unit
def test_export_int8_onnx_is_loadable(tmp_path):
    model = TinyDetector(width=8)
    out = export_onnx(model, tmp_path / "m.onnx", size=64)
    assert out.exists()
    # The exported model must satisfy the inference contract end to end.
    det = Detector(out, input_size=64)
    box = det.detect_box(Image.new("RGB", (64, 64), (128, 128, 128)))
    assert 0.0 <= box.x1 <= 1.0
    assert box.x1 <= box.x2
    assert not det.has_presence  # box-only model exposes no presence head


@pytest.mark.unit
def test_fit_refuses_all_pending_dataset(tmp_path):
    import json

    # A fresh auto-labeled dataset: every annotation pending human review.
    coco = {
        "images": [{"id": 1, "file_name": "images/a.png", "width": 64, "height": 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 16, 16], "review": "pending"}
        ],
        "categories": [{"id": 1, "name": "helicoil"}],
    }
    (tmp_path / "annotations.coco.json").write_text(json.dumps(coco))
    # The gate refuses to train -- and says why -- before touching any images.
    with pytest.raises(ValueError, match="pending human review"):
        fit(tmp_path, tmp_path / "m.onnx", epochs=1, size=64)


@pytest.mark.unit
def test_box_dataset_returns_consistent_tuple_shape(tmp_path):
    """Regression for CodeQL py/mixed-tuple-returns (alert #3).

    ``_BoxDataset.__getitem__`` used to return a 2-tuple ``(image, box)`` when
    ``presence=False`` but a 3-tuple ``(image, box, has_box)`` when ``presence=True``.
    Callers (``fit``'s training loop) had to branch on ``presence`` just to know how many
    values to unpack. It must now return the same shape in every mode -- since the class
    head landed that is ``(image, box, has_box, label)``, multiclass or not.
    """
    img_path = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img_path)
    box = BBox.from_pixels(1, 1, 4, 4, width=8, height=8)
    items = [(img_path, box), (img_path, None)]  # one positive, one negative

    for presence, multiclass in ((False, False), (True, False), (True, True)):
        ds = _BoxDataset(items, size=8, presence=presence, multiclass=multiclass)
        shapes = {len(ds[i]) for i in range(len(ds))}
        assert shapes == {4}, (
            f"presence={presence} multiclass={multiclass}: expected uniform 4-tuples, got {shapes}"
        )


@pytest.mark.unit
def test_presence_head_forward_shapes():
    model = TinyDetector(width=8, presence=True)
    box, presence = model(torch.zeros(3, 3, 64, 64))
    assert box.shape == (3, 4)
    assert presence.shape == (3, 1)  # one objectness logit per item


@pytest.mark.unit
def test_export_and_detect_presence_roundtrip(tmp_path):
    model = TinyDetector(width=8, presence=True)
    out = export_onnx(model, tmp_path / "m.onnx", size=64)
    det = Detector(out, input_size=64)
    assert det.has_presence
    score = det.detect_presence(Image.new("RGB", (64, 64), (128, 128, 128)))
    assert 0.0 <= score <= 1.0  # a probability, not a logit
    # box-only API still works on a presence-head model
    box = det.detect_box(Image.new("RGB", (64, 64), (10, 10, 10)))
    assert box.x1 <= box.x2


@pytest.mark.unit
def test_single_class_adds_no_class_head():
    # num_classes == 1 is the box-only/single-class path: no class head, no logits output,
    # forward returns exactly what it did before. Backward compatibility is the contract.
    box_only = TinyDetector(width=8)
    assert box_only.class_head is None
    assert box_only.output_names == ["box"]
    assert box_only(torch.zeros(2, 3, 64, 64)).shape == (2, 4)  # plain tensor, not a tuple

    presence_only = TinyDetector(width=8, presence=True, num_classes=1)
    assert presence_only.class_head is None
    assert presence_only.output_names == ["box", "presence"]


@pytest.mark.unit
def test_multiclass_forward_shapes_and_output_names():
    model = TinyDetector(width=8, presence=True, num_classes=3)
    assert model.output_names == ["box", "presence", "logits"]
    box, presence, logits = model(torch.zeros(2, 3, 64, 64))
    assert box.shape == (2, 4)
    assert presence.shape == (2, 1)
    assert logits.shape == (2, 3)  # one logit per class


@pytest.mark.unit
def test_multiclass_without_presence_emits_box_and_logits():
    # A class head can ride without a presence head: order is box, then logits.
    model = TinyDetector(width=8, num_classes=4)
    assert model.output_names == ["box", "logits"]
    box, logits = model(torch.zeros(1, 3, 64, 64))
    assert box.shape == (1, 4) and logits.shape == (1, 4)


@pytest.mark.unit
def test_export_and_detect_class_roundtrip(tmp_path):
    model = TinyDetector(width=8, presence=True, num_classes=3)
    out = export_onnx(model, tmp_path / "m.onnx", size=64)
    det = Detector(out, input_size=64)
    assert det.has_class and det.has_presence
    idx, conf = det.detect_class(Image.new("RGB", (64, 64), (128, 128, 128)))
    assert idx in (0, 1, 2)  # a valid 0-based class index
    assert 0.0 <= conf <= 1.0  # softmax confidence, not a raw logit


@pytest.mark.unit
def test_detect_class_raises_without_class_head(tmp_path):
    model = TinyDetector(width=8, presence=True)  # single-class: no logits output
    out = export_onnx(model, tmp_path / "m.onnx", size=64)
    det = Detector(out, input_size=64)
    assert not det.has_class
    with pytest.raises(RuntimeError, match="no class head"):
        det.detect_class(Image.new("RGB", (64, 64), (0, 0, 0)))


@pytest.mark.unit
def test_context_default_keeps_the_original_parameter_names():
    # Old checkpoints and exports must keep loading: context blocks are opt-in and appended.
    keys = {
        k.split(".")[1] for k in TinyDetector(width=8).state_dict() if k.startswith("features.")
    }
    assert keys == {"0", "1", "2", "3"}
    wide = TinyDetector(width=8, context=2).state_dict()
    assert {k.split(".")[1] for k in wide if k.startswith("features.")} == {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    }


@pytest.mark.unit
def test_context_widens_what_a_heatmap_cell_sees():
    # A change about 50 px from the top-left cell: outside the 31 px window of four stride-2
    # blocks, inside it once four context blocks widen the window to 159 px.
    def top_left_moves(context: int) -> bool:
        torch.manual_seed(0)
        model = TinyDetector(width=8, context=context).eval()
        base, poked = torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 224, 224)
        poked[:, :, 48:60, 48:60] = 1.0
        with torch.no_grad():
            a = model.heat(model.features(base))[0, :, 0, 0]
            b = model.heat(model.features(poked))[0, :, 0, 0]
        return not torch.equal(a, b)

    assert top_left_moves(0) is False
    assert top_left_moves(4) is True


@pytest.mark.unit
def test_context_keeps_the_heatmap_resolution():
    for context in (0, 3):
        model = TinyDetector(width=8, context=context)
        assert model.heat(model.features(torch.zeros(1, 3, 96, 96))).shape == (1, 4, 6, 6)


@pytest.mark.unit
@pytest.mark.parametrize("context", [-1, -100])
def test_negative_context_is_rejected(context):
    with pytest.raises(ValueError, match="context"):
        TinyDetector(context=context)


@pytest.mark.unit
def test_export_and_detect_with_context_roundtrip(tmp_path):
    model = TinyDetector(width=8, presence=True, context=2)
    out = export_onnx(model, tmp_path / "ctx.onnx", size=64)
    det = Detector(out, input_size=64)
    image = Image.new("RGB", (64, 64), (120, 90, 60))
    assert isinstance(det.detect_box(image), BBox)
    assert 0.0 <= det.detect_presence(image) <= 1.0


def _checker(tmp_path):
    # 1 px checkerboard: bilinear downscaling averages it to grey, nearest keeps pure 0/255.
    img = Image.new("L", (64, 64))
    img.putdata([255 * ((x + y) % 2) for y in range(64) for x in range(64)])
    path = tmp_path / "checker.png"
    img.convert("RGB").save(path)
    return path


@pytest.mark.unit
def test_resize_mode_changes_the_pixels_the_model_sees(tmp_path):
    path = _checker(tmp_path)
    nearest = _load_chw(path, 16, _resample("nearest", train=True))
    bilinear = _load_chw(path, 16, _resample("bilinear", train=True))
    assert set(nearest.unique().tolist()) <= {0.0, 1.0}
    assert 0.2 < float(bilinear.mean()) < 0.8 and float(bilinear.std()) < 0.2
    assert not torch.equal(nearest, bilinear)


@pytest.mark.unit
def test_mixed_resize_trains_on_both_and_validates_on_bilinear(tmp_path):
    path = _checker(tmp_path)
    data = _BoxDataset([(path, BBox(0.1, 0.1, 0.9, 0.9))] * 40, 16, resize="mixed")
    binary = [set(data[i][0].unique().tolist()) <= {0.0, 1.0} for i in range(40)]
    assert any(binary) and not all(binary)
    assert {_resample("mixed", train=False) for _ in range(20)} == {Image.Resampling.BILINEAR}


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["", "NEAREST", "bicubic", "area"])
def test_unknown_resize_mode_is_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match="resize"):
        _BoxDataset([], 16, resize=mode)
    with pytest.raises(ValueError, match="resize"):
        _resample(mode, train=False)


@pytest.mark.unit
def test_image_cache_returns_the_same_pixels_as_no_cache(tmp_path):
    path = _checker(tmp_path)
    cache = ImageCache()
    for mode in ("nearest", "bilinear"):
        resample = _resample(mode, train=False)
        plain = _load_chw(path, 16, resample)
        assert torch.equal(_load_chw(path, 16, resample, cache), plain)  # miss
        assert torch.equal(_load_chw(path, 16, resample, cache), plain)  # hit
    assert len(cache) == 2  # one entry per filter: they are different pixels


@pytest.mark.unit
def test_image_cache_decodes_a_rewritten_file_again(tmp_path):
    # The Studio overwrites a frame's file when it is re-labeled; a stale hit would train
    # on pixels that are no longer on disk.
    import os

    path = tmp_path / "frame.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(path)
    cache = ImageCache()
    resample = _resample("bilinear", train=False)
    assert float(_load_chw(path, 8, resample, cache).max()) == 0.0
    Image.new("RGB", (32, 32), (255, 255, 255)).save(path)
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 1_000_000_000))
    assert float(_load_chw(path, 8, resample, cache).min()) == 1.0
    assert len(cache) == 1 and cache.bytes == 8 * 8 * 3


@pytest.mark.unit
def test_image_cache_stops_growing_at_its_budget_but_still_serves(tmp_path):
    cache = ImageCache(max_bytes=8 * 8 * 3)  # room for exactly one 8 px image
    resample = _resample("bilinear", train=False)
    paths = []
    for i, v in enumerate((10, 200)):
        paths.append(tmp_path / f"{i}.png")
        Image.new("RGB", (32, 32), (v, v, v)).save(paths[-1])
    first = _load_chw(paths[0], 8, resample, cache)
    second = _load_chw(paths[1], 8, resample, cache)
    assert len(cache) == 1 and cache.bytes == 8 * 8 * 3
    assert float(first.mean()) == pytest.approx(10 / 255) and float(second.mean()) == pytest.approx(
        200 / 255
    )


@pytest.mark.unit
def test_image_cache_missing_file_raises_instead_of_serving_stale(tmp_path):
    path = _checker(tmp_path)
    cache = ImageCache()
    resample = _resample("bilinear", train=False)
    _load_chw(path, 16, resample, cache)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        _load_chw(path, 16, resample, cache)
