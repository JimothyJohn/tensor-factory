"""Stdlib JPEG decoder: libjpeg (via PIL) parity, adversarial bytes, pipeline hookup.

JPEG is lossy but decoding is deterministic: the same bytes through libjpeg and through
this decoder must agree to within the known implementation deltas (fixed-point vs float
IDCT, fancy vs nearest chroma upsampling) -- a couple of code values, far below the
detector's 1/255 quantization step. Hostile input must raise ``ImageError``, bounded
before allocation, exactly like the PNG path.
"""

import contextlib
import io
import struct

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tensor_factory.zerort import ImageError, decode, decode_jpeg, preprocess

PIL_Image = pytest.importorskip("PIL.Image")


def _photo(w=97, h=61, checker=False):
    """Smooth gradients; with ``checker``, hard 8x8 chroma edges (worst case for
    nearest-vs-fancy chroma upsampling -- only comparable to libjpeg at 4:4:4)."""
    img = PIL_Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            v = (x * 255) // w
            b = (255 - v if (x // 8 + y // 8) % 2 else v) if checker else (v + (y * 255) // h) // 2
            px[x, y] = (v, (y * 255) // h, b)
    return img


def _jpeg_bytes(img, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


@pytest.mark.unit
@pytest.mark.parametrize("subsampling", [0, 1, 2], ids=["4:4:4", "4:2:2", "4:2:0"])
@pytest.mark.parametrize("quality", [60, 95])
def test_jpeg_matches_pil_decode(subsampling, quality):
    blob = _jpeg_bytes(_photo(), quality=quality, subsampling=subsampling)
    w, h, rgb = decode_jpeg(blob)
    ref = PIL_Image.open(io.BytesIO(blob)).convert("RGB")
    assert (w, h) == ref.size
    refb = ref.tobytes()
    diffs = [abs(a - b) for a, b in zip(rgb, refb, strict=True)]
    mean = sum(diffs) / len(diffs)
    # Smooth chroma agrees to ~2 code values (IDCT rounding); libjpeg's fancy
    # upsampling only diverges from our nearest-neighbour on hard chroma edges,
    # which this image avoids (see the 4:4:4 checkerboard test for those).
    assert mean < 1.5, f"mean abs diff {mean}"
    assert sum(1 for d in diffs if d > 24) < len(diffs) * 0.01


@pytest.mark.unit
def test_jpeg_hard_edges_match_pil_at_444():
    """Hard luma+chroma edges: exact comparison territory only without subsampling."""
    blob = _jpeg_bytes(_photo(checker=True), quality=90, subsampling=0)
    _, _, rgb = decode_jpeg(blob)
    ref = PIL_Image.open(io.BytesIO(blob)).convert("RGB").tobytes()
    diffs = [abs(a - b) for a, b in zip(rgb, ref, strict=True)]
    assert sum(diffs) / len(diffs) < 1.0
    assert max(diffs) <= 4


@pytest.mark.unit
@pytest.mark.parametrize("subsampling", [1, 2], ids=["4:2:2", "4:2:0"])
def test_jpeg_chroma_alignment(subsampling):
    """A flat saturated color must survive subsampling exactly -- a chroma-siting or
    plane-indexing bug would smear it."""
    img = PIL_Image.new("RGB", (40, 24), (200, 60, 90))
    blob = _jpeg_bytes(img, quality=95, subsampling=subsampling)
    _, _, rgb = decode_jpeg(blob)
    ref = PIL_Image.open(io.BytesIO(blob)).convert("RGB").tobytes()
    diffs = [abs(a - b) for a, b in zip(rgb, ref, strict=True)]
    assert max(diffs) <= 2


@pytest.mark.unit
def test_jpeg_grayscale():
    blob = _jpeg_bytes(_photo().convert("L"), quality=90)
    w, h, rgb = decode_jpeg(blob)
    ref = PIL_Image.open(io.BytesIO(blob)).convert("RGB").tobytes()
    diffs = [abs(a - b) for a, b in zip(rgb, ref, strict=True)]
    assert sum(diffs) / len(diffs) < 1.0
    assert rgb[0::3] == rgb[1::3] == rgb[2::3]  # gray stays gray


@pytest.mark.unit
def test_jpeg_odd_dimensions_and_partial_mcus():
    # 4:2:0 with dimensions that are not multiples of 16 exercises edge-MCU cropping.
    blob = _jpeg_bytes(_photo(33, 17), quality=90, subsampling=2)
    w, h, _ = decode_jpeg(blob)
    assert (w, h) == (33, 17)


@pytest.mark.unit
def test_jpeg_restart_markers():
    img = _photo()
    try:
        blob = _jpeg_bytes(img, quality=85, subsampling=0, restart_marker_blocks=2)
    except TypeError:
        pytest.skip("this Pillow cannot emit restart markers")
    assert b"\xff\xdd" in blob  # DRI segment actually present
    w, h, rgb = decode_jpeg(blob)
    ref = PIL_Image.open(io.BytesIO(blob)).convert("RGB").tobytes()
    diffs = [abs(a - b) for a, b in zip(rgb, ref, strict=True)]
    assert sum(diffs) / len(diffs) < 1.5


@pytest.mark.unit
def test_progressive_is_rejected_clearly():
    blob = _jpeg_bytes(_photo(), quality=90, progressive=True)
    with pytest.raises(ImageError, match="progressive"):
        decode_jpeg(blob)


@pytest.mark.unit
def test_truncation_raises_imageerror():
    blob = _jpeg_bytes(_photo(), quality=90)
    for cut in (1, 4, 20, 200, len(blob) // 2):
        with pytest.raises(ImageError):
            decode_jpeg(blob[:cut])


@pytest.mark.unit
def test_hostile_sof_dimensions_rejected_before_allocation():
    # SOI + minimal SOF0 claiming a 32768x32768 grayscale image.
    sof = struct.pack(">BHHB", 8, 32768, 32768, 1) + bytes([1, 0x11, 0])
    blob = b"\xff\xd8\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
    with pytest.raises(ImageError, match="too large"):
        decode_jpeg(blob)


@pytest.mark.unit
@given(data=st.binary(min_size=0, max_size=400))
@settings(max_examples=200, deadline=None)
def test_jpeg_decoder_never_raises_uncontrolled(data):
    with contextlib.suppress(ImageError):
        decode_jpeg(b"\xff\xd8" + data)


@pytest.mark.unit
def test_jpeg_flows_through_decode_and_preprocess():
    blob = _jpeg_bytes(_photo(), quality=90)
    w, h, _ = decode(blob)  # sniffed by magic
    assert (w, h) == (97, 61)
    t = preprocess(blob, 32)
    assert t.shape == (1, 3, 32, 32)
    assert all(0.0 <= v <= 1.0 for v in t.data)
