"""Stdlib image decoder: PIL round-trips, adversarial bytes, resize invariants.

PNG is lossless, so a PIL-encoded image must decode to byte-identical RGB through the
zerort decoder -- that is the correctness oracle. The adversarial side: truncations,
bit flips, hostile headers, and random garbage must raise ``ImageError`` (or decode
cleanly), never an uncontrolled exception or a giant allocation.
"""

import contextlib
import io
import struct
import zlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tensor_factory.zerort import ImageError, decode, decode_png, decode_ppm, preprocess
from tensor_factory.zerort.imageio import resize_rgb
from tensor_factory.zerort.pb import WireError, parse_model

PIL_Image = pytest.importorskip("PIL.Image")


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gradient(w, h, mode="RGB"):
    img = PIL_Image.new(mode, (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            v = (x * 7 + y * 13) % 256
            if mode == "RGB":
                px[x, y] = (v, (v * 3) % 256, 255 - v)
            elif mode == "RGBA":
                px[x, y] = (v, (v * 3) % 256, 255 - v, (v * 5) % 256)
            elif mode == "L":
                px[x, y] = v
            else:
                px[x, y] = v % 8
    return img


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "LA"])
def test_png_roundtrip_matches_pil(mode):
    img = _gradient(23, 17, "RGBA" if mode == "LA" else mode).convert(mode)
    w, h, rgb = decode_png(_png_bytes(img))
    assert (w, h) == (23, 17)
    assert rgb == img.convert("RGB").tobytes()


@pytest.mark.unit
def test_png_palette_roundtrip():
    # 200 colors forces an 8-bit palette; tiny palettes get saved 4-bit, which the
    # decoder rejects by design (bit depth 8 only).
    img = _gradient(16, 16, "RGB").quantize(colors=200)  # mode P
    w, h, rgb = decode_png(_png_bytes(img))
    assert (w, h) == (16, 16)
    assert rgb == img.convert("RGB").tobytes()


@pytest.mark.unit
@given(
    w=st.integers(1, 12),
    h=st.integers(1, 12),
    data=st.data(),
)
@settings(max_examples=40, deadline=None)
def test_png_roundtrip_property(w, h, data):
    raw = bytes(data.draw(st.lists(st.integers(0, 255), min_size=w * h * 3, max_size=w * h * 3)))
    img = PIL_Image.frombytes("RGB", (w, h), raw)
    got = decode_png(_png_bytes(img))
    assert got == (w, h, raw)


@pytest.mark.unit
def test_ppm_roundtrip():
    raw = bytes((i * 31) % 256 for i in range(9 * 7 * 3))
    blob = b"P6\n# a comment\n9 7\n255\n" + raw
    assert decode_ppm(blob) == (9, 7, raw)
    assert decode(blob) == (9, 7, raw)


@pytest.mark.unit
def test_rejects_wrong_magic_and_truncations():
    img = _gradient(16, 16)
    blob = _png_bytes(img)
    with pytest.raises(ImageError, match="unrecognized"):
        decode(b"JFIF not a png")
    for cut in (4, 12, 40, len(blob) - 5):
        with pytest.raises(ImageError):
            decode_png(blob[:cut])
    with pytest.raises(ImageError, match="truncated"):
        decode_ppm(b"P6\n9 7\n255\nshort")


@pytest.mark.unit
def test_rejects_corrupt_crc_and_hostile_headers():
    blob = bytearray(_png_bytes(_gradient(16, 16)))
    blob[30] ^= 0xFF  # inside IHDR payload -> CRC mismatch
    with pytest.raises(ImageError, match="CRC"):
        decode_png(bytes(blob))

    # A crafted IHDR declaring absurd dimensions (valid CRC) must be rejected
    # before any allocation -- decompression-bomb guard.
    ihdr = struct.pack(">IIBBBBB", 30000, 30000, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    hostile = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk))
    )
    with pytest.raises(ImageError, match="too large"):
        decode_png(hostile)


@pytest.mark.unit
def test_idat_must_match_declared_dimensions():
    """An IDAT stream longer than the header promises is hostile, not tolerated."""
    img = _gradient(8, 8)
    blob = _png_bytes(img)
    # Rebuild with an IDAT that inflates to twice the expected size.
    raw = b"\x00" + bytes(8 * 3)
    payload = zlib.compress(raw * 16)  # 16 rows of data for an 8-row image
    out = bytearray(blob[:8])
    pos = 8
    while pos < len(blob):
        (length,), ctype = struct.unpack(">I", blob[pos : pos + 4]), blob[pos + 4 : pos + 8]
        if ctype == b"IDAT":
            chunk = b"IDAT" + payload
            out += struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk))
        else:
            out += blob[pos : pos + 12 + length]
        pos += 12 + length
    with pytest.raises(ImageError, match="declared dimensions"):
        decode_png(bytes(out))


@pytest.mark.unit
@given(data=st.binary(min_size=0, max_size=400))
@settings(max_examples=200, deadline=None)
def test_decode_never_raises_uncontrolled(data):
    with contextlib.suppress(ImageError):
        decode(data)


@pytest.mark.unit
@given(data=st.binary(min_size=0, max_size=400))
@settings(max_examples=200, deadline=None)
def test_model_parser_never_raises_uncontrolled(data):
    with contextlib.suppress(WireError):
        parse_model(data)


@pytest.mark.unit
def test_resize_invariants():
    # Constant image stays constant through the triangle filter, up or down.
    rgb = bytes([200, 100, 50] * (37 * 21))
    for out_w, out_h in ((8, 8), (64, 64)):
        out = resize_rgb(rgb, 37, 21, out_w, out_h)
        assert len(out) == out_w * out_h * 3
        for i in range(0, len(out), 3):
            assert out[i] == pytest.approx(200, abs=1e-6)
            assert out[i + 1] == pytest.approx(100, abs=1e-6)
            assert out[i + 2] == pytest.approx(50, abs=1e-6)


@pytest.mark.unit
def test_preprocess_shape_and_range(tmp_path):
    img = _gradient(31, 19)
    path = tmp_path / "g.png"
    img.save(path)
    t = preprocess(path, 16)
    assert t.shape == (1, 3, 16, 16)
    assert all(0.0 <= v <= 1.0 for v in t.data)
    # bytes input goes through the same path
    t2 = preprocess(_png_bytes(img), 16)
    assert t2.data == t.data
