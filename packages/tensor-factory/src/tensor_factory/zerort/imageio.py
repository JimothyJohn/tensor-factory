"""Stdlib-only image decoding and resizing for the zero-dependency path.

Decodes PNG (8-bit gray / RGB / palette / +alpha, non-interlaced -- what cameras and
screenshots actually produce) via ``zlib``, and binary PPM (P6). Resizing is a separable
triangle-filter resample (the same family as PIL's ``BILINEAR``), so downscales average
over the source footprint instead of skipping pixels. JPEG is deliberately out of scope
for now (see ROADMAP.md); with PIL installed the fast path handles it.

Defensive by construction: chunk CRCs are checked, the inflate size is bounded by the
header-declared dimensions (a truncated or bloated IDAT stream raises instead of
allocating unbounded memory), and dimensions are sanity-capped.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from .pb import FLOAT32
from .tensor import Tensor

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_MAX_DIM = 32768  # reject absurd headers before allocating anything
_MAX_PIXELS = 1 << 26  # ~67 MP: bounds inflate/allocation from a hostile header

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}  # PNG color type -> samples per pixel


class ImageError(ValueError):
    """Malformed or unsupported image data."""


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, width: int, height: int, ch: int) -> bytearray:
    stride = width * ch
    out = bytearray(height * stride)
    prev = bytes(stride)
    pos = 0
    for y in range(height):
        ft = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += 1 + stride
        if ft == 0:
            pass
        elif ft == 1:  # Sub
            for i in range(ch, stride):
                row[i] = (row[i] + row[i - ch]) & 0xFF
        elif ft == 2:  # Up
            row = bytearray((a + b) & 0xFF for a, b in zip(row, prev, strict=True))
        elif ft == 3:  # Average
            for i in range(stride):
                left = row[i - ch] if i >= ch else 0
                row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ft == 4:  # Paeth
            for i in range(stride):
                left = row[i - ch] if i >= ch else 0
                upleft = prev[i - ch] if i >= ch else 0
                row[i] = (row[i] + _paeth(left, prev[i], upleft)) & 0xFF
        else:
            raise ImageError(f"unknown PNG filter type {ft} on row {y}")
        out[y * stride : (y + 1) * stride] = row
        prev = bytes(row)
    return out


def _to_rgb(pixels: bytearray, width: int, height: int, color_type: int, plte: bytes) -> bytes:
    n = width * height
    if color_type == 2:
        return bytes(pixels)
    rgb = bytearray(n * 3)
    if color_type == 6:  # RGBA -> drop alpha
        for i in range(n):
            rgb[i * 3 : i * 3 + 3] = pixels[i * 4 : i * 4 + 3]
    elif color_type == 0:  # gray
        for i in range(n):
            g = pixels[i]
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = g
    elif color_type == 4:  # gray + alpha
        for i in range(n):
            g = pixels[i * 2]
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = g
    elif color_type == 3:  # palette
        if not plte:
            raise ImageError("palette PNG without a PLTE chunk")
        for i in range(n):
            p = pixels[i] * 3
            if p + 3 > len(plte):
                raise ImageError(f"palette index {pixels[i]} outside PLTE")
            rgb[i * 3 : i * 3 + 3] = plte[p : p + 3]
    else:
        raise ImageError(f"unsupported PNG color type {color_type}")
    return bytes(rgb)


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """Decode a PNG to ``(width, height, packed RGB bytes)``."""
    if data[:8] != _PNG_MAGIC:
        raise ImageError("not a PNG (bad signature)")
    pos = 8
    width = height = 0
    color_type = -1
    plte = b""
    idat = bytearray()
    seen_ihdr = seen_iend = False
    while pos + 8 <= len(data) and not seen_iend:
        length, ctype = struct.unpack(">I4s", data[pos : pos + 8])
        chunk = data[pos + 8 : pos + 8 + length]
        if len(chunk) != length or pos + 12 + length > len(data):
            raise ImageError(f"truncated {ctype!r} chunk")
        crc = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])[0]
        if zlib.crc32(ctype + chunk) != crc:
            raise ImageError(f"CRC mismatch in {ctype!r} chunk")
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if not (0 < width <= _MAX_DIM and 0 < height <= _MAX_DIM):
                raise ImageError(f"unreasonable dimensions {width}x{height}")
            if width * height > _MAX_PIXELS:
                raise ImageError(f"image too large ({width}x{height})")
            if depth != 8:
                raise ImageError(f"unsupported bit depth {depth} (only 8)")
            if color_type not in _CHANNELS:
                raise ImageError(f"unsupported color type {color_type}")
            if comp != 0 or filt != 0:
                raise ImageError("unsupported compression/filter method")
            if interlace != 0:
                raise ImageError("interlaced (Adam7) PNG is not supported")
            seen_ihdr = True
        elif ctype == b"PLTE":
            plte = chunk
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            seen_iend = True
    if not seen_ihdr or not idat or not seen_iend:
        raise ImageError("missing IHDR, IDAT or IEND -- truncated file?")
    ch = _CHANNELS[color_type]
    expected = height * (1 + width * ch)
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(bytes(idat), expected)
    except zlib.error as exc:
        raise ImageError(f"corrupt IDAT stream: {exc}") from exc
    if len(raw) != expected or (inflater.flush() or inflater.unconsumed_tail):
        raise ImageError("IDAT stream does not match the declared dimensions")
    pixels = _unfilter(raw, width, height, ch)
    return width, height, _to_rgb(pixels, width, height, color_type, plte)


def decode_ppm(data: bytes) -> tuple[int, int, bytes]:
    """Decode a binary PPM (P6, maxval 255) to ``(width, height, packed RGB bytes)``."""
    if data[:2] != b"P6":
        raise ImageError("not a binary PPM (P6)")
    fields: list[int] = []
    pos = 2
    while len(fields) < 3:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":  # comment to end of line
            while pos < len(data) and data[pos] != 0x0A:
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        try:
            fields.append(int(data[start:pos]))
        except ValueError as exc:
            raise ImageError("malformed PPM header") from exc
    width, height, maxval = fields
    if maxval != 255:
        raise ImageError(f"unsupported PPM maxval {maxval}")
    if not (0 < width <= _MAX_DIM and 0 < height <= _MAX_DIM) or width * height > _MAX_PIXELS:
        raise ImageError(f"unreasonable dimensions {width}x{height}")
    pos += 1  # single whitespace after maxval
    rgb = data[pos : pos + width * height * 3]
    if len(rgb) != width * height * 3:
        raise ImageError("truncated PPM pixel data")
    return width, height, rgb


def decode(data: bytes) -> tuple[int, int, bytes]:
    """Sniff and decode PNG or PPM bytes to ``(width, height, packed RGB)``."""
    if data[:8] == _PNG_MAGIC:
        return decode_png(data)
    if data[:2] == b"P6":
        return decode_ppm(data)
    raise ImageError("unrecognized image format (PNG and binary PPM are supported)")


def _filter_weights(src: int, dst: int) -> list[tuple[int, list[float]]]:
    """Per-output-index ``(first_source_index, normalized weights)`` for a triangle filter."""
    scale = src / dst
    support = max(scale, 1.0)
    out = []
    for i in range(dst):
        center = (i + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), src)
        ws = [max(0.0, 1.0 - abs((j + 0.5 - center) / support)) for j in range(lo, hi)]
        total = sum(ws) or 1.0
        out.append((lo, [w / total for w in ws]))
    return out


def resize_rgb(rgb: bytes, width: int, height: int, out_w: int, out_h: int) -> list[float]:
    """Separable triangle-filter resample; returns flat RGB floats in ``[0, 255]``."""
    xw = _filter_weights(width, out_w)
    yw = _filter_weights(height, out_h)
    # Horizontal pass: (height, out_w, 3) floats.
    mid = [0.0] * (height * out_w * 3)
    for y in range(height):
        row_off = y * width * 3
        out_off = y * out_w * 3
        for ox, (lo, ws) in enumerate(xw):
            r = g = b = 0.0
            base = row_off + lo * 3
            for w in ws:
                r += w * rgb[base]
                g += w * rgb[base + 1]
                b += w * rgb[base + 2]
                base += 3
            o = out_off + ox * 3
            mid[o] = r
            mid[o + 1] = g
            mid[o + 2] = b
    # Vertical pass: (out_h, out_w, 3).
    out = [0.0] * (out_h * out_w * 3)
    row_len = out_w * 3
    for oy, (lo, ws) in enumerate(yw):
        acc = [0.0] * row_len
        for k, w in enumerate(ws):
            src_off = (lo + k) * row_len
            row = mid[src_off : src_off + row_len]
            acc = [a + w * v for a, v in zip(acc, row, strict=True)]
        out[oy * row_len : (oy + 1) * row_len] = acc
    return out


def preprocess(source: bytes | str | Path, size: int) -> Tensor:
    """Decode + resize + normalize an image to the model's ``(1, 3, S, S)`` input tensor."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    width, height, rgb = decode(data)
    hwc = resize_rgb(rgb, width, height, size, size)
    n = size * size
    chw = [0.0] * (3 * n)
    for c in range(3):
        chw[c * n : (c + 1) * n] = [hwc[i * 3 + c] / 255.0 for i in range(n)]
    return Tensor((1, 3, size, size), FLOAT32, chw)
