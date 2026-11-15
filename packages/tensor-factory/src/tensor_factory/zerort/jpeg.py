"""Stdlib-only baseline JPEG decoder for the zero-dependency path.

Decodes baseline sequential JFIF/EXIF JPEGs -- what cameras and OpenCV pipelines
actually emit: 8-bit, Huffman-coded, grayscale or YCbCr, 4:4:4 / 4:2:2 / 4:2:0
subsampling, optional restart markers. Progressive (SOF2), arithmetic coding,
12-bit precision, and CMYK are rejected with a clear error; with PIL installed the
fast path handles those.

Fidelity note: the IDCT is exact floating point and chroma upsampling is nearest-
neighbour, so output differs from libjpeg (fixed-point IDCT, fancy upsampling) by a
couple of code values per pixel -- far below the detector's quantization step.
Bounds are enforced before allocation (hostile headers raise :class:`ImageError`).
"""

from __future__ import annotations

import math
import struct
from operator import mul

from .errors import ImageError

_MAX_DIM = 32768
_MAX_PIXELS = 1 << 26

# Zig-zag order: index = scan position, value = natural (row-major) position.
_ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
)  # fmt: skip

# IDCT basis: _COS[u][x] = C(u)/2 * cos((2x+1) u pi / 16); two separable passes
# yield the standard (1/4) C(u) C(v) normalization.
_COS = tuple(
    tuple(
        (math.sqrt(0.5) if u == 0 else 1.0) / 2.0 * math.cos((2 * x + 1) * u * math.pi / 16)
        for x in range(8)
    )
    for u in range(8)
)
_COS_T = tuple(tuple(_COS[u][x] for u in range(8)) for x in range(8))  # transposed


class _Huffman:
    """Canonical Huffman table with a two-level decoder.

    ``fast`` resolves any code of <= 8 bits from an 8-bit peek in one lookup (the
    overwhelmingly common case); longer codes fall back to ``lut`` keyed on
    ``(length, code)``.
    """

    __slots__ = ("fast", "lut")

    def __init__(self, counts: bytes, symbols: bytes) -> None:
        self.lut: dict[tuple[int, int], int] = {}
        self.fast: list[tuple[int, int] | None] = [None] * 256
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                if k >= len(symbols):
                    raise ImageError("Huffman table shorter than its counts")
                if code >= 1 << length:
                    raise ImageError("overfull Huffman table")
                sym = symbols[k]
                self.lut[(length, code)] = sym
                if length <= 8:
                    start = code << (8 - length)
                    for i in range(start, start + (1 << (8 - length))):
                        self.fast[i] = (sym, length)
                code += 1
                k += 1
            code <<= 1


class _BitReader:
    """MSB-first accumulator over entropy-coded data with 0xFF00 unstuffing.

    Consumers peek/consume multiple bits at a time (no per-bit calls). Once the scan
    hits a marker or the end of data, zero bits are fed so the final symbols -- whose
    peek window extends past the last real byte -- still decode; consuming more than a
    bounded amount of that padding means the stream was truncated.
    """

    __slots__ = ("data", "pos", "acc", "nbits", "marker", "pad")

    def __init__(self, data: bytes, pos: int) -> None:
        self.data = data
        self.pos = pos
        self.acc = 0
        self.nbits = 0
        self.marker = 0  # pending marker (0xD0..0xD7 restart, or scan terminator)
        self.pad = 0

    def _fill_byte(self) -> None:
        if self.marker or self.pos >= len(self.data):
            if not self.marker:
                self.marker = 0xD9  # ran off the end: treat as EOI
            self.pad += 8
            if self.pad > 512:
                raise ImageError("truncated entropy-coded data")
            self.acc <<= 8
            self.nbits += 8
            return
        b = self.data[self.pos]
        self.pos += 1
        if b == 0xFF:
            nxt = self.data[self.pos] if self.pos < len(self.data) else 0xD9
            if nxt == 0x00:
                self.pos += 1
            else:
                self.marker = nxt
                self.pos += 1
                self.acc <<= 8
                self.nbits += 8
                self.pad += 8
                return
        self.acc = (self.acc << 8) | b
        self.nbits += 8

    def receive(self, n: int) -> int:
        if n == 0:
            return 0
        while self.nbits < n:
            self._fill_byte()
        self.nbits -= n
        v = (self.acc >> self.nbits) & ((1 << n) - 1)
        self.acc &= (1 << self.nbits) - 1  # keep the accumulator a small int
        return v

    def decode(self, table: _Huffman) -> int:
        while self.nbits < 8:
            self._fill_byte()
        hit = table.fast[(self.acc >> (self.nbits - 8)) & 0xFF]
        if hit is not None:
            self.nbits -= hit[1]
            self.acc &= (1 << self.nbits) - 1
            return hit[0]
        while self.nbits < 16:
            self._fill_byte()
        code16 = (self.acc >> (self.nbits - 16)) & 0xFFFF
        lut = table.lut
        for length in range(9, 17):
            sym = lut.get((length, code16 >> (16 - length)))
            if sym is not None:
                self.nbits -= length
                self.acc &= (1 << self.nbits) - 1
                return sym
        raise ImageError("invalid Huffman code in entropy data")

    def restart(self) -> None:
        """Byte-align and consume an expected RSTn marker."""
        self.nbits = 0
        self.acc = 0
        self.pad = 0
        if self.marker:
            m, self.marker = self.marker, 0
        else:
            while self.pos < len(self.data) and self.data[self.pos] != 0xFF:
                self.pos += 1  # tolerate pad bytes before the marker
            if self.pos + 1 >= len(self.data):
                raise ImageError("missing restart marker")
            m = self.data[self.pos + 1]
            self.pos += 2
        if not 0xD0 <= m <= 0xD7:
            raise ImageError(f"expected restart marker, found 0x{m:02x}")


def _extend(v: int, n: int) -> int:
    """JPEG sign extension: an n-bit magnitude to a signed coefficient."""
    return v - (1 << n) + 1 if n and v < 1 << (n - 1) else v


def _idct8(s: list[float]) -> list[float]:
    if not any(s[1:]):
        v = s[0] * _COS[0][0]
        return [v] * 8
    return [sum(map(mul, row, s)) for row in _COS_T]


def _idct2d(coef: list[float]) -> list[float]:
    rows = [_idct8(coef[r * 8 : (r + 1) * 8]) for r in range(8)]
    cols = [_idct8([rows[y][x] for y in range(8)]) for x in range(8)]
    # cols[x][y] -> row-major samples, level-shifted and clamped to [0, 255]
    out = [0.0] * 64
    for x in range(8):
        cx = cols[x]
        for y in range(8):
            out[y * 8 + x] = cx[y]
    return out


class _Component:
    __slots__ = ("cid", "h", "v", "tq", "td", "ta", "pred", "plane", "pw", "ph")

    def __init__(self, cid: int, h: int, v: int, tq: int) -> None:
        self.cid, self.h, self.v, self.tq = cid, h, v, tq
        self.td = self.ta = 0
        self.pred = 0
        self.plane: bytearray = bytearray()
        self.pw = self.ph = 0


def decode_jpeg(data: bytes) -> tuple[int, int, bytes]:
    """Decode a baseline JPEG to ``(width, height, packed RGB bytes)``."""
    if data[:2] != b"\xff\xd8":
        raise ImageError("not a JPEG (bad SOI)")
    qt: dict[int, list[int]] = {}
    dc: dict[int, _Huffman] = {}
    ac: dict[int, _Huffman] = {}
    comps: list[_Component] = []
    width = height = 0
    restart_interval = 0
    pos = 2
    while True:
        if pos + 4 > len(data):
            raise ImageError("truncated JPEG (no SOS)")
        if data[pos] != 0xFF:
            raise ImageError(f"expected a marker at byte {pos}")
        marker = data[pos + 1]
        pos += 2
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue  # no payload
        (seglen,) = struct.unpack(">H", data[pos : pos + 2])
        seg = data[pos + 2 : pos + seglen]
        if seglen < 2 or len(seg) != seglen - 2:
            raise ImageError("truncated JPEG segment")
        body = pos + 2
        pos += seglen
        if marker == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                pq, tq_id = seg[p] >> 4, seg[p] & 15
                if pq != 0:
                    raise ImageError("16-bit quantization tables are not supported")
                if p + 65 > len(seg):
                    raise ImageError("truncated DQT")
                qt[tq_id] = list(seg[p + 1 : p + 65])
                p += 65
        elif marker == 0xC4:  # DHT
            p = 0
            while p + 17 <= len(seg):
                tc, th = seg[p] >> 4, seg[p] & 15
                counts = seg[p + 1 : p + 17]
                n = sum(counts)
                symbols = seg[p + 17 : p + 17 + n]
                if len(symbols) != n:
                    raise ImageError("truncated DHT")
                (dc if tc == 0 else ac)[th] = _Huffman(counts, symbols)
                p += 17 + n
        elif marker in (0xC0, 0xC1):  # SOF0/1: baseline & extended sequential
            prec, height, width, ncomp = struct.unpack(">BHHB", seg[:6])
            if prec != 8:
                raise ImageError(f"unsupported sample precision {prec}")
            if not (0 < width <= _MAX_DIM and 0 < height <= _MAX_DIM):
                raise ImageError(f"unreasonable dimensions {width}x{height}")
            if width * height > _MAX_PIXELS:
                raise ImageError(f"image too large ({width}x{height})")
            if ncomp not in (1, 3):
                raise ImageError(f"unsupported component count {ncomp} (gray or YCbCr only)")
            for i in range(ncomp):
                cid, hv, tq_id = seg[6 + i * 3 : 9 + i * 3]
                h, v = hv >> 4, hv & 15
                if h not in (1, 2) or v not in (1, 2):
                    raise ImageError(f"unsupported sampling factors {h}x{v}")
                comps.append(_Component(cid, h, v, tq_id))
        elif marker in (0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise ImageError("only baseline sequential JPEG is supported (progressive is not)")
        elif marker == 0xDD:  # DRI
            (restart_interval,) = struct.unpack(">H", seg[:2])
        elif marker == 0xDA:  # SOS
            if not comps:
                raise ImageError("SOS before SOF")
            ns = seg[0]
            if ns != len(comps):
                raise ImageError("partial-scan JPEG is not supported")
            by_id = {c.cid: c for c in comps}
            for i in range(ns):
                cid, tables = seg[1 + i * 2 : 3 + i * 2]
                if cid not in by_id:
                    raise ImageError(f"scan references unknown component {cid}")
                by_id[cid].td, by_id[cid].ta = tables >> 4, tables & 15
            scan_start = body + seglen - 2
            return _decode_scan(
                data, scan_start, width, height, comps, qt, dc, ac, restart_interval
            )
        elif marker == 0xD9:  # EOI before SOS
            raise ImageError("no image data (EOI before SOS)")
        # else: APPn/COM/DNL etc. -- skipped


def _decode_scan(
    data: bytes,
    pos: int,
    width: int,
    height: int,
    comps: list[_Component],
    qt: dict[int, list[int]],
    dc: dict[int, _Huffman],
    ac: dict[int, _Huffman],
    restart_interval: int,
) -> tuple[int, int, bytes]:
    hmax = max(c.h for c in comps)
    vmax = max(c.v for c in comps)
    mcux = -(-width // (8 * hmax))
    mcuy = -(-height // (8 * vmax))
    for c in comps:
        if c.tq not in qt:
            raise ImageError(f"component references missing quantization table {c.tq}")
        if c.td not in dc or c.ta not in ac:
            raise ImageError("scan references a missing Huffman table")
        c.pw = mcux * 8 * c.h
        c.ph = mcuy * 8 * c.v
        c.plane = bytearray(c.pw * c.ph)

    reader = _BitReader(data, pos)
    zz = _ZIGZAG
    todo = 0  # MCUs until the next expected restart marker
    for my in range(mcuy):
        for mx in range(mcux):
            if restart_interval:
                if todo == 0 and (my or mx):
                    reader.restart()
                    for c in comps:
                        c.pred = 0
                    todo = restart_interval
                elif todo == 0:
                    todo = restart_interval
                todo -= 1
            for c in comps:
                dqt = qt[c.tq]
                dct, act = dc[c.td], ac[c.ta]
                for by in range(c.v):
                    for bx in range(c.h):
                        coef = [0.0] * 64
                        t = reader.decode(dct)
                        c.pred += _extend(reader.receive(t), t)
                        coef[0] = c.pred * dqt[0]
                        k = 0
                        while k < 63:
                            rs = reader.decode(act)
                            r, s = rs >> 4, rs & 15
                            if s == 0:
                                if r != 15:
                                    break  # EOB
                                k += 16
                                continue
                            k += r + 1
                            if k > 63:
                                raise ImageError("AC coefficient index overran the block")
                            coef[zz[k]] = _extend(reader.receive(s), s) * dqt[k]
                        samples = _idct2d(coef)
                        ox = (mx * c.h + bx) * 8
                        oy = (my * c.v + by) * 8
                        plane, pw = c.plane, c.pw
                        for y in range(8):
                            off = (oy + y) * pw + ox
                            row = samples[y * 8 : (y + 1) * 8]
                            plane[off : off + 8] = bytes(
                                0 if t2 < 0 else 255 if t2 > 255 else t2
                                for t2 in (int(v + 128.5) for v in row)
                            )
    return width, height, _to_rgb_jpeg(width, height, comps, hmax, vmax)


def _to_rgb_jpeg(width: int, height: int, comps: list[_Component], hmax: int, vmax: int) -> bytes:
    if len(comps) == 1:
        y_ = comps[0]
        out = bytearray(width * height * 3)
        for row in range(height):
            line = y_.plane[row * y_.pw : row * y_.pw + width]
            for x, v in enumerate(line):
                o = (row * width + x) * 3
                out[o] = out[o + 1] = out[o + 2] = v
        return bytes(out)
    yc, cb, cr = comps
    out = bytearray(width * height * 3)
    for row in range(height):
        yoff = row * yc.pw
        cboff = (row * cb.v // vmax) * cb.pw
        croff = (row * cr.v // vmax) * cr.pw
        cbh, crh = cb.h, cr.h
        o = row * width * 3
        yplane, cbplane, crplane = yc.plane, cb.plane, cr.plane
        for x in range(width):
            y = yplane[yoff + x]
            u = cbplane[cboff + x * cbh // hmax] - 128
            v = crplane[croff + x * crh // hmax] - 128
            r = y + 1.402 * v
            g = y - 0.344136 * u - 0.714136 * v
            b = y + 1.772 * u
            out[o] = 0 if r < 0 else 255 if r > 255 else int(r + 0.5)
            out[o + 1] = 0 if g < 0 else 255 if g > 255 else int(g + 0.5)
            out[o + 2] = 0 if b < 0 else 255 if b > 255 else int(b + 0.5)
            o += 3
    return bytes(out)
