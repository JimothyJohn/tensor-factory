"""Operator kernels for the zero-dependency runtime -- stdlib only.

Implements exactly the ONNX ops a dynamically-quantized int8 export of the tiny
detector uses (opset 13-17 semantics). Unsupported ops or attribute combinations
raise ``UnsupportedOp`` loudly rather than guessing.

The one kernel where pure Python would normally be hopeless is ``ConvInteger`` --
~200M multiply-accumulates per frame at 480 px. It runs on a bigint-SIMD scheme:
an image plane is packed into one huge Python integer (4-byte lanes), so each
kernel tap becomes a *single* arbitrary-precision multiply-add executed by
CPython's C bigint machinery instead of tens of thousands of interpreted Python
operations. Positive and negative weights accumulate separately (lanes must stay
non-negative), and the shared ``x_zero_point`` term is folded out algebraically:

    sum((x - xzp) * (w - wzp)) == sum(x * w') - xzp * sum(w'),  w' = w - wzp

which holds everywhere because ONNX pads ``ConvInteger`` inputs with ``x_zero_point``
(we pad with ``xzp`` so padded taps contribute ``xzp * w'`` and the correction is
uniform). Lane width is chosen from the actual weight mass so overflow is impossible.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .pb import FLOAT32, INT32, INT64, UINT8, TensorDef
from .tensor import Tensor, binary_op, expand_data, numel, result_dtype


class UnsupportedOp(NotImplementedError):
    """The graph uses an op/attribute this minimal runtime does not implement."""


OpFn = Callable[[list["Tensor | None"], dict[str, object]], list[Tensor]]
OPS: dict[str, OpFn] = {}


def _register(name: str) -> Callable[[OpFn], OpFn]:
    def deco(fn: OpFn) -> OpFn:
        OPS[name] = fn
        return fn

    return deco


def _int_attr(attrs: dict[str, object], name: str, default: int) -> int:
    v = attrs.get(name, default)
    if isinstance(v, bool) or not isinstance(v, int):
        raise UnsupportedOp(f"attribute {name!r} is not an int")
    return v


def _ints_attr(attrs: dict[str, object], name: str, default: list[int]) -> list[int]:
    v = attrs.get(name)
    if v is None:
        return default
    if not isinstance(v, list):
        raise UnsupportedOp(f"attribute {name!r} is not an int list")
    out: list[int] = []
    for i in v:
        if isinstance(i, bool) or not isinstance(i, int):
            raise UnsupportedOp(f"attribute {name!r} is not an int list")
        out.append(i)
    return out


def _req(inputs: list[Tensor | None], i: int) -> Tensor:
    t = inputs[i] if i < len(inputs) else None
    if t is None:
        raise UnsupportedOp(f"missing required input #{i}")
    return t


def tensor_from_def(td: TensorDef) -> Tensor:
    return Tensor(td.dims, td.data_type, list(td.data))


# --- shape plumbing -------------------------------------------------------------------


@_register("Constant")
def _constant(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    value = attrs.get("value")
    if not isinstance(value, TensorDef):
        raise UnsupportedOp("Constant without a tensor 'value' attribute")
    return [tensor_from_def(value)]


@_register("Shape")
def _shape(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    return [Tensor.from_ints(list(_req(inputs, 0).shape))]


@_register("Reshape")
def _reshape(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    data = _req(inputs, 0)
    target = [int(v) for v in _req(inputs, 1).data]
    shape = [data.shape[i] if v == 0 else v for i, v in enumerate(target)]
    if shape.count(-1) > 1:
        raise UnsupportedOp("Reshape with multiple -1 dims")
    if -1 in shape:
        known = math.prod(d for d in shape if d != -1)
        shape[shape.index(-1)] = len(data.data) // known if known else 0
    if math.prod(shape) != len(data.data):
        raise ValueError(f"cannot reshape {data.shape} -> {tuple(shape)}")
    return [Tensor(tuple(shape), data.dtype, data.data)]


@_register("Unsqueeze")
def _unsqueeze(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    data = _req(inputs, 0)
    axes = [int(a) for a in _req(inputs, 1).data]
    rank = len(data.shape) + len(axes)
    axes = sorted(a + rank if a < 0 else a for a in axes)
    shape = list(data.shape)
    for a in axes:
        shape.insert(a, 1)
    return [Tensor(tuple(shape), data.dtype, data.data)]


@_register("Concat")
def _concat(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    ts = [_req(inputs, i) for i in range(len(inputs))]
    rank = len(ts[0].shape)
    axis = _int_attr(attrs, "axis", 0)
    axis += rank if axis < 0 else 0
    out_shape = list(ts[0].shape)
    out_shape[axis] = sum(t.shape[axis] for t in ts)
    inner = math.prod(ts[0].shape[axis + 1 :])
    outer = math.prod(ts[0].shape[:axis])
    if outer == 1:
        data: list = []
        for t in ts:
            data.extend(t.data)
    else:
        blocks = [t.shape[axis] * inner for t in ts]
        data = []
        for o in range(outer):
            for t, blk in zip(ts, blocks, strict=True):
                data.extend(t.data[o * blk : (o + 1) * blk])
    return [Tensor(tuple(out_shape), ts[0].dtype, data)]


@_register("Gather")
def _gather(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    data = _req(inputs, 0)
    indices = _req(inputs, 1)
    axis = _int_attr(attrs, "axis", 0)
    axis += len(data.shape) if axis < 0 else 0
    da = data.shape[axis]
    inner = math.prod(data.shape[axis + 1 :])
    outer = math.prod(data.shape[:axis])
    idx = [int(i) + da if int(i) < 0 else int(i) for i in indices.data]
    out: list = []
    for o in range(outer):
        base = o * da * inner
        for i in idx:
            out.append(data.data[base + i * inner : base + (i + 1) * inner])
    flat = [v for chunk in out for v in chunk]
    shape = data.shape[:axis] + indices.shape + data.shape[axis + 1 :]
    return [Tensor(shape, data.dtype, flat)]


@_register("Range")
def _range(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    start, limit, delta = (_req(inputs, i).item() for i in range(3))
    n = max(0, math.ceil((limit - start) / delta))
    dtype = _req(inputs, 0).dtype
    return [Tensor((n,), dtype, [start + i * delta for i in range(n)])]


@_register("Cast")
def _cast(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    data = _req(inputs, 0)
    to = _int_attr(attrs, "to", 0)
    if to == FLOAT32:
        return [Tensor(data.shape, FLOAT32, list(map(float, data.data)))]
    if to in (INT32, INT64):
        # ONNX Cast float->int truncates toward zero (int() on floats).
        return [Tensor(data.shape, to, list(map(int, data.data)))]
    raise UnsupportedOp(f"Cast to dtype {to}")


# --- elementwise math -----------------------------------------------------------------


def _int_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


@_register("Add")
def _add(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    a, b = _req(inputs, 0), _req(inputs, 1)
    return [binary_op(a, b, "add", result_dtype(a, b))]


@_register("Sub")
def _sub(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    a, b = _req(inputs, 0), _req(inputs, 1)
    return [binary_op(a, b, "sub", result_dtype(a, b))]


@_register("Mul")
def _mul(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    a, b = _req(inputs, 0), _req(inputs, 1)
    return [binary_op(a, b, "mul", result_dtype(a, b))]


@_register("Div")
def _div(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    a, b = _req(inputs, 0), _req(inputs, 1)
    if a.is_int and b.is_int:
        return [binary_op(a, b, _int_div, result_dtype(a, b))]
    return [binary_op(a, b, "div", result_dtype(a, b))]


@_register("Relu")
def _relu(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    zero = 0 if t.is_int else 0.0
    return [Tensor(t.shape, t.dtype, [v if v > zero else zero for v in t.data])]


@_register("Exp")
def _exp(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    return [Tensor(t.shape, FLOAT32, [math.exp(v) for v in t.data])]


@_register("Sigmoid")
def _sigmoid(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    return [Tensor(t.shape, FLOAT32, [1.0 / (1.0 + math.exp(-v)) for v in t.data])]


# --- reductions and softmax -----------------------------------------------------------


def _normalize_axes(axes: Sequence[int] | None, rank: int) -> tuple[int, ...]:
    if axes is None or len(axes) == 0:
        return tuple(range(rank))
    return tuple(sorted(a + rank if a < 0 else a for a in axes))


def _reduce(t: Tensor, axes: tuple[int, ...], keepdims: bool, kind: str) -> Tensor:
    rank = len(t.shape)
    out_shape = tuple(
        d if i not in axes else 1 for i, d in enumerate(t.shape) if keepdims or i not in axes
    )
    # Fast path: reducing a contiguous trailing block (global pooling, full reduce).
    if axes == tuple(range(rank - len(axes), rank)):
        inner = math.prod(t.shape[i] for i in axes)
        chunks = (t.data[i : i + inner] for i in range(0, len(t.data), inner))
        if kind == "sum":
            data = [sum(c) for c in chunks]
        elif kind == "mean":
            data = [sum(c) / inner for c in chunks]
        else:
            data = [max(c) for c in chunks]
        return Tensor(out_shape, FLOAT32 if kind == "mean" else t.dtype, data)
    # General path: per-element index decomposition (small tensors only).
    kept = [i for i in range(rank) if i not in axes]
    out_dims = [t.shape[i] for i in kept] or [1]
    out_strides = [1] * len(out_dims)
    for i in range(len(out_dims) - 2, -1, -1):
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1]
    in_strides = [1] * rank
    for i in range(rank - 2, -1, -1):
        in_strides[i] = in_strides[i + 1] * t.shape[i + 1]
    n_out = math.prod(out_dims)
    acc: list = [None] * n_out
    for flat, v in enumerate(t.data):
        o = 0
        for pos, i in enumerate(kept):
            o += ((flat // in_strides[i]) % t.shape[i]) * out_strides[pos]
        cur = acc[o]
        if cur is None:
            acc[o] = v
        elif kind == "max":
            acc[o] = v if v > cur else cur
        else:
            acc[o] = cur + v
    if kind == "mean":
        count = math.prod(t.shape[i] for i in axes)
        acc = [v / count for v in acc]
    return Tensor(out_shape, FLOAT32 if kind == "mean" else t.dtype, acc)


def _reduce_op(kind: str, axes_from_input: bool) -> OpFn:
    def fn(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
        t = _req(inputs, 0)
        if axes_from_input:
            axes_t = inputs[1] if len(inputs) > 1 else None
            axes = [int(a) for a in axes_t.data] if axes_t is not None else None
        else:
            axes = _ints_attr(attrs, "axes", []) or None
        keepdims = bool(attrs.get("keepdims", 1))
        return [_reduce(t, _normalize_axes(axes, len(t.shape)), keepdims, kind)]

    return fn


# Opset 13-17: ReduceSum takes axes as an input; ReduceMean/ReduceMax as an attribute.
OPS["ReduceSum"] = _reduce_op("sum", axes_from_input=True)
OPS["ReduceMean"] = _reduce_op("mean", axes_from_input=False)
OPS["ReduceMax"] = _reduce_op("max", axes_from_input=False)


@_register("Softmax")
def _softmax(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    rank = len(t.shape)
    axis = _int_attr(attrs, "axis", -1)
    axis += rank if axis < 0 else 0
    if axis != rank - 1:
        raise UnsupportedOp(f"Softmax over non-trailing axis {axis}")
    n = t.shape[-1]
    out: list[float] = []
    for i in range(0, len(t.data), n):
        row = t.data[i : i + n]
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        out.extend(e / s for e in exps)
    return [Tensor(t.shape, FLOAT32, out)]


# --- quantized kernels ----------------------------------------------------------------


def _round_f32(v: float) -> float:
    """Round a Python float (f64) to the nearest float32, like C's ``(float)v``."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


@_register("DynamicQuantizeLinear")
def _dynamic_quantize(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    lo = min(0.0, min(t.data))
    hi = max(0.0, max(t.data))
    scale = _round_f32((hi - lo) / 255.0)
    if scale == 0.0:
        scale = 1.0
    zp = round(-lo / scale)  # round() is half-to-even, matching the spec
    zp = 0 if zp < 0 else 255 if zp > 255 else zp
    # scale/zp are derived from the actual min/max, so values land in [0, 255] except
    # for the odd rounding overshoot at the extremes -- clip lazily, only if needed.
    q = [round(v / scale) + zp for v in t.data]
    if q and (min(q) < 0 or max(q) > 255):
        q = [0 if v < 0 else 255 if v > 255 else v for v in q]
    return [
        Tensor(t.shape, UINT8, q),
        Tensor.scalar(scale, FLOAT32),
        Tensor.scalar(zp, UINT8),
    ]


@dataclass
class ConvPrep:
    """Input-independent ConvInteger work, hoisted to model-load time.

    ``wq`` is the flat weight tensor with the weight zero point folded out; ``sum_w``
    the per-filter tap sums (the x-zero-point correction term); ``lane`` the
    accumulator lane width in bytes, sized from the worst-case one-sided weight mass
    so bigint lanes cannot overflow.
    """

    wq: list[int]
    sum_w: list[int]
    lane: int


def prepare_conv_integer(w: Tensor, wzp: int) -> ConvPrep:
    m = w.shape[0]
    taps = len(w.data) // m if m else 0
    wq = [int(v) - wzp for v in w.data]
    sum_w = [sum(wq[f * taps : (f + 1) * taps]) for f in range(m)]
    mass = 0
    for f in range(m):
        fw = wq[f * taps : (f + 1) * taps]
        mass = max(mass, sum(v for v in fw if v > 0), -sum(v for v in fw if v < 0))
    max_lane = 255 * mass
    if max_lane < 1 << 32:
        lane = 4
    elif max_lane < 1 << 64:
        lane = 8
    else:
        raise UnsupportedOp("integer conv accumulation exceeds 64-bit lanes")
    return ConvPrep(wq, sum_w, lane)


@_register("ConvInteger")
def _conv_integer(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    x = _req(inputs, 0)
    w = _req(inputs, 1)
    xzp = int(inputs[2].item()) if len(inputs) > 2 and inputs[2] is not None else 0
    wzp = int(inputs[3].item()) if len(inputs) > 3 and inputs[3] is not None else 0
    if x.dtype != UINT8:
        raise UnsupportedOp(f"ConvInteger input dtype {x.dtype} (expected uint8)")
    if len(x.shape) != 4 or x.shape[0] != 1:
        raise UnsupportedOp(f"ConvInteger input shape {x.shape} (expected (1, C, H, W))")
    if _int_attr(attrs, "group", 1) != 1:
        raise UnsupportedOp("grouped ConvInteger")
    if any(d != 1 for d in _ints_attr(attrs, "dilations", [1, 1])):
        raise UnsupportedOp("dilated ConvInteger")
    _, cin, h, wd = x.shape
    m, cin_w, kh, kw = w.shape
    if cin_w != cin:
        raise ValueError(f"weight channels {cin_w} != input channels {cin}")
    pt, pl, pb, pr = _ints_attr(attrs, "pads", [0, 0, 0, 0])
    sh, sw = _ints_attr(attrs, "strides", [1, 1])
    ho = (h + pt + pb - kh) // sh + 1
    wo = (wd + pl + pr - kw) // sw + 1
    nl = ho * wo

    # Input-independent weight prep is hoisted to Runtime init when the weights are
    # initializers (the usual case); recompute only for dynamic weights.
    prep = attrs.get("_prep")
    if not isinstance(prep, ConvPrep):
        prep = prepare_conv_integer(w, wzp)
    wq, sum_w, lane = prep.wq, prep.sum_w, prep.lane
    taps = cin * kh * kw
    fmt = f"<{nl}I" if lane == 4 else f"<{nl}Q"
    nbytes = nl * lane

    # Pad each input plane with the x zero point (ONNX ConvInteger padding semantics).
    # Planes are kept as per-row ``bytes`` so the shifted/strided views below are
    # C-speed slices and joins, not Python-level element loops.
    wp = wd + pl + pr
    xzp_b = bytes([xzp])
    pad_row = xzp_b * wp
    lpad, rpad = xzp_b * pl, xzp_b * pr
    planes: list[list[bytes]] = []
    for c in range(cin):
        off = c * h * wd
        rows = [pad_row] * pt
        rows += [lpad + bytes(x.data[off + r * wd : off + (r + 1) * wd]) + rpad for r in range(h)]
        rows += [pad_row] * pb
        planes.append(rows)

    acc_pos = [0] * m
    acc_neg = [0] * m
    span = (wo - 1) * sw + 1
    buf = bytearray(nbytes)  # uint8 plane values spread onto `lane`-byte lanes
    for c in range(cin):
        rows = planes[c]
        for ky in range(kh):
            for kx in range(kw):
                buf[::lane] = b"".join(rows[oy * sh + ky][kx : kx + span : sw] for oy in range(ho))
                plane_int = int.from_bytes(buf, "little")
                base = (c * kh + ky) * kw + kx
                for f in range(m):
                    wv = wq[f * taps + base]
                    if wv > 0:
                        acc_pos[f] += wv * plane_int
                    elif wv < 0:
                        acc_neg[f] -= wv * plane_int

    out: list[int] = []
    for f in range(m):
        pos = struct.unpack(fmt, acc_pos[f].to_bytes(nbytes, "little"))
        neg = struct.unpack(fmt, acc_neg[f].to_bytes(nbytes, "little"))
        corr = xzp * sum_w[f]
        out.extend([p - q - corr for p, q in zip(pos, neg, strict=True)])
    return [Tensor((1, m, ho, wo), INT32, out)]


@_register("MatMulInteger")
def _matmul_integer(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    a = _req(inputs, 0)
    b = _req(inputs, 1)
    azp = int(inputs[2].item()) if len(inputs) > 2 and inputs[2] is not None else 0
    bzp = int(inputs[3].item()) if len(inputs) > 3 and inputs[3] is not None else 0
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise UnsupportedOp(f"MatMulInteger shapes {a.shape} x {b.shape} (2-D only)")
    ma, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError(f"MatMulInteger inner dims {k} != {kb}")
    bt = [[int(b.data[r * n + col]) - bzp for r in range(k)] for col in range(n)]
    out: list[int] = []
    for r in range(ma):
        row = [int(v) - azp for v in a.data[r * k : (r + 1) * k]]
        for col in range(n):
            bc = bt[col]
            out.append(sum(x * y for x, y in zip(row, bc, strict=True)))
    return [Tensor((ma, n), INT32, out)]


def apply(op_type: str, inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    fn = OPS.get(op_type)
    if fn is None:
        raise UnsupportedOp(f"op {op_type!r} is not implemented by zerort")
    return fn(inputs, attrs)


def _identity(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    return [Tensor(t.shape, t.dtype, list(t.data))]


OPS["Identity"] = _identity


def _flatten(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    axis = _int_attr(attrs, "axis", 1)
    axis += len(t.shape) if axis < 0 else 0
    lead = math.prod(t.shape[:axis])
    return [Tensor((lead, len(t.data) // lead if lead else 0), t.dtype, t.data)]


OPS["Flatten"] = _flatten


def _expand_op(inputs: list[Tensor | None], attrs: dict[str, object]) -> list[Tensor]:
    t = _req(inputs, 0)
    from .tensor import broadcast_shapes

    target = tuple(int(v) for v in _req(inputs, 1).data)
    out_shape = broadcast_shapes(t.shape, target)
    return [Tensor(out_shape, t.dtype, expand_data(t, out_shape))]


OPS["Expand"] = _expand_op


__all__ = ["OPS", "UnsupportedOp", "apply", "numel", "tensor_from_def"]
