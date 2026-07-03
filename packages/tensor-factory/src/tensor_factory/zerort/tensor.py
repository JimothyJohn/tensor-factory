"""Flat-list tensors for the zero-dependency runtime.

A :class:`Tensor` is a flat Python list plus a shape and an ONNX dtype enum -- the
simplest representation that CPython executes quickly (C-speed ``list``/``map``/slice
machinery, no per-element objects beyond the numbers themselves). Broadcasting follows
numpy/ONNX rules; the hot shapes (same-shape, scalar, single-axis e.g. per-channel bias)
take vectorized fast paths and everything else falls back to an index-decomposition
path that is only used on small shape-computation tensors.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

from .pb import FLOAT32, INT64

_INT_TYPES = frozenset({2, 3, 6, 7, 9})  # UINT8, INT8, INT32, INT64, BOOL


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


@dataclass
class Tensor:
    """A dense tensor: ``shape``, ONNX ``dtype`` enum, and row-major flat ``data``."""

    shape: tuple[int, ...]
    dtype: int
    data: list

    def __post_init__(self) -> None:
        if len(self.data) != numel(self.shape):
            raise ValueError(f"data length {len(self.data)} != numel of shape {self.shape}")

    @property
    def is_int(self) -> bool:
        return self.dtype in _INT_TYPES

    def item(self) -> float | int:
        if len(self.data) != 1:
            raise ValueError(f"item() on tensor of shape {self.shape}")
        return self.data[0]

    @classmethod
    def scalar(cls, value: float | int, dtype: int) -> Tensor:
        return cls((), dtype, [value])

    @classmethod
    def from_ints(cls, values: list[int], dtype: int = INT64) -> Tensor:
        return cls((len(values),), dtype, list(values))


def broadcast_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Numpy-style broadcast of two shapes; raises on incompatibility."""
    rank = max(len(a), len(b))
    a = (1,) * (rank - len(a)) + a
    b = (1,) * (rank - len(b)) + b
    out = []
    for da, db in zip(a, b, strict=True):
        if da == db or db == 1:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            raise ValueError(f"cannot broadcast {a} with {b}")
    return tuple(out)


def _strides(shape: tuple[int, ...]) -> list[int]:
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * shape[i + 1]
    return out


def expand_data(t: Tensor, out_shape: tuple[int, ...]) -> list:
    """Materialize ``t``'s data broadcast to ``out_shape`` as a flat list."""
    if t.shape == out_shape:
        return t.data
    n = numel(out_shape)
    if len(t.data) == 1:
        return t.data * n
    rank = len(out_shape)
    shape = (1,) * (rank - len(t.shape)) + t.shape
    non1 = [i for i, d in enumerate(shape) if d != 1]
    if len(non1) == 1:
        # Single varying axis (e.g. a (1, C, 1, 1) bias against (1, C, H, W)):
        # repeat each value over the inner block, tile over the outer block.
        axis = non1[0]
        inner = math.prod(out_shape[axis + 1 :])
        outer = math.prod(out_shape[:axis])
        block: list = []
        for v in t.data:
            block.extend([v] * inner)
        return block * outer
    # General fallback -- per-element index decomposition. Only small tensors
    # (shape vectors, soft-argmax marginals) reach this path.
    in_strides = _strides(shape)
    out_strides = _strides(out_shape)
    out = []
    for flat in range(n):
        src = 0
        rem = flat
        for ax in range(rank):
            idx, rem = divmod(rem, out_strides[ax])
            if shape[ax] != 1:
                src += idx * in_strides[ax]
        out.append(t.data[src])
    return out


_PAIRWISE: dict[str, Callable] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


def _scalar_rhs(data: list, v, op: str) -> list:
    """``data <op> v`` with the operator inlined -- no per-element Python call."""
    if op == "add":
        return [x + v for x in data]
    if op == "sub":
        return [x - v for x in data]
    if op == "mul":
        return [x * v for x in data]
    return [x / v for x in data]


def _scalar_lhs(v, data: list, op: str) -> list:
    if op == "add":
        return [v + x for x in data]
    if op == "sub":
        return [v - x for x in data]
    if op == "mul":
        return [v * x for x in data]
    return [v / x for x in data]


def binary_op(a: Tensor, b: Tensor, op: str | Callable, dtype: int | None = None) -> Tensor:
    """Elementwise op over two broadcast tensors.

    ``op`` is one of the arithmetic names ("add"/"sub"/"mul"/"div"), dispatched to
    C-speed ``map`` / inlined-operator list comprehensions, or an arbitrary callable
    for the rare generic cases (integer division).
    """
    out_dtype = dtype if dtype is not None else (a.dtype if not a.is_int else b.dtype)
    fn = _PAIRWISE[op] if isinstance(op, str) else op
    if a.shape == b.shape:
        return Tensor(a.shape, out_dtype, list(map(fn, a.data, b.data)))
    out_shape = broadcast_shapes(a.shape, b.shape)
    if len(b.data) == 1:
        bv = b.data[0]
        da = expand_data(a, out_shape)
        data = _scalar_rhs(da, bv, op) if isinstance(op, str) else [fn(x, bv) for x in da]
        return Tensor(out_shape, out_dtype, data)
    if len(a.data) == 1:
        av = a.data[0]
        db = expand_data(b, out_shape)
        data = _scalar_lhs(av, db, op) if isinstance(op, str) else [fn(av, x) for x in db]
        return Tensor(out_shape, out_dtype, data)
    da = expand_data(a, out_shape)
    db = expand_data(b, out_shape)
    return Tensor(out_shape, out_dtype, list(map(fn, da, db)))


def result_dtype(a: Tensor, b: Tensor) -> int:
    """Dtype of a binary arithmetic result: float wins over int, else left operand."""
    if not a.is_int or not b.is_int:
        return FLOAT32 if a.is_int or b.is_int else a.dtype
    return a.dtype
