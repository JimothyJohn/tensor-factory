"""Minimal protobuf wire-format reader for ONNX model files -- stdlib only.

This is not a general protobuf library. It decodes exactly the subset of the ONNX
``ModelProto`` schema the :mod:`tensor_factory.zerort` engine needs -- graph, nodes,
attributes, initializers -- from the raw wire format (varints, length-delimited fields),
so loading a model requires neither ``protobuf`` nor ``onnx``. Field numbers below are
fixed by the ONNX .proto definition and stable across ONNX releases (protobuf fields are
append-only by contract).

Unknown fields are skipped, exactly as a generated protobuf parser would.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field

# TensorProto.DataType values (onnx.proto, stable enum).
FLOAT32 = 1
UINT8 = 2
INT8 = 3
INT32 = 6
INT64 = 7
BOOL = 9
DOUBLE = 11

_RAW_FMT = {FLOAT32: "f", UINT8: "B", INT8: "b", INT32: "i", INT64: "q", BOOL: "B", DOUBLE: "d"}


class WireError(ValueError):
    """Malformed protobuf wire data."""


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise WireError("truncated varint")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise WireError("varint too long")


def _signed(value: int) -> int:
    """Interpret a decoded varint as a two's-complement int64 (protobuf int32/int64)."""
    return value - (1 << 64) if value >= (1 << 63) else value


def _iter_fields(buf: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    """Yield ``(field_number, wire_type, payload)`` for each field in a message.

    Payload is an ``int`` for varint/fixed wire types and ``bytes`` for
    length-delimited ones.
    """
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        fnum, wtype = tag >> 3, tag & 7
        if wtype == 0:  # varint
            val, pos = _read_varint(buf, pos)
            yield fnum, wtype, val
        elif wtype == 1:  # 64-bit
            if pos + 8 > end:
                raise WireError("truncated fixed64")
            yield fnum, wtype, int.from_bytes(buf[pos : pos + 8], "little")
            pos += 8
        elif wtype == 2:  # length-delimited
            ln, pos = _read_varint(buf, pos)
            if pos + ln > end:
                raise WireError("truncated length-delimited field")
            yield fnum, wtype, buf[pos : pos + ln]
            pos += ln
        elif wtype == 5:  # 32-bit
            if pos + 4 > end:
                raise WireError("truncated fixed32")
            yield fnum, wtype, int.from_bytes(buf[pos : pos + 4], "little")
            pos += 4
        else:
            raise WireError(f"unsupported wire type {wtype}")


def _packed_varints(payload: bytes | int, wtype: int) -> list[int]:
    """A repeated int64 field arrives packed (one blob) or unpacked (one per tag)."""
    if wtype == 0:
        return [_signed(_as_int(payload))]
    payload = _as_bytes(payload)
    out: list[int] = []
    pos = 0
    while pos < len(payload):
        v, pos = _read_varint(payload, pos)
        out.append(_signed(v))
    return out


def _packed_f32(payload: bytes | int) -> list[float]:
    data = _as_bytes(payload)
    if len(data) % 4:
        raise WireError("packed float field length not a multiple of 4")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def _as_bytes(payload: bytes | int) -> bytes:
    if not isinstance(payload, bytes):
        raise WireError("expected a length-delimited field")
    return payload


def _utf8(payload: bytes | int) -> str:
    try:
        return _as_bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WireError(f"malformed UTF-8 in string field: {exc}") from exc


def _as_int(payload: bytes | int) -> int:
    if not isinstance(payload, int):
        raise WireError("expected a varint/fixed field")
    return payload


def _f32(bits: int) -> float:
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


@dataclass
class TensorDef:
    """A decoded ONNX TensorProto: shape, dtype enum, and a flat Python list."""

    name: str = ""
    dims: tuple[int, ...] = ()
    data_type: int = 0
    data: list[float] | list[int] = field(default_factory=list)


@dataclass
class NodeDef:
    op_type: str = ""
    name: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attrs: dict[str, object] = field(default_factory=dict)


@dataclass
class GraphDef:
    name: str = ""
    nodes: list[NodeDef] = field(default_factory=list)
    initializers: list[TensorDef] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


@dataclass
class ModelDef:
    ir_version: int = 0
    opset: int = 0
    graph: GraphDef = field(default_factory=GraphDef)


def _parse_tensor(buf: bytes) -> TensorDef:
    t = TensorDef()
    dims: list[int] = []
    raw = b""
    typed: list[float] | list[int] = []
    for fnum, wtype, payload in _iter_fields(buf):
        if fnum == 1:  # dims
            dims.extend(_packed_varints(payload, wtype))
        elif fnum == 2:  # data_type
            t.data_type = _as_int(payload)
        elif fnum == 4:  # float_data (packed fixed32)
            typed = _packed_f32(payload)
        elif fnum in (5, 7):  # int32_data / int64_data (packed varints)
            typed = list(_packed_varints(payload, wtype))
        elif fnum == 8:  # name
            t.name = _utf8(payload)
        elif fnum == 9:  # raw_data
            raw = _as_bytes(payload)
    t.dims = tuple(dims)
    if raw:
        fmt = _RAW_FMT.get(t.data_type)
        if fmt is None:
            raise WireError(f"unsupported tensor data_type {t.data_type} in {t.name!r}")
        width = struct.calcsize(fmt)
        if len(raw) % width:
            raise WireError(f"raw_data length {len(raw)} not a multiple of {width}")
        t.data = list(struct.unpack(f"<{len(raw) // width}{fmt}", raw))
    else:
        t.data = typed
    return t


def _parse_attribute(buf: bytes) -> tuple[str, object]:
    name = ""
    value: object = None
    ints: list[int] = []
    floats: list[float] = []
    for fnum, wtype, payload in _iter_fields(buf):
        if fnum == 1:
            name = _utf8(payload)
        elif fnum == 2:  # f (float, fixed32)
            value = _f32(_as_int(payload))
        elif fnum == 3:  # i
            value = _signed(_as_int(payload))
        elif fnum == 4:  # s
            value = _as_bytes(payload)
        elif fnum == 5:  # t (tensor)
            value = _parse_tensor(_as_bytes(payload))
        elif fnum == 7:  # floats (packed fixed32)
            floats.extend(_packed_f32(payload))
        elif fnum == 8:  # ints
            ints.extend(_packed_varints(payload, wtype))
    if ints:
        value = ints
    elif floats:
        value = floats
    return name, value


def _parse_node(buf: bytes) -> NodeDef:
    n = NodeDef()
    for fnum, _wtype, payload in _iter_fields(buf):
        if fnum == 1:
            n.inputs.append(_utf8(payload))
        elif fnum == 2:
            n.outputs.append(_utf8(payload))
        elif fnum == 3:
            n.name = _utf8(payload)
        elif fnum == 4:
            n.op_type = _utf8(payload)
        elif fnum == 5:
            k, v = _parse_attribute(_as_bytes(payload))
            n.attrs[k] = v
    return n


def _value_info_name(buf: bytes) -> str:
    for fnum, _wtype, payload in _iter_fields(buf):
        if fnum == 1:
            return _utf8(payload)
    return ""


def _parse_graph(buf: bytes) -> GraphDef:
    g = GraphDef()
    for fnum, _wtype, payload in _iter_fields(buf):
        if fnum == 1:
            g.nodes.append(_parse_node(_as_bytes(payload)))
        elif fnum == 2:
            g.name = _utf8(payload)
        elif fnum == 5:
            g.initializers.append(_parse_tensor(_as_bytes(payload)))
        elif fnum == 11:
            g.input_names.append(_value_info_name(_as_bytes(payload)))
        elif fnum == 12:
            g.output_names.append(_value_info_name(_as_bytes(payload)))
    return g


def parse_model(data: bytes) -> ModelDef:
    """Decode the bytes of an ``.onnx`` file into a :class:`ModelDef`."""
    m = ModelDef()
    for fnum, _wtype, payload in _iter_fields(data):
        if fnum == 1:
            m.ir_version = _as_int(payload)
        elif fnum == 7:
            m.graph = _parse_graph(_as_bytes(payload))
        elif fnum == 8:  # opset_import: OperatorSetIdProto {domain=1, version=2}
            for f2, _w2, p2 in _iter_fields(_as_bytes(payload)):
                if f2 == 2:
                    m.opset = max(m.opset, _signed(_as_int(p2)))
    if not m.graph.nodes and not m.graph.initializers:
        raise WireError("no graph found -- not an ONNX model?")
    return m
