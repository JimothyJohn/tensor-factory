"""``.tfm`` -- tensor-factory model container: flat, versioned, protobuf-free.

ONNX stays the interchange format; ``.tfm`` is the *deployment* container for targets
where even a protobuf reader is unwelcome (MCU loaders, other languages). The whole
format is this docstring -- parseable in any language in an afternoon. All integers are
little-endian; strings are UTF-8 with a u16 length prefix.

::

    file    := magic "TFM1" | u16 version (=1) | u32 opset | graph
    graph   := str name
             | u16 n_inputs  | str*   (graph input names, initializers excluded)
             | u16 n_outputs | str*
             | u32 n_initializers | tensor*
             | u32 n_nodes        | node*
    tensor  := str name | u8 dtype | u8 rank | u32 dims[rank] | u64 nbytes | raw
    node    := str op_type | str name
             | u16 n_inputs  | str*
             | u16 n_outputs | str*
             | u16 n_attrs   | attr*
    attr    := str name | u8 kind | payload
    kind    := 0 int (i64) | 1 float (f32) | 2 ints (u32 count, i64*)
             | 3 floats (u32 count, f32*) | 4 bytes (u32 len, raw) | 5 tensor

``dtype`` uses the ONNX TensorProto enum (1=f32, 2=u8, 3=i8, 6=i32, 7=i64, 9=bool,
11=f64); ``raw`` is the packed little-endian data. Loading is bounds-checked and
allocation-capped so a hostile file raises :class:`TfmError` instead of exhausting
memory. Convert with ``tensor-factory pack``; :class:`tensor_factory.zerort.Runtime`
loads either format by sniffing the magic.
"""

from __future__ import annotations

import struct
from pathlib import Path

from .pb import _RAW_FMT, GraphDef, ModelDef, NodeDef, TensorDef

MAGIC = b"TFM1"
VERSION = 1

_MAX_COUNT = 1 << 20  # nodes/tensors/attrs per collection: hostile-header cap
_MAX_BLOB = 1 << 31  # single raw tensor payload cap (2 GB)

_KIND_INT, _KIND_FLOAT, _KIND_INTS, _KIND_FLOATS, _KIND_BYTES, _KIND_TENSOR = range(6)


class TfmError(ValueError):
    """Malformed or unsupported .tfm data."""


# --- writing --------------------------------------------------------------------------


def _w_str(out: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    if len(b) > 0xFFFF:
        raise TfmError(f"string too long ({len(b)} bytes)")
    out += struct.pack("<H", len(b))
    out += b


def _w_tensor(out: bytearray, t: TensorDef) -> None:
    fmt = _RAW_FMT.get(t.data_type)
    if fmt is None:
        raise TfmError(f"unsupported tensor dtype {t.data_type} in {t.name!r}")
    _w_str(out, t.name)
    out += struct.pack("<BB", t.data_type, len(t.dims))
    out += struct.pack(f"<{len(t.dims)}I", *t.dims)
    raw = struct.pack(f"<{len(t.data)}{fmt}", *t.data)
    out += struct.pack("<Q", len(raw))
    out += raw


def _w_attr(out: bytearray, name: str, value: object) -> None:
    _w_str(out, name)
    if isinstance(value, bool):
        raise TfmError(f"attribute {name!r}: bool is not a .tfm attribute type")
    if isinstance(value, int):
        out += struct.pack("<Bq", _KIND_INT, value)
    elif isinstance(value, float):
        out += struct.pack("<Bf", _KIND_FLOAT, value)
    elif isinstance(value, bytes):
        out += struct.pack("<BI", _KIND_BYTES, len(value))
        out += value
    elif isinstance(value, TensorDef):
        out += struct.pack("<B", _KIND_TENSOR)
        _w_tensor(out, value)
    elif isinstance(value, list) and all(isinstance(v, int) for v in value):
        out += struct.pack(f"<BI{len(value)}q", _KIND_INTS, len(value), *value)
    elif isinstance(value, list) and all(isinstance(v, int | float) for v in value):
        out += struct.pack(f"<BI{len(value)}f", _KIND_FLOATS, len(value), *value)
    else:
        raise TfmError(f"attribute {name!r}: unsupported value {type(value).__name__}")


def dump(model: ModelDef) -> bytes:
    """Serialize a parsed model to ``.tfm`` bytes."""
    g = model.graph
    out = bytearray(MAGIC)
    out += struct.pack("<HI", VERSION, model.opset)
    _w_str(out, g.name)
    init_names = {t.name for t in g.initializers}
    inputs = [n for n in g.input_names if n not in init_names]
    out += struct.pack("<H", len(inputs))
    for name in inputs:
        _w_str(out, name)
    out += struct.pack("<H", len(g.output_names))
    for name in g.output_names:
        _w_str(out, name)
    out += struct.pack("<I", len(g.initializers))
    for t in g.initializers:
        _w_tensor(out, t)
    out += struct.pack("<I", len(g.nodes))
    for n in g.nodes:
        _w_str(out, n.op_type)
        _w_str(out, n.name)
        out += struct.pack("<H", len(n.inputs))
        for name in n.inputs:
            _w_str(out, name)
        out += struct.pack("<H", len(n.outputs))
        for name in n.outputs:
            _w_str(out, name)
        out += struct.pack("<H", len(n.attrs))
        for k, v in n.attrs.items():
            _w_attr(out, k, v)
    return bytes(out)


# --- reading --------------------------------------------------------------------------


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise TfmError("truncated .tfm file")
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt: str) -> tuple:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def r_str(self) -> str:
        (n,) = self.unpack("<H")
        try:
            return self.take(n).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TfmError(f"malformed UTF-8 string: {exc}") from exc

    def r_count(self, fmt: str) -> int:
        (n,) = self.unpack(fmt)
        if n > _MAX_COUNT:
            raise TfmError(f"unreasonable element count {n}")
        return n

    def r_tensor(self) -> TensorDef:
        name = self.r_str()
        dtype, rank = self.unpack("<BB")
        fmt = _RAW_FMT.get(dtype)
        if fmt is None:
            raise TfmError(f"unsupported tensor dtype {dtype} in {name!r}")
        dims = self.unpack(f"<{rank}I")
        (nbytes,) = self.unpack("<Q")
        if nbytes > _MAX_BLOB:
            raise TfmError(f"unreasonable tensor payload ({nbytes} bytes)")
        width = struct.calcsize(fmt)
        if nbytes % width:
            raise TfmError(f"tensor {name!r}: payload not a multiple of element size")
        raw = self.take(nbytes)
        data = list(struct.unpack(f"<{nbytes // width}{fmt}", raw))
        return TensorDef(name=name, dims=dims, data_type=dtype, data=data)

    def r_attr(self) -> tuple[str, object]:
        name = self.r_str()
        (kind,) = self.unpack("<B")
        if kind == _KIND_INT:
            return name, self.unpack("<q")[0]
        if kind == _KIND_FLOAT:
            return name, self.unpack("<f")[0]
        if kind == _KIND_INTS:
            n = self.r_count("<I")
            return name, list(self.unpack(f"<{n}q"))
        if kind == _KIND_FLOATS:
            n = self.r_count("<I")
            return name, list(self.unpack(f"<{n}f"))
        if kind == _KIND_BYTES:
            n = self.r_count("<I")
            return name, self.take(n)
        if kind == _KIND_TENSOR:
            return name, self.r_tensor()
        raise TfmError(f"unknown attribute kind {kind}")


def load(source: bytes | str | Path) -> ModelDef:
    """Parse ``.tfm`` bytes (or a path) back into a :class:`ModelDef`."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    r = _Reader(data)
    if r.take(4) != MAGIC:
        raise TfmError("not a .tfm file (bad magic)")
    version, opset = r.unpack("<HI")
    if version != VERSION:
        raise TfmError(f"unsupported .tfm version {version} (this reader speaks {VERSION})")
    g = GraphDef(name=r.r_str())
    g.input_names = [r.r_str() for _ in range(r.r_count("<H"))]
    g.output_names = [r.r_str() for _ in range(r.r_count("<H"))]
    g.initializers = [r.r_tensor() for _ in range(r.r_count("<I"))]
    for _ in range(r.r_count("<I")):
        node = NodeDef(op_type=r.r_str(), name=r.r_str())
        node.inputs = [r.r_str() for _ in range(r.r_count("<H"))]
        node.outputs = [r.r_str() for _ in range(r.r_count("<H"))]
        node.attrs = dict(r.r_attr() for _ in range(r.r_count("<H")))
        g.nodes.append(node)
    if r.pos != len(data):
        raise TfmError(f"{len(data) - r.pos} trailing bytes after the graph")
    if not g.nodes and not g.initializers:
        raise TfmError("empty graph")
    return ModelDef(ir_version=0, opset=opset, graph=g)
