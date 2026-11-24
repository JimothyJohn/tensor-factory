"""The .tfm container: exact round-trip against the ONNX parse, hostile bytes rejected.

A packed model is the *same graph* -- so the round-trip bar is exact equality of every
output on the real bundled models, not a tolerance. The adversarial side mirrors the
PNG/protobuf tests: truncations, bad magic, hostile counts, and fuzz must raise
``TfmError``, never an uncontrolled exception or a giant allocation.
"""

import contextlib
import random
import struct
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tensor_factory.cli import main as cli_main
from tensor_factory.zerort import Runtime, Tensor, TfmError, parse_model, tfm

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "packages/tensor-factory-mcp/src/tensor_factory_mcp/models"
CAM = MODELS / "helicoil-presence-cam-v1.onnx"
MOCK = MODELS / "helicoil-mock-v1.onnx"


def _pack(model_path: Path) -> bytes:
    return tfm.dump(parse_model(model_path.read_bytes()))


@pytest.mark.unit
@pytest.mark.parametrize("model_path", [MOCK, CAM], ids=lambda p: p.stem)
def test_tfm_roundtrip_runs_identically(model_path):
    blob = _pack(model_path)
    rng = random.Random(7)
    x = Tensor((1, 3, 480, 480), 1, [rng.random() for _ in range(3 * 480 * 480)])

    ref = Runtime(model_path).run({"image": x})
    got = Runtime(blob).run({"image": x})  # magic-sniffed

    assert set(got) == set(ref)
    for name in ref:
        assert got[name].shape == ref[name].shape
        assert got[name].data == ref[name].data  # same graph, same engine: exact


@pytest.mark.unit
def test_tfm_parse_preserves_the_graph():
    model = parse_model(CAM.read_bytes())
    again = tfm.load(tfm.dump(model))
    assert again.opset == model.opset
    g, g2 = model.graph, again.graph
    assert [n.op_type for n in g2.nodes] == [n.op_type for n in g.nodes]
    assert g2.output_names == g.output_names
    assert {t.name: (t.dims, t.data_type) for t in g2.initializers} == {
        t.name: (t.dims, t.data_type) for t in g.initializers
    }
    # dump() drops initializer names from the input list (IR<4 style graphs);
    # the runtime-visible inputs must survive exactly.
    inits = {t.name for t in g.initializers}
    assert [n for n in g2.input_names if n not in inits] == [
        n for n in g.input_names if n not in inits
    ]


@pytest.mark.unit
def test_cli_pack_produces_a_working_model(tmp_path, capsys):
    out = tmp_path / "cam.tfm"
    assert cli_main(["pack", "--model", str(CAM), "--out", str(out)]) == 0
    assert out.exists()
    assert f"{out}" in capsys.readouterr().out
    rt = Runtime(out)
    assert rt.output_names == ["box", "presence"]

    from tensor_factory.inference import Detector

    det = Detector(out)  # .tfm forces the zerort engine
    assert det.engine == "zerort"
    with pytest.raises(ValueError, match="zerort"):
        Detector(out, engine="ort")


@pytest.mark.unit
def test_tfm_rejects_hostile_input():
    blob = _pack(MOCK)
    with pytest.raises(TfmError, match="magic"):
        tfm.load(b"nope" + blob[4:])
    for cut in (2, 10, 100, len(blob) - 3):
        with pytest.raises(TfmError):
            tfm.load(blob[:cut])
    with pytest.raises(TfmError, match="version"):
        tfm.load(blob[:4] + struct.pack("<H", 99) + blob[6:])
    with pytest.raises(TfmError, match="trailing"):
        tfm.load(blob + b"\x00")
    # A crafted header declaring a giant tensor payload must be rejected before
    # any allocation is attempted.
    hostile = bytearray(tfm.MAGIC)
    hostile += struct.pack("<HI", tfm.VERSION, 17)
    hostile += struct.pack("<H", 0)  # graph name ""
    hostile += struct.pack("<H", 0)  # no inputs
    hostile += struct.pack("<H", 0)  # no outputs
    hostile += struct.pack("<I", 1)  # one initializer...
    hostile += struct.pack("<H", 1) + b"t"  # named "t"
    hostile += struct.pack("<BB", 1, 0)  # f32, rank 0
    hostile += struct.pack("<Q", 1 << 40)  # ...claiming a 1 TB payload
    with pytest.raises(TfmError, match="unreasonable"):
        tfm.load(bytes(hostile))


@pytest.mark.unit
@given(data=st.binary(min_size=0, max_size=300))
@settings(max_examples=200, deadline=None)
def test_tfm_loader_never_raises_uncontrolled(data):
    with contextlib.suppress(TfmError):
        tfm.load(tfm.MAGIC + data)


@pytest.mark.unit
def test_tfm_is_not_larger_than_the_onnx_it_came_from():
    for model_path in (MOCK, CAM):
        assert len(_pack(model_path)) <= model_path.stat().st_size * 1.02
