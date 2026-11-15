"""Op-level parity: every zerort kernel against onnxruntime on hand-built graphs.

onnxruntime is the reference implementation; each test builds a tiny single-op (or
few-op) ONNX graph, runs it on both engines with seeded random inputs, and compares.
Integer kernels must match exactly -- they are exact integer arithmetic on both sides.
Float kernels get a tight tolerance for f32-vs-f64 accumulation-order differences.
"""

import random

import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
np = pytest.importorskip("numpy")

from onnx import TensorProto, helper  # noqa: E402

from tensor_factory.zerort import Runtime, Tensor  # noqa: E402
from tensor_factory.zerort.pb import FLOAT32, INT64, UINT8  # noqa: E402

rng = random.Random(20260702)

_NP_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT64: np.int64,
}


def _run_both(nodes, inputs, outputs, feeds, initializers=()):
    """Build a model from nodes, run ort and zerort on identical feeds, return both."""
    graph = helper.make_graph(list(nodes), "t", list(inputs), list(outputs), list(initializers))
    # Pin the IR version opset 17 shipped with: onnx stamps its own newest IR by default,
    # which a locked onnxruntime one release behind refuses to load.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    blob = model.SerializeToString()

    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    np_feeds = {k: np.asarray(v[1], dtype=_NP_DTYPE[v[0]]).reshape(v[2]) for k, v in feeds.items()}
    ref = sess.run(None, np_feeds)

    rt = Runtime(blob)
    zr_feeds = {k: Tensor(tuple(v[2]), int(v[0]), list(v[1])) for k, v in feeds.items()}
    out = rt.run(zr_feeds)
    mine = [out[o.name] for o in outputs]
    for r, m in zip(ref, mine, strict=True):
        assert tuple(r.shape) == m.shape, f"shape {r.shape} != {m.shape}"
    return ref, mine


def _assert_exact(ref, mine):
    for r, m in zip(ref, mine, strict=True):
        assert r.reshape(-1).tolist() == m.data


def _assert_close(ref, mine, tol=1e-5):
    for r, m in zip(ref, mine, strict=True):
        diffs = [abs(a - b) for a, b in zip(r.reshape(-1).tolist(), m.data, strict=True)]
        assert max(diffs, default=0.0) <= tol


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cin", "cout", "hw", "k", "stride", "pad"),
    [(3, 4, 12, 3, 2, 1), (2, 3, 9, 3, 1, 1), (4, 2, 7, 1, 1, 0), (1, 5, 10, 3, 2, 0)],
)
def test_conv_integer_matches_ort_exactly(cin, cout, hw, k, stride, pad):
    x = [rng.randrange(256) for _ in range(cin * hw * hw)]
    w = [rng.randrange(256) for _ in range(cout * cin * k * k)]
    xzp, wzp = rng.randrange(256), rng.randrange(256)
    node = helper.make_node(
        "ConvInteger",
        ["x", "w", "xzp", "wzp"],
        ["y"],
        kernel_shape=[k, k],
        pads=[pad] * 4,
        strides=[stride, stride],
    )
    ho = (hw + 2 * pad - k) // stride + 1
    ref, mine = _run_both(
        [node],
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, cin, hw, hw])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [1, cout, ho, ho])],
        {"x": (TensorProto.UINT8, x, [1, cin, hw, hw])},
        [
            helper.make_tensor("w", TensorProto.UINT8, [cout, cin, k, k], w),
            helper.make_tensor("xzp", TensorProto.UINT8, [], [xzp]),
            helper.make_tensor("wzp", TensorProto.UINT8, [], [wzp]),
        ],
    )
    _assert_exact(ref, mine)


@pytest.mark.unit
def test_matmul_integer_matches_ort_exactly():
    m, k, n = 3, 17, 5
    a = [rng.randrange(256) for _ in range(m * k)]
    b = [rng.randrange(256) for _ in range(k * n)]
    node = helper.make_node("MatMulInteger", ["a", "b", "azp", "bzp"], ["y"])
    ref, mine = _run_both(
        [node],
        [helper.make_tensor_value_info("a", TensorProto.UINT8, [m, k])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [m, n])],
        {"a": (TensorProto.UINT8, a, [m, k])},
        [
            helper.make_tensor("b", TensorProto.UINT8, [k, n], b),
            helper.make_tensor("azp", TensorProto.UINT8, [], [7]),
            helper.make_tensor("bzp", TensorProto.UINT8, [], [131]),
        ],
    )
    _assert_exact(ref, mine)


@pytest.mark.unit
def test_dynamic_quantize_linear_matches_ort():
    data = [rng.uniform(-3.0, 5.0) for _ in range(1000)] + [0.0, -3.0, 5.0]
    data = [float(np.float32(v)) for v in data]
    node = helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "scale", "zp"])
    ref, mine = _run_both(
        [node],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [len(data)])],
        [
            helper.make_tensor_value_info("y", TensorProto.UINT8, [len(data)]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("zp", TensorProto.UINT8, []),
        ],
        {"x": (TensorProto.FLOAT, data, [len(data)])},
    )
    # Scale and zero point must agree exactly; quantized lanes may differ by 1 where
    # x/scale lands on a rounding boundary that f32 and f64 division resolve differently.
    assert float(ref[1]) == pytest.approx(mine[1].data[0], rel=1e-7)
    assert int(ref[2]) == mine[2].data[0]
    lane_diffs = [abs(int(a) - b) for a, b in zip(ref[0].tolist(), mine[0].data, strict=True)]
    assert max(lane_diffs) <= 1
    assert sum(1 for d in lane_diffs if d) <= len(data) // 100


@pytest.mark.unit
def test_softmax_and_reductions_match_ort():
    data = [rng.uniform(-4.0, 4.0) for _ in range(2 * 900)]
    nodes = [
        helper.make_node("Softmax", ["x"], ["sm"], axis=-1),
        helper.make_node("ReduceMean", ["x"], ["mean"], axes=[1], keepdims=0),
        helper.make_node("ReduceMax", ["x"], ["mx"], axes=[1], keepdims=1),
        helper.make_node("ReduceSum", ["sm", "axes"], ["s"], keepdims=0),
    ]
    ref, mine = _run_both(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 900])],
        [
            helper.make_tensor_value_info("sm", TensorProto.FLOAT, [2, 900]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("mx", TensorProto.FLOAT, [2, 1]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [2]),
        ],
        {"x": (TensorProto.FLOAT, data, [2, 900])},
        [helper.make_tensor("axes", TensorProto.INT64, [1], [1])],
    )
    _assert_close(ref, mine, tol=1e-5)


@pytest.mark.unit
def test_shape_plumbing_matches_ort():
    """Shape/Gather/Unsqueeze/Concat/Range/Reshape/Cast -- the soft-argmax scaffolding."""
    data = [rng.uniform(0.0, 1.0) for _ in range(2 * 3 * 4)]
    nodes = [
        helper.make_node("Shape", ["x"], ["shp"]),
        helper.make_node("Gather", ["shp", "gidx"], ["dim2"], axis=0),
        helper.make_node("Unsqueeze", ["dim2", "uax"], ["dim2u"]),
        helper.make_node("Concat", ["dim2u", "dim2u"], ["cc"], axis=0),
        helper.make_node("Cast", ["dim2"], ["dim2f"], to=TensorProto.FLOAT),
        helper.make_node("Range", ["zero", "dim2", "one"], ["rng_"]),
        helper.make_node("Reshape", ["x", "newshape"], ["xr"]),
    ]
    ref, mine = _run_both(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [
            helper.make_tensor_value_info("cc", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("dim2f", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("rng_", TensorProto.INT64, [4]),
            helper.make_tensor_value_info("xr", TensorProto.FLOAT, [2, 12]),
        ],
        {"x": (TensorProto.FLOAT, data, [2, 3, 4])},
        [
            helper.make_tensor("gidx", TensorProto.INT64, [], [2]),
            helper.make_tensor("uax", TensorProto.INT64, [1], [0]),
            helper.make_tensor("zero", TensorProto.INT64, [], [0]),
            helper.make_tensor("one", TensorProto.INT64, [], [1]),
            helper.make_tensor("newshape", TensorProto.INT64, [2], [2, -1]),
        ],
    )
    for r, m in zip(ref, mine, strict=True):
        assert [float(v) for v in r.reshape(-1)] == pytest.approx(m.data)


@pytest.mark.unit
def test_elementwise_broadcast_matches_ort():
    """Add/Sub/Mul/Div with the per-channel-bias broadcast shape the detector uses."""
    x = [rng.uniform(-2.0, 2.0) for _ in range(1 * 3 * 4 * 4)]
    bias = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    nodes = [
        helper.make_node("Add", ["x", "b"], ["a1"]),
        helper.make_node("Mul", ["a1", "scalar"], ["a2"]),
        helper.make_node("Sub", ["a2", "x"], ["a3"]),
        helper.make_node("Div", ["a3", "scalar"], ["y"]),
        helper.make_node("Relu", ["y"], ["yr"]),
        helper.make_node("Exp", ["scalar"], ["es"]),
    ]
    ref, mine = _run_both(
        nodes,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [
            helper.make_tensor_value_info("yr", TensorProto.FLOAT, [1, 3, 4, 4]),
            helper.make_tensor_value_info("es", TensorProto.FLOAT, [1]),
        ],
        {"x": (TensorProto.FLOAT, x, [1, 3, 4, 4])},
        [
            helper.make_tensor("b", TensorProto.FLOAT, [1, 3, 1, 1], bias),
            helper.make_tensor("scalar", TensorProto.FLOAT, [1], [0.37]),
        ],
    )
    _assert_close(ref, mine, tol=1e-5)


@pytest.mark.unit
def test_unsupported_op_raises_loudly(tmp_path):
    node = helper.make_node("LSTM", ["x", "w", "r"], ["y"], hidden_size=2)
    graph = helper.make_graph(
        [node],
        "t",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 8, 2]),
            helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 8, 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 2])],
    )
    # Pin the IR version opset 17 shipped with: onnx stamps its own newest IR by default,
    # which a locked onnxruntime one release behind refuses to load.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    from tensor_factory.zerort import UnsupportedOp

    with pytest.raises(UnsupportedOp, match="LSTM"):
        Runtime(model.SerializeToString())


@pytest.mark.unit
def test_zerort_tensor_validates_shape():
    with pytest.raises(ValueError, match="data length"):
        Tensor((2, 2), FLOAT32, [1.0, 2.0, 3.0])
    assert Tensor((3,), INT64, [1, 2, 3]).is_int
    assert not Tensor((1,), FLOAT32, [0.5]).is_int
    assert Tensor((1,), UINT8, [7]).item() == 7
