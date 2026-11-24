"""The browser's ONNX writer against the Python exporter, on the same weights.

``studio/js/nn/onnx.js`` fills a template with trained weights (batch norm folded, uint8
quantized). Here a torch model is exported both ways and the two files are compared tensor
by tensor and by what onnxruntime computes from them. Needs torch and node.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from tensor_factory.zerort import parse_model  # noqa: E402
from tensor_factory_train.model import TinyDetector  # noqa: E402
from tensor_factory_train.train import export_onnx  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CLI = REPO / "studio" / "tools" / "export_cli.mjs"
NODE = shutil.which("node") or ""
pytestmark = pytest.mark.skipif(not NODE, reason="node is not installed")


def _trained_looking(width: int, context: int, seed: int):
    """A detector whose batch norms and gain are off their initial values, like a trained one."""
    torch.manual_seed(seed)
    model = TinyDetector(width, presence=True, context=context)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".1." in name or name == "log_gain":
                p.add_(torch.randn_like(p) * 0.3)
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(torch.randn_like(b) * 0.5)
            elif name.endswith("running_var"):
                b.copy_(torch.rand_like(b) * 2 + 0.1)
    return model.eval()


def _both(tmp_path, width, context, seed, size=64):
    model = _trained_looking(width, context, seed)
    want = export_onnx(model, tmp_path / "python.onnx", size=size)
    state = {
        k: {"shape": list(v.shape), "data": v.reshape(-1).tolist()}
        for k, v in model.state_dict().items()
        if "num_batches" not in k
    }
    (tmp_path / "params.json").write_text(json.dumps(state))
    got = tmp_path / "browser.onnx"
    proc = subprocess.run(  # noqa: S603
        [NODE, str(CLI), str(tmp_path / "params.json"), str(got), str(size)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return want, got


def _run(path: Path, image: np.ndarray) -> dict[str, np.ndarray]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, {"image": image}), strict=True))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("width", "context", "size"), [(16, 4, 224), (8, 0, 64), (4, 1, 96), (12, 8, 160)]
)
def test_browser_export_matches_the_python_export(tmp_path, width, context, size):
    want, got = _both(tmp_path, width, context, seed=width + context, size=size)
    a = {t.name: t for t in parse_model(want.read_bytes()).graph.initializers}
    b = {t.name: t for t in parse_model(got.read_bytes()).graph.initializers}
    assert list(a) == list(b)  # same tensors, same order
    for name, ta in a.items():
        tb = b[name]
        assert tuple(ta.dims) == tuple(tb.dims), name
        assert ta.data_type == tb.data_type, name
        if ta.data_type == 2:  # uint8 weights and zero points: the same integers
            assert list(ta.data) == list(tb.data), name
        else:
            np.testing.assert_allclose(
                np.array(tb.data, dtype=np.float64),
                np.array(ta.data, dtype=np.float64),
                rtol=2e-6,
                atol=1e-7,
                err_msg=name,
            )
    rng = np.random.default_rng(0)
    image = rng.random((2, 3, size, size), dtype=np.float32)
    out_a, out_b = _run(want, image), _run(got, image)
    assert set(out_a) == set(out_b) == {"box", "presence"}
    for key in out_a:
        np.testing.assert_allclose(out_b[key], out_a[key], atol=1e-4, err_msg=key)
    # both files declare the trained input size, so a wrong-size image is refused by both
    for path in (want, got):
        shape = (
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            .get_inputs()[0]
            .shape
        )
        assert shape[1:] == [3, size, size], path.name


@pytest.mark.integration
def test_browser_export_loads_in_the_repo_runtime(tmp_path):
    from PIL import Image

    from tensor_factory.inference import Detector

    _, got = _both(tmp_path, 8, 2, seed=5)
    det = Detector(got, input_size=64)
    img = Image.new("RGB", (80, 60), (120, 90, 60))
    box = det.detect_box(img)
    assert det.has_presence and 0 <= box.x1 <= 1 and 0 <= box.y2 <= 1
    assert 0.0 <= det.detect_presence(img) <= 1.0


@pytest.mark.integration
def test_export_refuses_a_template_for_another_context(tmp_path):
    model = _trained_looking(4, 9, seed=1)  # no template ships for nine context blocks
    state = {
        k: {"shape": list(v.shape), "data": v.reshape(-1).tolist()}
        for k, v in model.state_dict().items()
        if "num_batches" not in k
    }
    (tmp_path / "p.json").write_text(json.dumps(state))
    proc = subprocess.run(  # noqa: S603
        [NODE, str(CLI), str(tmp_path / "p.json"), str(tmp_path / "o.onnx"), "64"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode != 0 and not (tmp_path / "o.onnx").exists()
