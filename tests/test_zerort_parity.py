"""Full-model parity: zerort vs onnxruntime on the bundled detectors.

Two claims are pinned here:

1. **Numerical parity.** Fed the identical input tensor, the pure-stdlib engine and
   onnxruntime produce the same outputs on the real bundled models. Tolerance is 5e-3
   normalized (~2.4 px at 480 px) -- above the observed worst case (~2e-3, a single
   quantization lane resolving differently at an f32/f64 rounding boundary) and just
   over the 1/255 wire-quantization step, so any real kernel bug trips it.

2. **Zero dependencies, provably.** A subprocess with an import hook that *blocks*
   numpy/onnxruntime/PIL/onnx runs the full pipeline -- PNG bytes to box -- on the
   stdlib alone, and its numbers match the onnxruntime run in this process.
"""

import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "packages/tensor-factory-mcp/src/tensor_factory_mcp/models"
MOCK = MODELS / "helicoil-mock-v1.onnx"
CAM = MODELS / "helicoil-presence-cam-v1.onnx"
SAMPLE = ROOT / "docs/sample-helicoil.png"

TOL = 5e-3


def _xyxy(box):
    return (box.x1, box.y1, box.x2, box.y2)


def _random_image_tensor(seed: int, size: int = 480):
    import numpy as np

    rng = random.Random(seed)
    data = [float(np.float32(rng.random())) for _ in range(3 * size * size)]
    return data, (1, 3, size, size)


@pytest.mark.unit
@pytest.mark.parametrize("model_path", [MOCK, CAM], ids=lambda p: p.stem)
def test_bundled_model_parity_with_onnxruntime(model_path):
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    from tensor_factory.zerort import Runtime, Tensor

    data, shape = _random_image_tensor(20260702)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ref = dict(
        zip(
            [o.name for o in sess.get_outputs()],
            sess.run(None, {"image": np.asarray(data, dtype=np.float32).reshape(shape)}),
            strict=True,
        )
    )

    rt = Runtime(model_path)
    out = rt.run({"image": Tensor(shape, 1, data)})

    assert set(out) == set(ref)
    for name in ref:
        theirs = [float(v) for v in ref[name].reshape(-1)]
        mine = out[name].data
        assert mine == pytest.approx(theirs, abs=TOL), f"output {name!r} diverged"


@pytest.mark.unit
def test_detector_engines_agree_on_the_sample_image():
    pytest.importorskip("onnxruntime")
    from tensor_factory.inference import Detector

    # Passing a *path* routes both engines through the same stdlib preprocessing,
    # so any disagreement is the inference engine, not the image pipeline.
    d_ort = Detector(CAM, engine="ort")
    d_zero = Detector(CAM, engine="zerort")
    assert d_zero.session is None and d_ort.session is not None

    box_ort = d_ort.detect_box(str(SAMPLE))
    box_zero = d_zero.detect_box(str(SAMPLE))
    for a, b in zip(_xyxy(box_ort), _xyxy(box_zero), strict=True):
        assert abs(a - b) <= TOL
    assert abs(d_ort.detect_presence(str(SAMPLE)) - d_zero.detect_presence(str(SAMPLE))) <= 0.01


_ZERO_DEP_SCRIPT = textwrap.dedent(
    """
    import json, sys

    BLOCKED = {"numpy", "onnxruntime", "PIL", "onnx"}

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"blocked third-party import: {name}", name=name)
            return None

    sys.meta_path.insert(0, Blocker())

    from tensor_factory.inference import Detector

    model, image = sys.argv[1], sys.argv[2]
    det = Detector(model)  # engine='auto' must fall back to zerort under the blocker
    assert det.engine == "zerort", det.engine
    box = det.detect_box(image)
    print(
        json.dumps(
            {
                "engine": det.engine,
                "box": [box.x1, box.y1, box.x2, box.y2],
                "uint8": det.detect_uint8(image),
                "presence": det.detect_presence(image) if det.has_presence else None,
            }
        )
    )
    """
)


@pytest.mark.unit
def test_full_pipeline_runs_with_third_party_imports_blocked(tmp_path):
    """PNG file -> detection on the stdlib alone; numbers match onnxruntime."""
    script = tmp_path / "zero_dep_detect.py"
    script.write_text(_ZERO_DEP_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script), str(CAM), str(SAMPLE)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["engine"] == "zerort"

    pytest.importorskip("onnxruntime")
    from tensor_factory.inference import Detector

    ref = Detector(CAM, engine="ort")
    ref_box = ref.detect_box(str(SAMPLE))
    for a, b in zip(got["box"], _xyxy(ref_box), strict=True):
        assert abs(a - b) <= TOL
    for a, b in zip(got["uint8"], ref.detect_uint8(str(SAMPLE)), strict=True):
        assert abs(a - b) <= 2
    assert abs(got["presence"] - ref.detect_presence(str(SAMPLE))) <= 0.01
