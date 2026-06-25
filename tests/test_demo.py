"""Contract tests for the in-browser demo (docs/demo.html).

The demo runs the same ONNX models the CLI runs, via onnxruntime-web. These tests pin the
web assets to the verified Python inference path so the demo can't silently drift from the
detection contract: same model bytes, same class names, a sample the bundled model detects,
and the structural wiring the page depends on.
"""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest
from PIL import Image

from tensor_factory.inference import Detector

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DEMO = DOCS / "demo.html"
DEMO_MODELS = DOCS / "models"
PKG_MODELS = ROOT / "packages/tensor-factory-mcp/src/tensor_factory_mcp/models"
SIZE = 480


@pytest.fixture(scope="module")
def html() -> str:
    return DEMO.read_text(encoding="utf-8")


@pytest.mark.unit
def test_demo_assets_exist():
    assert DEMO.is_file()
    assert (DEMO_MODELS / "helicoil-mock-v1.onnx").is_file()
    assert (DEMO_MODELS / "helicoil-presence-cam-v1.onnx").is_file()
    assert (DOCS / "sample-helicoil.png").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("name", ["helicoil-mock-v1.onnx", "helicoil-presence-cam-v1.onnx"])
def test_demo_models_match_the_bundled_package_models(name):
    # The demo must serve byte-identical models to what tensor-factory-mcp bundles -- not a
    # stale copy that quietly diverges from the shipped detector.
    assert filecmp.cmp(DEMO_MODELS / name, PKG_MODELS / name, shallow=False)


@pytest.mark.unit
def test_demo_models_load_and_satisfy_the_contract():
    for name in ("helicoil-mock-v1.onnx", "helicoil-presence-cam-v1.onnx"):
        det = Detector(DEMO_MODELS / name, input_size=SIZE)
        box = det.detect_box(Image.new("RGB", (SIZE, SIZE), (128, 128, 128)))
        assert 0.0 <= box.x1 <= 1.0 and box.x1 <= box.x2
        assert 0.0 <= box.y1 <= 1.0 and box.y1 <= box.y2


@pytest.mark.unit
def test_bundled_sample_is_detected_by_the_mock_model():
    # The demo auto-runs the mock model on sample-helicoil.png at load. Regenerate the exact
    # source (mock seed 17) to recover ground truth and assert the bundled model localizes it
    # tightly -- the demo's first impression must actually be a hit, not a shrug.
    synth = pytest.importorskip("tensor_factory_synth.generator")
    truth = synth.MockGenerator().generate("helicoil insert, microscope", 17, size=SIZE).box
    det = Detector(DEMO_MODELS / "helicoil-mock-v1.onnx", input_size=SIZE)
    with Image.open(DOCS / "sample-helicoil.png") as im:
        box = det.detect_box(im.convert("RGB"))
    pcx, pcy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    gcx, gcy = (truth.x1 + truth.x2) / 2, (truth.y1 + truth.y2) / 2
    err_px = ((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5 * SIZE
    assert err_px < 12, f"mock model missed its own sample by {err_px:.1f}px"


@pytest.mark.unit
def test_demo_presence_model_has_presence_head_and_no_class_metadata():
    # The presence model is YOLO-style: it exposes a "presence" output (one objectness
    # logit) and carries no class names -- absence is the low tail of one score, not a class.
    det = Detector(DEMO_MODELS / "helicoil-presence-cam-v1.onnx", input_size=SIZE)
    assert det.has_presence, "presence model must expose a 'presence' output"
    score = det.detect_presence(Image.new("RGB", (SIZE, SIZE), (128, 128, 128)))
    assert 0.0 <= score <= 1.0


@pytest.mark.unit
def test_demo_thresholds_presence_with_a_sigmoid(html):
    # The page must read the objectness output and sigmoid-threshold it (present vs. no box),
    # not decode class logits. Pins the wiring so the demo can't drift from the model.
    assert 'out["presence"]' in html, "demo.html must read the 'presence' output"
    assert "sigmoid" in html, "demo.html must sigmoid the objectness logit"
    assert "threshold" in html, "demo.html must threshold presence to decide box vs. no box"
    # The old class-logits path is gone (these strings are unique to it; "background" alone
    # would collide with CSS background: properties, so match the actual removed code).
    assert 'out["logits"]' not in html
    assert "softmaxArgmax" not in html and "classes:" not in html


@pytest.mark.unit
def test_demo_references_runtime_and_assets(html):
    assert "onnxruntime-web" in html  # WASM runtime
    assert "models/helicoil-mock-v1.onnx" in html
    assert "models/helicoil-presence-cam-v1.onnx" in html
    assert "sample-helicoil.png" in html
    assert "<canvas" in html and 'id="file"' in html  # the interactive surface


@pytest.mark.unit
def test_demo_uint8_formula_mirrors_codec(html):
    # The page must quantize with the same round-then-clamp as tensor_factory.codec._q;
    # assert the formula is present (the runtime equivalence is covered by the model tests).
    assert "Math.round(v * 255)" in html


@pytest.mark.unit
def test_demo_is_linked_from_every_doc_page():
    pages = [
        "index.html",
        "getting-started.html",
        "pipeline.html",
        "cli.html",
        "api.html",
        "architecture.html",
    ]
    for p in pages:
        assert '<a href="demo.html">Demo</a>' in (DOCS / p).read_text(encoding="utf-8"), p
    # demo.html marks itself current in the nav
    assert '<a href="demo.html" class="here">Demo</a>' in DEMO.read_text(encoding="utf-8")


@pytest.mark.unit
def test_no_stale_model_reference_in_docs():
    # v5 is the bundled default; nothing user-facing should still point at a superseded one.
    for p in DOCS.glob("*.html"):
        text = p.read_text(encoding="utf-8")
        assert "presence-v3" not in text, p.name
        assert "presence-v4" not in text, p.name
