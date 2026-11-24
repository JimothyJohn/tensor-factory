import io
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest
from PIL import Image

from tensor_factory_mcp import http_server


@pytest.fixture
def server() -> Iterator[str]:
    """A real ThreadingHTTPServer on an ephemeral port; yields its base URL."""
    httpd = http_server.make_server("127.0.0.1", 0, max_bytes=4096)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def _png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _post(url: str, data: bytes) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.mark.unit
def test_health(server):
    status, body = _get(f"{server}/health")
    assert status == 200 and body == {"status": "ok"}


@pytest.mark.unit
def test_model_info(server):
    status, body = _get(f"{server}/model_info")
    assert status == 200
    assert body["input_size"] == 480
    assert "CPUExecutionProvider" in body["providers"]


@pytest.mark.unit
def test_detect_returns_box_and_presence(server):
    status, body = _post(f"{server}/detect", _png_bytes())
    assert status == 200
    # Default bundled model has a presence head: present/score always set, box or null.
    assert isinstance(body["present"], bool)
    assert 0.0 <= body["score"] <= 1.0
    if body["present"]:
        assert body["box_norm"] is not None and len(body["uint8"]) == 4
    else:
        assert body["box_norm"] is None


@pytest.mark.unit
def test_detect_matches_core_path(server):
    # The HTTP body path must return the same JSON as the in-process core.detect_bytes.
    from tensor_factory_mcp import core

    data = _png_bytes()
    _, body = _post(f"{server}/detect", data)
    assert body == core.detect_bytes(data)


@pytest.mark.unit
def test_detect_bad_image_is_400(server):
    status, body = _post(f"{server}/detect", b"not a real image")
    assert status == 400 and "could not decode" in body["error"]


@pytest.mark.unit
def test_detect_empty_body_is_400(server):
    status, body = _post(f"{server}/detect", b"")
    assert status == 400


@pytest.mark.unit
def test_detect_oversized_body_is_413(server):
    # Fixture caps max_bytes at 4096; an over-cap body is refused before decode, not OOM'd.
    status, body = _post(f"{server}/detect", b"x" * 5000)
    assert status == 413


@pytest.mark.unit
def test_unknown_route_is_404(server):
    assert _get(f"{server}/nope")[0] == 404
    assert _post(f"{server}/nope", b"x")[0] == 404


_ZERO_DEP_SERVER_SCRIPT = """
import json, sys, threading, urllib.request

BLOCKED = {"numpy", "onnxruntime", "PIL", "onnx"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked third-party import: {name}", name=name)
        return None


sys.meta_path.insert(0, Blocker())

from tensor_factory_mcp import http_server

httpd = http_server.make_server("127.0.0.1", 0)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()

data = open(sys.argv[1], "rb").read()
req = urllib.request.Request(f"http://127.0.0.1:{port}/detect", data=data, method="POST")
with urllib.request.urlopen(req, timeout=120) as resp:
    detect = json.loads(resp.read())
with urllib.request.urlopen(f"http://127.0.0.1:{port}/model_info", timeout=30) as resp:
    info = json.loads(resp.read())
httpd.shutdown()
print(json.dumps({"detect": detect, "info": info}))
"""


@pytest.mark.unit
def test_http_serving_with_third_party_imports_blocked(tmp_path):
    """A bare CPython can *serve* detections: real socket round-trip, stdlib only.

    The subprocess blocks numpy/PIL/onnxruntime/onnx, boots the HTTP server, POSTs the
    real sample image to /detect, and its JSON must agree with the in-process
    onnxruntime path on the same bytes.
    """
    import subprocess
    import sys as _sys
    from pathlib import Path

    from tensor_factory_mcp import core

    sample = Path(__file__).resolve().parents[3] / "docs/sample-helicoil.png"
    script = tmp_path / "zero_dep_serve.py"
    script.write_text(_ZERO_DEP_SERVER_SCRIPT)
    proc = subprocess.run(
        [_sys.executable, str(script), str(sample)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["info"]["engine"] == "zerort"
    assert got["info"]["providers"] == ["zerort"]

    # The ort path decodes via PIL (different resampler), so scores may differ slightly;
    # presence/absence and the JSON shape must agree, scores within a few percent.
    ref = core.detect_bytes(sample.read_bytes())
    zd = got["detect"]
    assert zd["image_size"] == ref["image_size"]
    assert zd["present"] == ref["present"]
    assert abs(zd["score"] - ref["score"]) <= 0.05
    if ref["present"]:
        for k in ("x1", "y1", "x2", "y2"):
            assert abs(zd["box_norm"][k] - ref["box_norm"][k]) <= 0.02
    else:
        assert zd["box_norm"] is None and zd["uint8"] is None
