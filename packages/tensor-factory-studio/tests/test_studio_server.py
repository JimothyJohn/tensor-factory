"""HTTP routing tests against a real ThreadingHTTPServer (no mocks).

The Trainer is constructed but never started, so torch is never imported — these stay
unit-fast. The training round itself is covered by test_training_integration.py.
"""

import io
import json
import threading
import urllib.request
from urllib.error import HTTPError

import pytest
from PIL import Image

from tensor_factory_studio.dataset import Dataset
from tensor_factory_studio.server import make_server, serve
from tensor_factory_studio.trainer import Trainer


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 110, 120)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def server(tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html><title>studio</title>", encoding="utf-8")
    ds = Dataset(tmp_path / "data")
    tr = Trainer(ds, tmp_path / "data" / "models")  # not started -> no torch
    httpd = make_server("127.0.0.1", 0, dataset=ds, trainer=tr, ui_dir=ui, input_size=480)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", ds
    httpd.shutdown()


def _get(base, path):
    with urllib.request.urlopen(f"{base}{path}") as r:
        return r.status, json.loads(r.read())


def _post(base, path, body=b""):
    req = urllib.request.Request(f"{base}{path}", data=body, method="POST")
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


@pytest.mark.unit
def test_status_and_metrics(server):
    base, _ = server
    code, status = _get(base, "/status")
    assert code == 200 and "counts" in status and status["hasModel"] is False
    code, metrics = _get(base, "/metrics")
    assert code == 200 and metrics["hasModel"] is False


@pytest.mark.unit
def test_static_serves_index(server):
    base, _ = server
    with urllib.request.urlopen(f"{base}/") as r:
        assert r.status == 200 and b"studio" in r.read()


@pytest.mark.unit
def test_static_rejects_traversal(server):
    base, _ = server
    with pytest.raises(HTTPError) as exc:
        urllib.request.urlopen(f"{base}/../../etc/hosts")
    assert exc.value.code == 404


@pytest.mark.unit
def test_post_sample_positive_writes_dataset(server):
    base, ds = server
    code, body = _post(base, "/samples?id=7&present=1&box=0.1,0.2,0.3,0.4", _png())
    assert code == 200 and body["counts"]["positives"] == 1
    assert ds.samples[7]["present"] is True


@pytest.mark.unit
def test_post_sample_negative(server):
    base, ds = server
    code, body = _post(base, "/samples?id=8&present=0", _png())
    assert code == 200 and body["counts"]["negatives"] == 1


@pytest.mark.unit
def test_predict_before_model_is_not_ready(server):
    base, _ = server
    code, body = _post(base, "/predict", _png())
    assert code == 200 and body == {"ready": False}


@pytest.mark.unit
def test_reset_clears(server):
    base, ds = server
    _post(base, "/samples?id=1&present=1&box=0,0,0.5,0.5", _png())
    code, _ = _post(base, "/reset")
    assert code == 200 and ds.counts()["total"] == 0


@pytest.mark.unit
def test_bad_box_rejected(server):
    base, _ = server
    with pytest.raises(HTTPError) as exc:
        _post(base, "/samples?id=9&present=1&box=0.1,0.2", _png())
    assert exc.value.code == 400


@pytest.mark.unit
def test_unknown_route_404(server):
    base, _ = server
    with pytest.raises(HTTPError) as exc:
        _post(base, "/nope")
    assert exc.value.code == 404


@pytest.mark.unit
def test_serve_fails_cleanly_when_port_in_use(tmp_path):
    # Regression: a busy port must return 1 with guidance, not crash with a traceback
    # while a zombie server keeps shadowing this one (the "nothing changed" bug).
    import socket

    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("x", encoding="utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        rc = serve("127.0.0.1", port, data_dir=tmp_path / "data", ui_dir=ui)
        assert rc == 1
    finally:
        sock.close()


@pytest.mark.unit
def test_handler_exception_becomes_500_not_empty_response(tmp_path, monkeypatch):
    # A raising handler must return a 500 with a body, never close the socket empty
    # (the bug that made every /metrics poll fail silently in the browser).
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("x", encoding="utf-8")
    ds = Dataset(tmp_path / "data")
    tr = Trainer(ds, tmp_path / "data" / "models")

    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(tr, "metrics", boom)  # force a handler failure
    httpd = make_server("127.0.0.1", 0, dataset=ds, trainer=tr, ui_dir=ui, input_size=480)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(HTTPError) as exc:
            _get(f"http://127.0.0.1:{port}", "/metrics")
        assert exc.value.code == 500
        assert b"boom" in exc.value.read()
    finally:
        httpd.shutdown()


# --- adversarial input: the HTTP surface must never 500/hang on bad input ---


@pytest.mark.unit
def test_malformed_image_body_is_400_not_500(server):
    base, ds = server
    with pytest.raises(HTTPError) as exc:
        _post(base, "/samples?id=1&present=1&box=0,0,0.5,0.5", b"not a real png")
    assert exc.value.code == 400
    assert ds.counts()["total"] == 0  # nothing persisted from a bad upload


@pytest.mark.unit
def test_non_integer_id_is_400(server):
    base, _ = server
    with pytest.raises(HTTPError) as exc:
        _post(base, "/samples?id=abc&present=0", _png())
    assert exc.value.code == 400


@pytest.mark.unit
def test_empty_body_is_rejected(server):
    base, _ = server
    with pytest.raises(HTTPError) as exc:
        _post(base, "/samples?id=1&present=0", b"")
    assert exc.value.code in (400, 411)


@pytest.mark.unit
def test_oversized_body_is_413(tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("x", encoding="utf-8")
    ds = Dataset(tmp_path / "data")
    tr = Trainer(ds, tmp_path / "data" / "models")
    httpd = make_server(
        "127.0.0.1", 0, dataset=ds, trainer=tr, ui_dir=ui, input_size=480, max_bytes=16
    )
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(HTTPError) as exc:
            _post(f"http://127.0.0.1:{port}", "/samples?id=1&present=0", b"x" * 1000)
        assert exc.value.code == 413
    finally:
        httpd.shutdown()


@pytest.mark.unit
def test_head_request_on_index(server):
    base, _ = server
    req = urllib.request.Request(f"{base}/", method="HEAD")
    with urllib.request.urlopen(req) as r:
        assert r.status == 200
        assert r.read() == b""  # HEAD has no body
