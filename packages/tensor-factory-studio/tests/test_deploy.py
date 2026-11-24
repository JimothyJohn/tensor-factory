"""Device export and deploy: the packed P4 image format, the one-file bundle, the HTTP
routes, and the deploy job's orchestration.

No board is needed here. The job runs real subprocesses against stand-in tool scripts that
print what the real tools print; the real tools and a real board are covered by
``test_p4_board_takes_the_model`` (integration, needs ``TENSOR_FACTORY_P4_PORT``).
"""

import importlib.util
import io
import json
import os
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path
from urllib.error import HTTPError

import pytest
from PIL import Image

from tensor_factory_studio import device
from tensor_factory_studio.dataset import Dataset
from tensor_factory_studio.device import Deployer, DeviceError
from tensor_factory_studio.server import make_server
from tensor_factory_studio.trainer import Trainer

REPO = Path(__file__).resolve().parents[3]
BUNDLED = (
    REPO / "packages/tensor-factory-mcp/src/tensor_factory_mcp/models/helicoil-presence-cam-v1.onnx"
)


def _deploy_tool():
    spec = importlib.util.spec_from_file_location(
        "p4_deploy", REPO / "firmware/esp32-p4/tools/deploy.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _deploy_tool()


# --- the packed image: shared byte-for-byte with the firmware's ModelHeader ---


@pytest.mark.unit
def test_pack_layout_is_what_the_firmware_reads():
    blob = bytes(range(256)) * 3
    image = tool.pack(blob, size=224, gain=1.5, name="pets")
    magic, n, crc, size, gain, name = struct.unpack_from("<8sIIIf40s", image)
    assert (magic, n, crc, size, gain) == (b"TFMODEL1", len(blob), zlib.crc32(blob), 224, 1.5)
    assert name.rstrip(b"\0") == b"pets"
    assert struct.calcsize("<8sIIIf40s") == 64
    assert image[4096:] == blob and set(image[64:4096]) == {0xFF}
    assert tool.unpack(image) == {
        "name": "pets",
        "size": 224,
        "gain": 1.5,
        "bytes": len(blob),
        "crc32": f"{zlib.crc32(blob):08x}",
    }


@pytest.mark.unit
def test_pack_keeps_the_name_terminated_and_ascii():
    image = tool.pack(b"x", size=96, gain=1.0, name="é" + "n" * 100)
    name = struct.unpack_from("<8sIIIf40s", image)[5]
    assert name[-1] == 0 and name.isascii()  # the firmware prints it as a C string


@pytest.mark.unit
@pytest.mark.parametrize(
    ("blob", "size"),
    [(b"", 224), (b"x", 0), (b"x", 7), (b"x", 100_000), (b"\0" * (4 * 1024 * 1024), 224)],
)
def test_pack_refuses_what_the_board_could_not_load(blob, size):
    with pytest.raises(ValueError):
        tool.pack(blob, size=size, gain=1.0, name="m")


@pytest.mark.unit
def test_unpack_rejects_truncated_corrupt_and_foreign_files():
    image = tool.pack(b"model-bytes" * 50, size=224, gain=1.0, name="m")
    flipped = bytearray(image)
    flipped[-1] ^= 1
    for bad in (b"", image[:64], image[:-1], bytes(flipped), b"PK\x03\x04" + image[4:]):
        with pytest.raises(ValueError):
            tool.unpack(bad)


# --- probe frame ---


def _png(size=(64, 48), color=(100, 110, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.unit
@pytest.mark.parametrize("size", [(1, 1), (9, 700), (333, 517), (4000, 3000)])
def test_probe_jpeg_is_small_baseline_and_a_multiple_of_8(size):
    jpeg = device.probe_jpeg(_png(size))
    with Image.open(io.BytesIO(jpeg)) as im:
        assert im.format == "JPEG" and not im.info.get("progressive")
        assert im.width % 8 == 0 and im.height % 8 == 0
        assert 8 <= max(im.size) <= device.PROBE_MAX_SIDE


@pytest.mark.unit
def test_probe_jpeg_rejects_bytes_that_are_not_an_image():
    with pytest.raises(Exception):  # noqa: B017, PT011 -- PIL's error type is not the contract
        device.probe_jpeg(b"not an image")


# --- edge.pyz ---


@pytest.mark.unit
def test_build_pyz_packs_the_model_with_its_input_size(tmp_path):
    out = device.build_pyz(BUNDLED, size=480, out=tmp_path / "edge.pyz")
    assert out.read_bytes().startswith(b"#!/usr/bin/env python3\n")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        manifest = next(json.loads(z.read(n)) for n in names if n.endswith(".json"))
        assert BUNDLED.name in names and "__main__.py" in names
    assert manifest["input_size"] == 480 and manifest["bytes"] == BUNDLED.stat().st_size


# --- stand-in tools: real subprocesses that print what the real tools print ---

_EXPORT = """
import argparse, json, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--size"); ap.add_argument("--onnx"); ap.add_argument("--calib")
ap.add_argument("--out")
a = ap.parse_args()
pathlib.Path(a.out).write_bytes(b"ESPDL" + pathlib.Path(a.onnx).read_bytes()[:64])
pathlib.Path(a.out).with_suffix(".report.json").write_text(
    json.dumps({"meta": {"size": int(a.size), "gain": 1.25}}))
with open(pathlib.Path(a.out).parent.parent / "exports.log", "a") as f:
    f.write(str(a.calib) + "\\n")
"""

_FLASH_OK = """
import json, sys
real = __import__("importlib.util").util.spec_from_file_location("real", {real!r})
mod = __import__("importlib.util").util.module_from_spec(real); real.loader.exec_module(mod)
if sys.argv[1] == "pack":
    raise SystemExit(mod.main(sys.argv[1:]))
image = open(sys.argv[-1], "rb").read()
want = mod.unpack(image)
out = {{"ok": True, "want": want, "model": {{"ok": True, "name": want["name"],
       "size": want["size"], "bytes": want["bytes"], "crc32": want["crc32"],
       "self_test": True, "net_ms": 15.5}}}}
if "--image" in sys.argv:
    out["detection"] = {{"ok": True, "decode_ms": 15.0, "resize_ms": 2.0, "net_ms": 15.5,
                        "post_ms": 0.5, "presence": 4.0, "box": [0.25, 0.25, 0.75, 0.75]}}
print("esptool chatter")
print("RESULT " + json.dumps(out))
"""

_FLASH_REFUSED = """
import json, sys
real = __import__("importlib.util").util.spec_from_file_location("real", {real!r})
mod = __import__("importlib.util").util.module_from_spec(real); real.loader.exec_module(mod)
if sys.argv[1] == "pack":
    raise SystemExit(mod.main(sys.argv[1:]))
print("RESULT " + json.dumps({{"ok": False, "error": "This chip is ESP32, not ESP32-P4"}}))
raise SystemExit(1)
"""


def _tools(tmp_path, flash_src: str) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "export_espdl.py").write_text(_EXPORT)
    (tools / "deploy.py").write_text(
        flash_src.format(real=str(REPO / "firmware/esp32-p4/tools/deploy.py"))
    )
    return tools


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """Point the device module at stand-in tools run by this interpreter."""

    def use(flash_src: str = _FLASH_OK) -> Path:
        tools = _tools(tmp_path, flash_src)
        monkeypatch.setattr(device, "P4_TOOLS", tools)
        monkeypatch.setenv("TENSOR_FACTORY_P4_PYTHON", sys.executable)
        return tools

    return use


def _deployer(tmp_path, *, model=BUNDLED, probe=None, desktop=None) -> Deployer:
    return Deployer(
        tmp_path / "work",
        size=224,
        model=lambda: model,
        calib_dir=lambda: None,
        probe=lambda: probe,
        desktop_box=lambda onnx, jpeg: desktop,
    )


def _finish(dep: Deployer) -> dict:
    deadline = time.time() + 60
    while dep.state()["state"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    return dep.state()


@pytest.mark.unit
def test_p4_image_is_built_once_per_model_and_is_a_valid_image(tmp_path, stand_in):
    stand_in()
    dep = _deployer(tmp_path)
    first = dep.p4_image()
    meta = tool.unpack(first.read_bytes())
    assert meta["size"] == 224 and meta["gain"] == 1.25
    assert dep.p4_image() == first
    log = (tmp_path / "work" / "exports.log").read_text().splitlines()
    assert len(log) == 1, "the same model must not be quantized twice"


@pytest.mark.unit
def test_deploy_reports_the_board_and_compares_it_with_the_desktop(tmp_path, stand_in):
    stand_in()
    dep = _deployer(tmp_path, probe=_png((640, 480)), desktop=[0.25, 0.25, 0.75, 0.75])
    assert dep.start("/dev/null") is True
    state = _finish(dep)
    assert state["state"] == "done", json.dumps(state)
    r = state["result"]
    assert r["selfTest"] is True and r["netMs"] == 15.5 and r["size"] == 224
    assert r["frame"] == {
        "ms": 33.0,
        "present": True,
        "box": [0.25, 0.25, 0.75, 0.75],
        "desktopBox": [0.25, 0.25, 0.75, 0.75],
        "iou": 1.0,
    }
    json.dumps(state)  # the browser polls this


@pytest.mark.unit
def test_deploy_without_a_labeled_frame_still_confirms_the_model(tmp_path, stand_in):
    stand_in()
    dep = _deployer(tmp_path)
    dep.start("/dev/null")
    state = _finish(dep)
    assert state["state"] == "done" and "frame" not in state["result"], json.dumps(state)


@pytest.mark.unit
def test_deploy_surfaces_the_tools_own_error(tmp_path, stand_in):
    stand_in(_FLASH_REFUSED)
    dep = _deployer(tmp_path)
    dep.start("/dev/null")
    state = _finish(dep)
    assert state == {
        "state": "error",
        "port": "/dev/null",
        "error": "This chip is ESP32, not ESP32-P4",
    }


@pytest.mark.unit
def test_deploy_with_no_model_is_an_error_not_a_crash(tmp_path, stand_in):
    stand_in()
    dep = _deployer(tmp_path, model=None)
    dep.start("/dev/null")
    assert _finish(dep)["error"] == "no model trained yet"
    with pytest.raises(DeviceError):
        dep.pyz()


@pytest.mark.unit
def test_only_one_deploy_runs_at_a_time(tmp_path, stand_in):
    stand_in()
    dep = _deployer(tmp_path)
    started = [dep.start("/dev/null") for _ in range(8)]
    assert started.count(True) == 1
    assert _finish(dep)["state"] == "done"
    assert dep.start("/dev/null") is True  # and it can run again afterwards
    _finish(dep)


@pytest.mark.unit
def test_missing_p4_tools_say_how_to_install_them(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSOR_FACTORY_P4_PYTHON", str(tmp_path / "nope" / "python"))
    assert device.p4_python() is None
    with pytest.raises(DeviceError, match="uv pip install esp-ppq"):
        _deployer(tmp_path).p4_image()


# --- HTTP routes ---


@pytest.fixture
def server(tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("<!doctype html><title>studio</title>", encoding="utf-8")
    ds = Dataset(tmp_path / "data")
    tr = Trainer(ds, tmp_path / "data" / "models")  # not started -> no torch
    httpd = make_server("127.0.0.1", 0, dataset=ds, trainer=tr, ui_dir=ui, input_size=480)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tr
    httpd.shutdown()


def _status(base, path, method="GET"):
    req = urllib.request.Request(f"{base}{path}", data=b"" if method == "POST" else None)
    req.method = method
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except HTTPError as exc:
        return exc.code, exc.read()


@pytest.mark.unit
def test_devices_lists_ports_and_tool_state(server, monkeypatch, tmp_path):
    base, _ = server
    monkeypatch.setenv("TENSOR_FACTORY_P4_PYTHON", str(tmp_path / "missing"))
    code, body = _status(base, "/devices")
    payload = json.loads(body)
    assert code == 200 and isinstance(payload["ports"], list)
    assert payload["p4Tools"] is False and "esptool" in payload["hint"]


@pytest.mark.unit
def test_deploy_state_starts_idle(server):
    base, _ = server
    assert json.loads(_status(base, "/deploy")[1]) == {"state": "idle"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "port",
    ["", "/etc/passwd", "/dev/cu.usbmodem1;rm -rf /", "../../dev/null", "--after=no_reset", "%00"],
)
def test_deploy_refuses_a_port_this_machine_does_not_list(server, port, monkeypatch):
    base, tr = server
    tr.served = BUNDLED
    monkeypatch.setattr(device, "ports", lambda: ["/dev/cu.usbmodemREAL"])
    code, body = _status(base, f"/deploy?port={urllib.parse.quote(port, safe='')}", "POST")
    assert code == 400, body
    assert json.loads(_status(base, "/deploy")[1]) == {"state": "idle"}


@pytest.mark.unit
def test_deploy_needs_a_model_and_the_tools(server, monkeypatch, tmp_path):
    base, tr = server
    monkeypatch.setattr(device, "ports", lambda: ["/dev/cu.usbmodemREAL"])
    assert _status(base, "/deploy?port=/dev/cu.usbmodemREAL", "POST")[0] == 404
    tr.served = BUNDLED
    monkeypatch.setenv("TENSOR_FACTORY_P4_PYTHON", str(tmp_path / "missing"))
    code, body = _status(base, "/deploy?port=/dev/cu.usbmodemREAL", "POST")
    assert code == 503 and "esp-ppq" in json.loads(body)["error"]


@pytest.mark.unit
def test_model_download_formats(server, monkeypatch, tmp_path):
    base, tr = server
    for fmt in ("onnx", "pyz", "p4"):
        assert _status(base, f"/model?format={fmt}")[0] == 404  # nothing trained yet
    tr.served = BUNDLED
    assert _status(base, "/model?format=exe")[0] == 400
    assert _status(base, "/model")[1] == BUNDLED.read_bytes()
    code, pyz = _status(base, "/model?format=pyz")
    assert code == 200 and BUNDLED.name in zipfile.ZipFile(io.BytesIO(pyz)).namelist()
    monkeypatch.setenv("TENSOR_FACTORY_P4_PYTHON", str(tmp_path / "missing"))
    code, body = _status(base, "/model?format=p4")
    assert code == 503 and "esp-ppq" in json.loads(body)["error"]


@pytest.mark.unit
def test_p4_download_is_a_valid_packed_image(server, stand_in):
    base, tr = server
    stand_in()
    tr.served = BUNDLED
    code, image = _status(base, "/model?format=p4")
    assert code == 200 and tool.unpack(image)["size"] == 480


def _post_bytes(base, path, body):
    req = urllib.request.Request(f"{base}{path}", data=body, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.mark.unit
def test_a_browser_trained_model_becomes_the_served_model(server):
    base, tr = server  # this server's input size is 480, the bundled model's
    assert tr.served is None
    code, body = _post_bytes(base, "/model", BUNDLED.read_bytes())
    assert code == 200, body
    assert tr.served is not None and tr.served.read_bytes() == BUNDLED.read_bytes()
    assert _status(base, "/model")[1] == BUNDLED.read_bytes()
    assert json.loads(_status(base, "/status")[1])["hasModel"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "body",
    [b"not a model", b"\x08\x08" * 2000, BUNDLED.read_bytes()[:5000], b"PK\x03\x04" + b"\0" * 64],
    ids=["text", "protobuf-noise", "truncated", "zip"],
)
def test_bytes_that_are_not_a_detector_do_not_replace_the_model(server, body):
    base, tr = server
    _post_bytes(base, "/model", BUNDLED.read_bytes())
    served = tr.served.read_bytes()
    code, payload = _post_bytes(base, "/model", body)
    assert code == 400 and "not a detector model" in payload["error"], payload
    assert tr.served.read_bytes() == served


@pytest.mark.unit
def test_a_model_for_another_input_size_is_refused(tmp_path):
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("x", encoding="utf-8")
    ds = Dataset(tmp_path / "data")
    tr = Trainer(ds, tmp_path / "data" / "models")
    # this server works at 224 px; the bundled model declares 480
    httpd = make_server("127.0.0.1", 0, dataset=ds, trainer=tr, ui_dir=ui, input_size=224)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, payload = _post_bytes(base, "/model", BUNDLED.read_bytes())
        assert code == 400, payload
        assert tr.served is None
    finally:
        httpd.shutdown()


@pytest.mark.unit
def test_an_empty_or_oversized_model_upload_is_refused(server):
    base, tr = server
    assert _post_bytes(base, "/model", b"")[0] in (400, 411)
    assert tr.served is None


@pytest.mark.unit
def test_status_says_whether_the_backend_holds_source_data(server):
    base, tr = server
    assert json.loads(_status(base, "/status")[1])["sources"] == 0
    tr.sources = [Path("/somewhere")]
    assert json.loads(_status(base, "/status")[1])["sources"] == 1


@pytest.mark.unit
def test_status_reports_what_is_being_trained(server):
    base, _ = server
    config = json.loads(_status(base, "/status")[1])["config"]
    assert config == {"size": 224, "width": 16, "context": 4, "resize": "mixed"}


# --- the real tools and a real board ---


@pytest.mark.integration
def test_p4_board_takes_the_model(tmp_path):
    port = os.environ.get("TENSOR_FACTORY_P4_PORT")
    if not port or device.p4_python() is None:
        pytest.skip("set TENSOR_FACTORY_P4_PORT to a flashed ESP32-P4 and install the P4 tools")
    frame = _png((320, 240), (90, 90, 90))
    dep = Deployer(
        tmp_path / "work",
        size=480,
        model=lambda: BUNDLED,
        calib_dir=lambda: None,
        probe=lambda: frame,
        desktop_box=lambda onnx, jpeg: None,
    )
    dep.start(port)
    deadline = time.time() + 300
    while dep.state()["state"] == "running" and time.time() < deadline:
        time.sleep(0.5)
    state = dep.state()
    assert state["state"] == "done", json.dumps(state)
    assert state["result"]["selfTest"] is True, json.dumps(state)
    assert state["result"]["frame"]["ms"] > 0, json.dumps(state)
