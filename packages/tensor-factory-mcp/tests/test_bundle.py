"""The single-file edge bundle: build it, then run it on an interpreter with nothing else.

The run tests start the real ``.pyz`` in a subprocess with ``-S -I`` (no site-packages,
no environment, no user site), so they prove the file is self-contained rather than
borrowing this venv's packages.
"""

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import pytest

from tensor_factory_mcp import bundle, core, edge_main

MODELS = core.default_model_path().parent
SAMPLE = Path(__file__).resolve().parents[3] / "docs" / "sample-helicoil.png"


@pytest.fixture(scope="module")
def pyz(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bundle") / "edge.pyz"
    bundle.build(out=out)
    return out


# --- building -----------------------------------------------------------------------


@pytest.mark.unit
def test_bundle_is_an_executable_zip_with_a_shebang(pyz):
    assert pyz.read_bytes().startswith(b"#!/usr/bin/env python3\n")
    assert zipfile.is_zipfile(pyz)
    assert pyz.stat().st_mode & stat.S_IXUSR


@pytest.mark.unit
def test_bundle_carries_runtime_server_model_and_manifest(pyz):
    with zipfile.ZipFile(pyz) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("bundle.json"))
        model = z.read(manifest["model"])
    assert {"__main__.py", "bundle.json", "tensor_factory/__init__.py"} <= names
    assert "tensor_factory/zerort/engine.py" in names
    assert "tensor_factory_mcp/http_server.py" in names
    assert manifest["model"] == core.DEFAULT_MODEL_NAME
    assert manifest["sha256"] == hashlib.sha256(model).hexdigest()
    assert manifest["bytes"] == len(model)
    assert manifest["input_size"] == core.DEFAULT_INPUT_SIZE


@pytest.mark.unit
def test_bundle_leaves_out_what_the_edge_cannot_run(pyz):
    with zipfile.ZipFile(pyz) as z:
        names = z.namelist()
    assert "tensor_factory_mcp/server.py" not in names  # needs the MCP SDK
    assert "tensor_factory_mcp/bundle.py" not in names  # build-time only
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    # exactly one model: the other bundled demo model must not ride along
    assert len([n for n in names if n.endswith((".onnx", ".tfm"))]) == 1


@pytest.mark.unit
def test_bundle_build_is_reproducible(tmp_path):
    a = tmp_path / "a.pyz"
    b = tmp_path / "b.pyz"
    bundle.build(out=a)
    bundle.build(out=b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.unit
def test_bundle_takes_another_model_and_input_size(tmp_path):
    out = tmp_path / "mock.pyz"
    result = bundle.build(MODELS / "helicoil-mock-v1.onnx", out, input_size=320)
    assert result["model"] == "helicoil-mock-v1.onnx"
    assert edge_main.read_manifest(out)["input_size"] == 320


@pytest.mark.unit
def test_bundle_refuses_a_file_that_is_not_a_model(tmp_path, capsys):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model at all")
    out = tmp_path / "bad.pyz"
    assert bundle.main(["--model", str(bad), "--out", str(out)]) == 1
    assert "bundle failed" in capsys.readouterr().out
    assert not out.exists()  # nothing half-built is left to deploy


@pytest.mark.unit
def test_bundle_refuses_a_missing_model(tmp_path):
    assert bundle.main(["--model", str(tmp_path / "nope.onnx"), "--out", str(tmp_path / "x")]) == 1


# --- unpacking on the device ----------------------------------------------------------


@pytest.mark.unit
def test_extract_writes_the_model_and_reuses_it(pyz, tmp_path):
    path, manifest = edge_main.extract_model(pyz, tmp_path / "cache")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    mtime = path.stat().st_mtime_ns
    again, _ = edge_main.extract_model(pyz, tmp_path / "cache")
    assert again == path
    assert path.stat().st_mtime_ns == mtime  # verified and reused, not rewritten


@pytest.mark.unit
def test_extract_repairs_a_corrupted_cached_model(pyz, tmp_path):
    path, manifest = edge_main.extract_model(pyz, tmp_path / "cache")
    path.write_bytes(b"truncated")
    again, _ = edge_main.extract_model(pyz, tmp_path / "cache")
    assert hashlib.sha256(again.read_bytes()).hexdigest() == manifest["sha256"]


@pytest.mark.unit
def test_extract_falls_back_when_the_cache_dir_is_unwritable(pyz, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the cache directory should be")
    path, manifest = edge_main.extract_model(pyz, blocked / "cache")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]


def _rewrite(pyz: Path, out: Path, *, manifest: dict | None = None, model: bytes | None = None):
    """Copy a bundle, substituting its manifest and/or model bytes (a tampered bundle)."""
    with zipfile.ZipFile(pyz) as src, zipfile.ZipFile(out, "w") as dst:
        original = json.loads(src.read("bundle.json"))
        for name in src.namelist():
            data = src.read(name)
            if name == "bundle.json" and manifest is not None:
                data = json.dumps(manifest).encode()
            if name == original["model"] and model is not None:
                data = model
            dst.writestr(name, data)
    return original


@pytest.mark.unit
def test_extract_rejects_a_model_that_does_not_match_its_manifest(pyz, tmp_path):
    tampered = tmp_path / "tampered.pyz"
    _rewrite(pyz, tampered, model=b"swapped model bytes")
    with pytest.raises(ValueError, match="sha256"):
        edge_main.extract_model(tampered, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()  # rejected before anything is written


@pytest.mark.unit
@pytest.mark.parametrize("name", ["../../evil.onnx", "/etc/evil.onnx", "a/b.onnx", "..", ""])
def test_manifest_cannot_steer_the_unpack_outside_the_cache(pyz, tmp_path, name):
    evil = tmp_path / "evil.pyz"
    original = _rewrite(pyz, evil, manifest={"model": name, "sha256": "0" * 64, "input_size": 480})
    assert original["model"] != name
    with pytest.raises(ValueError, match="unsafe model name"):
        edge_main.extract_model(evil, tmp_path / "cache")


@pytest.mark.unit
def test_manifest_missing_a_field_is_rejected(pyz, tmp_path):
    broken = tmp_path / "broken.pyz"
    _rewrite(pyz, broken, manifest={"model": "m.onnx"})
    with pytest.raises(ValueError, match="missing"):
        edge_main.read_manifest(broken)


@pytest.mark.unit
def test_argv_defaults_come_from_the_bundle_and_user_flags_win(pyz, tmp_path, monkeypatch):
    monkeypatch.setenv("TENSOR_FACTORY_CACHE", str(tmp_path / "cache"))
    argv = edge_main.build_argv(["--port", "9000"], pyz)
    assert argv[0] == "--model" and Path(argv[1]).is_file()
    assert argv[2:4] == ["--input-size", "480"]
    assert argv[-2:] == ["--port", "9000"]

    external = edge_main.build_argv(["--model", "/data/new.onnx", "--input-size=320"], pyz)
    assert external == ["--model", "/data/new.onnx", "--input-size=320"]  # nothing unpacked
    assert edge_main.build_argv(["--model=/data/new.onnx"], pyz)[-1] == "--model=/data/new.onnx"


@pytest.mark.unit
def test_cache_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSOR_FACTORY_CACHE", str(tmp_path / "explicit"))
    assert edge_main.cache_dir() == tmp_path / "explicit"
    monkeypatch.delenv("TENSOR_FACTORY_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert edge_main.cache_dir() == tmp_path / "xdg" / "tensor-factory"


# --- running it bare ------------------------------------------------------------------


def _start(pyz: Path, home: Path, *extra: str) -> tuple[subprocess.Popen, str]:
    """Run the bundle on a bare interpreter; return the process and its base URL."""
    proc = subprocess.Popen(
        # -S: no site-packages. -I: no env vars, no user site, no cwd on sys.path.
        [sys.executable, "-S", "-I", "-u", str(pyz), "--port", "0", *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={"HOME": str(home), "PATH": os.defpath},
        cwd=home,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    match = re.search(r"http://127\.0\.0\.1:(\d+)", line)
    if not match:
        proc.kill()
        raise AssertionError(f"bundle did not start: {line!r}{proc.stdout.read()!r}")
    return proc, f"http://127.0.0.1:{match.group(1)}"


def _json(url: str, data: bytes | None = None) -> dict:
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


@pytest.mark.unit
def test_bundle_serves_detections_on_a_bare_interpreter(pyz, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    proc, base = _start(pyz, home)
    try:
        assert _json(f"{base}/health") == {"status": "ok"}
        info = _json(f"{base}/model_info")
        served = _json(f"{base}/detect", SAMPLE.read_bytes())
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # nothing but the standard library was available, so this is the zerort engine
    assert info["engine"] == "zerort"
    assert info["sha256"] == edge_main.read_manifest(pyz)["sha256"]
    # and it answers exactly like the in-process core on the same engine
    local = core.detect_bytes(SAMPLE.read_bytes(), engine="zerort")
    assert served["present"] == local["present"]
    assert served["score"] == pytest.approx(local["score"], abs=1e-9)
    assert served["image_size"] == local["image_size"]
    # the model was unpacked under the device's own cache, not beside the bundle
    assert Path(info["model"]).is_relative_to(home / ".cache" / "tensor-factory")


@pytest.mark.unit
def test_bundle_serves_an_external_model_when_asked(pyz, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    external = MODELS / "helicoil-mock-v1.onnx"
    proc, base = _start(pyz, home, "--model", str(external))
    try:
        info = _json(f"{base}/model_info")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert info["sha256"] == hashlib.sha256(external.read_bytes()).hexdigest()
    assert not (home / ".cache").exists()  # the bundled model was never unpacked
