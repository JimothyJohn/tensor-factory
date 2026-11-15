#!/usr/bin/env python3
"""The Doom test: the full detection stack on a bare CPython, no wheels, no venv.

Run from the repo root (CI runs it in a python:3.12-slim container with nothing
installed). Uses only the standard library and the in-repo sources:

1. detect on the sample PNG through the serving core (zerort engine, stdlib decode)
2. pack the model to .tfm and detect again -- identical outputs required
3. detect on synthetic PPM bytes
4. boot the stdlib HTTP endpoint and round-trip /detect over a real socket
5. build the single-file edge bundle and serve a detection from it in a fresh
   interpreter started with no site-packages at all (``-S -I``)

If third-party packages happen to be importable (a dev venv), an import blocker is
installed so the run still exercises the zero-dependency path; in the CI container
the environment is genuinely bare.
"""

import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKED = ("numpy", "onnxruntime", "PIL", "onnx")


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked third-party import: {name}", name=name)
        return None


def main() -> int:
    present = [m for m in BLOCKED if importlib.util.find_spec(m) is not None]
    if present:
        print(f"note: {', '.join(present)} importable -- blocking them for this run")
        sys.meta_path.insert(0, _Blocker())
    else:
        print("environment is bare: no third-party packages importable")

    sys.path.insert(0, str(ROOT / "packages/tensor-factory/src"))
    sys.path.insert(0, str(ROOT / "packages/tensor-factory-mcp/src"))

    from tensor_factory.cli import main as cli_main
    from tensor_factory_mcp import core, http_server

    model = str(
        ROOT
        / "packages/tensor-factory-mcp/src/tensor_factory_mcp/models/helicoil-presence-cam-v1.onnx"
    )
    sample = ROOT / "docs/sample-helicoil.png"

    # 1. PNG through the serving core.
    result = core.detect_bytes(sample.read_bytes(), model)
    info = core.model_info(model)
    assert info["engine"] == "zerort", info
    assert "present" in result and "score" in result, result
    print(f"1. PNG detect: present={result['present']} score={result['score']:.3f}")

    # 2. .tfm round trip must be output-identical (same graph, same engine).
    tfm_path = str(
        ROOT / "packages/tensor-factory-mcp/src/tensor_factory_mcp/models/.portability-check.tfm"
    )
    try:
        assert cli_main(["pack", "--model", model, "--out", tfm_path]) == 0
        again = core.detect_bytes(sample.read_bytes(), tfm_path)
        for key in ("present", "score", "box_norm", "uint8"):
            assert again[key] == result[key], (key, again[key], result[key])
    finally:
        Path(tfm_path).unlink(missing_ok=True)
    print("2. .tfm pack + detect: outputs identical to the ONNX parse")

    # 3. Synthetic PPM bytes (no file, no PNG machinery).
    ppm = b"P6\n64 64\n255\n" + bytes([128]) * (64 * 64 * 3)
    r3 = core.detect_bytes(ppm, model)
    assert "score" in r3, r3
    print(f"3. PPM detect: score={r3['score']:.3f}")

    # 4. HTTP serving over a real socket.
    httpd = http_server.make_server("127.0.0.1", 0, model_path=model)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/detect", data=sample.read_bytes(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            served = json.loads(resp.read())
    finally:
        httpd.shutdown()
    assert served["present"] == result["present"], served
    assert abs(served["score"] - result["score"]) < 1e-9, served
    print(f"4. HTTP /detect on :{port}: matches the in-process result")

    # 5. The edge bundle: one file, started the way a device would start it.
    from tensor_factory_mcp import bundle

    with tempfile.TemporaryDirectory() as tmp:
        pyz = Path(tmp) / "edge.pyz"
        built = bundle.build(model, pyz)
        proc = subprocess.Popen(
            [sys.executable, "-S", "-I", "-u", str(pyz), "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={"HOME": tmp, "PATH": os.defpath},
            cwd=tmp,
        )
        try:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            found = re.search(r"http://127\.0\.0\.1:(\d+)", line)
            assert found, f"bundle did not start: {line!r}"
            base = f"http://127.0.0.1:{found.group(1)}"
            with urllib.request.urlopen(f"{base}/model_info", timeout=60) as resp:
                edge_info = json.loads(resp.read())
            req = urllib.request.Request(f"{base}/detect", data=sample.read_bytes(), method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                edge = json.loads(resp.read())
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    assert edge_info["engine"] == "zerort", edge_info
    assert edge_info["sha256"] == built["sha256"], edge_info
    assert edge["present"] == result["present"], edge
    assert abs(edge["score"] - result["score"]) < 1e-9, edge
    print(
        f"5. edge bundle ({built['bundle_bytes'] // 1024} KB, one file): "
        "served the same detection from a bare interpreter"
    )

    print(
        f"portability check passed on {platform.system()} {platform.machine()}: "
        "image -> detection on the stdlib alone"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
