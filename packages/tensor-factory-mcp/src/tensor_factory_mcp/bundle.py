"""Build a single-file edge bundle: runtime + HTTP server + model in one ``.pyz``.

    tensor-factory-bundle --model model.onnx --out edge.pyz
    scp edge.pyz device:            # the whole deployment
    python3 edge.pyz --host 0.0.0.0 # on the device: serves POST /detect

The bundle is a standard Python zip application holding the pure-Python ``tensor_factory``
package (zerort included), the stdlib HTTP server from this package, the model, and a
manifest. It needs nothing but CPython >= 3.11 on the device. If onnxruntime happens to be
installed there it is used (~200 fps); otherwise the bundled zerort engine runs (~1 fps).

Builds are reproducible: entries are sorted and carry a fixed timestamp, so the same
sources and model always produce byte-identical bundles.
"""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from collections.abc import Sequence
from pathlib import Path

import tensor_factory

from . import core
from .edge_main import MANIFEST

SHEBANG = b"#!/usr/bin/env python3\n"
# Only these modules of this package go in: server.py needs the MCP SDK, which the edge
# does not have, and bundle.py itself is a build-time tool.
_MCP_MODULES = ("__init__.py", "core.py", "http_server.py", "edge_main.py")
_MAIN = "from tensor_factory_mcp.edge_main import main\n\nraise SystemExit(main())\n"
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # the zip epoch: no build time leaks into the bytes


def _sources() -> list[tuple[str, bytes]]:
    """``(archive name, bytes)`` for every Python source the bundle needs, sorted."""
    core_root = Path(tensor_factory.__file__).resolve().parent
    mcp_root = Path(__file__).resolve().parent
    entries = [
        (f"tensor_factory/{p.relative_to(core_root).as_posix()}", p.read_bytes())
        for p in core_root.rglob("*.py")
    ]
    entries += [
        (f"tensor_factory_mcp/{name}", (mcp_root / name).read_bytes()) for name in _MCP_MODULES
    ]
    return sorted(entries)


def build(
    model: str | Path | None = None,
    out: str | Path = "edge.pyz",
    *,
    input_size: int = core.DEFAULT_INPUT_SIZE,
) -> dict:
    """Write the bundle to ``out`` and return its manifest (plus ``path`` and ``bytes``).

    ``model`` defaults to the detector shipped with this package. The model is loaded once
    before bundling, so a file that is not a runnable model fails here, at build time, not
    on the device.
    """
    model_path = Path(core.resolve_model(str(model) if model else None))
    data = model_path.read_bytes()
    from tensor_factory.inference import Detector

    Detector(model_path, input_size=input_size, model_bytes=data)  # raises if unusable

    manifest = {
        "model": model_path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "input_size": input_size,
    }
    entries = [
        ("__main__.py", _MAIN.encode()),
        (MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()),
        (manifest["model"], data),
        *_sources(),
    ]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(SHEBANG)
        with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
            for name, payload in entries:
                info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, payload)
    out_path.chmod(out_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {**manifest, "path": str(out_path), "bundle_bytes": out_path.stat().st_size}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tensor-factory-bundle",
        description="Pack the runtime, the HTTP server and a model into one runnable file.",
    )
    parser.add_argument("--model", default=None, help="ONNX or .tfm model (default: bundled)")
    parser.add_argument("--out", default="edge.pyz", help="output file (default: edge.pyz)")
    parser.add_argument(
        "--input-size", type=int, default=core.DEFAULT_INPUT_SIZE, help="square model input px"
    )
    args = parser.parse_args(argv)
    try:
        result = build(args.model, args.out, input_size=args.input_size)
    except Exception as exc:  # noqa: BLE001 -- a build tool reports, it does not trace
        print(f"bundle failed: {type(exc).__name__}: {exc}")
        return 1
    print(
        f"wrote {result['path']} ({result['bundle_bytes'] / 1024:.0f} KB)\n"
        f"  model  {result['model']}  sha256 {result['sha256']}\n"
        f"  run    python3 {result['path']} --host 0.0.0.0 --port 8088\n"
        f"  check  curl http://DEVICE:8088/model_info   # sha256 must match the line above"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
