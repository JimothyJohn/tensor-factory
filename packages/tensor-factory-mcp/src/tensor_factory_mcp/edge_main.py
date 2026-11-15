"""Entry point of a single-file edge bundle (see :mod:`tensor_factory_mcp.bundle`).

``python3 edge.pyz`` lands here. The bundle carries the model inside the archive, but the
inference engines want a real file, so the model is unpacked once into a cache directory
(keyed by its sha256, verified on every start) and the ordinary HTTP server is started on
it. Everything here is standard library: the bundle must run on a bare CPython.

Any ``tensor-factory-http`` flag passes straight through, so ``--host 0.0.0.0``,
``--port`` and ``--engine`` work as usual, and ``--model PATH`` serves an external file
instead of the bundled one (the way to hot-swap models on a device).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

MANIFEST = "bundle.json"


def cache_dir() -> Path:
    """Where unpacked models live: ``$TENSOR_FACTORY_CACHE``, else the XDG cache dir."""
    explicit = os.environ.get("TENSOR_FACTORY_CACHE")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tensor-factory"


def read_manifest(archive: str | Path) -> dict:
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read(MANIFEST))
    for key in ("model", "sha256", "input_size"):
        if key not in manifest:
            raise ValueError(f"bundle manifest is missing {key!r}")
    # The name is joined onto the cache dir below; a manifest must not be able to steer
    # that write somewhere else.
    if Path(manifest["model"]).name != manifest["model"] or manifest["model"] in ("", ".", ".."):
        raise ValueError(f"bundle manifest has an unsafe model name: {manifest['model']!r}")
    return manifest


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _write_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".unpack-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, target)  # readers see the old file or the whole new one
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def extract_model(archive: str | Path, cache: Path | None = None) -> tuple[Path, dict]:
    """Unpack the bundled model to disk and return ``(path, manifest)``.

    A cached copy is reused only when its sha256 matches the manifest; anything else
    (truncated, tampered, a different model under the same name) is overwritten. If the
    cache directory cannot be written, a private temporary directory is used instead, so
    a read-only home never stops the server from starting.
    """
    manifest = read_manifest(archive)
    with zipfile.ZipFile(archive) as z:
        data = z.read(manifest["model"])
    if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
        raise ValueError("bundled model does not match its manifest sha256 (corrupt bundle)")
    name = f"{manifest['sha256'][:16]}-{manifest['model']}"
    target = (cache or cache_dir()) / name
    if _sha256(target) == manifest["sha256"]:
        return target, manifest
    try:
        _write_atomic(target, data)
    except OSError:
        target = Path(tempfile.mkdtemp(prefix="tensor-factory-")) / name
        target.write_bytes(data)
    return target, manifest


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def build_argv(argv: Sequence[str], archive: str | Path) -> list[str]:
    """The ``tensor-factory-http`` argv for this bundle: defaults first, user flags win."""
    args = list(argv)
    defaults: list[str] = []
    if _has_flag(args, "--model"):
        manifest = read_manifest(archive)
    else:
        path, manifest = extract_model(archive)
        defaults += ["--model", str(path)]
    if not _has_flag(args, "--input-size"):
        defaults += ["--input-size", str(manifest["input_size"])]
    return defaults + args


def main(argv: Sequence[str] | None = None, archive: str | Path | None = None) -> int:
    from . import http_server

    # Inside a zipapp this module's file is "<archive>/tensor_factory_mcp/edge_main.py".
    archive = archive or Path(__file__).resolve().parents[1]
    return http_server.main(build_argv(sys.argv[1:] if argv is None else argv, archive))
