"""Turn the served model into something a device runs, and put it there.

Two targets:

* **ESP32-P4.** The detector is quantized to a static int8 ``.espdl`` (ESP-PPQ) and wrapped
  in the header the firmware in ``firmware/esp32-p4`` reads from its ``model`` flash
  partition. Writing it is one ``esptool`` call; no firmware rebuild. The board then runs
  its self-test, times the network and detects one frame, which is compared with what the
  desktop model says about the same file.
* **Linux / any CPython.** ``edge.pyz``: runtime, HTTP server and model in one file
  (:mod:`tensor_factory_mcp.bundle`).

The P4 tools run in their own interpreter (ESP-PPQ pins its own torch), found through
``TENSOR_FACTORY_P4_PYTHON`` or ``~/esp/ppq-venv/bin/python``. Without it the P4 routes
answer with what to install; everything else keeps working.
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

# packages/tensor-factory-studio/src/tensor_factory_studio/device.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
P4_TOOLS = REPO_ROOT / "firmware" / "esp32-p4" / "tools"
DEFAULT_P4_PYTHON = Path.home() / "esp" / "ppq-venv" / "bin" / "python"
P4_SETUP_HINT = (
    "ESP32-P4 tools not found. One-time setup: "
    "`uv venv ~/esp/ppq-venv --python 3.11 && "
    "VIRTUAL_ENV=~/esp/ppq-venv uv pip install esp-ppq torch pillow esptool` "
    "(or point TENSOR_FACTORY_P4_PYTHON at an interpreter that has them)."
)
_PORT_GLOBS = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/ttyACM*", "/dev/ttyUSB*")
PROBE_MAX_SIDE = 320  # px; the probe frame travels over a 115200 baud console


class DeviceError(RuntimeError):
    """A deploy step failed in a way the user can act on; the message is shown as is."""


def p4_python() -> Path | None:
    """The interpreter that has ESP-PPQ and esptool, or ``None`` when it is not set up."""
    path = Path(os.environ.get("TENSOR_FACTORY_P4_PYTHON") or DEFAULT_P4_PYTHON)
    return path if path.is_file() and (P4_TOOLS / "deploy.py").is_file() else None


def ports() -> list[str]:
    """Serial ports a board could be on, sorted. Which chip is behind each is not known
    until ``esptool`` talks to it; a wrong one is refused there."""
    return sorted({p for pattern in _PORT_GLOBS for p in glob.glob(pattern)})


def probe_jpeg(image: bytes | Path) -> bytes:
    """A small baseline JPEG of ``image``: what the board's decoder accepts.

    ESP-DL decodes baseline files only; sides are rounded down to a multiple of 8 so the
    chip's hardware decoder would take it too.
    """
    from PIL import Image

    src = io.BytesIO(image) if isinstance(image, bytes) else image
    with Image.open(src) as im:
        rgb = im.convert("RGB")
    rgb.thumbnail((PROBE_MAX_SIDE, PROBE_MAX_SIDE))
    w, h = max(8, rgb.width // 8 * 8), max(8, rgb.height // 8 * 8)
    out = io.BytesIO()
    rgb.resize((w, h)).save(out, "JPEG", quality=85, progressive=False, subsampling="4:2:0")
    return out.getvalue()


def _result(stdout: str) -> dict[str, Any] | None:
    """The tool's last ``RESULT {json}`` line, if it printed one."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            try:
                found = json.loads(line[len("RESULT ") :])
            except ValueError:
                return None
            return found if isinstance(found, dict) else None
    return None


def _run(argv: list[str], *, cwd: Path, timeout: float, what: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        str(REPO_ROOT / "packages" / pkg / "src")
        for pkg in ("tensor-factory-train", "tensor-factory")
    )
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv built here, no shell
            argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise DeviceError(f"{what} timed out after {timeout:.0f} s") from exc
    if proc.returncode != 0:
        found = _result(proc.stdout)
        if found and found.get("error"):
            raise DeviceError(str(found["error"]))
        tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-3:])
        raise DeviceError(f"{what} failed: {tail}")
    return proc.stdout


def build_p4_image(
    onnx: Path,
    *,
    size: int,
    calib_dir: Path | None,
    out_dir: Path,
    name: str = "studio",
    python: Path | None = None,
    tools: Path | None = None,
) -> Path:
    """Quantize ``onnx`` for the P4 and pack it for the model partition.

    The result is cached by the model's sha256, so asking twice for the same model
    quantizes once. Older images in ``out_dir`` are removed.
    """
    python = python or p4_python()
    if python is None:
        raise DeviceError(P4_SETUP_HINT)
    tools = tools or P4_TOOLS
    digest = hashlib.sha256(onnx.read_bytes()).hexdigest()[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    image = out_dir / f"{digest}.p4.bin"
    if image.is_file():
        return image
    espdl = out_dir / f"{digest}.espdl"
    argv = [str(python), str(tools / "export_espdl.py"), "--size", str(size), "--onnx", str(onnx)]
    if calib_dir is not None:
        argv += ["--calib", str(calib_dir)]
    _run([*argv, "--out", str(espdl)], cwd=out_dir, timeout=600, what="quantizing for the P4")
    tmp = out_dir / f"{digest}.p4.bin.part"
    _run(
        [
            str(python),
            str(tools / "deploy.py"),
            "pack",
            "--espdl",
            str(espdl),
            "--report",
            str(espdl.with_suffix(".report.json")),
            "--name",
            name,
            "--out",
            str(tmp),
        ],
        cwd=out_dir,
        timeout=60,
        what="packing the model",
    )
    for stale in out_dir.iterdir():
        if stale != tmp:
            stale.unlink(missing_ok=True)
    os.replace(tmp, image)
    return image


def flash_p4(
    port: str,
    image: Path,
    probe: Path | None = None,
    *,
    python: Path | None = None,
    tools: Path | None = None,
) -> dict[str, Any]:
    """Write ``image`` to the board on ``port``; returns what the board reported."""
    python = python or p4_python()
    if python is None:
        raise DeviceError(P4_SETUP_HINT)
    tools = tools or P4_TOOLS
    argv = [str(python), str(tools / "deploy.py"), "flash", "--port", port]
    if probe is not None:
        argv += ["--image", str(probe)]
    out = _run([*argv, str(image)], cwd=image.parent, timeout=300, what="writing to the board")
    found = _result(out)
    if not found or not found.get("ok"):
        raise DeviceError("the board did not confirm the model")
    return found


def build_pyz(onnx: Path, *, size: int, out: Path) -> Path:
    """``edge.pyz`` for ``onnx``: one file that serves ``/detect`` on bare CPython."""
    from tensor_factory_mcp.bundle import build

    build(onnx, out, input_size=size)
    return out


def _iou(a: list[float], b: list[float]) -> float:
    from tensor_factory.geometry import BBox

    return BBox.clamped(*a).iou(BBox.clamped(*b))


class Deployer:
    """One deploy at a time, in the background; the browser polls :meth:`state`.

    ``model`` returns the served ONNX (or ``None``), ``calib_dir`` a directory of training
    images to calibrate the quantizer with, ``probe`` the bytes of one labeled frame to try
    on the board, ``desktop_box`` what the desktop model says about a JPEG.
    """

    def __init__(
        self,
        work_dir: Path,
        *,
        size: int,
        model: Callable[[], Path | None],
        calib_dir: Callable[[], Path | None],
        probe: Callable[[], bytes | None],
        desktop_box: Callable[[Path, bytes], list[float] | None],
    ) -> None:
        self.work_dir = Path(work_dir)
        self.size = size
        self._model, self._calib_dir = model, calib_dir
        self._probe, self._desktop_box = probe, desktop_box
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"state": "idle"}

    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._state = fields

    def p4_image(self) -> Path:
        onnx = self._model()
        if onnx is None:
            raise DeviceError("no model trained yet")
        return build_p4_image(
            onnx, size=self.size, calib_dir=self._calib_dir(), out_dir=self.work_dir / "p4"
        )

    def pyz(self) -> Path:
        onnx = self._model()
        if onnx is None:
            raise DeviceError("no model trained yet")
        return build_pyz(onnx, size=self.size, out=self.work_dir / "edge.pyz")

    def start(self, port: str) -> bool:
        """Begin a P4 deploy; ``False`` when one is already running."""
        with self._lock:
            if self._state.get("state") == "running":
                return False
            self._state = {"state": "running", "step": "quantizing for the P4", "port": port}
        threading.Thread(target=self._deploy, args=(port,), daemon=True).start()
        return True

    def _deploy(self, port: str) -> None:
        try:
            onnx = self._model()
            if onnx is None:
                raise DeviceError("no model trained yet")
            image = self.p4_image()
            frame = self._probe()
            probe_path, jpeg = None, None
            if frame is not None:
                jpeg = probe_jpeg(frame)
                probe_path = self.work_dir / "probe.jpg"
                probe_path.write_bytes(jpeg)
            self._set(state="running", step="writing to the board", port=port)
            found = flash_p4(port, image, probe_path)
            self._set(state="done", port=port, result=self._shape(found, onnx, jpeg))
        except DeviceError as exc:
            self._set(state="error", port=port, error=str(exc))
        except Exception as exc:  # noqa: BLE001 -- surface to the UI, never kill the server
            self._set(state="error", port=port, error=f"{type(exc).__name__}: {exc}")

    def _shape(self, found: dict[str, Any], onnx: Path, jpeg: bytes | None) -> dict[str, Any]:
        model = found.get("model") or {}
        out: dict[str, Any] = {
            "name": model.get("name"),
            "size": model.get("size"),
            "bytes": model.get("bytes"),
            "crc32": model.get("crc32"),
            "selfTest": bool(model.get("self_test")),
            "netMs": model.get("net_ms"),
        }
        det = found.get("detection")
        if isinstance(det, dict) and det.get("ok") and jpeg is not None:
            board = [float(v) for v in det["box"]]
            desktop = self._desktop_box(onnx, jpeg)
            out["frame"] = {
                "ms": round(sum(float(det.get(k, 0)) for k in _STAGES), 2),
                "present": float(det.get("presence", 0)) >= 0,
                "box": board,
                "desktopBox": desktop,
                "iou": None if desktop is None else round(_iou(board, desktop), 3),
            }
        elif isinstance(det, dict):
            out["frameError"] = str(det.get("error") or "the board returned no detection")
        return out


_STAGES = ("decode_ms", "resize_ms", "net_ms", "post_ms")
