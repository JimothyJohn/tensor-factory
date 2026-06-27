"""Tensor Factory Studio backend over the stdlib ``http.server``.

Serves the browser UI (the ``studio/`` directory) and the training API. Same lightweight
shape as ``tensor-factory-http`` -- ``ThreadingHTTPServer`` + ``BaseHTTPRequestHandler``,
no web framework. Binds ``127.0.0.1`` by default: a trainer with a writable dataset on a
dev box is not public-by-default.

Routes:
  ``GET  /``, ``/<static>``  -> the Studio UI files
  ``GET  /metrics``          -> latest training metrics (polled by the UI)
  ``GET  /status``           -> device, running flag, dataset counts
  ``GET  /model``            -> download the served int8 ONNX
  ``POST /samples?id&present&box``  -> body = raw PNG; add/replace a labeled frame
  ``POST /predict``          -> body = raw PNG; auto-label with the best served model
  ``POST /reset``            -> wipe the dataset and trainer state
  ``POST /pause`` / ``/resume`` -> gate the training loop
"""

from __future__ import annotations

import contextlib
import errno
import functools
import json
import mimetypes
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .dataset import Dataset
from .trainer import Trainer

DEFAULT_MAX_BYTES = 25 * 1024 * 1024
# packages/tensor-factory-studio/src/tensor_factory_studio/server.py -> repo root is parents[4]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_UI_DIR = REPO_ROOT / "studio"


@functools.lru_cache(maxsize=4)
def _detector(model_path: str, input_size: int):
    from tensor_factory.inference import Detector  # lazy: onnxruntime

    return Detector(model_path, input_size=input_size)


def _predict(model_path: str, input_size: int, png: bytes) -> dict[str, Any]:
    import io

    from PIL import Image

    det = _detector(model_path, input_size)
    with Image.open(io.BytesIO(png)) as im:
        img = im.convert("RGB")
        score = det.detect_presence(img) if det.has_presence else 1.0
        present = score >= 0.5
        box = None
        if present:
            b = det.detect_box(img)
            box = [b.x1, b.y1, b.x2, b.y2]
    return {"ready": True, "present": present, "score": score, "box": box}


def _make_handler(
    dataset: Dataset, trainer: Trainer, ui_dir: Path, input_size: int, max_bytes: int
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "tensor-factory-studio"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return  # quiet; never interpolate client input into logs

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _read_body(self) -> bytes | None:
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(411, {"error": "Content-Length required"})
                return None
            if length <= 0:
                self._json(400, {"error": "empty body"})
                return None
            if length > max_bytes:
                self._json(413, {"error": f"body exceeds {max_bytes} bytes"})
                return None
            return self.rfile.read(length)

        # --- static UI ---
        def _serve_static(self, path: str) -> None:
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (ui_dir / rel).resolve()
            try:
                target.relative_to(ui_dir)  # reject path traversal outside the UI dir
            except ValueError:
                self._json(404, {"error": f"no such file: {path}"})
                return
            if not target.is_file():
                self._json(404, {"error": f"no such file: {path}"})
                return
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _guard(self, fn) -> None:
            # Never let a handler exception close the socket with no response -- a silent
            # empty response (ERR_EMPTY_RESPONSE) is far harder to diagnose than a 500.
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                # headers may already be sent (can't send a 500 then) -- suppress and move on
                with contextlib.suppress(Exception):
                    self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:  # noqa: N802
            self._guard(self._get)

        def do_POST(self) -> None:  # noqa: N802
            self._guard(self._post)

        def _get(self) -> None:
            route = urlsplit(self.path).path
            if route == "/metrics":
                self._json(200, trainer.metrics())
            elif route == "/status":
                self._json(
                    200,
                    {
                        "backend": trainer.device,
                        "running": not trainer.paused,
                        "hasModel": trainer.served is not None,
                        "counts": dataset.counts(),
                    },
                )
            elif route == "/model":
                if trainer.served is None:
                    self._json(404, {"error": "no model trained yet"})
                    return
                data = Path(trainer.served).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", "attachment; filename=tinydetector.onnx")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._serve_static(route)

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def _post(self) -> None:
            route = urlsplit(self.path).path
            if route == "/samples":
                self._post_samples()
            elif route == "/predict":
                self._post_predict()
            elif route == "/reset":
                dataset.clear()
                trainer.reset()
                self._json(200, {"ok": True})
            elif route == "/pause":
                trainer.pause()
                self._json(200, {"ok": True, "running": False})
            elif route == "/resume":
                trainer.resume()
                self._json(200, {"ok": True, "running": True})
            else:
                self._json(404, {"error": f"no such route: {route}"})

        def _post_samples(self) -> None:
            q = parse_qs(urlsplit(self.path).query)
            try:
                fid = int(q["id"][0])
            except (KeyError, ValueError):
                self._json(400, {"error": "id query param required"})
                return
            present = q.get("present", ["1"])[0] in ("1", "true", "True")
            box = None
            if present and q.get("box", [""])[0]:
                try:
                    box = [float(v) for v in q["box"][0].split(",")]
                    if len(box) != 4:
                        raise ValueError
                except ValueError:
                    self._json(400, {"error": "box must be x1,y1,x2,y2"})
                    return
            body = self._read_body()
            if body is None:
                return
            try:
                dataset.upsert(fid, present, box, body)
            except Exception as exc:  # noqa: BLE001 -- bad image -> 400
                self._json(400, {"error": f"could not store sample: {type(exc).__name__}"})
                return
            trainer.mark_dirty()
            self._json(200, {"ok": True, "counts": dataset.counts()})

        def _post_predict(self) -> None:
            body = self._read_body()
            if body is None:
                return
            if trainer.served is None:
                self._json(200, {"ready": False})
                return
            try:
                self._json(200, _predict(str(trainer.served), input_size, body))
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"error": f"predict failed: {type(exc).__name__}"})

    return Handler


class _StudioServer(ThreadingHTTPServer):
    daemon_threads = True  # don't let in-flight requests block process exit
    request_queue_size = 64  # tolerate short bursts of connections without dropping SYNs


def make_server(
    host: str,
    port: int,
    *,
    dataset: Dataset,
    trainer: Trainer,
    ui_dir: Path,
    input_size: int,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ThreadingHTTPServer:
    handler = _make_handler(dataset, trainer, Path(ui_dir).resolve(), input_size, max_bytes)
    return _StudioServer((host, port), handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 8089,
    *,
    data_dir: str | Path | None = None,
    ui_dir: str | Path | None = None,
    size: int = 480,
    width: int = 16,
    epochs: int = 20,
    batch: int = 16,
) -> int:
    # Resolve to absolute: _serve_static compares against resolved targets, and a relative
    # ui_dir would make every relative_to() check raise and 404 the whole UI.
    ui = (Path(ui_dir) if ui_dir else DEFAULT_UI_DIR).resolve()
    if not (ui / "index.html").is_file():
        print(f"UI not found at {ui} -- pass --ui-dir to point at the studio/ directory")
        return 2
    data = Path(data_dir) if data_dir else ui / ".data"
    dataset = Dataset(data)
    trainer = Trainer(dataset, data / "models", size=size, width=width, epochs=epochs, batch=batch)
    # Bind BEFORE starting the trainer so a port clash fails loudly instead of leaving a
    # zombie server (or a previous instance) shadowing this one -- you'd be talking to the
    # wrong process and never see your changes.
    try:
        httpd = make_server(
            host, port, dataset=dataset, trainer=trainer, ui_dir=ui, input_size=size
        )
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"port {port} is already in use -- another server is holding it. Free it "
                f"(`lsof -nP -iTCP:{port} -sTCP:LISTEN`, then kill the PID) or pass --port."
            )
            return 1
        raise
    trainer.start()
    bound = httpd.server_address[1]
    print(f"tensor-factory-studio on http://{host}:{bound}  (data: {data})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    finally:
        trainer.stop()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="tensor-factory-studio",
        description="Serve the Studio labeling UI and train continuously on labeled frames.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--data-dir", default=None, help="dataset dir (default: <ui>/.data)")
    p.add_argument("--ui-dir", default=None, help="studio UI dir (default: repo studio/)")
    p.add_argument("--size", type=int, default=480, help="square model input px")
    p.add_argument("--width", type=int, default=16, help="base channel width")
    p.add_argument("--epochs", type=int, default=20, help="epochs per retrain round")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args(argv)
    return serve(
        a.host,
        a.port,
        data_dir=a.data_dir,
        ui_dir=a.ui_dir,
        size=a.size,
        width=a.width,
        epochs=a.epochs,
        batch=a.batch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
