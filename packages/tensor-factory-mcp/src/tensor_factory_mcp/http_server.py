"""A minimal HTTP detection endpoint over the stdlib ``http.server``.

A lighter-weight alternative to the MCP stdio server for callers that just want JSON over
HTTP -- no MCP client, no framework, zero extra dependencies (``ThreadingHTTPServer`` +
``BaseHTTPRequestHandler``). Wraps the same :mod:`tensor_factory_mcp.core` functions the
MCP tools use, so the JSON is byte-identical.

Routes:
  ``GET  /health``     -> ``{"status": "ok"}``
  ``GET  /model_info`` -> resolved model path, input size, IO name, ORT providers
  ``POST /detect``     -> request body is raw image bytes; returns the detection JSON

Example::

    tensor-factory-http --port 8088 &
    curl --data-binary @frame.png http://127.0.0.1:8088/detect

Binds ``127.0.0.1`` by default -- a detector on a dev box is not public-by-default; pass
``--host 0.0.0.0`` to expose it deliberately.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import core

# Cap the request body so a single client can't exhaust memory with a giant upload. A
# 480px frame is well under a megabyte; 25 MB is generous headroom for full-res photos.
DEFAULT_MAX_BYTES = 25 * 1024 * 1024


def _make_handler(
    model_path: str | None, input_size: int, max_bytes: int
) -> type[BaseHTTPRequestHandler]:
    class DetectHandler(BaseHTTPRequestHandler):
        # Advertised in the Server: header; keep the version out of it (don't leak internals).
        server_version = "tensor-factory-http"
        sys_version = ""

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            # Quiet by default: per-request stderr spam makes the server unusable in tests
            # and noisy in the foreground. Parameterized, never interpolating client input.
            return

        def do_GET(self) -> None:  # noqa: N802 -- http.server dispatch name
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            elif self.path == "/model_info":
                try:
                    self._send_json(200, core.model_info(model_path, input_size))
                except Exception as exc:  # noqa: BLE001 -- surfaced as JSON, not a 500 trace
                    self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            else:
                self._send_json(404, {"error": f"no such route: {self.path}"})

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self._send_json(404, {"error": f"no such route: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(411, {"error": "Content-Length required"})
                return
            if length <= 0:
                self._send_json(400, {"error": "empty request body"})
                return
            if length > max_bytes:
                self._send_json(413, {"error": f"body exceeds {max_bytes} bytes"})
                return
            data = self.rfile.read(length)
            try:
                result = core.detect_bytes(data, model_path, input_size)
            except Exception as exc:  # noqa: BLE001 -- bad image -> 400, not a 500 trace
                self._send_json(400, {"error": f"could not decode image: {type(exc).__name__}"})
                return
            self._send_json(200, result)

    return DetectHandler


def make_server(
    host: str,
    port: int,
    *,
    model_path: str | None = None,
    input_size: int = core.DEFAULT_INPUT_SIZE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ThreadingHTTPServer:
    """Build (but don't start) the detection HTTP server. ``port=0`` binds an ephemeral port."""
    handler = _make_handler(model_path, input_size, max_bytes)
    return ThreadingHTTPServer((host, port), handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 8088,
    *,
    model_path: str | None = None,
    input_size: int = core.DEFAULT_INPUT_SIZE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> int:
    httpd = make_server(
        host, port, model_path=model_path, input_size=input_size, max_bytes=max_bytes
    )
    model = core.resolve_model(model_path)
    bound = httpd.server_address[1]
    print(f"tensor-factory-http on http://{host}:{bound}  (model: {model})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tensor-factory-http",
        description="Lightweight HTTP detection endpoint (POST raw image bytes to /detect).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8088, help="bind port (default 8088)")
    parser.add_argument("--model", default=None, help="ONNX model path (default: bundled model)")
    parser.add_argument(
        "--input-size", type=int, default=core.DEFAULT_INPUT_SIZE, help="square model input px"
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_MAX_BYTES / (1024 * 1024),
        help="reject request bodies larger than this many MB",
    )
    args = parser.parse_args(argv)
    return serve(
        args.host,
        args.port,
        model_path=args.model,
        input_size=args.input_size,
        max_bytes=int(args.max_mb * 1024 * 1024),
    )


if __name__ == "__main__":
    raise SystemExit(main())
