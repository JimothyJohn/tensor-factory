#!/usr/bin/env python3
"""Static file server with wide-open CORS — for serving dataset images to Label Studio.

Label Studio loads each image into a ``<canvas crossorigin="anonymous">`` so it can draw
boxes on it. The browser refuses that cross-origin canvas read unless the image server
sends ``Access-Control-Allow-Origin``. Plain ``python -m http.server`` doesn't, which
surfaces in LS as "There was an issue loading URL from $image value" even though the URL
is valid and returns 200. This server adds the header so the canvas load succeeds.

Usage:
    scripts/cors_server.py <port> <directory>
"""

from __future__ import annotations

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that allows any origin to read the served files."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server dispatch name
        self.send_response(204)
        self.end_headers()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = partial(CORSRequestHandler, directory=directory)
    httpd = ThreadingHTTPServer(("", port), handler)
    print(f"CORS static server on :{port} serving {directory}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
