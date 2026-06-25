#!/usr/bin/env python3
"""Dual-stack CORS static server for serving dataset images to Label Studio.

Same wide-open CORS as scripts/cors_server.py, but binds an IPv6 socket with
IPV6_V6ONLY disabled so it answers on BOTH 127.0.0.1 and ::1. The plain server binds
IPv4 only; macOS /etc/hosts maps ``localhost`` to ``::1`` as well, and browsers commonly
try ``::1`` first -- so ``http://localhost:<port>`` fails in the browser (connection
refused on IPv6) even though curl, which falls back to IPv4, returns 200. That surfaces
in Label Studio as "There was an issue loading URL from $image value".

Usage:
    python3 examples/helicoils/scripts/cors_server_dualstack.py <port> <directory>
"""

from __future__ import annotations

import socket
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


class DualStackHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        # Accept IPv4-mapped addresses too, so one socket serves 127.0.0.1 and ::1.
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = partial(CORSRequestHandler, directory=directory)
    httpd = DualStackHTTPServer(("", port), handler)
    print(f"dual-stack CORS server on :{port} (v4+v6) serving {directory}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
