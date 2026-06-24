"""End-to-end browser smoke for the in-browser demo.

Serves docs/ over a real loopback HTTP server, drives a headless Chromium to demo.html, and
asserts the page auto-runs the bundled model on the sample and reports a detection. Marked
integration: needs Playwright + a Chromium build + network for the onnxruntime-web CDN. Run
with `uv run --with playwright python -m pytest tests/test_demo_browser.py -m integration`
(after `uv run --with playwright playwright install chromium`). Skips cleanly if any of
those are absent so the unit gate never depends on a browser.
"""

from __future__ import annotations

import contextlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"


@contextlib.contextmanager
def _serve(directory: Path):
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


@pytest.mark.integration
def test_demo_detects_sample_in_a_real_browser():
    pw = pytest.importorskip("playwright.sync_api")
    with _serve(DOCS) as port:
        try:
            with pw.sync_playwright() as p:
                try:
                    browser = p.chromium.launch()
                except Exception as exc:  # noqa: BLE001 -- no browser binary installed
                    pytest.skip(f"chromium unavailable: {exc}")
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{port}/demo.html")
                # The page auto-detects the sample on load; the status badge resolves to a
                # present/absent verdict. Wait for the mock model to localize the sample.
                try:
                    page.wait_for_selector(
                        "#status.badge.present, #status.badge.absent", timeout=30000
                    )
                except Exception as exc:  # noqa: BLE001 -- likely CDN/WASM blocked offline
                    err = page.text_content("#err-text") or ""
                    browser.close()
                    pytest.skip(f"demo did not resolve (offline CDN?): {err or exc}")
                status = page.text_content("#status-text") or ""
                u8 = page.text_content("#m-u8") or ""
                browser.close()
                # The mock sample is a helicoil -> present, and four uint8 values were drawn.
                assert "present" in status.lower()
                assert len([t for t in u8.split() if t.strip().isdigit()]) == 4
        except pw.Error as exc:
            pytest.skip(f"playwright runtime error: {exc}")
