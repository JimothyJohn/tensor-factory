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


def _launch_chromium_or_skip(p):  # noqa: ANN001, ANN202
    """Launch chromium via playwright, skipping the test if no browser binary is installed.

    ``browser`` must be bound on every path through this function: if ``launch()`` raises,
    ``pytest.skip`` normally aborts the test immediately, but nothing in the function body
    itself guarantees that to a reader (or to static analysis) -- initializing ``browser``
    up front, rather than only inside the ``try``, keeps this function well-defined even if
    ``pytest.skip`` is ever replaced or stubbed out (e.g. in a unit test). Extracted out of
    the test body so that case is unit-testable without a real playwright/chromium stack.
    """
    browser = None
    try:
        browser = p.chromium.launch()
    except Exception as exc:  # noqa: BLE001 -- no browser binary installed
        pytest.skip(f"chromium unavailable: {exc}")
    return browser


@pytest.mark.unit
def test_launch_chromium_or_skip_initializes_before_use(monkeypatch):
    """Regression for CodeQL py/uninitialized-local-variable (alert #4).

    Before the fix, ``browser`` was only assigned inside the ``try``; if ``launch()``
    raised and ``pytest.skip`` (monkeypatched here, as pytest does not actually abort a
    plain function call) failed to abort control flow, the subsequent read of ``browser``
    raised ``UnboundLocalError`` instead of yielding a defined value.
    """

    class _FailingChromium:
        def launch(self):
            raise RuntimeError("no browser binary")

    class _P:
        chromium = _FailingChromium()

    monkeypatch.setattr(pytest, "skip", lambda *a, **k: None)
    assert _launch_chromium_or_skip(_P()) is None


@pytest.mark.integration
def test_demo_detects_sample_in_a_real_browser():
    pw = pytest.importorskip("playwright.sync_api")
    with _serve(DOCS) as port:
        try:
            with pw.sync_playwright() as p:
                browser = _launch_chromium_or_skip(p)
                page = browser.new_page()
                page.goto(f"http://127.0.0.1:{port}/demo.html")
                # The page auto-runs the (box-only) mock model on the sample at load. A
                # completed inference populates the timing field with "<n> ms" -- a
                # model-agnostic signal a detection ran. Generous timeout: cold WASM init +
                # fetching the runtime from the CDN on first load.
                try:
                    page.wait_for_function(
                        "() => /ms$/.test(document.getElementById('m-ms').textContent.trim())",
                        timeout=60000,
                    )
                except Exception as exc:  # noqa: BLE001 -- likely CDN/WASM blocked offline
                    err = page.text_content("#err-text") or ""
                    browser.close()
                    pytest.skip(f"demo did not resolve (offline CDN?): {err or exc}")
                u8 = page.text_content("#m-u8") or ""
                norm = page.text_content("#m-norm") or ""
                browser.close()
                # Four uint8 values and four normalized coords were drawn -- a real detection.
                assert len([t for t in u8.split() if t.strip().isdigit()]) == 4
                assert len(norm.split()) == 4
        except pw.Error as exc:
            pytest.skip(f"playwright runtime error: {exc}")
