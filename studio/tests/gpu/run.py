"""Check the WebGPU trainer against the CPU reference in a real, headless Chromium.

    python3 studio/tests/gpu/run.py            # needs: pip install playwright; playwright install chromium

Serves ``studio/`` on a free local port, opens ``tests/gpu/harness.html`` with WebGPU
enabled and fails if any value is further from torch's than ``TOLERANCE`` (float32 on the
GPU against float64 ground truth). Not part of the gate: it needs a browser and a GPU.
Exit status 0 = all checks passed, 1 = a check failed, 2 = no WebGPU adapter here.
"""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
from pathlib import Path

TOLERANCE = 2e-4
LOOSE = 1e-3  # outputs after three chained float32 optimizer steps
STUDIO = Path(__file__).resolve().parents[2]
FLAGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=metal"]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # noqa: ANN002
        return


def main() -> int:
    from playwright.sync_api import sync_playwright

    handler = functools.partial(Quiet, directory=str(STUDIO))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/tests/gpu/harness.html"
    with sync_playwright() as p:
        browser = p.chromium.launch(args=FLAGS)
        page = browser.new_page()
        errors: list[str] = []
        page.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_function("window.report !== undefined", timeout=300_000)
        report = page.evaluate("window.report")
        browser.close()
    httpd.shutdown()
    if report.get("error") == "no WebGPU adapter":
        print("no WebGPU adapter in this browser; nothing checked")
        return 2
    failed = [f"{k}: {v:.2e}" for k, v in report.get("checks", {}).items() if not v <= TOLERANCE]
    failed += [f"{k}: {v:.2e}" for k, v in report.get("loose", {}).items() if not v <= LOOSE]
    worst = max(report.get("checks", {"-": float("inf")}).values())
    print(json.dumps({k: v for k, v in report.items() if k not in ("checks", "loose")}))
    print(f"{len(report.get('checks', {}))} checks, worst error {worst:.2e} (tolerance {TOLERANCE:.0e})")
    for line in [*failed[:12], *errors[:3]]:
        print("  " + line[:600])
    ok = not failed and not errors and "error" not in report and "gpuError" not in report
    ok = ok and len(report.get("checks", {})) >= 60 and len(report.get("loose", {})) == 2
    return 0 if ok and report.get("finite") else 1


if __name__ == "__main__":
    sys.exit(main())
