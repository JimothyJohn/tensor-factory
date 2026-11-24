"""Reset the board and print its serial output until a marker line (or a timeout).

python tools/capture.py /dev/cu.usbmodemXXXX --until BENCH_DONE --timeout 300
"""

from __future__ import annotations

import argparse
import sys
import time

import serial  # pyserial; present in the ESP-IDF Python environment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--until", default="BENCH_DONE")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--grep", default="", help="only print lines containing this")
    args = ap.parse_args()

    port = serial.Serial(args.port, 115200, timeout=1)
    port.dtr, port.rts = False, True  # pulse reset, boot the app
    time.sleep(0.2)
    port.rts = False
    deadline, pending = time.time() + args.timeout, b""
    while time.time() < deadline:
        pending += port.read(4096)
        *lines, pending = pending.split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", "replace").rstrip()
            if args.grep in line:
                print(line, flush=True)
            if args.until in line:
                return 0
    print(f"timeout: no {args.until!r} within {args.timeout:.0f}s", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
