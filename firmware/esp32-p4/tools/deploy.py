"""Put a model on an ESP32-P4 that already runs this firmware: no rebuild, one flash write.

The firmware reads its model from the ``model`` flash partition. This tool wraps an
``.espdl`` in the 64-byte header the firmware expects, writes it there, resets the board
and reports what the board says about it. It can also send a JPEG file and print the box.

    python tools/deploy.py pack   --espdl m.espdl --report m.report.json --out m.p4.bin
    python tools/deploy.py flash  --port PORT m.p4.bin [--image photo.jpg]
    python tools/deploy.py detect --port PORT photo.jpg

``pack`` is stdlib only. ``flash`` and ``detect`` need ``esptool`` (which brings
``pyserial``): ``VIRTUAL_ENV=~/esp/ppq-venv uv pip install esptool``. The last line of
output is ``RESULT {json}``; the exit status is 0 only when the board accepted the model.

The JPEG must be baseline (not progressive); ESP-DL's decoder takes nothing else.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

MAGIC = b"TFMODEL1"
HEADER = struct.Struct("<8sIIIf40s")  # magic, bytes, crc32, size, gain, name
MODEL_OFFSET = 4096  # the .espdl starts here, inside the partition
PARTITION_OFFSET = 0xB10000  # "model" in partitions.csv
PARTITION_SIZE = 4 * 1024 * 1024
BAUD = 115200


def pack(espdl: bytes, *, size: int, gain: float, name: str) -> bytes:
    """The bytes to write at the start of the model partition."""
    if not espdl:
        raise ValueError("empty model")
    if len(espdl) > PARTITION_SIZE - MODEL_OFFSET:
        raise ValueError(
            f"model is {len(espdl)} B; the partition holds {PARTITION_SIZE - MODEL_OFFSET} B"
        )
    if not 8 <= size <= 4096:
        raise ValueError(f"input size {size} px is not plausible")
    label = name.encode("ascii", "replace")[:39]
    header = HEADER.pack(MAGIC, len(espdl), zlib.crc32(espdl), size, gain, label)
    return header + b"\xff" * (MODEL_OFFSET - len(header)) + espdl


def unpack(image: bytes) -> dict:
    """The header fields of a packed image; raises ``ValueError`` if it is not one."""
    if len(image) < MODEL_OFFSET or image[:8] != MAGIC:
        raise ValueError("not a packed model image (no TFMODEL1 header)")
    _, n, crc, size, gain, name = HEADER.unpack_from(image)
    blob = image[MODEL_OFFSET : MODEL_OFFSET + n]
    if len(blob) != n or zlib.crc32(blob) != crc:
        raise ValueError("packed model image is truncated or corrupt")
    return {
        "name": name.rstrip(b"\0").decode("ascii"),
        "size": size,
        "gain": gain,
        "bytes": n,
        "crc32": f"{crc:08x}",
    }


class Board:
    """The board's console: reset it, read its lines, send it image files."""

    def __init__(self, port: str) -> None:
        import serial  # ty: ignore[import-not-found]

        self.port = serial.Serial(port, BAUD, timeout=0.2)
        self.pending = b""
        self.queue: list[str] = []

    def reset(self) -> None:
        self.port.dtr, self.port.rts = False, True
        time.sleep(0.2)
        self.port.rts = False
        self.port.reset_input_buffer()
        self.pending, self.queue = b"", []

    def wait(self, prefix: str, timeout: float) -> str | None:
        """The next line starting with ``prefix``, without it; ``None`` on timeout.

        Lines after the match stay queued for the next call.
        """
        deadline = time.time() + timeout
        while True:
            while self.queue:
                line = self.queue.pop(0)
                if line.startswith(prefix):
                    return line[len(prefix) :].strip()
            if time.time() >= deadline:
                return None
            self.pending += self.port.read(4096)
            *whole, self.pending = self.pending.split(b"\n")
            self.queue += [raw.decode("utf-8", "replace").strip() for raw in whole]

    def ready(self, timeout: float = 30.0) -> bool:
        """True once the firmware answers a PING carrying a fresh token.

        The token is what makes this a sync point: the board may still be booting when the
        port opens, and reset noise can leave a partial line in its buffer. Everything it
        printed before the echoed token is dropped.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            token = f"{time.monotonic_ns():x}"
            self.port.write(f"PING {token}\n".encode())
            if self.wait(f"READY {token}", 1.5) is not None:
                return True
        return False

    def detect(self, jpeg: bytes, timeout: float = 60.0) -> dict:
        # PING first: it ends any partial line left in the board's buffer (reset noise),
        # so the image line is parsed from its first byte.
        if not self.ready():
            return {"ok": False, "error": "the board did not answer PING"}
        line = b"IMG " + base64.b64encode(jpeg) + b"\n"
        for i in range(0, len(line), 1024):  # no flow control: stay inside the RX buffer
            self.port.write(line[i : i + 1024])
            self.port.flush()
        reply = self.wait("DET ", timeout)
        if reply is None:
            return {"ok": False, "error": "no answer from the board"}
        return json.loads(reply)


def _finish(result: dict) -> int:
    print("RESULT " + json.dumps(result))
    return 0 if result.get("ok") else 1


def cmd_pack(args: argparse.Namespace) -> int:
    meta = json.loads(Path(args.report).read_text())["meta"] if args.report else {}
    size = args.size or meta.get("size")
    gain = args.gain if args.gain is not None else meta.get("gain")
    if size is None or gain is None:
        raise SystemExit("pack: give --report (written by export_espdl.py) or --size and --gain")
    espdl = Path(args.espdl).read_bytes()
    image = pack(espdl, size=int(size), gain=float(gain), name=args.name or Path(args.espdl).stem)
    Path(args.out).write_bytes(image)
    return _finish({"ok": True, "out": args.out, "image_bytes": len(image), **unpack(image)})


def cmd_flash(args: argparse.Namespace) -> int:
    image = Path(args.image_file).read_bytes()
    want = unpack(image)
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [
            sys.executable,
            "-m",
            "esptool",
            "--chip",
            "esp32p4",
            "-p",
            args.port,
            "-b",
            "921600",
            "write-flash",
            hex(PARTITION_OFFSET),
            args.image_file,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        lines = [ln.strip() for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
        fatal = [ln for ln in lines if "fatal error" in ln.lower()]
        why = fatal[-1].split("occurred:", 1)[-1].strip() if fatal else " | ".join(lines[-3:])
        return _finish({"ok": False, "error": f"could not write to the board: {why}"})
    board = Board(args.port)
    board.reset()
    reply = board.wait("MODEL ", 60.0)
    if reply is None:
        return _finish(
            {
                "ok": False,
                "error": "the board printed no MODEL line: flash this firmware once "
                "(idf.py -p PORT flash in firmware/esp32-p4)",
            }
        )
    model = json.loads(reply)
    result: dict = {"ok": bool(model.get("ok")), "model": model, "want": want}
    if result["ok"] and model.get("crc32") != want["crc32"]:
        result.update(ok=False, error="the board is running a different model than was written")
    elif not result["ok"]:
        result["error"] = model.get("error") or "the board rejected the model (self-test failed)"
    if result["ok"] and args.image:
        result["detection"] = board.detect(Path(args.image).read_bytes())
    return _finish(result)


def cmd_detect(args: argparse.Namespace) -> int:
    board = Board(args.port)
    return _finish(board.detect(Path(args.image).read_bytes()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="wrap an .espdl in the header the firmware reads")
    p.add_argument("--espdl", required=True)
    p.add_argument("--report", default=None, help="the .report.json export_espdl.py wrote")
    p.add_argument("--size", type=int, default=None, help="square input px (else from --report)")
    p.add_argument("--gain", type=float, default=None, help="soft-argmax gain (else --report)")
    p.add_argument("--name", default=None, help="label the board reports (default: file stem)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser("flash", help="write a packed image to the board and check it runs")
    p.add_argument("--port", required=True)
    p.add_argument("--image", default=None, help="baseline JPEG to run once on the board")
    p.add_argument("image_file", help="output of `pack`")
    p.set_defaults(fn=cmd_flash)
    p = sub.add_parser("detect", help="send a baseline JPEG to the board, print its box")
    p.add_argument("--port", required=True)
    p.add_argument("image")
    p.set_defaults(fn=cmd_detect)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
