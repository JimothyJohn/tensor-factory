"""`helicoils` CLI -- the inference harness (detect, bench).

Light by default: importing this module does not pull onnxruntime. The detect/bench
subcommands lazy-import inference, so `helicoils --help` works without the infer extra.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _cmd_detect(args: argparse.Namespace) -> int:
    from PIL import Image

    from .inference import Detector

    det = Detector(args.model, input_size=args.size)
    image = Image.open(args.image)
    box = det.detect_box(image)
    u = det.detect_uint8(image)
    print(f"box   (norm xyxy): {box.x1:.4f} {box.y1:.4f} {box.x2:.4f} {box.y2:.4f}")
    print(f"uint8 (x1 y1 x2 y2): {u[0]} {u[1]} {u[2]} {u[3]}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from PIL import Image

    from .inference import Detector, benchmark

    det = Detector(args.model, input_size=args.size)
    image = (
        Image.open(args.image)
        if args.image
        else Image.new("RGB", (args.size, args.size), (128, 128, 128))
    )
    fps = benchmark(det, image, n=args.n)
    print(f"{fps:.1f} fps  ({args.n} runs, {args.size}px, CPU)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helicoils")
    parser.add_argument("--size", type=int, default=480, help="model input size (px)")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="detect a helicoil in one image")
    d.add_argument("--model", required=True, help="path to the ONNX model")
    d.add_argument("--image", required=True, help="path to the input image")
    d.set_defaults(func=_cmd_detect)

    b = sub.add_parser("bench", help="measure CPU throughput (fps)")
    b.add_argument("--model", required=True, help="path to the ONNX model")
    b.add_argument("--image", default=None, help="image to loop on (default: gray)")
    b.add_argument("--n", type=int, default=100, help="iterations")
    b.set_defaults(func=_cmd_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
