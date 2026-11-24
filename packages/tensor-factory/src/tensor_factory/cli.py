"""`tensor-factory` CLI -- the inference harness (detect, bench).

Light by default: importing this module does not pull onnxruntime. The detect/bench
subcommands lazy-import inference, so `tensor-factory --help` works without the infer extra.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _load_image(path: str):
    """Open via PIL when installed (any format); else hand the path to the stdlib
    decoder (PNG / binary PPM), keeping `detect` usable on a bare Python install."""
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return path
    return Image.open(path)


def _cmd_detect(args: argparse.Namespace) -> int:
    from .inference import Detector

    det = Detector(args.model, input_size=args.size, engine=args.engine)
    image = _load_image(args.image)
    # A presence-head model can say "nothing here": below threshold, emit no box at all.
    if det.has_presence:
        score = det.detect_presence(image)
        if score < 0.5:
            print(f"absent (presence {score:.2f}) -- no box")
            return 0
        print(f"present (presence {score:.2f})")
    box = det.detect_box(image)
    u = det.detect_uint8(image)
    print(f"box   (norm xyxy): {box.x1:.4f} {box.y1:.4f} {box.x2:.4f} {box.y2:.4f}")
    print(f"uint8 (x1 y1 x2 y2): {u[0]} {u[1]} {u[2]} {u[3]}")
    return 0


def _gray_frame(size: int):
    """A synthetic mid-gray frame: a PIL image when PIL is installed (keeps the ort
    bench on the fast PIL preprocessing), else binary PPM bytes for the stdlib path."""
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return b"P6\n%d %d\n255\n" % (size, size) + bytes([128]) * (size * size * 3)
    return Image.new("RGB", (size, size), (128, 128, 128))


def _cmd_bench(args: argparse.Namespace) -> int:
    from .inference import Detector, benchmark

    det = Detector(args.model, input_size=args.size, engine=args.engine)
    image = _load_image(args.image) if args.image else _gray_frame(args.size)
    fps = benchmark(det, image, n=args.n)
    print(f"{fps:.1f} fps  ({args.n} runs, {args.size}px, CPU)")
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .zerort import parse_model
    from .zerort.tfm import dump

    src = Path(args.model)
    out = Path(args.out) if args.out else src.with_suffix(".tfm")
    data = src.read_bytes()
    blob = dump(parse_model(data))
    out.write_bytes(blob)
    print(f"{src} ({len(data)} B) -> {out} ({len(blob)} B)")
    return 0


def _cmd_store(args: argparse.Namespace) -> int:
    import sys

    from .store import S3Store, StoreError

    try:
        store = S3Store.from_env(args.store)
        if args.action == "push-model":
            sha = store.push_model(args.path, args.name)
            print(f"{sha}  {args.path}")
        elif args.action == "pull-model":
            sha = store.pull_model(args.ref, args.out)
            print(f"{sha}  {args.out}")
        elif args.action == "push-dataset":
            manifest = store.push_dataset(args.path, args.name)
            size = sum(f["size"] for f in manifest["files"])
            print(f"{manifest['name']}: {len(manifest['files'])} files, {size} B")
        elif args.action == "pull-dataset":
            manifest = store.pull_dataset(args.name, args.out)
            print(f"{args.name}: {len(manifest['files'])} files -> {args.out}")
        else:
            for key, size in store.list(args.prefix):
                print(f"{size:>12}  {key}")
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensor-factory")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="detect the target object in one image")
    d.add_argument("--model", required=True, help="path to the ONNX model")
    d.add_argument("--image", required=True, help="path to the input image")
    d.add_argument("--size", type=int, default=480, help="model input size (px)")
    d.add_argument(
        "--engine",
        choices=("auto", "ort", "zerort"),
        default="auto",
        help="inference engine: onnxruntime, the pure-stdlib zerort, or auto-fallback",
    )
    d.set_defaults(func=_cmd_detect)

    b = sub.add_parser("bench", help="measure CPU throughput (fps)")
    b.add_argument("--model", required=True, help="path to the ONNX model")
    b.add_argument("--image", default=None, help="image to loop on (default: gray)")
    b.add_argument("--n", type=int, default=100, help="iterations")
    b.add_argument("--size", type=int, default=480, help="model input size (px)")
    b.add_argument(
        "--engine",
        choices=("auto", "ort", "zerort"),
        default="auto",
        help="inference engine: onnxruntime, the pure-stdlib zerort, or auto-fallback",
    )
    b.set_defaults(func=_cmd_bench)

    k = sub.add_parser("pack", help="convert an ONNX model to the flat .tfm container")
    k.add_argument("--model", required=True, help="path to the ONNX model")
    k.add_argument("--out", default=None, help="output path (default: <model>.tfm)")
    k.set_defaults(func=_cmd_pack)

    s = sub.add_parser("store", help="push and pull models and datasets (S3, stdlib only)")
    s.add_argument(
        "--store", default=None, help="s3://bucket[/prefix]; default $TENSOR_FACTORY_STORE"
    )
    act = s.add_subparsers(dest="action", required=True)
    a = act.add_parser("push-model", help="upload a model; prints its sha256")
    a.add_argument("path", help="model file (.onnx or .tfm)")
    a.add_argument("--name", default=None, help="name to record it under (default: file stem)")
    a = act.add_parser("pull-model", help="download a model by sha256 or name, verified")
    a.add_argument("ref", help="sha256 (64 hex) or a model name")
    a.add_argument("--out", required=True, help="where to write it (replaced atomically)")
    a = act.add_parser("push-dataset", help="upload every file under a dataset directory")
    a.add_argument("path", help="dataset directory")
    a.add_argument("--name", default=None, help="dataset name (default: directory name)")
    a = act.add_parser("pull-dataset", help="download a dataset by name")
    a.add_argument("name", help="dataset name")
    a.add_argument("--out", required=True, help="directory to write into")
    a = act.add_parser("ls", help="list what is in the store")
    a.add_argument("prefix", nargs="?", default="", help="e.g. models/ or datasets/")
    s.set_defaults(func=_cmd_store)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
