"""`tensor-factory-train` CLI: fit the tiny detector and export int8 ONNX."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _cmd_fit(args: argparse.Namespace) -> int:
    from .train import fit

    out = fit(
        args.data,
        args.out,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        size=args.size,
        width=args.width,
        device=args.device,
        classify=args.classify,
        val_frac=args.val_frac,
        box_weight=args.box_weight,
        cls_weight=args.cls_weight,
        augment=args.augment,
        require_review=not args.allow_unreviewed,
        negatives=args.negatives,
    )
    print(f"wrote int8 ONNX -> {out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensor-factory-train")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit", help="train on a COCO dataset, export int8 ONNX")
    f.add_argument("--data", required=True, help="dataset dir (annotations.coco.json + images/)")
    f.add_argument("--out", default="model.onnx", help="output ONNX path")
    f.add_argument("--epochs", type=int, default=10)
    f.add_argument("--batch", type=int, default=16)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--size", type=int, default=480)
    f.add_argument("--width", type=int, default=16, help="model channel width")
    f.add_argument("--device", default=None, help="cuda/mps/cpu (default: auto)")
    f.add_argument(
        "--classify",
        action="store_true",
        help="train a class head too (categories from the COCO file)",
    )
    f.add_argument("--val-frac", type=float, default=0.0, help="held-out fraction for metrics")
    f.add_argument("--box-weight", type=float, default=1.0, help="weight on the box loss")
    f.add_argument("--cls-weight", type=float, default=1.0, help="weight on the class loss")
    f.add_argument("--augment", action="store_true", help="random flip augmentation")
    f.add_argument(
        "--negatives",
        nargs="+",
        default=None,
        help="dirs of raw no-object images -> a trailing 'background' class "
        "(forces the presence head on; box loss is masked for them)",
    )
    f.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="train on un-validated labels too (default: only human-approved annotations)",
    )
    f.set_defaults(func=_cmd_fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
