"""`tensor-factory-synth` CLI: sample for prompt iteration, dataset for the real run.

Defaults to the torch-free mock backend so both subcommands run anywhere. Pass
``--backend gemini`` to use Nano Banana (gemini-2.5-flash-image) generation via the
Gemini API (needs the ``gemini`` extra + a ``GEMINI_API_KEY``) plus GroundingDINO
auto-labeling (needs the ``gpu`` extra).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .generator import DEFAULT_SIZE, Generator


def _make_generator(backend: str) -> Generator:
    if backend == "mock":
        from .generator import MockGenerator

        return MockGenerator()
    if backend == "gemini":
        from .generator import NanoBananaGenerator

        return NanoBananaGenerator()
    raise ValueError(f"unknown backend: {backend!r}")


def _cmd_sample(args: argparse.Namespace) -> int:
    from .sampling import sample_grid

    gen = _make_generator(args.backend)
    sheet = sample_grid(
        gen,
        args.prompt,
        n=args.n,
        seed_start=args.seed,
        size=args.size,
        cols=args.cols,
    )
    sheet.save(args.out)
    print(f"wrote contact sheet ({args.n} samples) -> {args.out}")
    return 0


def _cmd_triage(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from tensor_factory.review import review_summary

    coco = json.loads((Path(args.data) / "annotations.coco.json").read_text())
    s = review_summary(coco)
    img, ann = s["images"], s["annotations"]
    print(f"dataset: {args.data}")
    print(
        f"  images       {img['total']:>5}  "
        f"(approved {img['approved']}, pending {img['pending']}, rejected {img['rejected']})"
    )
    print(
        f"  annotations  {ann['total']:>5}  "
        f"(approved {ann['approved']}, pending {ann['pending']}, rejected {ann['rejected']})"
    )
    print(f"  trainable    {ann['trainable']:>5}  annotations would enter training")
    if ann["pending"]:
        print(
            f"  -> {ann['pending']} annotations awaiting human review "
            "(tensor-factory-label push -> correct -> pull)"
        )
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    from .pipeline import synth_dataset

    gen = _make_generator(args.backend)
    labeler = None
    if args.backend == "gemini" and not args.no_label:
        from .autolabel import GroundingDinoAutoLabeler

        labeler = GroundingDinoAutoLabeler()

    records = synth_dataset(
        gen,
        args.prompt,
        args.features,
        n=args.n,
        out_dir=args.out,
        seed_start=args.seed,
        size=args.size,
        labeler=labeler,
    )
    n_dets = sum(len(d) for _, _, _, d in records)
    print(f"wrote {len(records)} images, {n_dets} annotations -> {args.out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensor-factory-synth")
    parser.add_argument(
        "--backend", choices=("mock", "gemini"), default="mock", help="generation backend"
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="image size (px)")
    parser.add_argument("--seed", type=int, default=0, help="starting seed")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="generate a contact sheet for prompt iteration")
    s.add_argument("--prompt", required=True)
    s.add_argument("--n", type=int, default=9, help="number of samples")
    s.add_argument("--cols", type=int, default=3)
    s.add_argument("--out", default="samples.png")
    s.set_defaults(func=_cmd_sample)

    d = sub.add_parser("dataset", help="generate + label a COCO dataset")
    d.add_argument("--prompt", required=True)
    d.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="features to extract, e.g. --features helicoil thread",
    )
    d.add_argument("--n", type=int, default=64, help="number of images")
    d.add_argument("--out", default="dataset", help="output directory")
    d.add_argument(
        "--no-label",
        action="store_true",
        help="skip auto-labeling (gemini backend only)",
    )
    d.set_defaults(func=_cmd_dataset)

    t = sub.add_parser("triage", help="report review/validation state of a dataset")
    t.add_argument("--data", required=True, help="dataset dir (annotations.coco.json)")
    t.set_defaults(func=_cmd_triage)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
