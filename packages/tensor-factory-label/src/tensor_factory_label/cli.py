"""`tensor-factory-label` CLI: push candidates into Label Studio, pull labels back to COCO.

URL/token default to the LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY env vars.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from .client import LabelStudioClient
from .config import bbox_config
from .convert import coco_to_tasks, http_image_url, local_storage_url, ls_export_to_coco


def _client(args: argparse.Namespace) -> LabelStudioClient:
    url = args.url or os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
    token = args.token or os.environ.get("LABEL_STUDIO_API_KEY")
    if not token:
        raise SystemExit("No API token: pass --token or set LABEL_STUDIO_API_KEY")
    return LabelStudioClient(url, token)


def _cmd_config(args: argparse.Namespace) -> int:
    print(bbox_config(args.labels))
    return 0


def _cmd_push(args: argparse.Namespace) -> int:
    coco = json.loads((Path(args.data) / "annotations.coco.json").read_text())
    image_url = http_image_url(args.image_base) if args.image_base else local_storage_url()
    tasks = coco_to_tasks(coco, image_url)
    labels = [c["name"] for c in coco["categories"]] or ["helicoil"]

    client = _client(args)
    project_id = client.create_project(args.title, bbox_config(labels))
    client.import_tasks(project_id, tasks)
    base = (args.url or os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")).rstrip("/")
    print(f"created project {project_id} with {len(tasks)} tasks")
    print(f"label at: {base}/projects/{project_id}/data")
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    client = _client(args)
    export = client.export_json(args.project)
    coco = ls_export_to_coco(export)
    Path(args.out).write_text(json.dumps(coco, indent=2), encoding="utf-8")
    n_img, n_ann = len(coco["images"]), len(coco["annotations"])
    print(f"wrote {n_img} images, {n_ann} annotations -> {args.out}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensor-factory-label")
    parser.add_argument("--url", default=None, help="Label Studio base URL")
    parser.add_argument("--token", default=None, help="API token")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("config", help="print the labeling config XML")
    c.add_argument("--labels", nargs="+", default=["helicoil"])
    c.set_defaults(func=_cmd_config)

    p = sub.add_parser("push", help="create a project and import a COCO dataset with predictions")
    p.add_argument("--data", required=True, help="dataset dir (annotations.coco.json + images/)")
    p.add_argument("--title", default="tensor_factory", help="project title")
    p.add_argument("--image-base", default=None, help="HTTP base URL serving the images")
    p.set_defaults(func=_cmd_push)

    q = sub.add_parser("pull", help="export corrected annotations back to COCO")
    q.add_argument("--project", type=int, required=True, help="project id")
    q.add_argument("--out", default="annotations.coco.json", help="output COCO path")
    q.set_defaults(func=_cmd_pull)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
