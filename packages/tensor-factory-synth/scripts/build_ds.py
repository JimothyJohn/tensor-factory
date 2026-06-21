"""Build a real photoreal helicoil dataset: Nano Banana generate -> GroundingDINO label -> COCO.

Two phases:
  1. Generate concurrently via the Gemini API (I/O bound, ThreadPoolExecutor).
  2. Label sequentially via GroundingDINO on MPS/CUDA (GPU bound).

Insert-present states only (drops ``missing``), so the model trains on holes that
actually contain a coil. Output is a COCO dataset:  <out>/images/*.png + annotations.coco.json.

Usage:
    uv run python packages/tensor-factory-synth/scripts/build_ds.py [OUT_DIR] [N_REQUEST]

This is a long (minutes) GPU + API job. Over SSH, run it under tmux so a dropped
connection can't SIGHUP it mid-run:

    scripts/run-bg tf-build \
        uv run python packages/tensor-factory-synth/scripts/build_ds.py
    tmux attach -t tf-build      # watch; detach with Ctrl-b then d

OUT_DIR defaults to the durable (gitignored) home under examples/helicoils/images/real_ds.
Needs GEMINI_API_KEY in the env for generation and the `gpu` extra for GroundingDINO.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

# gen_samples is a sibling script (not an installed module) -> add its dir to the path.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import gen_samples as G  # noqa: E402

from tensor_factory_synth.autolabel import Detection, GroundingDinoAutoLabeler  # noqa: E402
from tensor_factory_synth.export import build_coco, write_json  # noqa: E402
from tensor_factory_synth.generator import NanoBananaGenerator  # noqa: E402

# Repo root is three levels up from this script (packages/tensor-factory-synth/scripts).
REPO_ROOT = SCRIPTS_DIR.parents[2]
DEFAULT_OUT = REPO_ROOT / "examples" / "helicoils" / "images" / "real_ds"

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
N_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 170  # request extra; we drop label failures
SIZE = 480
SEED0 = 100000  # disjoint from the sample seeds
FEATURE = "threaded hole"

(OUT / "images").mkdir(parents=True, exist_ok=True)

# Varied prompts, insert-present states only.
plan = [p for p in G.build_plan(N_REQ, SEED0) if p["qc_state"] != "missing"]
print(f"plan: {len(plan)} insert-present images -> {OUT}")

# --- Phase 1: generate concurrently (API, I/O bound) ---
gen = NanoBananaGenerator()


def make(item: dict) -> dict:
    err = "unknown"
    for _ in range(2):
        try:
            img = gen.generate(item["prompt"], item["seed"], size=SIZE).image
            img.save(OUT / "images" / item["file"])
            return {**item, "ok": True}
        except Exception as e:  # noqa: BLE001 - record and retry, then drop
            err = str(e)
    return {**item, "ok": False, "err": err}


t0 = time.time()
gend: list[dict] = []
with ThreadPoolExecutor(max_workers=6) as ex:
    for fut in as_completed([ex.submit(make, it) for it in plan]):
        r = fut.result()
        gend.append(r)
        if len(gend) % 20 == 0:
            print(f"  generated {len(gend)}/{len(plan)}  ({time.time() - t0:.0f}s)")
gend = [r for r in gend if r["ok"]]
print(f"generated {len(gend)} images in {time.time() - t0:.0f}s")

# --- Phase 2: label sequentially (MPS/CUDA) ---
labeler = GroundingDinoAutoLabeler(box_threshold=0.25, text_threshold=0.2)


def pick(dets: list[Detection]) -> Detection | None:
    good = [
        d for d in dets if 0.03 < d.box.area < 0.85 and d.box.width < 0.95 and d.box.height < 0.95
    ]
    return max(good, key=lambda d: d.score) if good else None


records = []
t1 = time.time()
labeled = 0
for r in gend:
    img = Image.open(OUT / "images" / r["file"]).convert("RGB")
    d = pick(labeler.label(img, [FEATURE]))
    if d is None:
        continue
    det = Detection(label="helicoil", box=d.box, score=d.score)
    records.append((f"images/{r['file']}", SIZE, SIZE, [det]))
    labeled += 1
    if labeled % 20 == 0:
        print(f"  labeled {labeled}/{len(gend)}  ({time.time() - t1:.0f}s)")

write_json(OUT / "annotations.coco.json", build_coco(records, ["helicoil"]))
print(f"\nDATASET: {labeled} labeled / {len(gend)} generated -> {OUT}")
print(f"  ({len(gend) - labeled} dropped for no qualifying box)")
