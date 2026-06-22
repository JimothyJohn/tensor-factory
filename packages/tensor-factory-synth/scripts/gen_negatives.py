#!/usr/bin/env python
"""Generate generic machined-part negatives -- holes and features but NO helicoils.

Same visual domain as the helicoil samples (macro inspection-microscope photos of worn
machined metal), reusing the shared material / surface / lighting / photoreal vocabulary
from ``gen_samples`` so the negatives are indistinguishable in style -- only the subject
differs: bare drilled/tapped/counterbored holes, slots, pockets, chamfers, with no wire-
thread insert of any kind. These are raw, *unlabeled* background images meant to be added
to the dataset as negatives.

    uv run --with google-genai python packages/tensor-factory-synth/scripts/gen_negatives.py
    uv run --with google-genai python packages/tensor-factory-synth/scripts/gen_negatives.py \
        --n 110 --out examples/helicoils/images/negatives_pool

Output: <out>/images/neg_*.png + <out>/manifest.json (provenance only -- no annotations).
Needs GEMINI_API_KEY in the env and google-genai (the `gemini` extra, or --with).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import gen_samples as G  # noqa: E402 -- sibling script, shares the prompt vocabulary

from tensor_factory_synth.generator import NanoBananaGenerator  # noqa: E402

# Hammer the point home: every prompt must render bare metal, never a coiled insert.
NO_INSERT = (
    "absolutely no wire-thread insert, no coiled wire, no helicoil, no spring -- "
    "just bare machined metal"
)

# (feature phrasing, preferred angles) -- generic machined features, all insert-free.
FEATURES = [
    ("an empty drilled hole with smooth unthreaded walls", ["top_down_deep", "oblique"]),
    (
        "a bare tapped hole showing clean cut internal threads, nothing installed",
        ["top_down_deep", "oblique"],
    ),
    ("a countersunk screw hole with a crisp chamfered rim, empty", ["oblique", "grazing_flush"]),
    ("a counterbored hole with a flat shoulder and a plain bore", ["top_down_deep", "oblique"]),
    ("a milled rectangular pocket with rounded internal corners", ["oblique", "grazing_flush"]),
    ("a straight milled slot with crisp machined edges", ["grazing_flush", "oblique"]),
    ("a chamfered through-hole, plain smooth bore", ["top_down_deep", "oblique"]),
    ("a blind hole with a flat bottom and faint drill swirl marks", ["top_down_deep", "oblique"]),
    ("two plain unthreaded bolt holes side by side", ["oblique", "grazing_flush"]),
    ("a cross-drilled intersecting hole exposing a bare bore", ["oblique", "top_down_deep"]),
    ("a reamed dowel-pin hole, smooth empty bore", ["top_down_deep", "oblique"]),
    ("a deburred bevelled edge along a machined block face", ["grazing_flush", "oblique"]),
]


def build_plan(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    plan = []
    turn: dict[int, int] = {}
    for i in range(n):
        fi = i % len(FEATURES)
        phrasing, angles = FEATURES[fi]
        angle_key = angles[turn.get(fi, 0) % len(angles)]
        turn[fi] = turn.get(fi, 0) + 1
        prompt = (
            f"macro inspection photo {G.ANGLE_TXT[angle_key]}, {phrasing}, {NO_INSERT}, "
            f"in {rng.choice(G.MATERIALS)}, {rng.choice(G.SURFACE)}, "
            f"{rng.choice(G.LIGHTING)}, {G.PHOTOREAL}"
        )
        plan.append(
            {
                "file": f"neg_{i:05d}.png",
                "seed": seed + i,
                "feature": phrasing,
                "angle": angle_key,
                "prompt": prompt,
            }
        )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n", type=int, default=110, help="number of negatives (match the labeled set)"
    )
    ap.add_argument("--out", default="examples/helicoils/images/negatives_pool")
    ap.add_argument("--seed", type=int, default=200000, help="seed base (disjoint from samples)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    plan = build_plan(args.n, args.seed)
    gen = NanoBananaGenerator()
    print(f"plan: {len(plan)} machined-part negatives -> {out}")

    def render(item: dict) -> dict:
        last = "unknown"
        for _ in range(2):
            try:
                s = gen.generate(item["prompt"], item["seed"], size=480)
                s.image.save(out / "images" / item["file"])
                return {**item, "ok": True}
            except Exception as exc:  # noqa: BLE001 -- batch survives one refusal; record + retry
                last = str(exc)
        return {**item, "ok": False, "error": last}

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(render, it) for it in plan]):
            r = fut.result()
            results.append(r)
            if len(results) % 10 == 0 or not r["ok"]:
                tag = "ok " if r["ok"] else "FAIL"
                print(
                    f"  [{tag}] {len(results)}/{len(plan)}  {r['file']}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    results.sort(key=lambda r: r["file"])
    ok = [r for r in results if r["ok"]]
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "machined-part negatives (no helicoil), raw and unlabeled",
                "seed_base": args.seed,
                "model": gen.model,
                "count": len(ok),
                "samples": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(ok)}/{args.n} negatives generated -> {out}/images/")
    return 0 if len(ok) == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
