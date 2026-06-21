#!/usr/bin/env python
"""Generate a batch of helicoil QC-inspection samples with Nano Banana.

The reusable batch tool behind SAMPLES.md. Text-only by default; pass --reference PATH to
condition every generation on a real installed-helicoil photo (the fix for subjects the
model renders wrong from text alone -- proud / cross-threaded inserts especially).

    uv run --with google-genai python packages/helicoils-synth/scripts/gen_samples.py
    uv run --with google-genai python packages/helicoils-synth/scripts/gen_samples.py \
        --reference ref.png --n 25 --out images

QC state is recorded in <out>/manifest.json only, so the PNGs can be reviewed blind.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from helicoils_synth.generator import NanoBananaGenerator

PHOTOREAL = (
    "real macro photograph taken through an inspection microscope, natural uneven "
    "lighting, fine sensor grain, shallow depth of field and slightly soft focus, used "
    "and worn, gritty and imperfect, not a clean studio render"
)

# Reused everywhere an insert is present: forces the tight-coil, not-a-spring read.
TIGHT = (
    "the wire wound tight with no gaps between turns, forming continuous internal threads, "
    "not a loose spring"
)

ANGLE_TXT = {
    "top_down_deep": "looking straight down into a tapped hole, threads visible deep in the bore",
    "oblique": "at an oblique three-quarter angle into a tapped hole",
    "grazing_flush": "at a low grazing angle, almost flush along the top surface across the rim",
}

# state -> (phrasing, preferred angles). Installed states bias to top-down/oblique (the
# angles that render the coil as threads); proud uses grazing so the slight rise reads.
STATES = {
    "flush_pass": (
        f"a correctly installed wire thread insert seated flush with the surface, {TIGHT}, "
        "looking like a clean finely-threaded hole, nothing protruding",
        ["top_down_deep", "oblique"],
    ),
    "slightly_recessed": (
        f"an installed wire thread insert sitting slightly too deep, {TIGHT}, the tight "
        "coil recessed a little below the rim",
        ["top_down_deep", "oblique"],
    ),
    "slightly_proud": (
        f"an installed wire thread insert seated slightly too high, {TIGHT}, the tight coil "
        "raised just a hair proud of the surface, still threaded into the bore",
        ["grazing_flush", "oblique"],
    ),
    "missing": (
        "no insert installed, an empty tapped hole showing bare cut threads, the rim "
        "slightly dinged and dirty",
        ["top_down_deep", "grazing_flush"],
    ),
    "cross_threaded": (
        f"a wire thread insert installed crooked and cross-threaded, {TIGHT}, the coil "
        "seated visibly off-axis and uneven but still down in the bore",
        ["oblique", "top_down_deep"],
    ),
    "damaged_coil": (
        f"an installed wire thread insert with a damaged thread, {TIGHT}, one coil loop "
        "nicked or deformed out of round down inside the bore",
        ["top_down_deep", "oblique"],
    ),
}

DISTRIBUTION = (
    ["flush_pass"] * 5
    + ["slightly_recessed"] * 4
    + ["slightly_proud"] * 4
    + ["missing"] * 4
    + ["cross_threaded"] * 4
    + ["damaged_coil"] * 4
)

MATERIALS = [
    "dull worn aluminum",
    "clear-anodized aluminum, matte and scuffed",
    "bare stainless steel with a brushed finish",
    "grey tool steel",
]
SURFACE = [
    "machining marks and faint oxidation",
    "fine swarf chips and shop grime around the hole",
    "dust, smudges, and a faint oil film",
    "light corrosion, scuffs, and tiny scratches",
]
LIGHTING = [
    "uneven shop lighting",
    "a harsh single LED with glare and hard shadow",
    "dim angled microscope light with hotspots",
    "soft directional light with deep shadows",
]


def build_plan(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    states = (DISTRIBUTION * (n // len(DISTRIBUTION) + 1))[:n]
    rng.shuffle(states)
    angle_turn: dict[str, int] = {}
    plan = []
    for i in range(n):
        state_key = states[i]
        phrasing, angles = STATES[state_key]
        angle_key = angles[angle_turn.get(state_key, 0) % len(angles)]
        angle_turn[state_key] = angle_turn.get(state_key, 0) + 1
        prompt = (
            f"macro inspection photo {ANGLE_TXT[angle_key]}, {phrasing}, "
            f"in {rng.choice(MATERIALS)}, {rng.choice(SURFACE)}, {rng.choice(LIGHTING)}, "
            f"{PHOTOREAL}"
        )
        plan.append(
            {
                "file": f"sample_{i:02d}.png",
                "seed": seed + i,
                "qc_state": state_key,
                "angle": angle_key,
                "prompt": prompt,
            }
        )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=25, help="number of samples")
    ap.add_argument("--out", default="images", help="output directory")
    ap.add_argument("--seed", type=int, default=20260620, help="master seed")
    ap.add_argument("--reference", help="path to a real photo to condition every sample on")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sample_*.png"):
        old.unlink()
    (out / "manifest.json").unlink(missing_ok=True)

    reference = Image.open(args.reference) if args.reference else None
    if reference is not None:
        print(f"conditioning on reference: {args.reference} ({reference.size})")

    plan = build_plan(args.n, args.seed)
    gen = NanoBananaGenerator()

    def render(item: dict) -> dict:
        last: Exception | None = None
        for _ in range(2):
            try:
                s = gen.generate(item["prompt"], item["seed"], size=480, reference=reference)
                s.image.save(out / item["file"])
                return {**item, "ok": True}
            except Exception as exc:  # noqa: BLE001 -- batch survives one refusal
                last = exc
        return {**item, "ok": False, "error": str(last)}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(render, it) for it in plan]):
            r = fut.result()
            results.append(r)
            print(f"[{'ok ' if r['ok'] else 'FAIL'}] {r['file']}  {r['qc_state']:18} {r['angle']}")

    results.sort(key=lambda r: r["file"])
    ok = [r for r in results if r["ok"]]
    counts: dict[str, int] = {}
    for r in ok:
        counts[r["qc_state"]] = counts.get(r["qc_state"], 0) + 1
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "master_seed": args.seed,
                "model": gen.model,
                "reference": args.reference,
                "count": len(ok),
                "state_counts": counts,
                "samples": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\n{len(ok)}/{args.n} generated -> {out}/   states: {counts}")
    return 0 if len(ok) == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
