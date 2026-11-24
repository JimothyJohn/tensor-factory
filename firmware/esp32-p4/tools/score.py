"""Score the board's "DET {json}" lines: boxes vs the labels and vs the desktop int8 graph.

python tools/score.py results/pets-*.txt
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0.0) * max(h, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_err(a: list[float], b: list[float]) -> float:
    return math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    out: dict = {"files": len(rows), "decoded": len(ok)}
    if not ok:
        return out
    for who in ("box", "desktop"):
        label = "board" if who == "box" else "desktop"
        ious = [iou(r[who], r["truth"]) for r in ok]
        out[f"{label}_mean_iou"] = round(mean(ious), 4)
        out[f"{label}_iou_ge_0.5"] = round(mean([i >= 0.5 for i in ious]), 4)
        out[f"{label}_center_err_frac_median"] = round(
            median([center_err(r[who], r["truth"]) for r in ok]), 4
        )
    out["board_vs_desktop_mean_iou"] = round(mean([iou(r["box"], r["desktop"]) for r in ok]), 4)
    for key in ("decode_ms", "resize_ms", "net_ms", "post_ms"):
        out[f"{key}_mean"] = round(mean([r[key] for r in ok]), 2)
    total = [r["decode_ms"] + r["resize_ms"] + r["net_ms"] + r["post_ms"] for r in ok]
    out["total_ms_mean"], out["total_ms_max"] = round(mean(total), 2), round(max(total), 2)
    out["jpeg_kb_mean"] = round(mean([r["bytes"] for r in ok]) / 1000, 1)
    out["megapixels_mean"] = round(mean([r["w"] * r["h"] for r in ok]) / 1e6, 3)
    return out


def main() -> None:
    rows = []
    for path in sys.argv[1:]:
        for line in Path(path).read_text(errors="replace").splitlines():
            if line.startswith("DET "):
                rows.append(json.loads(line[4:]))
    if not rows:
        raise SystemExit("no DET lines found")
    for decoder in sorted({r.get("decoder", "sw") for r in rows}):
        print(
            decoder,
            json.dumps(summarize([r for r in rows if r.get("decoder", "sw") == decoder]), indent=2),
        )


if __name__ == "__main__":
    main()
