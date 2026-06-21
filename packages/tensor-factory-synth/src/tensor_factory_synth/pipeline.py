"""End-to-end: generate seeded images, label them, write a COCO dataset.

If a labeler is given, it produces the detections (the real path). Otherwise, when the
generator already knows the ground-truth box (the mock), that box is used directly --
so a full, perfectly-labeled toy dataset can be produced with no torch at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tensor_factory import review

from .autolabel import AutoLabeler, Detection
from .export import (
    Record,
    build_coco,
    build_label_studio_predictions,
    write_json,
)
from .generator import DEFAULT_SIZE, Generator


def synth_dataset(
    generator: Generator,
    prompt: str,
    features: Sequence[str],
    *,
    n: int,
    out_dir: str | Path,
    seed_start: int = 0,
    size: int = DEFAULT_SIZE,
    labeler: AutoLabeler | None = None,
) -> list[Record]:
    """Generate ``n`` images, label them, and write images + COCO + Label Studio JSON.

    Returns the records (one per image). Files written under ``out_dir``:
    ``images/``, ``annotations.coco.json``, ``label_studio.json``.
    """
    if not features:
        raise ValueError("at least one feature is required")

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    records: list[Record] = []
    for i in range(n):
        seed = seed_start + i
        sample = generator.generate(prompt, seed, size=size)
        file_name = f"img_{seed:08d}.png"
        sample.image.save(out / "images" / file_name)

        if labeler is not None:
            # AI guesses: PENDING + groundingdino provenance (the Detection defaults) --
            # they must be human-validated before they can train.
            dets = labeler.label(sample.image, features)
        elif sample.box is not None:
            # The mock generator's box is exact ground truth, not a guess, so it is
            # APPROVED on creation -- trainable without a review pass.
            dets = [
                Detection(
                    label=features[0],
                    box=sample.box,
                    score=1.0,
                    review=review.APPROVED,
                    source=review.SYNTHETIC_GT,
                )
            ]
        else:
            dets = []
        records.append((f"images/{file_name}", size, size, dets))

    write_json(out / "annotations.coco.json", build_coco(records, list(features)))
    write_json(out / "label_studio.json", build_label_studio_predictions(records))
    return records
