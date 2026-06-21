"""Validation-state vocabulary for human-in-the-loop label triage.

AI-generated labels are never trainable until a human has validated them. This module is
the single source of truth for the review states and which ones may enter a training set,
so the gate is identical across generation, labeling, and training.

Each COCO annotation (and image) carries two extra keys:

- ``review``: the triage decision -- one of :data:`PENDING`, :data:`APPROVED`,
  :data:`REJECTED`.
- ``source``: provenance -- where the label came from.

Only :data:`APPROVED` annotations are trainable. AI labels (``source == GROUNDINGDINO``)
start :data:`PENDING`; a human flips them to :data:`APPROVED` by correcting them in Label
Studio and pulling the result back. Synthetic ground truth (the mock generator,
``source == SYNTHETIC_GT``) is exact by construction -- not a guess -- so it is
:data:`APPROVED` on creation. A **missing** ``review`` key counts as :data:`PENDING`: an
unmarked label is, by default, untrusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Triage states.
PENDING = "pending"  # awaiting human review -- NOT trainable
APPROVED = "approved"  # human-validated (or exact synthetic GT) -- trainable
REJECTED = "rejected"  # reviewed and discarded -- excluded

STATES: tuple[str, ...] = (PENDING, APPROVED, REJECTED)

# Provenance.
GROUNDINGDINO = "groundingdino"  # open-vocabulary auto-labeler -- a guess, must be reviewed
HUMAN = "human"  # corrected/approved by a person in Label Studio
SYNTHETIC_GT = "synthetic_gt"  # mock generator's known box -- exact, not a guess

_TRAINABLE = frozenset({APPROVED})


def normalize(review: str | None) -> str:
    """Coerce a (possibly missing/unknown) review value to a known state.

    Anything not explicitly recognized -- including ``None`` -- is treated as
    :data:`PENDING`, so untrusted-by-default is the safe failure mode.
    """
    return review if review in STATES else PENDING


def is_trainable(review: str | None) -> bool:
    """True iff an annotation with this ``review`` value may enter a training set."""
    return review in _TRAINABLE


def review_summary(coco: Mapping[str, Any]) -> dict[str, Any]:
    """Count images and annotations by review state -- the triage progress report.

    Returns ``{"images": {...}, "annotations": {...}}`` where each inner dict has a
    ``total`` plus a count per state, and the annotations dict also reports ``trainable``
    (the count that would actually enter training).
    """
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))

    img_counts = {s: 0 for s in STATES}
    for im in images:
        img_counts[normalize(im.get("review"))] += 1

    ann_counts = {s: 0 for s in STATES}
    for ann in annotations:
        ann_counts[normalize(ann.get("review"))] += 1

    return {
        "images": {"total": len(images), **img_counts},
        "annotations": {
            "total": len(annotations),
            "trainable": ann_counts[APPROVED],
            **ann_counts,
        },
    }
