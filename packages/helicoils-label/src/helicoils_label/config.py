"""Label Studio labeling-config XML for helicoil bounding boxes.

The ``name``/``toName`` ("label"/"image") must match the ``from_name``/``to_name`` in the
prediction results emitted by :mod:`helicoils_label.convert`, or Label Studio rejects the
pre-annotations.
"""

from __future__ import annotations

from collections.abc import Sequence

_PALETTE = ("#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE")


def bbox_config(labels: Sequence[str] = ("helicoil",)) -> str:
    """Build a RectangleLabels config for the given class labels."""
    rows = "\n    ".join(
        f'<Label value="{label}" background="{_PALETTE[i % len(_PALETTE)]}"/>'
        for i, label in enumerate(labels)
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"    {rows}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )
