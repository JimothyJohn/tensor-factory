"""helicoils-label -- Label Studio integration for the helicoil annotation loop.

The flow: synth + GroundingDINO produce a COCO dataset with candidate boxes -> push it
into Label Studio as tasks with *predictions* (a human starts from the candidates and
corrects) -> pull the corrected annotations back as COCO for training. The conversion
functions (:mod:`helicoils_label.convert`) are pure and testable; the REST client
(:mod:`helicoils_label.client`) talks to a running Label Studio over its HTTP API.
"""

from .client import LabelStudioClient
from .config import bbox_config
from .convert import (
    coco_to_tasks,
    http_image_url,
    local_storage_url,
    ls_export_to_coco,
)

__all__ = [
    "LabelStudioClient",
    "bbox_config",
    "coco_to_tasks",
    "http_image_url",
    "local_storage_url",
    "ls_export_to_coco",
]
