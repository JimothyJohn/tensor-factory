"""helicoils-synth -- synthesize and auto-label a helicoil detection dataset.

The light path (no torch) is fully functional: :class:`MockGenerator` renders
deterministic, seeded coil images with known ground-truth boxes, and the export +
pipeline code turns those into COCO datasets and Label Studio pre-annotations.

The heavy path is behind the ``gpu`` extra: :class:`FluxGenerator` (FLUX.1-schnell
via diffusers) and :class:`GroundingDinoAutoLabeler` (GroundingDINO via transformers)
lazy-import torch, so importing this package never requires it.
"""

from .autolabel import AutoLabeler, Detection, GroundingDinoAutoLabeler
from .device import enable_mps_fallback, resolve_device
from .generator import (
    DEFAULT_SIZE,
    FluxGenerator,
    GeneratedSample,
    Generator,
    MockGenerator,
)
from .pipeline import synth_dataset

__all__ = [
    "DEFAULT_SIZE",
    "AutoLabeler",
    "Detection",
    "FluxGenerator",
    "GeneratedSample",
    "Generator",
    "GroundingDinoAutoLabeler",
    "MockGenerator",
    "enable_mps_fallback",
    "resolve_device",
    "synth_dataset",
]
