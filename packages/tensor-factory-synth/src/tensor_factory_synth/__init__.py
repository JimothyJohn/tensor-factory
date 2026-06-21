"""tensor-factory-synth -- synthesize and auto-label a helicoil detection dataset.

The light path (no torch) is fully functional: :class:`MockGenerator` renders
deterministic, seeded coil images with known ground-truth boxes, and the export +
pipeline code turns those into COCO datasets and Label Studio pre-annotations.

Real generation is behind the ``gemini`` extra: :class:`NanoBananaGenerator` calls the
Gemini API (no GPU). Auto-labeling is behind the ``gpu`` extra:
:class:`GroundingDinoAutoLabeler` (GroundingDINO via transformers) lazy-imports torch,
so importing this package never requires it.
"""

from .autolabel import AutoLabeler, Detection, GroundingDinoAutoLabeler
from .device import enable_mps_fallback, resolve_device
from .generator import (
    DEFAULT_SIZE,
    GeneratedSample,
    Generator,
    MockGenerator,
    NanoBananaGenerator,
)
from .pipeline import synth_dataset

__all__ = [
    "DEFAULT_SIZE",
    "AutoLabeler",
    "Detection",
    "GeneratedSample",
    "Generator",
    "GroundingDinoAutoLabeler",
    "MockGenerator",
    "NanoBananaGenerator",
    "enable_mps_fallback",
    "resolve_device",
    "synth_dataset",
]
