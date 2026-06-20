"""Auto-labeling: turn a generated image + a feature list into detections.

The feature list is the "features I want to extract" half of the brief, kept separate
from the generation prompt. GroundingDINO is open-vocabulary, so the same model labels
"helicoil", "thread", "hole", etc. without retraining. Detections become COCO
annotations and Label Studio pre-annotations for a fast human review pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from helicoils.geometry import BBox


@dataclass
class Detection:
    label: str
    box: BBox
    score: float


class AutoLabeler(Protocol):
    def label(self, image: Image.Image, features: Sequence[str]) -> list[Detection]: ...


class GroundingDinoAutoLabeler:
    """Open-vocabulary auto-labeler via transformers' GroundingDINO (Apache-2.0).

    Uses the transformers implementation, not the IDEA-Research repo, specifically to
    avoid the custom CUDA op that will not build on Metal -- this runs on MPS/CPU. Behind
    the ``gpu`` extra and not yet runtime-verified; checked in the heavy-run step.
    """

    def __init__(
        self,
        model: str = "IDEA-Research/grounding-dino-tiny",
        device: str | None = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            import torch  # ty: ignore[unresolved-import]
            from transformers import (  # ty: ignore[unresolved-import]
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GroundingDinoAutoLabeler needs the 'gpu' extra: uv sync --extra gpu"
            ) from exc

        from .device import enable_mps_fallback, resolve_device

        enable_mps_fallback()
        self._torch = torch
        self.device = device or resolve_device()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(self.device)

    def label(self, image: Image.Image, features: Sequence[str]) -> list[Detection]:
        # GroundingDINO wants lowercase, period-delimited phrases.
        text = ". ".join(f.strip().lower() for f in features) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        width, height = image.size
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        # transformers renamed the label key to "text_labels"; tolerate both.
        labels = results.get("text_labels", results.get("labels", []))
        detections: list[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], labels, strict=False):
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    label=str(label),
                    box=BBox.from_pixels(x1, y1, x2, y2, width=width, height=height),
                    score=float(score),
                )
            )
        return detections
