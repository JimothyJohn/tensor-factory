"""Image generators -- the light mock and the heavy FLUX path behind one interface."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageDraw, ImageFilter

from helicoils.geometry import BBox

# 480 is a multiple of 16 (FLUX's tiling requirement) and the target throughput res.
DEFAULT_SIZE = 480


@dataclass
class GeneratedSample:
    """One generated image plus its provenance.

    ``box`` is the ground-truth helicoil box when the generator knows it (the mock
    does); for real generators it is ``None`` and an :class:`AutoLabeler` fills it in.
    """

    image: Image.Image
    prompt: str
    seed: int
    box: BBox | None = None


class Generator(Protocol):
    def generate(self, prompt: str, seed: int, *, size: int = DEFAULT_SIZE) -> GeneratedSample: ...


class MockGenerator:
    """Deterministic, torch-free stand-in for the diffusion model.

    Renders a noisy aluminum field with a concentric-ring "coil" filling ~80% of the
    frame at a seeded position, and returns the exact bounding box it drew. This makes
    the whole downstream pipeline (export, Label Studio, even toy training) runnable and
    testable with no weights, and gives a clean A/B baseline against real generation.
    """

    def generate(self, prompt: str, seed: int, *, size: int = DEFAULT_SIZE) -> GeneratedSample:
        rng = random.Random(seed)

        base = Image.new("RGB", (size, size), (176, 178, 182))
        noise = Image.frombytes("L", (size, size), rng.randbytes(size * size)).convert("RGB")
        img = Image.blend(base, noise, 0.12)

        draw = ImageDraw.Draw(img)
        margin = size * 0.1
        max_r = size / 2 - margin
        r = rng.uniform(max_r * 0.85, max_r)
        cx = rng.uniform(size / 2 - margin / 2, size / 2 + margin / 2)
        cy = rng.uniform(size / 2 - margin / 2, size / 2 + margin / 2)

        rings = 9
        line_w = max(2, int(size * 0.012))
        for i in range(rings):
            rr = r * (1.0 - i / (rings + 2))
            shade = 90 + i * 8
            draw.ellipse(
                [cx - rr, cy - rr, cx + rr, cy + rr],
                outline=(shade, shade, min(255, shade + 4)),
                width=line_w,
            )
        img = img.filter(ImageFilter.GaussianBlur(0.6))

        box = BBox.from_pixels(cx - r, cy - r, cx + r, cy + r, width=size, height=size)
        return GeneratedSample(image=img, prompt=prompt, seed=seed, box=box)


class FluxGenerator:
    """FLUX.1-schnell text-to-image (Apache-2.0 weights), behind the ``gpu`` extra.

    Lazy-imports torch/diffusers in ``__init__`` so the module stays import-light. Not
    yet runtime-verified -- exercised in the dedicated heavy-run step.
    """

    def __init__(
        self,
        model: str = "black-forest-labs/FLUX.1-schnell",
        device: str | None = None,
        steps: int = 4,
    ) -> None:
        try:
            import torch  # ty: ignore[unresolved-import]
            from diffusers import FluxPipeline  # ty: ignore[unresolved-import]
        except ModuleNotFoundError as exc:
            raise RuntimeError("FluxGenerator needs the 'gpu' extra: uv sync --extra gpu") from exc

        from .device import enable_mps_fallback, resolve_device

        enable_mps_fallback()
        self._torch = torch
        self.device = device or resolve_device()
        self.steps = steps
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.pipe = FluxPipeline.from_pretrained(model, torch_dtype=dtype).to(self.device)

    def generate(self, prompt: str, seed: int, *, size: int = DEFAULT_SIZE) -> GeneratedSample:
        generator = self._torch.Generator(device="cpu").manual_seed(seed)
        # schnell is distilled: 0.0 guidance, ~4 steps.
        out = self.pipe(
            prompt,
            num_inference_steps=self.steps,
            guidance_scale=0.0,
            height=size,
            width=size,
            generator=generator,
        )
        return GeneratedSample(image=out.images[0], prompt=prompt, seed=seed, box=None)
