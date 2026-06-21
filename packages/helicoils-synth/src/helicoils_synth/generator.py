"""Image generators -- the light mock and the Nano Banana API path behind one interface."""

from __future__ import annotations

import io
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

    Renders a noisy aluminum field with a concentric-ring "coil" (55-85% of the frame)
    at a widely-varying seeded position, and returns the exact box it drew. This makes
    the whole downstream pipeline (export, Label Studio, even toy training) runnable and
    testable with no weights, and gives a clean A/B baseline against real generation.
    """

    def generate(self, prompt: str, seed: int, *, size: int = DEFAULT_SIZE) -> GeneratedSample:
        rng = random.Random(seed)

        base = Image.new("RGB", (size, size), (176, 178, 182))
        noise = Image.frombytes("L", (size, size), rng.randbytes(size * size)).convert("RGB")
        img = Image.blend(base, noise, 0.12)

        draw = ImageDraw.Draw(img)
        # Wide positional + scale variance so the model must actually localize, not
        # memorize a near-centered box. The coil may clip the frame edges (realistic
        # for an off-center microscope); the ground-truth box is its clamped extent.
        r = rng.uniform(0.28, 0.42) * size
        cx = rng.uniform(0.30, 0.70) * size
        cy = rng.uniform(0.30, 0.70) * size

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


class NanoBananaGenerator:
    """Nano Banana (Gemini ``gemini-2.5-flash-image``) text-to-image via the Gemini API.

    No GPU and no local weights -- generation is a hosted API call, so the whole heavy
    diffusion stack is gone. Needs ``google-genai`` (the ``gemini`` extra) and a
    ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) in the environment. The model emits 1:1
    images at a fixed native resolution (1024 px today); we downscale to ``size``, the
    480 px throughput target. ``seed`` is forwarded best-effort for provenance.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-image",
        api_key: str | None = None,
    ) -> None:
        try:
            # `google` is a namespace package (protobuf et al. occupy it), so an absent
            # `genai` raises ImportError, not ModuleNotFoundError -- catch both.
            from google import genai  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError(
                "NanoBananaGenerator needs the 'gemini' extra: uv sync --extra gemini"
            ) from exc

        self._genai = genai
        # Client() reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model

    def generate(
        self,
        prompt: str,
        seed: int,
        *,
        size: int = DEFAULT_SIZE,
        reference: Image.Image | None = None,
    ) -> GeneratedSample:
        from google.genai import types  # ty: ignore[unresolved-import]

        # A reference photo conditions the model on the real part's appearance (the fix for
        # subjects it renders wrong from text alone). Image first, then the instruction.
        contents = [types.Part.from_text(text=prompt)]
        if reference is not None:
            buf = io.BytesIO()
            reference.convert("RGB").save(buf, format="PNG")
            contents.insert(0, types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
                seed=seed,
            ),
        )
        image = _first_image(resp)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return GeneratedSample(image=image, prompt=prompt, seed=seed, box=None)


def _first_image(response: object) -> Image.Image:
    """Pull the first inline image out of a Gemini ``generate_content`` response.

    Raises with the model's text part (often a safety refusal) when no image came back,
    so a blocked prompt fails loudly instead of silently writing nothing.
    """
    candidates = getattr(response, "candidates", None) or []
    text_parts: list[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            blob = getattr(part, "inline_data", None)
            if blob is not None and blob.data:
                return Image.open(io.BytesIO(blob.data))
            if getattr(part, "text", None):
                text_parts.append(part.text)
    detail = f": {' '.join(text_parts)}" if text_parts else ""
    raise RuntimeError(f"Gemini returned no image{detail}")
