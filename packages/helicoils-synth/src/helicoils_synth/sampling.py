"""Contact sheets for the prompt-iteration loop.

Generate a grid of seeded samples for one prompt so a human can eyeball whether the
images match the target microscope aesthetic before committing to a full dataset run.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from .generator import DEFAULT_SIZE, Generator


def make_contact_sheet(
    images: Sequence[Image.Image],
    *,
    cols: int = 3,
    pad: int = 6,
    bg: tuple[int, int, int] = (24, 24, 28),
) -> Image.Image:
    """Tile ``images`` into a single grid image (assumes uniform tile size)."""
    if not images:
        raise ValueError("no images to tile")
    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols
    tw, th = images[0].size
    sheet = Image.new(
        "RGB",
        (cols * tw + pad * (cols + 1), rows * th + pad * (rows + 1)),
        bg,
    )
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + pad)
        sheet.paste(img, (x, y))
    return sheet


def sample_grid(
    generator: Generator,
    prompt: str,
    *,
    n: int = 9,
    seed_start: int = 0,
    size: int = DEFAULT_SIZE,
    cols: int = 3,
) -> Image.Image:
    """Generate ``n`` seeded samples for ``prompt`` and return a contact sheet."""
    images = [generator.generate(prompt, seed_start + i, size=size).image for i in range(n)]
    return make_contact_sheet(images, cols=cols)
