"""Errors shared by the zero-dependency decoders.

Kept in its own module so :mod:`.imageio` and :mod:`.jpeg` can both raise it without
importing each other.
"""

from __future__ import annotations


class ImageError(ValueError):
    """Malformed or unsupported image data."""
