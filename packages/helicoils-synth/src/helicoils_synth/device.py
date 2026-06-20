"""Device resolution for the heavy (torch) paths.

Resolves ``cuda -> mps -> cpu`` so the same code runs on a CUDA box, this Mac
Studio's Metal (MPS) backend, or plain CPU. Importing this module never imports
torch; resolution is attempted lazily and degrades to ``"cpu"`` when torch is absent.
"""

from __future__ import annotations

import os

_VALID = ("cuda", "mps", "cpu")


def resolve_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    ``prefer`` (one of ``cuda``/``mps``/``cpu``) wins when it is actually available;
    otherwise availability is probed in ``cuda -> mps -> cpu`` order. Returns ``"cpu"``
    if torch is not installed.
    """
    try:
        import torch  # ty: ignore[unresolved-import]
    except ModuleNotFoundError:
        return "cpu"

    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def enable_mps_fallback() -> None:
    """Let unimplemented MPS ops fall back to CPU instead of raising.

    Several diffusers/transformers ops still lack Metal kernels; without this an
    otherwise-fine pipeline crashes on the first such op. Set before model load.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
