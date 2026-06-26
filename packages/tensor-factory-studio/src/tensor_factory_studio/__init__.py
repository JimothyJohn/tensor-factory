"""Local backend for Tensor Factory Studio.

Serves the browser labeling UI and runs continuous training on the on-disk COCO
dataset via :mod:`tensor_factory_train`, streaming live metrics back to the UI and
serving the best checkpoint for auto-labeling. torch is imported lazily inside the
trainer thread so the package (and its unit tests) import without the ``serve`` extra.
"""

from .dataset import Dataset

__all__ = ["Dataset"]
