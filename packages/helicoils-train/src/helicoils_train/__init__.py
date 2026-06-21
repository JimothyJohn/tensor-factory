"""helicoils-train -- train the tiny detector and export an int8 ONNX model.

Only the torch-free data loader is exported eagerly; the model and training loop
import torch and are pulled in lazily by the CLI / :mod:`helicoils_train.train`, so
this package imports cleanly without the ``train`` extra installed.
"""

from .data import load_coco_boxes, load_coco_labeled

__all__ = ["load_coco_boxes", "load_coco_labeled"]
