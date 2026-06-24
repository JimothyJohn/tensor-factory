"""Stage-2 crop head: the SBC-tier state model that rides on top of the v5 detector.

v5 (:class:`~tensor_factory_train.model.TinyDetector`, "stage 1") localizes the insert and
reports presence. Stage 2 takes the *crop* of that predicted box -- where the discriminative
detail lives, at a resolution the 30x30 stage-1 feature map can't carry -- and adds the two
things the roadmap (see ``MODEL_SIZE.md``) calls for:

* **damage classification** (:data:`DAMAGE_CLASSES`): ``ok`` vs the failure modes. A category
  problem, so a softmax head.
* **insertion measurement** (``seating``): a *measurement*, not a category -- a single signed,
  normalized protrusion of the coil relative to the chamfer plane. Positive = proud (the
  insert sits high: *under*-inserted); negative = recessed (*over*-inserted); ~0 = flush.
  :func:`insertion_state` thresholds it into under/correct/over. Modelling depth as a
  continuous scalar generalizes and labels far better than guessing the three classes direct.

Two-stage keeps stage 1 tiny and MCU-portable while spending crop resolution only where it
matters, and lets each head train and quantize independently. This module is deliberately
self-contained -- its own conv stem, not coupled to stage-1 internals -- so stage 2 can scale
its backbone per deployment tier without touching the detector. It is architecture only: the
heads are untrained until the per-failure-mode dataset exists (see ``TODO.md`` v6 item 3).
"""

from __future__ import annotations

from pathlib import Path

import torch  # ty: ignore[unresolved-import]
import torch.nn.functional as F  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

# Damage taxonomy in head-logit order, so an exported model stays self-describing (the names
# travel as ONNX metadata exactly as the stage-1 presence model carries its own contract).
DAMAGE_CLASSES: tuple[str, ...] = ("ok", "deformed", "cross_threaded", "broken_tang", "burr")


def _block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def crop_to_box(image, box, *, out_size: int = 128, context: float = 0.25):  # noqa: ANN001, ANN201
    """Crop a normalized ``xyxy`` box out of a ``(3, H, W)`` image -> ``(3, out_size, out_size)``.

    This is the stage-1 -> stage-2 bridge: stage 1 emits a box, this lifts the pixels under it
    to a fixed-size crop for the head. ``context`` pads the box by that fraction of its
    width/height on each side -- a tight stage-1 box clips the coil's outer turns, and a little
    context buys the head the chamfer rim it needs to judge seating. The expanded box is
    clamped to the image, and a degenerate (zero-area or fully out-of-frame) box still yields a
    valid >=1px crop rather than crashing, so a bad stage-1 prediction degrades instead of
    raising.
    """
    _, h, w = image.shape
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * context, x2 + bw * context
    y1, y2 = y1 - bh * context, y2 + bh * context
    # To pixels, clamped into frame. px2/py2 are kept strictly > px1/py1 so the slice is never
    # empty -- interpolate always has at least one row/column to resample.
    px1 = max(0, min(w - 1, round(x1 * w)))
    px2 = max(px1 + 1, min(w, round(x2 * w)))
    py1 = max(0, min(h - 1, round(y1 * h)))
    py2 = max(py1 + 1, min(h, round(y2 * h)))
    crop = image[:, py1:py2, px1:px2].unsqueeze(0)
    resized = F.interpolate(crop, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def insertion_state(seating: float, *, tol: float = 0.1) -> str:
    """Map a signed normalized protrusion to ``under`` / ``correct`` / ``over``.

    Positive ``seating`` = the insert sits proud of the chamfer (not driven in far enough:
    *under*-inserted); negative = recessed below it (*over*-inserted); within ``+/-tol`` of
    flush = *correct*. The continuous scalar is the model's real output; this is only the
    reporting threshold, kept pure (no torch) so the policy is unit-testable without a model
    and the tolerance is tunable per application.
    """
    if seating > tol:
        return "under"
    if seating < -tol:
        return "over"
    return "correct"


class Stage2Head(nn.Module):
    """Crop-conditioned damage classifier + insertion-depth regressor.

    Four stride-2 blocks take an ``out_size`` crop down to a small map; a concat of global
    *mean* and *max* pooling feeds two linear heads. The max half matters for the same reason
    it does on the stage-1 presence head: a coil and a smooth bore have near-identical channel
    means, so only a spatial *peak* exposes the discriminative texture. ``forward`` returns
    ``(damage_logits, seating)`` -- ``damage_logits`` over :data:`DAMAGE_CLASSES`, ``seating``
    one signed normalized protrusion scalar (feed it to :func:`insertion_state` to report).
    """

    def __init__(self, width: int = 16, num_damage_classes: int = len(DAMAGE_CLASSES)) -> None:
        super().__init__()
        c = width
        self.num_damage_classes = num_damage_classes
        self.features = nn.Sequential(
            _block(3, c),
            _block(c, 2 * c),
            _block(2 * c, 4 * c),
            _block(4 * c, 4 * c),
        )
        self.damage = nn.Linear(8 * c, num_damage_classes)
        self.seating = nn.Linear(8 * c, 1)

    def forward(self, x):  # noqa: ANN001, ANN201
        f = self.features(x)
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)  # B,8c
        return self.damage(pooled), self.seating(pooled)


def export_stage2_onnx(  # noqa: ANN201
    model: nn.Module,
    out_path: str | Path,
    *,
    size: int = 128,
    quantize: bool = True,
):
    """Export the stage-2 head to ONNX on CPU, then dynamic-quantize weights to uint8.

    Outputs are named ``damage`` and ``seating`` so the runtime reads each by name regardless
    of graph order -- the same name-is-the-contract discipline the stage-1 export uses for
    ``box``/``presence``.
    """
    out = Path(out_path)
    model = model.to("cpu").eval()
    dummy = torch.zeros(1, 3, size, size)
    fp32 = out.with_suffix(".fp32.onnx") if quantize else out
    output_names = ["damage", "seating"]
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),  # args as a 1-tuple -- torch's documented form (a bare tensor also works)
            str(fp32),
            input_names=["crop"],
            output_names=output_names,
            opset_version=17,
            dynamic_axes={n: {0: "batch"} for n in ["crop", *output_names]},
            dynamo=False,
        )
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QUInt8)
    return out
