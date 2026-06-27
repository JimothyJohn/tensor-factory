"""The tiny single-object box regressor with a soft-argmax coordinate head.

Four stride-2 conv blocks take the 480px input to a 30x30 feature map (no global pool,
so spatial information survives). A 1x1 conv makes one heatmap *per box edge*; each is
spatially soft-maxed and a soft-argmax of the relevant marginal yields that edge --
``x1``/``x2`` from the x-marginal, ``y1``/``y2`` from the y-marginal. Every coordinate is
thus localized sub-pixel by the same mechanism (no size regression, which was the
precision bottleneck). Output is normalized ``xyxy``; tiny enough to int8-quantize and
run well above 10 fps.

The soft-argmax carries a *learnable gain* (inverse softmax temperature). A plain softmax
of a diffuse or multimodal heatmap has its marginal expectation pulled toward 0.5 -- the
image centre -- so on ambiguous real-world features the box regresses to the middle. That
centre-bias is invisible on clean synthetic data (sharp peaks already) but is the dominant
localization error on real photos. The gain lets the head sharpen the heatmap before
taking the expectation, pushing the soft-argmax toward the true peak; at ``gain == 1`` it
reproduces the original behaviour exactly, so existing checkpoints are unaffected and the
optimizer is free to anneal it up. It is one scalar parameter and exports cleanly to ONNX
(no change to the box/presence output contract).

With ``presence=True`` a single global-pooled objectness logit rides alongside the box
head -- YOLO-style: sigmoid(logit) is the probability the target is present, so the runtime
returns one box when it clears a threshold and *nothing* when it doesn't. ``forward`` then
returns ``(box, presence_logit)``; with ``presence=False`` it returns just the box, so
existing single-output models and their exports are unchanged. Absence is the low tail of
the objectness score, not a "background" class.

With ``num_classes > 1`` a classification head (the same pooled mean+max features) rides
alongside, emitting one logit *per class* for the single detected object -- the model is
still single-object, it just also says *which* class that object is. ``forward`` appends
``class_logits`` to its output tuple, exported as ``logits``; softmax-argmax at the runtime
gives the class. ``num_classes == 1`` adds no head at all, so single-class/box-only models
and their exports are unchanged. The output order is always ``box``, then ``presence`` (if
any), then ``logits`` (if any); :attr:`output_names` is the authoritative contract and the
runtime reads each output by name.
"""

from __future__ import annotations

import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]


def _block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def soft_argmax_xyxy(heat, gain=1.0):  # noqa: ANN001, ANN201
    """Marginal soft-argmax of a ``(B, 4, H, W)`` edge-heatmap -> ``(B, 4)`` normalized xyxy.

    ``gain`` scales the logits before the spatial softmax (inverse temperature): ``gain > 1``
    sharpens the distribution so the marginal expectation tracks the peak instead of being
    dragged toward the centre; ``gain == 1`` is the plain softmax. Pulled out of ``forward``
    so the centre-bias behaviour is unit-testable in isolation.
    """
    b, _, h, w = heat.shape
    prob = torch.softmax((heat * gain).reshape(b, 4, h * w), dim=2).reshape(b, 4, h, w)
    xs = torch.linspace(0.0, 1.0, w, device=heat.device, dtype=heat.dtype)
    ys = torch.linspace(0.0, 1.0, h, device=heat.device, dtype=heat.dtype)
    edge_x = (prob.sum(dim=2) * xs).sum(dim=2)  # B,4 -> x position per channel
    edge_y = (prob.sum(dim=3) * ys).sum(dim=2)  # B,4 -> y position per channel
    # x1 from ch0's x-marginal, y1 from ch1's y, x2 from ch2's x, y2 from ch3's y.
    return torch.stack([edge_x[:, 0], edge_y[:, 1], edge_x[:, 2], edge_y[:, 3]], dim=1)


class TinyDetector(nn.Module):
    def __init__(
        self,
        width: int = 16,
        presence: bool = False,
        learn_gain: bool = True,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        c = width
        self.presence = presence
        self.num_classes = num_classes
        # 480 -> 240 -> 120 -> 60 -> 30 (4 stride-2 blocks, no final pool).
        self.features = nn.Sequential(
            _block(3, c),
            _block(c, 2 * c),
            _block(2 * c, 4 * c),
            _block(4 * c, 4 * c),
        )
        # One heatmap per edge: channels are (x1, y1, x2, y2).
        self.heat = nn.Conv2d(4 * c, 4, kernel_size=1)
        # Inverse-temperature, stored in log-space so gain = exp(.) stays positive. Init 0
        # -> gain 1 -> identical to a plain softmax at the start of training. Learnable by
        # default; ``learn_gain=False`` freezes it at 1 (a registered buffer) for ablation.
        log_gain = torch.zeros(1)
        if learn_gain:
            self.log_gain = nn.Parameter(log_gain)
        else:
            self.register_buffer("log_gain", log_gain)
        # Optional presence head: one objectness logit from a concat of global *mean* and
        # *max* pooling. Mean alone washes out texture (a coil and a smooth bore have
        # near-identical channel averages); max captures whether a discriminative texture
        # fires anywhere. sigmoid(logit) is P(target present).
        self.presence_head = nn.Linear(8 * c, 1) if presence else None
        # Optional classification head: one logit per class for the single detected object,
        # from the same pooled mean+max features. Only built with >1 class; a single-class
        # detector has nothing to classify, so it adds no head and no output.
        self.class_head = nn.Linear(8 * c, num_classes) if num_classes > 1 else None

    @property
    def output_names(self) -> list[str]:
        """The ONNX output contract, in graph order: box, then presence, then logits.

        The authoritative source for both :meth:`forward`'s tuple order and the export's
        ``output_names`` -- the runtime reads each output by name, so the order only has to
        be self-consistent here.
        """
        names = ["box"]
        if self.presence_head is not None:
            names.append("presence")
        if self.class_head is not None:
            names.append("logits")
        return names

    def forward(self, x):  # noqa: ANN001, ANN201
        f = self.features(x)
        box = soft_argmax_xyxy(self.heat(f), gain=self.log_gain.exp())
        if self.presence_head is None and self.class_head is None:
            return box
        # Pooled features feed both optional heads; compute once when either is present.
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)  # B,8c
        outs = [box]
        if self.presence_head is not None:
            outs.append(self.presence_head(pooled))  # B,1
        if self.class_head is not None:
            outs.append(self.class_head(pooled))  # B,num_classes
        return tuple(outs)
