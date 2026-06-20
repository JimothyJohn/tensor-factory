"""The tiny single-object box regressor with a soft-argmax coordinate head.

Four stride-2 conv blocks take the 480px input to a 30x30 feature map (no global pool,
so spatial information survives). A 1x1 conv makes one heatmap *per box edge*; each is
spatially soft-maxed and a soft-argmax of the relevant marginal yields that edge --
``x1``/``x2`` from the x-marginal, ``y1``/``y2`` from the y-marginal. Every coordinate is
thus localized sub-pixel by the same mechanism (no size regression, which was the
precision bottleneck). Output is normalized ``xyxy``; tiny enough to int8-quantize and
run well above 10 fps.
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


class TinyDetector(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        c = width
        # 480 -> 240 -> 120 -> 60 -> 30 (4 stride-2 blocks, no final pool).
        self.features = nn.Sequential(
            _block(3, c),
            _block(c, 2 * c),
            _block(2 * c, 4 * c),
            _block(4 * c, 4 * c),
        )
        # One heatmap per edge: channels are (x1, y1, x2, y2).
        self.heat = nn.Conv2d(4 * c, 4, kernel_size=1)

    def forward(self, x):  # noqa: ANN001, ANN201
        f = self.features(x)
        b, _, h, w = f.shape

        # Per-channel spatial softmax -> 4 probability maps.
        prob = torch.softmax(self.heat(f).reshape(b, 4, h * w), dim=2).reshape(b, 4, h, w)

        xs = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype)
        ys = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype)
        # Soft-argmax of each channel's marginal along its axis.
        edge_x = (prob.sum(dim=2) * xs).sum(dim=2)  # B,4 -> x position per channel
        edge_y = (prob.sum(dim=3) * ys).sum(dim=2)  # B,4 -> y position per channel

        # x1 from ch0's x-marginal, y1 from ch1's y, x2 from ch2's x, y2 from ch3's y.
        return torch.stack([edge_x[:, 0], edge_y[:, 1], edge_x[:, 2], edge_y[:, 3]], dim=1)
