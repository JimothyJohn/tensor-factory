"""The tiny single-object box regressor.

Five stride-2 conv blocks shrink the 480px input to a small feature map, global pool,
one linear layer to four coordinates, sigmoid to ``[0, 1]``. At width=16 this is a few
hundred KB of weights -- small enough to int8-quantize and run well above 10 fps on CPU,
and plausibly on an SBC. Output is normalized ``xyxy``, matching the inference contract.
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
        self.features = nn.Sequential(
            _block(3, c),
            _block(c, 2 * c),
            _block(2 * c, 4 * c),
            _block(4 * c, 4 * c),
            _block(4 * c, 4 * c),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * c, 4))

    def forward(self, x):  # noqa: ANN001, ANN201
        return torch.sigmoid(self.head(self.features(x)))
