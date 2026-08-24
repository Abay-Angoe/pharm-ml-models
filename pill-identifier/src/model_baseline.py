"""Phase 1 baseline CNN (section 5).

Four Conv2d -> BatchNorm -> ReLU -> MaxPool blocks, then dense layers to N
logits. Trained from scratch. The point is a working pipeline and an honest
benchmark number for Phase 2 to beat, not peak accuracy.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """224 -> 112 -> 56 -> 28 -> 14, then global pool -> dense -> logits."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        # global pooling keeps the head small; with only 840 training images a
        # flattened 14x14x256 -> dense layer would be ~50M params and overfit hard
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Last conv block's ReLU output feeds Grad-CAM (section 8)."""
        return self.features[-1].block[2]


def build_model(num_classes: int) -> BaselineCNN:
    return BaselineCNN(num_classes)


if __name__ == "__main__":
    m = build_model(30)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"BaselineCNN params: {n_params:,}")
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print(f"forward: {tuple(x.shape)} -> {tuple(out.shape)}")
    feats = m.features(x)
    print(f"feature map before pooling: {tuple(feats.shape)}")
    print(f"grad-cam target layer: {m.gradcam_target_layer}")
