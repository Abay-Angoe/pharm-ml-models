"""Phase 2 transfer learning (section 6).

ResNet18 or EfficientNet-B0 pretrained on ImageNet, with a fresh head sized to our
class count. Training runs in two stages driven by train.py:

  1. freeze the whole backbone, train only the head at lr=1e-3
  2. unfreeze the last 1-2 blocks and fine-tune at lr=1e-4

**Reported results use ResNet18.** Section 6 asks for one of the two backbones; the
EfficientNet-B0 path is implemented and smoke-tested (run this file directly) but was
never trained, so no EfficientNet numbers exist anywhere in the outputs.

Pill imprints and colours are fine-grained detail, which is where pretrained
low/mid-level features tend to pay off relative to training the Phase 1 CNN from
scratch on 819 images - measured here as +44.6pp top-1.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)


class TransferModel(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            in_feats = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
            # coarse-to-fine stages, last first - used by set_trainable
            self._stages = [net.layer4, net.layer3, net.layer2, net.layer1]
            self._last_conv = net.layer4
        elif backbone == "efficientnet_b0":
            net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_feats = net.classifier[1].in_features
            net.classifier = nn.Identity()
            self.backbone = net
            feats = net.features
            self._stages = [feats[-1], feats[-2], feats[-3], feats[-4]]
            self._last_conv = feats[-1]
        else:
            raise ValueError(f"unknown backbone {backbone!r}")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Deepest conv stage - the standard Grad-CAM tap point."""
        return self._last_conv


def set_trainable(model: TransferModel, unfreeze_last: int = 0) -> None:
    """Freeze the backbone, then re-enable the last `unfreeze_last` stages.

    The head is always trainable. BatchNorm layers inside frozen stages are put
    in eval mode so their running statistics are not updated by our small
    batches - otherwise the frozen features drift even with requires_grad=False.
    """
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    unfrozen = set()
    for stage in model._stages[:unfreeze_last]:
        for p in stage.parameters():
            p.requires_grad = True
        unfrozen.add(id(stage))

    # freeze BN statistics everywhere except in unfrozen stages
    def freeze_bn(module: nn.Module, inside_unfrozen: bool) -> None:
        for child in module.children():
            child_unfrozen = inside_unfrozen or id(child) in unfrozen
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                child.train(child_unfrozen)
                child.track_running_stats = child_unfrozen
            freeze_bn(child, child_unfrozen)

    freeze_bn(model.backbone, False)


def build_model(num_classes: int, backbone: str = "resnet18") -> TransferModel:
    return TransferModel(num_classes, backbone=backbone)


if __name__ == "__main__":
    from classes import CLASSES

    for name in ("resnet18", "efficientnet_b0"):
        m = build_model(len(CLASSES), backbone=name)
        total = sum(p.numel() for p in m.parameters())
        for k in (0, 1, 2):
            set_trainable(m, unfreeze_last=k)
            train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"{name:16s} unfreeze_last={k}: "
                  f"{train_p:>11,} / {total:,} trainable "
                  f"({100 * train_p / total:5.1f}%)")
        out = m(torch.randn(2, 3, 224, 224))
        print(f"{name:16s} forward -> {tuple(out.shape)}, "
              f"gradcam layer {type(m.gradcam_target_layer).__name__}\n")
