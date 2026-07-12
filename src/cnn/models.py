"""CNN models: SimpleCNN (from-scratch baseline) and EfficientNet-B0 transfer.

SimpleCNN follows PROJECT_SPEC 5 exactly: 4 conv blocks (3->32->64->128->256),
each Conv(k=3,p=1) + BatchNorm + ReLU + MaxPool(2), then GAP -> Dropout(0.4)
-> Linear(256 -> num_classes). ~1.2M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

from config import NUM_CLASSES


class SimpleCNN(nn.Module):
    """From-scratch baseline CNN (spec section 5)."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        chans = [3, 32, 64, 128, 256]
        blocks: list[nn.Module] = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_effnet(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with a fresh classification head (spec 'FoodNet')."""
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model


def build_model(name: str, num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """Factory: 'simple' -> SimpleCNN, 'effnet' -> EfficientNet-B0 transfer."""
    if name == "simple":
        return SimpleCNN(num_classes)
    if name == "effnet":
        return build_effnet(num_classes, pretrained)
    raise ValueError(f"Unknown model: {name}")


def count_params(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ("simple", "effnet"):
        m = build_model(name, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        print(f"{name}: params={count_params(m):,} out={tuple(m(x).shape)}")
