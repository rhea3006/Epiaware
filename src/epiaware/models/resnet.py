
"""ResNet-18 architecture adapted for CIFAR-10."""

import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18(num_classes: int = 10) -> nn.Module:
    """Create a ResNet-18 adapted for 32x32 RGB images.

    Args:
        num_classes: Number of output classes.

    Returns:
        A ResNet-18 model with a CIFAR-10-compatible input stem
        and a classifier producing one logit per class.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    model = resnet18(weights=None)

    # Preserve spatial detail in small 32x32 CIFAR-10 images.
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # Output raw class logits; CrossEntropyLoss handles softmax internally.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
