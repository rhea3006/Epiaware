
"""Tests for the CIFAR-10 ResNet-18 architecture."""

import pytest
import torch
from torch import nn

from epiaware.models.resnet import create_resnet18


def test_resnet18_output_shape() -> None:
    """The model should produce ten logits for each CIFAR-10 image."""
    model = create_resnet18(num_classes=10)
    model.eval()

    images = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, 10)


def test_resnet18_classifier_size() -> None:
    """The final layer should match the requested class count."""
    model = create_resnet18(num_classes=10)

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 10


@pytest.mark.parametrize("num_classes", [0, -1])
def test_resnet18_rejects_invalid_class_count(num_classes: int) -> None:
    """Non-positive class counts should raise ValueError."""
    with pytest.raises(ValueError, match="num_classes must be positive"):
        create_resnet18(num_classes=num_classes)


def test_resnet18_rejects_non_integer_class_count() -> None:
    """Class counts must be integers rather than booleans or other types."""
    with pytest.raises(TypeError, match="num_classes must be an integer"):
        create_resnet18(num_classes=True)
