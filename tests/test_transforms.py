import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from epiaware.data.cifar10 import (CIFAR10_MEAN,CIFAR10_STD,get_eval_transform,
                                   get_train_transform,)


def test_normalization_constants_have_three_channels():
    assert len(CIFAR10_MEAN) == 3
    assert len(CIFAR10_STD) == 3
    assert all(std > 0 for std in CIFAR10_STD)


def test_evaluation_transform_produces_normalized_tensor():
    transform = get_eval_transform()

    # Create a synthetic 32x32 RGB image with pixel value 128.
    image_array = np.full((32, 32, 3), 128, dtype=np.uint8)
    pil_image = Image.fromarray(image_array)

    result = transform(pil_image)

    assert result.shape == (3, 32, 32)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_training_transform_includes_augmentation():
    transform = get_train_transform()

    assert isinstance(transform.transforms[0], transforms.RandomCrop)
    assert isinstance(transform.transforms[1], transforms.RandomHorizontalFlip)