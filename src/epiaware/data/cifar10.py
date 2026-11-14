"""CIFAR-10 dataset loading and preprocessing."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Standard CIFAR-10 channel statistics.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_train_transform() -> transforms.Compose:
    """Return the augmentation and preprocessing pipeline for training."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

def get_eval_transform() -> transforms.Compose:
    """Return deterministic preprocessing for validation and test images."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

def create_train_val_split(dataset_size: int = 50_000,val_size: int = 5_000,
                           seed: int = 101,) -> tuple[list[int], list[int]]:
    """Create reproducible, non-overlapping training and validation indices.

    Args:
        dataset_size: Total number of examples in the dataset.
        val_size: Number of examples reserved for validation.
        seed: Random seed used to generate the split.

    Returns:
        A tuple containing the training indices and validation indices.

    Raises:
        ValueError: If dataset_size is not positive, val_size is invalid,
            or seed is negative.
    """

    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int):
        raise TypeError("dataset_size must be an integer")

    if isinstance(val_size, bool) or not isinstance(val_size, int):
        raise TypeError("val_size must be an integer")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")

    if val_size <= 0 or val_size >= dataset_size:
        raise ValueError("val_size must be between 1 and dataset_size - 1")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    train_indices = indices[:-val_size]
    validation_indices = indices[-val_size:]

    return train_indices, validation_indices


def get_cifar10_dataloaders(
    data_dir: str | Path = "data/raw",
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 101,
    split_seed: int | None = None,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create reproducible training, validation, and test loaders.

    The official 50,000-image training pool is divided into 45,000
    training images and 5,000 validation images. The official 10,000-image
    test set remains separate.

    Args:
        data_dir: Directory where CIFAR-10 is stored.
        batch_size: Number of images per batch.
        num_workers: Number of worker processes for data loading.
        seed: Seed used to generate the train-validation split.
        download: Whether to download CIFAR-10 if it is unavailable.

    Returns:
        A tuple containing the training, validation, and test DataLoaders.

    Raises:
        ValueError: If batch_size is not positive, num_workers is
            negative, or seed is negative.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    if split_seed is None:
        split_seed = seed

    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise TypeError("split_seed must be an integer")

    if split_seed < 0:
        raise ValueError("split_seed must be non-negative")

    data_dir = Path(data_dir)

    # Training augmentation is applied only to training images.
    train_transform = get_train_transform()
    

    # Validation and test images use deterministic preprocessing.
    eval_transform = get_eval_transform()

    # Separate dataset instances allow training augmentation to differ
    # from validation preprocessing while sharing the same source images.
    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=train_transform,
        download=download,
    )

    validation_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=eval_transform,
        download=False,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=eval_transform,
        download=download,
    )

    # Generate a reproducible train-validation split.
    train_indices, validation_indices = create_train_val_split(
        dataset_size=len(train_dataset),
        val_size=5_000,
        seed=split_seed,
    )

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    # Seed the training loader's shuffle order.
    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader