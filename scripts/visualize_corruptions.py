"""Validate corruptions on real CIFAR-10 samples and save comparison grids."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from epiaware.data.corruptions import gaussian_blur, gaussian_noise, rotation


CORRUPTIONS = {
    "Gaussian noise": lambda image, severity: gaussian_noise(
        image, severity, seed=1000 + severity
    ),
    "Gaussian blur": gaussian_blur,
    "Rotation (custom)": rotation,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/corruption_grid.png")
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CIFAR10(root=args.data_dir, train=False, download=args.download)
    image, label = dataset[args.sample_index]
    clean = np.asarray(image, dtype=np.uint8)
    _validate_real_cifar10_image(clean)

    figure, axes = plt.subplots(len(CORRUPTIONS), 6, figsize=(12, 6))
    for row, (name, corrupt) in enumerate(CORRUPTIONS.items()):
        axes[row, 0].imshow(clean)
        axes[row, 0].set_title("Clean")
        axes[row, 0].set_ylabel(name)
        for severity in range(1, 6):
            corrupted = corrupt(clean, severity)
            _validate_real_cifar10_image(corrupted)
            axes[row, severity].imshow(corrupted)
            axes[row, severity].set_title(f"Severity {severity}")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(f"CIFAR-10 test sample {args.sample_index}; class label {label}")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Validated CIFAR-10 sample and saved grid to {args.output}")


def _validate_real_cifar10_image(image: np.ndarray) -> None:
    if image.shape != (32, 32, 3) or image.dtype != np.uint8:
        raise ValueError("expected a uint8 CIFAR-10 image with shape (32, 32, 3)")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("image values must be in the range [0, 255]")


if __name__ == "__main__":
    main()
