"""Synthetic tests for controlled CIFAR-10 image corruptions."""

import numpy as np
import pytest

from epiaware.data.corruptions import (
    GAUSSIAN_NOISE_STD_NORMALIZED,
    GAUSSIAN_NOISE_STD_UINT8,
    gaussian_noise,
    gaussian_noise_std,
)


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """A deterministic, valid CIFAR-10-format image."""
    return np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)


def test_gaussian_noise_preserves_shape_and_dtype(synthetic_image):
    corrupted = gaussian_noise(synthetic_image, severity=3, seed=101)

    assert corrupted.shape == (32, 32, 3)
    assert corrupted.dtype == np.uint8


def test_gaussian_noise_clips_pixel_values_to_uint8_range():
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    corrupted = gaussian_noise(image, severity=5, seed=101)

    assert corrupted.min() >= 0
    assert corrupted.max() <= 255
    assert np.any(corrupted == 255)


def test_gaussian_noise_is_reproducible_with_seed(synthetic_image):
    first = gaussian_noise(synthetic_image, severity=2, seed=42)
    second = gaussian_noise(synthetic_image, severity=2, seed=42)

    np.testing.assert_array_equal(first, second)


def test_gaussian_noise_does_not_modify_input(synthetic_image):
    original = synthetic_image.copy()

    gaussian_noise(synthetic_image, severity=1, seed=7)

    np.testing.assert_array_equal(synthetic_image, original)


@pytest.mark.parametrize(
    ("severity", "normalized_std", "uint8_std"),
    [
        (1, 0.08, 20.4),
        (2, 0.12, 30.6),
        (3, 0.18, 45.9),
        (4, 0.26, 66.3),
        (5, 0.38, 96.9),
    ],
)
def test_gaussian_noise_severity_mapping(severity, normalized_std, uint8_std):
    assert GAUSSIAN_NOISE_STD_NORMALIZED[severity - 1] == normalized_std
    assert GAUSSIAN_NOISE_STD_UINT8[severity - 1] == pytest.approx(uint8_std)
    assert gaussian_noise_std(severity) == pytest.approx(uint8_std)


@pytest.mark.parametrize("severity", [0, 6, -1])
def test_gaussian_noise_rejects_out_of_range_severity(synthetic_image, severity):
    with pytest.raises(ValueError, match="severity"):
        gaussian_noise(synthetic_image, severity=severity, seed=1)


@pytest.mark.parametrize("severity", [1.0, "1", True])
def test_gaussian_noise_rejects_non_integer_severity(synthetic_image, severity):
    with pytest.raises(TypeError, match="severity"):
        gaussian_noise(synthetic_image, severity=severity, seed=1)


def test_gaussian_noise_rejects_non_cifar10_shape():
    with pytest.raises(ValueError, match="shape"):
        gaussian_noise(np.zeros((31, 32, 3), dtype=np.uint8), severity=1, seed=1)
