"""Controlled image corruptions for CIFAR-10 evaluation images.

The first implementation deliberately operates on one RGB CIFAR-10 image at a
time.  Dataset code is responsible for keeping each returned image paired with
its original label.
"""

from __future__ import annotations

import numbers

import numpy as np
from scipy import ndimage


# Values established for CIFAR-10-C Gaussian noise.  The uint8 values are the
# normalized values multiplied by 255 and are used because this module accepts
# uint8 images in the inclusive range [0, 255].
GAUSSIAN_NOISE_STD_NORMALIZED: tuple[float, ...] = (0.04, 0.06, 0.08, 0.09, 0.10)
GAUSSIAN_NOISE_STD_UINT8: tuple[float, ...] = tuple(
    value * 255 for value in GAUSSIAN_NOISE_STD_NORMALIZED
)
GAUSSIAN_BLUR_SIGMAS: tuple[float, ...] = (0.4, 0.6, 0.7, 0.8, 1.0)
ROTATION_ANGLES: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0)

_CIFAR10_IMAGE_SHAPE = (32, 32, 3)


def gaussian_noise_std(severity: int) -> float:
    """Return the uint8-scale standard deviation for a severity level.

    Args:
        severity: Integer severity from 1 (least severe) to 5 (most severe).

    Raises:
        TypeError: If ``severity`` is not an integer.
        ValueError: If ``severity`` is outside the inclusive range 1--5.
    """
    return _parameter_for_severity(GAUSSIAN_NOISE_STD_UINT8, severity)


def gaussian_noise(
    image: np.ndarray, severity: int, *, seed: int | None = None
) -> np.ndarray:
    """Add seeded Gaussian noise to one uint8 CIFAR-10 RGB image.

    Args:
        image: An RGB image with shape ``(32, 32, 3)``, dtype ``uint8``, and
            values in the inclusive range [0, 255].
        severity: Integer severity from 1 (least severe) to 5 (most severe).
        seed: Optional NumPy random-generator seed.  Reusing it produces the
            same corruption for the same image and severity.

    Returns:
        A newly allocated uint8 image with the same shape as ``image``.

    Raises:
        TypeError: If the image type/dtype or seed type is invalid.
        ValueError: If the image shape or pixel range is invalid, or severity
            is outside the supported range.
    """
    _validate_image(image)
    _validate_seed(seed)

    noise = np.random.default_rng(seed).normal(
        loc=0.0, scale=gaussian_noise_std(severity), size=image.shape
    )
    shifted = image.astype(np.float64) + noise
    return np.clip(np.rint(shifted), 0, 255).astype(np.uint8)


def gaussian_blur_sigma(severity: int) -> float:
    """Return the approved CIFAR-10-C Gaussian-filter sigma for severity."""
    return _parameter_for_severity(GAUSSIAN_BLUR_SIGMAS, severity)


def gaussian_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """Blur one uint8 CIFAR-10 image using the approved Gaussian sigma.

    The Gaussian filter is applied independently to each colour channel.  It
    uses nearest-edge boundary handling, matching the CIFAR-10-C generator's
    Gaussian-filter defaults.  Unlike Gaussian noise, blur is deterministic
    and therefore has no seed argument.
    """
    _validate_image(image)
    sigma = gaussian_blur_sigma(severity)
    blurred = ndimage.gaussian_filter(
        image.astype(np.float64), sigma=(sigma, sigma, 0.0), mode="nearest"
    )
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def rotation_angle(severity: int) -> float:
    """Return the approved counter-clockwise rotation angle in degrees."""
    return _parameter_for_severity(ROTATION_ANGLES, severity)


def rotation(image: np.ndarray, severity: int) -> np.ndarray:
    """Rotate one uint8 CIFAR-10 image around its centre.

    Positive angles are counter-clockwise.  Bilinear interpolation is used and
    pixels uncovered at the corners are filled with black (value 0).
    """
    _validate_image(image)
    rotated = ndimage.rotate(
        image.astype(np.float64),
        angle=rotation_angle(severity),
        axes=(1, 0),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.clip(np.rint(rotated), 0, 255).astype(np.uint8)


def _parameter_for_severity(parameters: tuple[float, ...], severity: int) -> float:
    """Select a five-level corruption parameter after validating severity."""
    if isinstance(severity, bool) or not isinstance(severity, numbers.Integral):
        raise TypeError("severity must be an integer from 1 to 5")
    if not 1 <= severity <= len(parameters):
        raise ValueError("severity must be between 1 and 5")
    return parameters[int(severity) - 1]


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.shape != _CIFAR10_IMAGE_SHAPE:
        raise ValueError("image must have shape (32, 32, 3)")
    if image.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, numbers.Integral)
    ):
        raise TypeError("seed must be an integer or None")
