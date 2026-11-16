
# Calibration metrics for ensemble predictions

import numpy as np


def expected_calibration_error(confidences, correctness, num_bins=15):
    """Calculate Expected Calibration Error (ECE)."""

    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness)

    if confidences.ndim != 1 or correctness.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(confidences) == 0 or len(correctness) == 0:
        raise ValueError("Inputs cannot be empty")

    if len(confidences) != len(correctness):
        raise ValueError("Inputs must have equal lengths")

    if not np.all(np.isfinite(confidences)):
        raise ValueError("Confidences must contain finite values")

    if np.any(confidences < 0) or np.any(confidences > 1):
        raise ValueError("Confidences must be between 0 and 1")

    if not np.all(np.isin(correctness, [0, 1, False, True])):
        raise ValueError("Correctness values must be 0 or 1")

    if (
        not isinstance(num_bins, (int, np.integer))
        or isinstance(num_bins, (bool, np.bool_))
        or num_bins <= 0
    ):
        raise ValueError("Number of bins must be a positive integer")

    correctness = correctness.astype(float)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0

    for bin_index in range(num_bins):
        lower_bound = bin_edges[bin_index]
        upper_bound = bin_edges[bin_index + 1]

        if bin_index == num_bins - 1:
            bin_mask = (
                (confidences >= lower_bound)
                & (confidences <= upper_bound)
            )
        else:
            bin_mask = (
                (confidences >= lower_bound)
                & (confidences < upper_bound)
            )

        if np.any(bin_mask):
            bin_accuracy = np.mean(correctness[bin_mask])
            bin_confidence = np.mean(confidences[bin_mask])

            bin_weight = np.sum(bin_mask) / len(confidences)

            ece += bin_weight * abs(
                bin_accuracy - bin_confidence
            )

    return ece

