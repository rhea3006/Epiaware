
# Selective prediction and risk-coverage metrics

import numpy as np


def risk_coverage_curve(error_labels, uncertainty_scores):
    """Calculate risk at increasing coverage levels."""

    error_labels = np.asarray(error_labels)
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=float)

    if error_labels.ndim != 1 or uncertainty_scores.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(error_labels) == 0 or len(uncertainty_scores) == 0:
        raise ValueError("Inputs cannot be empty")

    if len(error_labels) != len(uncertainty_scores):
        raise ValueError("Inputs must have equal lengths")

    if not np.all(np.isin(error_labels, [0, 1])):
        raise ValueError("Error labels must contain only 0 and 1")

    if not np.all(np.isfinite(uncertainty_scores)):
        raise ValueError("Uncertainty scores must be finite")

    # Sort samples from lowest uncertainty to highest uncertainty.
    sorted_indices = np.argsort(
        uncertainty_scores,
        kind="stable"
    )

    sorted_errors = error_labels[sorted_indices]

    # Calculate cumulative errors among retained predictions.
    cumulative_errors = np.cumsum(sorted_errors)

    number_of_samples = len(error_labels)

    coverage = np.arange(
        1,
        number_of_samples + 1
    ) / number_of_samples

    risk = cumulative_errors / np.arange(
        1,
        number_of_samples + 1
    )

    return coverage, risk


def calculate_aurc(risk):
    """Calculate AURC using the mean of discrete risk values."""

    risk = np.asarray(risk, dtype=float)

    if risk.ndim != 1:
        raise ValueError("Risk must be one-dimensional")

    if len(risk) == 0:
        raise ValueError("Risk cannot be empty")

    if not np.all(np.isfinite(risk)):
        raise ValueError("Risk values must be finite")

    if np.any(risk < 0) or np.any(risk > 1):
        raise ValueError("Risk values must be between 0 and 1")

    # Discrete right-endpoint approximation.
    return np.mean(risk)

