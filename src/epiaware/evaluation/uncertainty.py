
"""Uncertainty metrics for deep ensemble predictions."""

import numpy as np


def predictive_entropy(probabilities):
    """Calculate predictive entropy for each probability distribution."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must have shape (samples, classes)")

    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("Probabilities cannot have empty dimensions")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must contain finite values")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1")

    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Each probability distribution must sum to 1")

    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)

    entropy = -np.sum(
        probabilities * np.log(safe_probabilities),
        axis=1
    )

    return entropy


def variation_ratio(predictions):
    """Calculate ensemble disagreement using variation ratio."""
    predictions = np.asarray(predictions)

    if predictions.ndim != 2:
        raise ValueError("Predictions must have shape (models, samples)")

    if predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("Predictions cannot have empty dimensions")

    if not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Predicted class labels must be integers")

    if np.any(predictions < 0):
        raise ValueError("Predicted class labels cannot be negative")

    ensemble_size = predictions.shape[0]
    variation_ratios = []

    for sample_index in range(predictions.shape[1]):
        sample_predictions = predictions[:, sample_index]

        _, counts = np.unique(
            sample_predictions,
            return_counts=True
        )

        majority_votes = np.max(counts)
        disagreement = ensemble_size - majority_votes

        variation_ratios.append(disagreement / ensemble_size)

    return np.array(variation_ratios)


def jensen_shannon_divergence(probabilities):
    """Calculate mean KL divergence from members to ensemble mean."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 3:
        raise ValueError(
            "Probabilities must have shape (models, samples, classes)"
        )

    if any(dimension == 0 for dimension in probabilities.shape):
        raise ValueError("Probabilities cannot have empty dimensions")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must contain finite values")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1")

    if not np.allclose(probabilities.sum(axis=2), 1.0):
        raise ValueError("Each model probability distribution must sum to 1")

    ensemble_mean = np.mean(probabilities, axis=0)

    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    safe_mean = np.clip(ensemble_mean, 1e-12, 1.0)

    kl_divergence = np.sum(
        probabilities * np.log(safe_probabilities / safe_mean),
        axis=2
    )

    return np.mean(kl_divergence, axis=0)

