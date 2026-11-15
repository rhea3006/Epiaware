
# Error detection metrics for ensemble predictions

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def get_ensemble_probabilities(probabilities):
    """Average class probabilities across ensemble members."""

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 3:
        raise ValueError(
            "Probabilities must have shape (models, samples, classes)"
        )

    if any(dimension == 0 for dimension in probabilities.shape):
        raise ValueError("Probabilities cannot have empty dimensions")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1")

    if not np.allclose(probabilities.sum(axis=2), 1.0):
        raise ValueError("Each probability distribution must sum to 1")

    return np.mean(probabilities, axis=0)


def get_predictions(probabilities):
    """Get ensemble class predictions and confidence."""

    ensemble_probabilities = get_ensemble_probabilities(probabilities)

    predictions = np.argmax(ensemble_probabilities, axis=1)
    confidence = np.max(ensemble_probabilities, axis=1)

    return predictions, confidence


def get_error_labels(predictions, ground_truth):
    """Return 0 for correct predictions and 1 for incorrect predictions."""

    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    if predictions.ndim != 1 or ground_truth.ndim != 1:
        raise ValueError("Predictions and labels must be one-dimensional")

    if len(predictions) == 0 or len(ground_truth) == 0:
        raise ValueError("Predictions and labels cannot be empty")

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and labels must have equal lengths")

    if not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Predictions must contain integer class labels")

    if not np.issubdtype(ground_truth.dtype, np.integer):
        raise ValueError("Ground truth must contain integer class labels")

    correct = predictions == ground_truth
    error_labels = (~correct).astype(int)

    return correct, error_labels


def calculate_error_detection_metrics(error_labels, uncertainty_scores):
    """Calculate AUROC and AUPRC for identifying incorrect predictions."""

    error_labels = np.asarray(error_labels)
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=float)

    if error_labels.ndim != 1 or uncertainty_scores.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(error_labels) == 0:
        raise ValueError("Inputs cannot be empty")

    if len(error_labels) != len(uncertainty_scores):
        raise ValueError("Inputs must have equal lengths")

    if not np.all(np.isin(error_labels, [0, 1])):
        raise ValueError("Error labels must contain only 0 and 1")

    if not np.all(np.isfinite(uncertainty_scores)):
        raise ValueError("Uncertainty scores must be finite")

    if len(np.unique(error_labels)) < 2:
        return np.nan, np.nan

    auroc = roc_auc_score(error_labels, uncertainty_scores)
    auprc = average_precision_score(error_labels, uncertainty_scores)

    return auroc, auprc

