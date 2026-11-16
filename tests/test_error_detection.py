
# Tests for ensemble error detection metrics

import numpy as np
import pytest

from epiaware.evaluation.error_detection import (
    get_ensemble_probabilities,
    get_predictions,
    get_error_labels,
    calculate_error_detection_metrics
)


def test_get_ensemble_probabilities():
    probabilities = np.array([
        [[0.8, 0.2], [0.3, 0.7]],
        [[0.6, 0.4], [0.5, 0.5]],
        [[0.7, 0.3], [0.4, 0.6]]
    ])

    result = get_ensemble_probabilities(probabilities)

    expected = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    assert result.shape == (2, 2)
    assert np.allclose(result, expected)


def test_get_ensemble_probabilities_invalid_input():
    probabilities = np.array([
        [[0.8, 0.5]],
        [[0.2, 0.8]]
    ])

    with pytest.raises(ValueError):
        get_ensemble_probabilities(probabilities)


def test_get_predictions():
    probabilities = np.array([
        [[0.9, 0.1], [0.2, 0.8]],
        [[0.8, 0.2], [0.3, 0.7]],
        [[0.7, 0.3], [0.4, 0.6]]
    ])

    predictions, confidence = get_predictions(probabilities)

    assert np.array_equal(predictions, [0, 1])
    assert np.allclose(confidence, [0.8, 0.7])


def test_get_error_labels():
    predictions = np.array([0, 1, 2, 1])
    ground_truth = np.array([0, 2, 2, 0])

    correct, error_labels = get_error_labels(
        predictions,
        ground_truth
    )

    assert np.array_equal(correct, [True, False, True, False])
    assert np.array_equal(error_labels, [0, 1, 0, 1])


def test_get_error_labels_invalid_lengths():
    predictions = np.array([0, 1])
    ground_truth = np.array([0])

    with pytest.raises(ValueError):
        get_error_labels(predictions, ground_truth)


def test_error_detection_metrics_perfect_ranking():
    error_labels = np.array([0, 0, 1, 1])
    uncertainty_scores = np.array([0.1, 0.2, 0.8, 0.9])

    auroc, auprc = calculate_error_detection_metrics(
        error_labels,
        uncertainty_scores
    )

    assert auroc == pytest.approx(1.0)
    assert auprc == pytest.approx(1.0)


def test_error_detection_metrics_reverse_ranking():
    error_labels = np.array([0, 0, 1, 1])
    uncertainty_scores = np.array([0.9, 0.8, 0.2, 0.1])

    auroc, auprc = calculate_error_detection_metrics(
        error_labels,
        uncertainty_scores
    )

    assert auroc == pytest.approx(0.0)
    assert auprc == pytest.approx(0.4166666667)


def test_error_detection_metrics_all_correct():
    error_labels = np.array([0, 0, 0, 0])
    uncertainty_scores = np.array([0.1, 0.2, 0.3, 0.4])

    auroc, auprc = calculate_error_detection_metrics(
        error_labels,
        uncertainty_scores
    )

    assert np.isnan(auroc)
    assert np.isnan(auprc)


def test_error_detection_metrics_all_incorrect():
    error_labels = np.array([1, 1, 1, 1])
    uncertainty_scores = np.array([0.1, 0.2, 0.3, 0.4])

    auroc, auprc = calculate_error_detection_metrics(
        error_labels,
        uncertainty_scores
    )

    assert np.isnan(auroc)
    assert np.isnan(auprc)


def test_error_detection_metrics_invalid_labels():
    error_labels = np.array([0, 1, 2])
    uncertainty_scores = np.array([0.1, 0.5, 0.9])

    with pytest.raises(ValueError):
        calculate_error_detection_metrics(
            error_labels,
            uncertainty_scores
        )

