
# Cell 1: Tests for ensemble uncertainty metrics

import numpy as np
import pytest

from epiaware.evaluation.uncertainty import (
    predictive_entropy,
    variation_ratio,
    jensen_shannon_divergence
)


def test_predictive_entropy():
    probabilities = np.array([
        [0.9, 0.1],
        [0.5, 0.5]
    ])

    entropy = predictive_entropy(probabilities)

    assert entropy.shape == (2,)
    assert entropy[0] < entropy[1]


def test_predictive_entropy_for_certain_prediction():
    probabilities = np.array([
        [1.0, 0.0]
    ])

    entropy = predictive_entropy(probabilities)

    assert np.allclose(entropy, [0.0])


def test_predictive_entropy_invalid_input():
    probabilities = np.array([
        [0.8, 0.5]
    ])

    with pytest.raises(ValueError):
        predictive_entropy(probabilities)


def test_variation_ratio():
    predictions = np.array([
        [0, 1, 2],
        [0, 1, 1],
        [0, 0, 2]
    ])

    variation = variation_ratio(predictions)

    assert variation.shape == (3,)
    assert np.allclose(variation, [0.0, 1/3, 1/3])


def test_variation_ratio_with_different_ensemble_size():
    predictions = np.array([
        [0, 1],
        [0, 1],
        [0, 2],
        [0, 2],
        [0, 2]
    ])

    variation = variation_ratio(predictions)

    assert np.allclose(variation, [0.0, 2/5])


def test_variation_ratio_invalid_input():
    predictions = np.array([
        [0.5, 1.0],
        [0.0, 1.0]
    ])

    with pytest.raises(ValueError):
        variation_ratio(predictions)


def test_jensen_shannon_divergence_when_models_agree():
    probabilities = np.array([
        [[0.8, 0.2]],
        [[0.8, 0.2]],
        [[0.8, 0.2]]
    ])

    divergence = jensen_shannon_divergence(probabilities)

    assert divergence.shape == (1,)
    assert np.allclose(divergence, [0.0])


def test_jensen_shannon_divergence_when_models_disagree():
    probabilities = np.array([
        [[0.9, 0.1]],
        [[0.1, 0.9]],
        [[0.8, 0.2]]
    ])

    divergence = jensen_shannon_divergence(probabilities)

    assert divergence.shape == (1,)
    assert divergence[0] > 0


def test_jensen_shannon_divergence_invalid_input():
    probabilities = np.array([
        [[0.8, 0.5]],
        [[0.2, 0.8]]
    ])

    with pytest.raises(ValueError):
        jensen_shannon_divergence(probabilities)

