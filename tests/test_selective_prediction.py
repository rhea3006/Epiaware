
# Tests for selective prediction and AURC

import numpy as np
import pytest

from epiaware.evaluation.selective_prediction import (
    risk_coverage_curve,
    calculate_aurc
)


def test_risk_coverage_curve():
    error_labels = np.array([0, 1, 0, 1, 1])
    uncertainty_scores = np.array([0.1, 0.8, 0.2, 0.9, 0.7])

    coverage, risk = risk_coverage_curve(
        error_labels,
        uncertainty_scores
    )

    expected_coverage = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    expected_risk = np.array([0.0, 0.0, 1/3, 0.5, 0.6])

    assert np.allclose(coverage, expected_coverage)
    assert np.allclose(risk, expected_risk)


def test_risk_coverage_output_lengths():
    error_labels = np.array([0, 1, 0, 1])
    uncertainty_scores = np.array([0.1, 0.8, 0.2, 0.9])

    coverage, risk = risk_coverage_curve(
        error_labels,
        uncertainty_scores
    )

    assert len(coverage) == 4
    assert len(risk) == 4


def test_low_uncertainty_samples_retained_first():
    error_labels = np.array([0, 1, 0])
    uncertainty_scores = np.array([0.1, 0.9, 0.2])

    coverage, risk = risk_coverage_curve(
        error_labels,
        uncertainty_scores
    )

    assert risk[0] == pytest.approx(0.0)
    assert risk[1] == pytest.approx(0.0)
    assert risk[2] == pytest.approx(1/3)


def test_risk_coverage_all_correct():
    error_labels = np.array([0, 0, 0])
    uncertainty_scores = np.array([0.3, 0.1, 0.2])

    coverage, risk = risk_coverage_curve(
        error_labels,
        uncertainty_scores
    )

    assert np.allclose(risk, [0.0, 0.0, 0.0])
    assert np.allclose(coverage, [1/3, 2/3, 1.0])


def test_risk_coverage_all_incorrect():
    error_labels = np.array([1, 1, 1])
    uncertainty_scores = np.array([0.3, 0.1, 0.2])

    coverage, risk = risk_coverage_curve(
        error_labels,
        uncertainty_scores
    )

    assert np.allclose(risk, [1.0, 1.0, 1.0])


def test_risk_coverage_invalid_lengths():
    error_labels = np.array([0, 1])
    uncertainty_scores = np.array([0.1])

    with pytest.raises(ValueError):
        risk_coverage_curve(error_labels, uncertainty_scores)


def test_risk_coverage_invalid_labels():
    error_labels = np.array([0, 2, 1])
    uncertainty_scores = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        risk_coverage_curve(error_labels, uncertainty_scores)


def test_risk_coverage_empty_input():
    error_labels = np.array([])
    uncertainty_scores = np.array([])

    with pytest.raises(ValueError):
        risk_coverage_curve(error_labels, uncertainty_scores)


def test_calculate_aurc():
    risk = np.array([0.0, 0.0, 1/3, 0.5, 0.6])

    aurc = calculate_aurc(risk)

    expected = np.mean(risk)

    assert aurc == pytest.approx(expected)


def test_calculate_aurc_invalid_input():
    risk = np.array([0.1, 0.5, 1.2])

    with pytest.raises(ValueError):
        calculate_aurc(risk)


def test_calculate_aurc_empty_input():
    risk = np.array([])

    with pytest.raises(ValueError):
        calculate_aurc(risk)

