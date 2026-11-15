
# Tests for Expected Calibration Error

import numpy as np
import pytest

from epiaware.evaluation.calibration import expected_calibration_error


def test_ece_perfect_calibration():
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    correctness = np.array([1, 1, 1, 1])

    ece = expected_calibration_error(
        confidences,
        correctness
    )

    assert ece == pytest.approx(0.25)


def test_ece_with_known_result():
    confidences = np.array([0.9, 0.8, 0.2, 0.1])
    correctness = np.array([1, 0, 1, 0])

    ece = expected_calibration_error(
        confidences,
        correctness,
        num_bins=2
    )

    assert ece == pytest.approx(0.35)


def test_ece_zero_for_exact_confidence():
    confidences = np.array([1.0, 0.0, 1.0, 0.0])
    correctness = np.array([1, 0, 1, 0])

    ece = expected_calibration_error(confidences, correctness)

    assert ece == pytest.approx(0.0)


def test_ece_default_number_of_bins():
    confidences = np.array([0.9, 0.8, 0.7])
    correctness = np.array([1, 1, 0])

    result_default = expected_calibration_error(
        confidences,
        correctness
    )

    result_explicit = expected_calibration_error(
        confidences,
        correctness,
        num_bins=15
    )

    assert result_default == pytest.approx(result_explicit)


def test_ece_includes_confidence_one_in_final_bin():
    confidences = np.array([1.0])
    correctness = np.array([1])

    ece = expected_calibration_error(
        confidences,
        correctness,
        num_bins=10
    )

    assert ece == pytest.approx(0.0)


def test_ece_invalid_confidence_range():
    confidences = np.array([0.5, 1.2])
    correctness = np.array([1, 0])

    with pytest.raises(ValueError):
        expected_calibration_error(confidences, correctness)


def test_ece_mismatched_lengths():
    confidences = np.array([0.5, 0.8])
    correctness = np.array([1])

    with pytest.raises(ValueError):
        expected_calibration_error(confidences, correctness)


def test_ece_invalid_correctness_values():
    confidences = np.array([0.5, 0.8])
    correctness = np.array([1, 2])

    with pytest.raises(ValueError):
        expected_calibration_error(confidences, correctness)


def test_ece_invalid_number_of_bins():
    confidences = np.array([0.5, 0.8])
    correctness = np.array([1, 0])

    with pytest.raises(ValueError):
        expected_calibration_error(
            confidences,
            correctness,
            num_bins=0
        )


def test_ece_empty_input():
    confidences = np.array([])
    correctness = np.array([])

    with pytest.raises(ValueError):
        expected_calibration_error(confidences, correctness)

