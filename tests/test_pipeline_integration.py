
# Tests for individual entropy, logits integration and result storage

import numpy as np
import pandas as pd
import pytest

from epiaware.evaluation.pipeline import (
    evaluate_ensemble,
    evaluate_ensemble_logits,
    save_evaluation_results
)


def create_test_logits():
    return np.array([
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [1.0, 0.0, 4.0],
            [2.0, 1.0, 0.0]
        ],
        [
            [3.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 3.0],
            [1.0, 2.0, 0.0]
        ],
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0]
        ]
    ])


def test_individual_entropy_columns():
    logits = create_test_logits()

    ground_truth = np.array([0, 1, 2, 0])

    summary, sample_results = evaluate_ensemble_logits(
        logits,
        ground_truth
    )

    for model_number in range(1, 4):
        assert f"model_{model_number}_entropy" in summary
        assert f"model_{model_number}_entropy" in sample_results.columns

    assert "ensemble_entropy" in summary
    assert "ensemble_entropy" in sample_results.columns

    assert len(sample_results) == len(ground_truth)


def test_logits_integration_returns_valid_results():
    logits = create_test_logits()

    ground_truth = np.array([0, 1, 2, 0])

    summary, sample_results = evaluate_ensemble_logits(
        logits,
        ground_truth
    )

    assert isinstance(summary, dict)
    assert isinstance(sample_results, pd.DataFrame)

    assert np.isfinite(summary["ensemble_accuracy"])


def test_logits_invalid_dimensions():
    logits = np.zeros((4, 10))

    ground_truth = np.array([0, 1, 2, 0])

    with pytest.raises(ValueError):
        evaluate_ensemble_logits(logits, ground_truth)


def test_logits_non_finite_values():
    logits = create_test_logits()
    logits[0, 0, 0] = np.nan

    ground_truth = np.array([0, 1, 2, 0])

    with pytest.raises(ValueError):
        evaluate_ensemble_logits(logits, ground_truth)


def test_logits_empty_dimensions():
    logits = np.empty((3, 0, 10))

    ground_truth = np.array([])

    with pytest.raises(ValueError):
        evaluate_ensemble_logits(logits, ground_truth)


def test_probability_evaluator_still_works():
    logits = create_test_logits()

    shifted_logits = logits - np.max(
        logits,
        axis=2,
        keepdims=True
    )

    exponential_values = np.exp(shifted_logits)

    probabilities = exponential_values / exponential_values.sum(
        axis=2,
        keepdims=True
    )

    ground_truth = np.array([0, 1, 2, 0])

    summary, sample_results = evaluate_ensemble(
        probabilities,
        ground_truth
    )

    assert "ensemble_entropy" in summary
    assert len(sample_results) == 4


def test_multiple_saves_do_not_overwrite(tmp_path):
    logits = create_test_logits()
    ground_truth = np.array([0, 1, 2, 0])

    summary, sample_results = evaluate_ensemble_logits(
        logits,
        ground_truth,
        experiment_name="test_experiment",
        corruption_type="gaussian_noise",
        severity=1
    )

    first_summary_path, first_samples_path = save_evaluation_results(
        summary,
        sample_results,
        output_directory=tmp_path
    )

    second_summary_path, second_samples_path = save_evaluation_results(
        summary,
        sample_results,
        output_directory=tmp_path
    )

    assert first_summary_path == second_summary_path
    assert first_samples_path != second_samples_path

    assert first_samples_path.exists()
    assert second_samples_path.exists()

    saved_summary = pd.read_csv(first_summary_path)

    assert len(saved_summary) == 2


def test_sample_csv_contains_entropy_columns(tmp_path):
    logits = create_test_logits()
    ground_truth = np.array([0, 1, 2, 0])

    summary, sample_results = evaluate_ensemble_logits(
        logits,
        ground_truth,
        experiment_name="clean_test",
        corruption_type="clean"
    )

    _, samples_path = save_evaluation_results(
        summary,
        sample_results,
        output_directory=tmp_path
    )

    saved_samples = pd.read_csv(samples_path)

    assert "model_1_entropy" in saved_samples.columns
    assert "model_2_entropy" in saved_samples.columns
    assert "model_3_entropy" in saved_samples.columns
    assert "ensemble_entropy" in saved_samples.columns

