
# Tests for the main ensemble evaluation pipeline

import numpy as np
import pandas as pd
import pytest

from epiaware.evaluation.pipeline import (
    evaluate_ensemble,
    save_evaluation_results
)


def create_test_data():
    """Create deterministic ensemble probabilities for testing."""

    probabilities = np.array([
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2]
        ],
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.6, 0.1],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2]
        ],
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.7, 0.1]
        ]
    ])

    ground_truth = np.array([0, 1, 2, 1])

    return probabilities, ground_truth


def test_evaluate_ensemble_returns_summary_and_dataframe():
    probabilities, ground_truth = create_test_data()

    summary, sample_results = evaluate_ensemble(
        probabilities,
        ground_truth
    )

    assert isinstance(summary, dict)
    assert isinstance(sample_results, pd.DataFrame)
    assert len(sample_results) == 4


def test_evaluate_ensemble_summary_contains_metrics():
    probabilities, ground_truth = create_test_data()

    summary, sample_results = evaluate_ensemble(
        probabilities,
        ground_truth
    )

    expected_keys = [
        "ensemble_accuracy",
        "ensemble_ece",
        "model_1_accuracy",
        "model_2_accuracy",
        "model_3_accuracy",
        "entropy_auroc",
        "entropy_auprc",
        "entropy_aurc",
        "variation_ratio_auroc",
        "variation_ratio_auprc",
        "variation_ratio_aurc",
        "jsd_auroc",
        "jsd_auprc",
        "jsd_aurc"
    ]

    for key in expected_keys:
        assert key in summary


def test_evaluate_ensemble_sample_columns():
    probabilities, ground_truth = create_test_data()

    summary, sample_results = evaluate_ensemble(
        probabilities,
        ground_truth
    )

    expected_columns = [
        "ground_truth",
        "prediction",
        "correct",
        "error",
        "ensemble_confidence",
        "ensemble_entropy",
        "variation_ratio",
        "jsd"
    ]

    for column in expected_columns:
        assert column in sample_results.columns


def test_evaluate_ensemble_correctness_labels():
    probabilities, ground_truth = create_test_data()

    summary, sample_results = evaluate_ensemble(
        probabilities,
        ground_truth
    )

    expected_errors = (
        sample_results["prediction"].to_numpy() != ground_truth
    ).astype(int)

    assert np.array_equal(
        sample_results["error"].to_numpy(),
        expected_errors
    )

    assert np.array_equal(
        sample_results["correct"].to_numpy(),
        expected_errors == 0
    )


def test_evaluate_ensemble_invalid_probability_shape():
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7]
    ])

    ground_truth = np.array([0, 1])

    with pytest.raises(ValueError):
        evaluate_ensemble(probabilities, ground_truth)


def test_evaluate_ensemble_invalid_ground_truth_length():
    probabilities, ground_truth = create_test_data()

    wrong_ground_truth = np.array([0, 1])

    with pytest.raises(ValueError):
        evaluate_ensemble(probabilities, wrong_ground_truth)


def test_evaluate_ensemble_invalid_ground_truth_class():
    probabilities, ground_truth = create_test_data()

    wrong_ground_truth = np.array([0, 1, 2, 5])

    with pytest.raises(ValueError):
        evaluate_ensemble(probabilities, wrong_ground_truth)


def test_evaluate_ensemble_invalid_severity():
    probabilities, ground_truth = create_test_data()

    with pytest.raises(ValueError):
        evaluate_ensemble(
            probabilities,
            ground_truth,
            corruption_type="gaussian_noise",
            severity=6
        )


def test_evaluate_ensemble_metadata():
    probabilities, ground_truth = create_test_data()

    summary, sample_results = evaluate_ensemble(
        probabilities,
        ground_truth,
        experiment_name="test_experiment",
        corruption_type="gaussian_noise",
        severity=3
    )

    assert summary["experiment_name"] == "test_experiment"
    assert summary["corruption_type"] == "gaussian_noise"
    assert summary["severity"] == 3


def test_save_evaluation_results(tmp_path):
    probabilities, ground_truth = create_test_data()

    summary, sample_results = evaluate_ensemble(
        probabilities,
        ground_truth
    )

    summary_path, samples_path = save_evaluation_results(
        summary,
        sample_results,
        output_directory=tmp_path
    )

    assert summary_path.exists()
    assert samples_path.exists()

    saved_summary = pd.read_csv(summary_path)
    saved_samples = pd.read_csv(samples_path)

    assert len(saved_summary) == 1
    assert len(saved_samples) == 4

