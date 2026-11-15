
# Main evaluation pipeline for deep ensemble predictions

from pathlib import Path

import numpy as np
import pandas as pd

from epiaware.evaluation.uncertainty import (
    predictive_entropy,
    variation_ratio,
    jensen_shannon_divergence
)

from epiaware.evaluation.error_detection import (
    get_ensemble_probabilities,
    get_predictions,
    get_error_labels,
    calculate_error_detection_metrics
)

from epiaware.evaluation.calibration import (
    expected_calibration_error
)

from epiaware.evaluation.selective_prediction import (
    risk_coverage_curve,
    calculate_aurc
)


def evaluate_ensemble(
    probabilities,
    ground_truth,
    experiment_name="experiment",
    corruption_type="clean",
    severity=None
):
    """
    Evaluate ensemble predictions using all implemented metrics.

    Args:
        probabilities: Array with shape (models, samples, classes).
        ground_truth: True class labels for each sample.
        experiment_name: Name of the experiment.
        corruption_type: Type of corruption, or clean.
        severity: Corruption severity from 1 to 5, or None.

    Returns:
        summary: Dictionary containing aggregate evaluation metrics.
        sample_results: DataFrame containing sample-level results.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    ground_truth = np.asarray(ground_truth)

    # Validate probability dimensions.
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
        raise ValueError("Each probability distribution must sum to 1")

    number_of_models, number_of_samples, number_of_classes = (
        probabilities.shape
    )

    # Validate ground-truth labels.
    if ground_truth.ndim != 1:
        raise ValueError("Ground truth must be one-dimensional")

    if len(ground_truth) != number_of_samples:
        raise ValueError(
            "Ground truth length must match the number of samples"
        )

    if not np.issubdtype(ground_truth.dtype, np.integer):
        raise ValueError("Ground-truth labels must be integers")

    if np.any(ground_truth < 0) or np.any(
        ground_truth >= number_of_classes
    ):
        raise ValueError("Ground-truth labels are outside the class range")

    if severity is not None:
        if (
            not isinstance(severity, (int, np.integer))
            or isinstance(severity, (bool, np.bool_))
            or not 1 <= severity <= 5
        ):
            raise ValueError("Severity must be an integer from 1 to 5")

    # Average probabilities across ensemble members.
    ensemble_probabilities = get_ensemble_probabilities(probabilities)

    # Calculate ensemble predictions and confidence.
    ensemble_predictions, ensemble_confidence = get_predictions(
        probabilities
    )

    # Calculate correctness and error labels.
    correct, error_labels = get_error_labels(
        ensemble_predictions,
        ground_truth
    )

    # Calculate individual member accuracies.
    individual_member_accuracies = []

    for model_index in range(number_of_models):
        model_predictions = np.argmax(
            probabilities[model_index],
            axis=1
        )

        model_accuracy = np.mean(
            model_predictions == ground_truth
        )

        individual_member_accuracies.append(model_accuracy)

    # Calculate ensemble accuracy.
    ensemble_accuracy = np.mean(correct)

    # Calculate ensemble predictive entropy.
    ensemble_entropy = predictive_entropy(
        ensemble_probabilities
    )

    # Calculate variation ratio using member predictions.
    member_predictions = np.argmax(
        probabilities,
        axis=2
    )

    ensemble_variation_ratio = variation_ratio(
        member_predictions
    )

    # Calculate Jensen-Shannon divergence.
    ensemble_jsd = jensen_shannon_divergence(
        probabilities
    )

    # Collect uncertainty scores.
    uncertainty_scores = {
        "entropy": ensemble_entropy,
        "variation_ratio": ensemble_variation_ratio,
        "jsd": ensemble_jsd
    }

    # Calculate error detection and selective prediction metrics.
    summary = {
        "experiment_name": experiment_name,
        "corruption_type": corruption_type,
        "severity": severity,
        "number_of_samples": number_of_samples,
        "number_of_models": number_of_models,
        "ensemble_accuracy": ensemble_accuracy
    }

    for model_index, model_accuracy in enumerate(
        individual_member_accuracies
    ):
        summary[f"model_{model_index + 1}_accuracy"] = model_accuracy

    # Calculate Expected Calibration Error.
    summary["ensemble_ece"] = expected_calibration_error(
        ensemble_confidence,
        correct.astype(int),
        num_bins=15
    )

    # Calculate AUROC, AUPRC and AURC for each uncertainty metric.
    for metric_name, scores in uncertainty_scores.items():

        auroc, auprc = calculate_error_detection_metrics(
            error_labels,
            scores
        )

        coverage, risk = risk_coverage_curve(
            error_labels,
            scores
        )

        aurc = calculate_aurc(risk)

        summary[f"{metric_name}_auroc"] = auroc
        summary[f"{metric_name}_auprc"] = auprc
        summary[f"{metric_name}_aurc"] = aurc

    # Create sample-level results.
    sample_results = pd.DataFrame({
        "ground_truth": ground_truth,
        "prediction": ensemble_predictions,
        "correct": correct,
        "error": error_labels,
        "ensemble_confidence": ensemble_confidence,
        "ensemble_entropy": ensemble_entropy,
        "variation_ratio": ensemble_variation_ratio,
        "jsd": ensemble_jsd
    })

    return summary, sample_results


def save_evaluation_results(
    summary,
    sample_results,
    output_directory="results"
):
    """Save summary and sample-level results as CSV files."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_path = output_directory / "evaluation_summary.csv"
    samples_path = output_directory / "sample_results.csv"

    summary_dataframe = pd.DataFrame([summary])

    summary_dataframe.to_csv(
        summary_path,
        index=False
    )

    sample_results.to_csv(
        samples_path,
        index=False
    )

    return summary_path, samples_path

