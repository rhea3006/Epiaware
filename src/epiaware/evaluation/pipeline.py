
# Main evaluation pipeline for deep ensemble predictions

from pathlib import Path
import re

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
    Evaluate ensemble probabilities.

    probabilities shape: (models, samples, classes)
    """

    probabilities = np.asarray(probabilities, dtype=float)
    ground_truth = np.asarray(ground_truth)

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

    if ground_truth.ndim != 1:
        raise ValueError("Ground truth must be one-dimensional")

    if len(ground_truth) != number_of_samples:
        raise ValueError("Ground truth length must match sample count")

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

    # Get ensemble predictions and confidence.
    ensemble_predictions, ensemble_confidence = get_predictions(
        probabilities
    )

    # Determine correct and incorrect predictions.
    correct, error_labels = get_error_labels(
        ensemble_predictions,
        ground_truth
    )

    # Calculate accuracy for each individual member.
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

    ensemble_accuracy = np.mean(correct)

    # Calculate entropy separately for every ensemble member.
    individual_member_entropies = []

    for model_index in range(number_of_models):
        model_entropy = predictive_entropy(
            probabilities[model_index]
        )

        individual_member_entropies.append(model_entropy)

    # Calculate entropy of the averaged ensemble probabilities.
    ensemble_entropy = predictive_entropy(
        ensemble_probabilities
    )

    # Calculate ensemble disagreement.
    member_predictions = np.argmax(
        probabilities,
        axis=2
    )

    ensemble_variation_ratio = variation_ratio(
        member_predictions
    )

    ensemble_jsd = jensen_shannon_divergence(
        probabilities
    )

    # Ensemble uncertainty metrics.
    uncertainty_scores = {
        "entropy": ensemble_entropy,
        "variation_ratio": ensemble_variation_ratio,
        "jsd": ensemble_jsd
    }

    # Build experiment summary.
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

    for model_index, model_entropy in enumerate(
        individual_member_entropies
    ):
        summary[f"model_{model_index + 1}_entropy"] = np.mean(
            model_entropy
        )

    summary["ensemble_entropy"] = np.mean(ensemble_entropy)

    # Calculate ensemble calibration.
    summary["ensemble_ece"] = expected_calibration_error(
        ensemble_confidence,
        correct.astype(int),
        num_bins=15
    )

    # Calculate error detection and selective prediction metrics.
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

    # Build sample-level results.
    sample_data = {
        "ground_truth": ground_truth,
        "prediction": ensemble_predictions,
        "correct": correct,
        "error": error_labels,
        "ensemble_confidence": ensemble_confidence,
        "ensemble_entropy": ensemble_entropy,
        "variation_ratio": ensemble_variation_ratio,
        "jsd": ensemble_jsd
    }

    for model_index, model_entropy in enumerate(
        individual_member_entropies
    ):
        sample_data[f"model_{model_index + 1}_entropy"] = model_entropy

    sample_results = pd.DataFrame(sample_data)

    return summary, sample_results


def evaluate_ensemble_logits(
    logits,
    ground_truth,
    experiment_name="experiment",
    corruption_type="clean",
    severity=None
):
    """
    Convert raw ensemble logits to probabilities and evaluate them.

    Accepts NumPy arrays or PyTorch tensors with shape:
    (models, samples, classes).
    """

    # Convert PyTorch tensors to NumPy when necessary.
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()

    logits = np.asarray(logits, dtype=float)

    if logits.ndim != 3:
        raise ValueError(
            "Logits must have shape (models, samples, classes)"
        )

    if any(dimension == 0 for dimension in logits.shape):
        raise ValueError("Logits cannot have empty dimensions")

    if not np.all(np.isfinite(logits)):
        raise ValueError("Logits must contain finite values")

    # Numerically stable softmax along the class dimension.
    shifted_logits = logits - np.max(
        logits,
        axis=2,
        keepdims=True
    )

    exponential_values = np.exp(shifted_logits)

    probabilities = exponential_values / np.sum(
        exponential_values,
        axis=2,
        keepdims=True
    )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Converted probabilities must be finite")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Converted probabilities are outside [0, 1]")

    if not np.allclose(probabilities.sum(axis=2), 1.0):
        raise ValueError("Converted probabilities must sum to 1")

    return evaluate_ensemble(
        probabilities,
        ground_truth,
        experiment_name=experiment_name,
        corruption_type=corruption_type,
        severity=severity
    )


def save_evaluation_results(
    summary,
    sample_results,
    output_directory="results"
):
    """
    Append one experiment to the cumulative summary CSV.

    Save sample-level results in a unique file for each run.
    Existing experiment results are preserved.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_path = output_directory / "evaluation_summary.csv"

    # Create a descriptive filename.
    experiment_name = str(
        summary.get("experiment_name", "experiment")
    )

    corruption_type = str(
        summary.get("corruption_type", "clean")
    )

    severity = summary.get("severity")

    filename_parts = [
        experiment_name,
        corruption_type
    ]

    if severity is not None:
        filename_parts.append(f"severity_{severity}")

    filename = "_".join(filename_parts)

    # Remove characters that are unsuitable for filenames.
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", filename)

    samples_path = output_directory / f"{filename}_samples.csv"

    # Avoid overwriting results from an earlier run.
    run_number = 2

    while samples_path.exists():
        samples_path = output_directory / (
            f"{filename}_run_{run_number}_samples.csv"
        )
        run_number += 1

    # Append the new experiment summary.
    summary_dataframe = pd.DataFrame([summary])

    summary_dataframe.to_csv(
        summary_path,
        mode="a",
        header=not summary_path.exists(),
        index=False
    )

    # Save sample-level results separately.
    sample_results.to_csv(
        samples_path,
        index=False
    )

    return summary_path, samples_path

