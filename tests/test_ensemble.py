
"""Tests for EpiAware ensemble inference."""

import pytest
import torch
import torch.nn as nn

from epiaware.models.ensemble import predict_ensemble_logits


class DummyModel(nn.Module):
    """A lightweight model that returns fixed logits for testing."""

    def __init__(self, logit_value: float):
        super().__init__()
        self.logit_value = logit_value

    def forward(self, inputs):
        return torch.full(
            (inputs.shape[0], 10),
            self.logit_value,
            device=inputs.device,
        )


@pytest.fixture
def dummy_ensemble():
    """Create three lightweight models with different outputs."""
    return [
        DummyModel(1.0),
        DummyModel(2.0),
        DummyModel(3.0),
    ]


def test_ensemble_logits_shape(dummy_ensemble):
    """Verify the output has shape (ensemble_size, N, num_classes)."""
    inputs = torch.randn(8, 3, 32, 32)

    logits = predict_ensemble_logits(
        models=dummy_ensemble,
        inputs=inputs,
        device=torch.device("cpu"),
    )

    assert logits.shape == (3, 8, 10)


def test_ensemble_returns_raw_logits(dummy_ensemble):
    """Verify outputs are raw logits, not softmax probabilities."""
    inputs = torch.randn(4, 3, 32, 32)

    logits = predict_ensemble_logits(
        models=dummy_ensemble,
        inputs=inputs,
        device=torch.device("cpu"),
    )

    assert torch.all(logits[0] == 1.0)
    assert torch.all(logits[1] == 2.0)
    assert torch.all(logits[2] == 3.0)

    assert not torch.allclose(
        logits[0].sum(dim=1),
        torch.ones(4),
    )


def test_empty_ensemble_raises_error():
    """Verify an empty ensemble is rejected."""
    inputs = torch.randn(4, 3, 32, 32)

    with pytest.raises(ValueError, match="At least one model"):
        predict_ensemble_logits(
            models=[],
            inputs=inputs,
            device=torch.device("cpu"),
        )


def test_invalid_input_dimensions_raise_error(dummy_ensemble):
    """Verify inputs must have four dimensions."""
    inputs = torch.randn(3, 32, 32)

    with pytest.raises(ValueError, match="shape"):
        predict_ensemble_logits(
            models=dummy_ensemble,
            inputs=inputs,
            device=torch.device("cpu"),
        )


def test_empty_input_batch_raises_error(dummy_ensemble):
    """Verify an empty batch is rejected."""
    inputs = torch.randn(0, 3, 32, 32)

    with pytest.raises(ValueError, match="cannot be empty"):
        predict_ensemble_logits(
            models=dummy_ensemble,
            inputs=inputs,
            device=torch.device("cpu"),
        )
