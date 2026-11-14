
"""Integration tests for loading trained EpiAware ensemble checkpoints."""

from pathlib import Path

import pytest
import torch

from epiaware.models.ensemble import load_ensemble_models


CHECKPOINT_PATHS = [
    Path("checkpoints/ensemble/member_1_seed_404.pt"),
    Path("checkpoints/ensemble/member_2_seed_505.pt"),
    Path("checkpoints/ensemble/member_3_seed_606.pt"),
]

EXPECTED_PARAMETER_COUNT = 11_173_962

CHECKPOINTS_AVAILABLE = all(
    path.is_file() for path in CHECKPOINT_PATHS
)


@pytest.mark.skipif(
    not CHECKPOINTS_AVAILABLE,
    reason="Trained ensemble checkpoints are not available",
)
def test_load_all_ensemble_checkpoints():
    """Verify that all three trained ensemble members load."""
    models, device = load_ensemble_models(CHECKPOINT_PATHS)

    assert len(models) == 3
    assert isinstance(device, torch.device)

    for model in models:
        assert model.training is False


@pytest.mark.skipif(
    not CHECKPOINTS_AVAILABLE,
    reason="Trained ensemble checkpoints are not available",
)
def test_loaded_models_have_expected_parameter_count():
    """Verify the architecture of each loaded ensemble member."""
    models, _ = load_ensemble_models(CHECKPOINT_PATHS)

    for model in models:
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        assert parameter_count == EXPECTED_PARAMETER_COUNT


def test_missing_checkpoint_raises_file_not_found_error(tmp_path):
    """Verify that missing checkpoint files are rejected."""
    missing_checkpoint = tmp_path / "missing_model.pt"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_ensemble_models([missing_checkpoint])
