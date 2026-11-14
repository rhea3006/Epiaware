
"""Utilities for loading and running the EpiAware deep ensemble."""

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from epiaware.models.resnet import create_resnet18
from epiaware.training.engine import get_device


def load_ensemble_models(
    checkpoint_paths: Sequence[str | Path],
    device_name: str = "auto",
) -> tuple[list[nn.Module], torch.device]:
    """Load trained ensemble members from their checkpoints."""

    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required")

    device = get_device(device_name)
    models = []

    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Missing model_state_dict in {checkpoint_path}"
            )

        metadata = checkpoint.get("metadata", {})
        num_classes = metadata.get("num_classes", 10)

        model = create_resnet18(num_classes=num_classes)
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        model.to(device)
        model.eval()

        models.append(model)

    return models, device


def predict_ensemble_logits(
    models: Sequence[nn.Module],
    inputs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate and stack raw logits from all ensemble members."""

    if not models:
        raise ValueError("At least one model is required")

    if inputs.ndim != 4:
        raise ValueError(
            "Inputs must have shape (N, C, H, W)"
        )

    if inputs.shape[0] == 0:
        raise ValueError("Input batch cannot be empty")

    inputs = inputs.to(device)

    member_logits = []

    with torch.inference_mode():
        for model in models:
            model.eval()
            logits = model(inputs)
            member_logits.append(logits)

    return torch.stack(member_logits, dim=0)
