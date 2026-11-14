
"""Training and validation engine for EpiAware."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def get_device(device_name: str = "auto") -> torch.device:
    """Select the requested device, with automatic accelerator detection."""
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    return torch.device(device_name)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Training loader is empty")

    return total_loss / total, correct / total


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model without updating its parameters."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    if total == 0:
        raise ValueError("Evaluation loader is empty")

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device_name: str = "auto",
    checkpoint_path: str | Path = "checkpoints/best_model.pt",
    seed: int = 101,
    metadata: dict[str, Any] | None = None,
    scheduler_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a model, select its best validation checkpoint, and return history."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    device = get_device(device_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    scheduler = None

    if scheduler_config is not None:
        scheduler_name = scheduler_config.get("name")

        if scheduler_name != "step_lr":
            raise ValueError(
                f"Unsupported scheduler: {scheduler_name}"
            )

        step_size = scheduler_config.get("step_size")
        gamma = scheduler_config.get("gamma")

        if (
            isinstance(step_size, bool)
            or not isinstance(step_size, int)
            or step_size <= 0
        ):
            raise ValueError("scheduler step_size must be a positive integer")

        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, (int, float))
            or not 0 < gamma < 1
        ):
            raise ValueError("scheduler gamma must be between 0 and 1")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    history: dict[str, Any] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    best_accuracy = -1.0
    best_epoch = 0
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        validation_loss, validation_accuracy = evaluate_model(
            model, validation_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {validation_loss:.4f} | "
            f"Val acc: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "validation_accuracy": best_accuracy,
                    "validation_loss": validation_loss,
                    "seed": seed,
                    "metadata": metadata or {},
                },
                checkpoint,
            )

        if scheduler is not None:
            scheduler.step()

    # Restore the model weights from the best validation checkpoint.
    best_checkpoint = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    history["best_epoch"] = best_epoch
    history["best_validation_accuracy"] = best_accuracy
    history["checkpoint_path"] = str(checkpoint)
    history["device"] = str(device)

    return history
