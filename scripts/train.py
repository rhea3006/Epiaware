
"""Run EpiAware CIFAR-10 model training experiments."""

import argparse
import json
from pathlib import Path

import yaml

from epiaware.data.cifar10 import get_cifar10_dataloaders
from epiaware.models.resnet import create_resnet18
from epiaware.reproducibility import set_seed
from epiaware.training.engine import train_model


def load_config(config_path: str | Path) -> dict:
    """Load experiment configuration from a YAML file."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an EpiAware CIFAR-10 ResNet-18 model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to experiment configuration.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured number of epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the experiment random seed.",
    )

    return parser.parse_args()


def main() -> None:
    """Execute a single-model training experiment."""
    args = parse_args()
    config = load_config(args.config)

    experiment = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    seed = args.seed if args.seed is not None else experiment["seed"]
    epochs = (
        args.epochs if args.epochs is not None else training_config["epochs"]
    )

    set_seed(seed)

    print(f"Experiment: {experiment['name']}")
    print(f"Seed: {seed}")
    print(f"Epochs: {epochs}")

    train_loader, validation_loader, _ = get_cifar10_dataloaders(
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        seed=seed,
    )

    model = create_resnet18(
        num_classes=data_config["num_classes"]
    )

    checkpoint_path = (
        Path("checkpoints")
        / experiment["name"]
        / f"resnet18_seed_{seed}.pt"
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=epochs,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        device_name=experiment["device"],
        checkpoint_path=checkpoint_path,
        seed=seed,
        metadata={
            "experiment_name": experiment["name"],
            "architecture": model_config["architecture"],
            "num_classes": data_config["num_classes"],
            "batch_size": data_config["batch_size"],
        },
    )

    results_dir = Path("results") / experiment["name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    history_path = results_dir / f"history_seed_{seed}.json"

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print("\nTraining completed.")
    print(f"Best epoch: {history['best_epoch']}")
    print(
        "Best validation accuracy: "
        f"{history['best_validation_accuracy']:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
