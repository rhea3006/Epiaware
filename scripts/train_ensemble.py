
"""Train the EpiAware deep ensemble sequentially."""

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml

from epiaware.data.cifar10 import get_cifar10_dataloaders
from epiaware.models.resnet import create_resnet18
from epiaware.reproducibility import set_seed
from epiaware.training.engine import train_model


DEFAULT_SEEDS = [404, 505, 606]


def load_config(config_path: str | Path) -> dict:
    """Load experiment configuration from YAML."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    return config


def parse_args() -> argparse.Namespace:
    """Parse ensemble training arguments."""
    parser = argparse.ArgumentParser(
        description="Train the EpiAware ResNet-18 deep ensemble."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/scheduler_experiment.yaml",
        help="Path to the training configuration.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds for ensemble members.",
    )

    return parser.parse_args()


def main() -> None:
    """Train ensemble members sequentially."""
    args = parse_args()
    config = load_config(args.config)

    experiment = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    seeds = args.seeds

    if not seeds:
        raise ValueError("At least one ensemble seed is required")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble seeds must be unique")

    if len(seeds) != model_config["ensemble_size"]:
        raise ValueError(
            "Number of seeds must match configured ensemble_size"
        )

    checkpoint_dir = Path("checkpoints") / "ensemble"
    results_dir = Path("results") / "ensemble"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {experiment['name']}")
    print(f"Ensemble size: {len(seeds)}")
    print(f"Seeds: {seeds}")
    print(f"Epochs per member: {training_config['epochs']}")

    for member_index, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 60)
        print(f"Training ensemble member {member_index}/{len(seeds)}")
        print(f"Seed: {seed}")
        print("=" * 60)

        # Reset randomness independently for this member.
        set_seed(seed)

        train_loader, validation_loader, _ = get_cifar10_dataloaders(
            batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        seed=seed,
        split_seed=707,
        )

        model = create_resnet18(
            num_classes=data_config["num_classes"]
        )

        checkpoint_path = (
            checkpoint_dir / f"member_{member_index}_seed_{seed}.pt"
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=training_config["epochs"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            device_name=experiment["device"],
            checkpoint_path=checkpoint_path,
            seed=seed,
            metadata={
                "experiment_name": "epiaware_ensemble",
                "member_index": member_index,
                "architecture": model_config["architecture"],
                "num_classes": data_config["num_classes"],
                "batch_size": data_config["batch_size"],
            },
            scheduler_config=training_config.get("scheduler"),
        )

        history["member_index"] = member_index
        history["seed"] = seed

        history_path = results_dir / f"history_seed_{seed}.json"

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

        recorded_config = {
            **config,
            "experiment": {
                **experiment,
                "seed": seed,
            },
            "ensemble": {
                "member_index": member_index,
                "ensemble_size": len(seeds),
                "seeds": seeds,
            },
            "data": {
                **data_config,
                "split_seed": 707,
            },
        }

        config_path = results_dir / f"config_seed_{seed}.yaml"

        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(recorded_config, file, sort_keys=False)

        print(f"\nMember {member_index} completed.")
        print(f"Best epoch: {history['best_epoch']}")
        print(
            "Best validation accuracy: "
            f"{history['best_validation_accuracy']:.4f}"
        )
        print(f"Checkpoint: {checkpoint_path}")
        print(f"History: {history_path}")

        # Release resources before training the next member.
        del model
        del train_loader
        del validation_loader
        del history

        gc.collect()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETED")
    print(f"Members trained: {len(seeds)}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Results: {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
