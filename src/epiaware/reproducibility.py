"""Utilities for reproducible EpiAware experiments."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Non-negative integer used to initialize random generators.

    Raises:
        TypeError: If seed is not an integer.
        ValueError: If seed is negative.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True