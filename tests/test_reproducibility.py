import random

import numpy as np
import pytest
import torch


def test_python_random_reproducibility():
    random.seed(101)
    first = random.random()

    random.seed(101)
    second = random.random()

    assert first == second


def test_numpy_random_reproducibility():
    np.random.seed(101)
    first = np.random.rand(5)

    np.random.seed(101)
    second = np.random.rand(5)

    np.testing.assert_array_equal(first, second)


def test_torch_random_reproducibility():
    torch.manual_seed(101)
    first = torch.rand(5)

    torch.manual_seed(101)
    second = torch.rand(5)

    assert torch.equal(first, second)


def test_negative_seed_rejected():
    from epiaware.reproducibility import set_seed

    with pytest.raises(ValueError):
        set_seed(-1)