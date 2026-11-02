import pytest
from epiaware.data.cifar10 import (create_train_val_split,
    get_cifar10_dataloaders,)


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_invalid_batch_size_raises_error(batch_size):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        get_cifar10_dataloaders(batch_size=batch_size)


def test_negative_num_workers_raises_error():
    with pytest.raises(
        ValueError,
        match="num_workers must be non-negative",
    ):
        get_cifar10_dataloaders(num_workers=-1)


def test_negative_seed_raises_error():
    with pytest.raises(
        ValueError,
        match="seed must be non-negative",
    ):
        get_cifar10_dataloaders(seed=-1)

def test_train_val_split_sizes():
    train_indices, val_indices = create_train_val_split(
        dataset_size=100,
        val_size=20,
        seed=101,
    )

    assert len(train_indices) == 80
    assert len(val_indices) == 20


def test_train_val_split_has_no_overlap():
    train_indices, val_indices = create_train_val_split(
        dataset_size=100,
        val_size=20,
        seed=101,
    )

    assert set(train_indices).isdisjoint(set(val_indices))


def test_train_val_split_is_reproducible():
    first_train, first_val = create_train_val_split(
        dataset_size=100,
        val_size=20,
        seed=101,
    )

    second_train, second_val = create_train_val_split(
        dataset_size=100,
        val_size=20,
        seed=101,
    )

    assert first_train == second_train
    assert first_val == second_val

@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_size": 10.5},
        {"val_size": 2.5},
        {"seed": True},
    ],
)
def test_train_val_split_rejects_invalid_types(kwargs):
    with pytest.raises(TypeError):
        create_train_val_split(**kwargs)