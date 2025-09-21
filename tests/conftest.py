"""
Shared pytest fixtures for continual learning tests.

This module provides common fixtures for creating synthetic DataLoaders
and testing infrastructure components.
"""

from typing import Callable

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def make_loader() -> Callable:
    """
    Factory fixture for creating synthetic DataLoaders.

    Returns:
        Function that creates DataLoader with synthetic data
    """

    def _make_loader(
        n_samples: int = 96,
        in_dim: int = 32,
        n_classes: int = 5,
        batch_size: int = 32,
        seed: int = 0,
    ) -> DataLoader:
        """
        Create a synthetic DataLoader with controlled randomness.

        Args:
            n_samples: Number of samples in dataset
            in_dim: Input feature dimension
            n_classes: Number of output classes
            batch_size: Batch size for DataLoader
            seed: Random seed for reproducibility

        Returns:
            DataLoader with synthetic data
        """
        # Use manual seed for reproducibility
        g = torch.Generator().manual_seed(seed)

        # Generate synthetic features and labels
        X = torch.randn(n_samples, in_dim, generator=g)
        y = torch.randint(0, n_classes, (n_samples,), generator=g)

        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),  # For consistent shuffling
        )

    return _make_loader


@pytest.fixture
def synthetic_train_val_loaders(make_loader) -> tuple:
    """
    Create a pair of train/validation DataLoaders for testing.

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = make_loader(n_samples=128, seed=42)
    val_loader = make_loader(n_samples=32, seed=123)  # Different seed for val
    return train_loader, val_loader


@pytest.fixture
def random_labels_loader(make_loader) -> DataLoader:
    """
    Create a DataLoader with random labels (for leakage detection).

    Returns:
        DataLoader with random labels that should yield ~chance performance
    """
    return make_loader(n_samples=96, seed=999)  # Different seed for random behavior
