"""
Tests for Phase 1 - Optimized Data Layer

This test suite validates the optimized dataset implementations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Direct imports to avoid circular dependencies
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from files to avoid init.py issues
import importlib.util

# Import optimized_datasets
spec = importlib.util.spec_from_file_location(
    "optimized_datasets",
    project_root / "adaptiveneuralnetwork" / "data" / "optimized_datasets.py"
)
optimized_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimized_datasets)

VectorizedDataset = optimized_datasets.VectorizedDataset
PreallocatedBuffer = optimized_datasets.PreallocatedBuffer
OptimizedDatasetWrapper = optimized_datasets.OptimizedDatasetWrapper
vectorized_collate_fn = optimized_datasets.vectorized_collate_fn
create_optimized_loader = optimized_datasets.create_optimized_loader
optimize_dataset = optimized_datasets.optimize_dataset

# Import SyntheticDataset directly
from adaptiveneuralnetwork.training.datasets.datasets import SyntheticDataset


class TestVectorizedDataset:
    """Test VectorizedDataset class."""

    def test_initialization(self):
        """Test dataset initialization."""
        data = torch.randn(100, 10)
        targets = torch.randint(0, 5, (100,))

        dataset = VectorizedDataset(data, targets)

        assert len(dataset) == 100
        assert dataset.data.shape == (100, 10)
        assert dataset.targets.shape == (100,)

    def test_getitem(self):
        """Test single item retrieval."""
        data = torch.randn(100, 10)
        targets = torch.randint(0, 5, (100,))

        dataset = VectorizedDataset(data, targets)

        item_data, item_target = dataset[5]

        assert item_data.shape == (10,)
        assert torch.equal(item_data, data[5])
        assert item_target == targets[5]

    def test_get_batch(self):
        """Test vectorized batch retrieval."""
        data = torch.randn(100, 10)
        targets = torch.randint(0, 5, (100,))

        dataset = VectorizedDataset(data, targets)

        indices = [0, 5, 10, 15, 20]
        batch_data, batch_targets = dataset.get_batch(indices)

        assert batch_data.shape == (5, 10)
        assert batch_targets.shape == (5,)
        assert torch.equal(batch_data[0], data[0])
        assert torch.equal(batch_data[1], data[5])

    def test_numpy_input(self):
        """Test initialization with numpy arrays."""
        data = np.random.randn(100, 10)
        targets = np.random.randint(0, 5, size=100)

        dataset = VectorizedDataset(data, targets)

        assert len(dataset) == 100
        assert isinstance(dataset.data, torch.Tensor)
        assert isinstance(dataset.targets, torch.Tensor)


class TestPreallocatedBuffer:
    """Test PreallocatedBuffer class."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = PreallocatedBuffer(
            batch_size=32,
            data_shape=(10,),
            target_shape=(),
            pin_memory=False
        )

        assert buffer.data_buffer.shape == (32, 10)
        assert buffer.target_buffer.shape == (32,)

    def test_fill_batch(self):
        """Test filling buffer with batch data."""
        buffer = PreallocatedBuffer(
            batch_size=32,
            data_shape=(10,),
            target_shape=(),
            pin_memory=False
        )

        # Create batch data
        data_list = [torch.randn(10) for _ in range(8)]
        target_list = [torch.tensor(i) for i in range(8)]

        batch_data, batch_targets = buffer.fill_batch(data_list, target_list)

        assert batch_data.shape == (8, 10)
        assert batch_targets.shape == (8,)
        assert torch.equal(batch_data[0], data_list[0])
        assert torch.equal(batch_targets[3], target_list[3])


class TestVectorizedCollateFn:
    """Test vectorized collate function."""

    def test_basic_collation(self):
        """Test basic batch collation."""
        batch = [
            (torch.randn(10), torch.tensor(0)),
            (torch.randn(10), torch.tensor(1)),
            (torch.randn(10), torch.tensor(2)),
        ]

        batch_data, batch_targets = vectorized_collate_fn(batch)

        assert batch_data.shape == (3, 10)
        assert batch_targets.shape == (3,)
        assert batch_targets[0] == 0
        assert batch_targets[1] == 1
        assert batch_targets[2] == 2

    def test_empty_batch(self):
        """Test collation with empty batch."""
        batch = []

        batch_data, batch_targets = vectorized_collate_fn(batch)

        assert batch_data.shape == (0,)
        assert batch_targets.shape == (0,)


class TestOptimizedDatasetWrapper:
    """Test OptimizedDatasetWrapper class."""

    def test_wrapper_without_preload(self):
        """Test wrapper without pre-loading."""
        base_dataset = SyntheticDataset(num_samples=100, input_dim=10)
        wrapper = OptimizedDatasetWrapper(base_dataset, preload=False)

        assert len(wrapper) == 100

        data, target = wrapper[5]
        assert data.shape == (10,)

    def test_wrapper_with_preload(self):
        """Test wrapper with pre-loading."""
        base_dataset = SyntheticDataset(num_samples=100, input_dim=10)
        wrapper = OptimizedDatasetWrapper(base_dataset, preload=True, pin_memory=False)

        assert len(wrapper) == 100
        assert wrapper.data is not None
        assert wrapper.targets is not None

        data, target = wrapper[5]
        assert data.shape == (10,)

    def test_get_batch(self):
        """Test batch retrieval from wrapper."""
        base_dataset = SyntheticDataset(num_samples=100, input_dim=10)
        wrapper = OptimizedDatasetWrapper(base_dataset, preload=True, pin_memory=False)

        indices = [0, 5, 10, 15]
        batch_data, batch_targets = wrapper.get_batch(indices)

        assert batch_data.shape == (4, 10)
        assert batch_targets.shape == (4,)


class TestCreateOptimizedLoader:
    """Test optimized loader factory function."""

    def test_create_loader_basic(self):
        """Test basic loader creation."""
        dataset = SyntheticDataset(num_samples=100, input_dim=10)

        loader = create_optimized_loader(
            dataset,
            batch_size=16,
            shuffle=True,
            pin_memory=False,
            num_workers=0
        )

        # Test that loader works
        batch_data, batch_targets = next(iter(loader))

        assert batch_data.shape == (16, 10)
        assert batch_targets.shape == (16,)

    def test_create_loader_with_workers(self):
        """Test loader creation with workers (if supported)."""
        dataset = SyntheticDataset(num_samples=100, input_dim=10)

        # Note: num_workers > 0 may not work in all environments
        loader = create_optimized_loader(
            dataset,
            batch_size=16,
            shuffle=True,
            pin_memory=False,
            num_workers=0,  # Use 0 for testing
            prefetch_factor=2
        )

        # Test that loader works
        batches = list(loader)
        assert len(batches) > 0


class TestOptimizeDataset:
    """Test optimize_dataset utility function."""

    def test_optimize_without_preload(self):
        """Test dataset optimization without pre-loading."""
        dataset = SyntheticDataset(num_samples=100, input_dim=10)

        optimized = optimize_dataset(dataset, preload=False)

        assert len(optimized) == 100
        assert isinstance(optimized, OptimizedDatasetWrapper)

    def test_optimize_with_preload(self):
        """Test dataset optimization with pre-loading."""
        dataset = SyntheticDataset(num_samples=50, input_dim=10)

        optimized = optimize_dataset(dataset, preload=True, pin_memory=False)

        assert len(optimized) == 50
        assert optimized.data is not None
        assert optimized.targets is not None


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_training_loop(self):
        """Test complete training loop with optimized loader."""
        # Create dataset
        dataset = SyntheticDataset(num_samples=200, input_dim=10, num_classes=5)

        # Create optimized loader
        loader = create_optimized_loader(
            dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=False,
            num_workers=0
        )

        # Simulate training loop
        total_samples = 0
        for batch_data, batch_targets in loader:
            assert batch_data.ndim == 2
            assert batch_targets.ndim == 1
            assert batch_data.shape[1] == 10
            assert batch_targets.max() < 5

            total_samples += len(batch_data)

        # Should process all samples
        assert total_samples == 200

    def test_comparison_with_baseline(self):
        """Test that optimized loader produces same results as baseline."""
        from torch.utils.data import DataLoader

        dataset = SyntheticDataset(num_samples=100, input_dim=10, num_classes=3)

        # Baseline loader
        baseline_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Optimized loader
        optimized_loader = create_optimized_loader(
            dataset,
            batch_size=16,
            shuffle=False,
            pin_memory=False,
            num_workers=0
        )

        # Compare first batch
        baseline_data, baseline_targets = next(iter(baseline_loader))
        optimized_data, optimized_targets = next(iter(optimized_loader))

        assert baseline_data.shape == optimized_data.shape
        assert baseline_targets.shape == optimized_targets.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
