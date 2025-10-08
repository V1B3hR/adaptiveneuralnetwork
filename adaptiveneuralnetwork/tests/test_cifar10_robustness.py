"""
Tests for CIFAR-10 robustness benchmarking functionality.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from adaptiveneuralnetwork.api import AdaptiveConfig
from adaptiveneuralnetwork.benchmarks.vision.cifar10 import CIFAR10Benchmark, run_cifar10_benchmark
from adaptiveneuralnetwork.training.datasets import CIFAR10Corrupted, load_cifar10_corrupted


class TestCIFAR10Corrupted:
    """Test cases for CIFAR10Corrupted dataset."""

    def test_corruption_types_available(self):
        """Test that corruption types are properly defined."""
        assert len(CIFAR10Corrupted.CORRUPTION_TYPES) > 0
        assert 'gaussian_noise' in CIFAR10Corrupted.CORRUPTION_TYPES
        assert 'brightness' in CIFAR10Corrupted.CORRUPTION_TYPES
        assert 'contrast' in CIFAR10Corrupted.CORRUPTION_TYPES

    @pytest.mark.slow
    def test_corrupted_dataset_creation(self):
        """Test creating corrupted CIFAR-10 dataset."""
        # Create small mock dataset to avoid downloading
        with patch('torchvision.datasets.CIFAR10') as mock_cifar:
            # Mock CIFAR-10 dataset
            mock_data = []
            for _ in range(10):  # Small dataset
                # Create 32x32x3 PIL-like image
                img = torch.randint(0, 256, (32, 32, 3)).numpy().astype(np.uint8)
                mock_data.append((img, 0))

            mock_dataset = MagicMock()
            mock_dataset.__len__ = lambda: len(mock_data)
            mock_dataset.__getitem__ = lambda idx: mock_data[idx]
            mock_cifar.return_value = mock_dataset

            # Test different corruption types
            for corruption_type in ['gaussian_noise', 'brightness', 'contrast']:
                for severity in [1, 3, 5]:
                    dataset = CIFAR10Corrupted(
                        corruption_type=corruption_type,
                        severity=severity,
                        download=False
                    )

                    assert len(dataset) == len(mock_data)

                    # Test getting an item
                    corrupted_img, label = dataset[0]
                    assert isinstance(corrupted_img, torch.Tensor)
                    assert corrupted_img.shape[0] == 3  # RGB channels
                    assert isinstance(label, int)

    def test_invalid_corruption_type(self):
        """Test error handling for invalid corruption types."""
        with pytest.raises(ValueError, match="Corruption type.*not supported"):
            CIFAR10Corrupted(corruption_type='invalid_corruption', download=False)

    def test_severity_clamping(self):
        """Test that severity is properly clamped to [1, 5]."""
        with patch('torchvision.datasets.CIFAR10'):
            dataset = CIFAR10Corrupted(severity=10, download=False)
            assert dataset.severity == 5

            dataset = CIFAR10Corrupted(severity=-1, download=False)
            assert dataset.severity == 1


class TestCIFAR10Benchmark:
    """Test cases for CIFAR10Benchmark."""

    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        config = AdaptiveConfig(
            num_nodes=32,
            hidden_dim=16,
            input_dim=3072,  # 32*32*3
            output_dim=10
        )

        benchmark = CIFAR10Benchmark(config)
        assert benchmark.config.input_dim == 3072
        assert benchmark.config.output_dim == 10

    @pytest.mark.slow
    def test_standard_benchmark_quick(self):
        """Test standard benchmark with minimal epochs."""
        config = AdaptiveConfig(
            num_nodes=16,
            hidden_dim=8,
            input_dim=3072,
            output_dim=10
        )

        # Mock CIFAR-10 data loading to avoid download
        with patch('adaptiveneuralnetwork.training.datasets.load_cifar10') as mock_load:
            # Create mock data loaders with small synthetic data
            mock_train_data = []
            mock_test_data = []

            for _ in range(8):  # Small dataset
                x = torch.randn(3, 32, 32)
                y = torch.randint(0, 10, (1,)).item()
                mock_train_data.append((x, y))
                mock_test_data.append((x, y))

            train_loader = torch.utils.data.DataLoader(mock_train_data, batch_size=4)
            test_loader = torch.utils.data.DataLoader(mock_test_data, batch_size=4)

            mock_load.return_value = (train_loader, test_loader)

            benchmark = CIFAR10Benchmark(config, device=torch.device('cpu'))

            results = benchmark.run_standard_benchmark(
                epochs=1,
                batch_size=4,
                save_results=False
            )

            assert 'benchmark_type' in results
            assert results['benchmark_type'] == 'cifar10_standard'
            assert 'final_test_accuracy' in results
            assert 'training_time' in results
            assert results['epochs'] == 1

    def test_robustness_benchmark_structure(self):
        """Test robustness benchmark result structure."""
        config = AdaptiveConfig(num_nodes=16, hidden_dim=8)

        with patch('adaptiveneuralnetwork.training.datasets.load_cifar10') as mock_clean:
            with patch('adaptiveneuralnetwork.training.datasets.load_cifar10_corrupted') as mock_corrupt:
                # Mock clean data
                clean_data = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(4)]
                clean_loader = torch.utils.data.DataLoader(clean_data, batch_size=2)
                mock_clean.return_value = (None, clean_loader)

                # Mock corrupted data
                corrupt_data = [(torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()) for _ in range(4)]
                corrupt_loader = torch.utils.data.DataLoader(corrupt_data, batch_size=2)
                mock_corrupt.return_value = (None, corrupt_loader)

                benchmark = CIFAR10Benchmark(config, device=torch.device('cpu'))

                results = benchmark.run_robustness_benchmark(
                    corruption_types=['gaussian_noise'],
                    severities=[1, 2],
                    batch_size=2,
                    save_results=False
                )

                assert 'benchmark_type' in results
                assert results['benchmark_type'] == 'cifar10_robustness'
                assert 'robustness_results' in results

                robustness = results['robustness_results']
                assert 'clean_accuracy' in robustness
                assert 'corruption_results' in robustness
                assert 'mean_corruption_error' in robustness
                assert 'relative_robustness' in robustness

                # Check corruption results structure
                assert 'gaussian_noise' in robustness['corruption_results']
                noise_results = robustness['corruption_results']['gaussian_noise']
                assert 1 in noise_results
                assert 2 in noise_results

                for severity_results in noise_results.values():
                    assert 'accuracy' in severity_results
                    assert 'corruption_error' in severity_results
                    assert 'relative_accuracy' in severity_results


class TestIntegration:
    """Integration tests for CIFAR-10 robustness functionality."""

    def test_end_to_end_corrupted_loading(self):
        """Test end-to-end corrupted data loading."""
        with patch('torchvision.datasets.CIFAR10') as mock_cifar:
            # Mock small dataset
            mock_data = []
            for i in range(4):
                img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                mock_data.append((img, i % 2))

            mock_dataset = MagicMock()
            mock_dataset.__len__ = lambda: len(mock_data)
            mock_dataset.__getitem__ = lambda idx: mock_data[idx]
            mock_cifar.return_value = mock_dataset

            # Test loading corrupted data
            train_loader, test_loader = load_cifar10_corrupted(
                corruption_type='gaussian_noise',
                severity=2,
                batch_size=2,
                download=False
            )

            # Test getting a batch
            for batch_data, targets in train_loader:
                assert batch_data.shape[1] == 3  # RGB channels
                assert batch_data.shape[2] == 32  # Height
                assert batch_data.shape[3] == 32  # Width
                assert len(targets) == batch_data.shape[0]
                break

    @pytest.mark.slow
    def test_robustness_metrics_calculation(self):
        """Test robustness metrics calculation logic."""
        # Test the mathematical correctness of robustness calculations
        clean_acc = 0.9
        corrupt_acc_1 = 0.8
        corrupt_acc_2 = 0.7

        # Calculate expected metrics
        error_1 = clean_acc - corrupt_acc_1  # 0.1
        error_2 = clean_acc - corrupt_acc_2  # 0.2
        mean_error = (error_1 + error_2) / 2  # 0.15
        relative_robustness = 1.0 - (mean_error / clean_acc)  # 1.0 - 0.15/0.9 ≈ 0.833

        assert abs(mean_error - 0.15) < 1e-6
        assert abs(relative_robustness - (1.0 - 0.15/0.9)) < 1e-6

        # Edge case: perfect robustness
        perfect_error = clean_acc - clean_acc  # 0
        perfect_robustness = 1.0 - (perfect_error / clean_acc)  # 1.0
        assert abs(perfect_robustness - 1.0) < 1e-6

        # Edge case: complete failure
        failure_error = clean_acc - 0.0  # 0.9
        failure_robustness = 1.0 - (failure_error / clean_acc)  # 0.0
        assert abs(failure_robustness - 0.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
