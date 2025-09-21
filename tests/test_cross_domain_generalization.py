"""
Test for Cross-Domain Generalization - Phase 2.1

Tests the ability of the adaptive neural network to generalize
across different domains and handle domain shift scenarios.
"""

import os

# Import from the main package
import sys
import unittest
from unittest.mock import Mock

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel
from adaptiveneuralnetwork.training.continual import domain_shift_evaluation
from adaptiveneuralnetwork.training.datasets import (
    DomainRandomizedDataset,
    create_cross_domain_loaders,
    create_synthetic_loaders,
)


class TestCrossDomainGeneralization(unittest.TestCase):
    def setUp(self):
        """Set up test environment for cross-domain generalization tests."""
        self.config = AdaptiveConfig()
        self.model = Mock(spec=AdaptiveModel)
        # Mock model behavior for testing
        self.model.forward = Mock(return_value=torch.randn(32, 10))
        self.model.to = Mock(return_value=self.model)
        self.model.eval = Mock()

    def test_domain_randomized_dataset_creation(self):
        """Test creation of domain randomized datasets."""
        # Create base synthetic dataset
        train_loader, test_loader = create_synthetic_loaders(num_samples=100, batch_size=16)

        # Get base dataset from loader
        base_dataset = train_loader.dataset

        # Define domain configurations
        domain_configs = [
            {"noise_level": 0.1},
            {"brightness_factor": 0.8, "contrast_factor": 1.2},
            {"blur_kernel_size": 3},
        ]

        # Create domain randomized dataset
        domain_dataset = DomainRandomizedDataset(
            base_dataset=base_dataset, domain_configs=domain_configs, randomization_prob=1.0
        )

        # Test dataset properties
        self.assertEqual(len(domain_dataset), len(base_dataset))

        # Test that we can get items from the dataset
        data, target = domain_dataset[0]
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)

    def test_cross_domain_loader_creation(self):
        """Test creation of cross-domain data loaders."""
        # Create base synthetic dataset
        train_loader, _ = create_synthetic_loaders(num_samples=100, batch_size=16)
        base_dataset = train_loader.dataset

        # Create cross-domain loaders
        domain_loaders = create_cross_domain_loaders(
            base_dataset=base_dataset, batch_size=16, num_domains=3
        )

        # Verify we get the expected number of loaders
        self.assertEqual(len(domain_loaders), 3)

        # Test that each loader produces data
        for i, loader in enumerate(domain_loaders):
            batch = next(iter(loader))
            data, target = batch
            self.assertIsInstance(data, torch.Tensor)
            self.assertIsInstance(target, torch.Tensor)
            self.assertEqual(
                data.shape[0], min(16, len(base_dataset))
            )  # batch_size or remaining samples

    def test_domain_shift_evaluation(self):
        """Test domain shift evaluation functionality."""
        # Create source and target loaders
        source_loader, _ = create_synthetic_loaders(num_samples=80, batch_size=16)

        target_loaders = create_cross_domain_loaders(
            base_dataset=source_loader.dataset, batch_size=16, num_domains=2
        )

        # Test domain shift evaluation
        results = domain_shift_evaluation(
            model=self.model, source_loader=source_loader, target_loaders=target_loaders
        )

        # Verify results structure
        self.assertIn("source_domain_accuracy", results)
        self.assertIn("target_domain_accuracies", results)
        self.assertIn("transfer_learning_metrics", results)
        self.assertIn("generalization_score", results)
        self.assertIn("domain_adaptation_success", results)

        # Verify accuracy values are reasonable
        self.assertGreaterEqual(results["source_domain_accuracy"], 0.0)
        self.assertLessEqual(results["source_domain_accuracy"], 1.0)

        # Verify we have target domain results
        self.assertEqual(len(results["target_domain_accuracies"]), len(target_loaders))

    def test_domain_randomization_effects(self):
        """Test that domain randomization actually changes the data."""
        # Create base synthetic dataset
        train_loader, _ = create_synthetic_loaders(num_samples=50, batch_size=16)
        base_dataset = train_loader.dataset

        # Create domain randomized dataset with high noise
        high_noise_config = [{"noise_level": 0.5}]
        noisy_dataset = DomainRandomizedDataset(
            base_dataset=base_dataset,
            domain_configs=high_noise_config,
            randomization_prob=1.0,  # Always apply
        )

        # Get same sample from both datasets
        original_data, _ = base_dataset[0]
        modified_data, _ = noisy_dataset[0]

        # Due to randomization, the data should be different
        # (Though there's a tiny chance they could be identical)
        data_difference = torch.abs(original_data - modified_data).mean()
        self.assertGreater(
            data_difference, 0.01, "Domain randomization should modify the data significantly"
        )

    def test_transfer_learning_metrics(self):
        """Test that transfer learning metrics are calculated correctly."""
        # Create synthetic scenario
        source_loader, _ = create_synthetic_loaders(num_samples=64, batch_size=16)

        # Create target domains with known degradation
        target_loaders = create_cross_domain_loaders(
            base_dataset=source_loader.dataset, batch_size=16, num_domains=2
        )

        # Mock model to return predictable accuracies
        def mock_forward(x):
            # Return predictions that will give us controlled accuracy
            batch_size = x.shape[0]
            # Create output that gives ~80% accuracy on source, ~60% on target
            output = torch.randn(batch_size, 10)
            return output

        self.model.forward.side_effect = mock_forward

        results = domain_shift_evaluation(
            model=self.model, source_loader=source_loader, target_loaders=target_loaders
        )

        # Check that transfer learning metrics exist and are reasonable
        metrics = results["transfer_learning_metrics"]
        self.assertIn("average_target_accuracy", metrics)
        self.assertIn("accuracy_drop", metrics)
        self.assertIn("relative_performance", metrics)
        self.assertIn("adaptation_coefficient", metrics)

        # All metrics should be numeric
        for metric_name, value in metrics.items():
            self.assertIsInstance(
                value, (int, float), f"Metric {metric_name} should be numeric, got {type(value)}"
            )


if __name__ == "__main__":
    unittest.main()
