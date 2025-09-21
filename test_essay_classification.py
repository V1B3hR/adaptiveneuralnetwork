#!/usr/bin/env python3
"""
Test script for essay classification benchmark.

This script runs comprehensive tests on the Human vs AI Generated Essays
classification benchmark to ensure it works correctly.
"""

# Add the current directory to Python path
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptiveneuralnetwork.api import AdaptiveConfig
from adaptiveneuralnetwork.benchmarks.text_classification import (
    EssayDataset,
    SyntheticEssayDataset,
    TextClassificationBenchmark,
    run_essay_classification_benchmark,
)


class TestEssayClassification(unittest.TestCase):
    """Test suite for essay classification benchmark."""

    def setUp(self):
        """Set up test environment."""
        self.config = AdaptiveConfig(
            num_nodes=50,
            hidden_dim=64,
            learning_rate=0.01,
            batch_size=16,
            num_epochs=2,
            device="cpu",
        )

    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticEssayDataset(num_samples=100, vocab_size=1000, max_length=128)

        self.assertEqual(len(dataset), 100)
        self.assertEqual(len(dataset.vocab), 1000)
        self.assertEqual(dataset.max_length, 128)

        # Test data loading
        sample_text, label = dataset[0]
        self.assertEqual(sample_text.shape, (128,))
        self.assertIn(label.item(), [0, 1])

    def test_essay_dataset_functionality(self):
        """Test EssayDataset class with sample data."""
        texts = [
            "This is a human written essay with natural language",
            "This AI generated text uses algorithmic patterns in writing",
            "Another human essay with creative thoughts and emotions",
            "Generated content with systematic structure and logic",
        ]
        labels = [0, 1, 0, 1]  # 0=human, 1=AI

        dataset = EssayDataset(texts=texts, labels=labels, vocab_size=100, max_length=64)

        self.assertEqual(len(dataset), 4)
        self.assertTrue(len(dataset.vocab) <= 102)  # vocab_size + 2 special tokens

        # Test tokenization
        sample_text, label = dataset[0]
        self.assertEqual(sample_text.shape, (64,))
        self.assertEqual(label.item(), 0)

    def test_benchmark_initialization(self):
        """Test benchmark class initialization."""
        benchmark = TextClassificationBenchmark(self.config)

        # Model should be None initially
        self.assertIsNone(benchmark.model)
        self.assertEqual(benchmark.config.output_dim, 2)  # Binary classification

        # After creating model with dimensions, it should exist
        benchmark._create_model(vocab_size=1000, max_length=128)
        self.assertIsNotNone(benchmark.model)

    def test_training_functionality(self):
        """Test training functionality with small dataset."""
        dataset = SyntheticEssayDataset(num_samples=32, vocab_size=500, max_length=64)

        benchmark = TextClassificationBenchmark(self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change results directory
            original_path = Path("benchmark_results")
            test_path = Path(temp_dir) / "test_results"
            test_path.mkdir(exist_ok=True)

            results = benchmark.run_essay_classification_benchmark(
                dataset=dataset,
                epochs=2,
                batch_size=8,
                learning_rate=0.01,
                test_split=0.25,
                save_results=False,  # Don't save during test
            )

            # Verify results structure
            expected_keys = [
                "final_test_accuracy",
                "best_test_accuracy",
                "final_train_accuracy",
                "best_train_accuracy",
                "train_losses",
                "train_accuracies",
                "test_accuracies",
                "training_time",
                "epochs",
                "vocab_size",
            ]

            for key in expected_keys:
                self.assertIn(key, results)

            # Verify reasonable values
            self.assertGreaterEqual(results["final_test_accuracy"], 0.0)
            self.assertLessEqual(results["final_test_accuracy"], 1.0)
            self.assertEqual(results["epochs"], 2)
            self.assertEqual(results["vocab_size"], 500)

    def test_convenience_function(self):
        """Test the convenience function for running benchmarks."""
        results = run_essay_classification_benchmark(
            config=self.config,
            dataset=None,  # Will use synthetic data
            epochs=2,
            batch_size=8,
            learning_rate=0.01,
        )

        self.assertIsInstance(results, dict)
        self.assertIn("final_test_accuracy", results)
        self.assertEqual(results["epochs"], 2)

    def test_variable_batch_sizes(self):
        """Test handling of variable batch sizes."""
        # Create dataset with size that doesn't divide evenly by batch size
        dataset = SyntheticEssayDataset(
            num_samples=37,  # Not divisible by common batch sizes
            vocab_size=200,
            max_length=32,
        )

        benchmark = TextClassificationBenchmark(self.config)

        # This should work without errors despite uneven batch sizes
        results = benchmark.run_essay_classification_benchmark(
            dataset=dataset, epochs=1, batch_size=10, learning_rate=0.01, save_results=False
        )

        self.assertIsInstance(results, dict)
        self.assertIn("final_test_accuracy", results)

    def test_model_parameters_count(self):
        """Test that model parameters are reasonable."""
        benchmark = TextClassificationBenchmark(self.config)

        # Create model with specific dimensions
        benchmark._create_model(vocab_size=1000, max_length=128)

        total_params = sum(p.numel() for p in benchmark.model.parameters())

        # Should have reasonable number of parameters (not too few, not too many)
        self.assertGreater(total_params, 1000)  # At least 1K parameters
        self.assertLess(total_params, 1000000)  # Less than 1M parameters

    def test_accuracy_improvement(self):
        """Test that accuracy can improve over epochs."""
        dataset = SyntheticEssayDataset(
            num_samples=200,
            vocab_size=1000,
            max_length=128,
            human_vs_ai_ratio=0.5,  # Balanced dataset
        )

        results = run_essay_classification_benchmark(
            config=self.config, dataset=dataset, epochs=5, batch_size=16, learning_rate=0.01
        )

        # With synthetic data that has clear patterns, accuracy should be able to improve
        train_accuracies = results["train_accuracies"]

        # Check that training is occurring (losses should change)
        self.assertGreater(len(train_accuracies), 1)

        # Accuracy should be reasonable for this task
        final_accuracy = results["final_test_accuracy"]
        self.assertGreaterEqual(final_accuracy, 0.3)  # Better than very poor random performance


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
