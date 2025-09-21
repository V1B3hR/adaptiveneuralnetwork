"""
CIFAR-10 benchmark for adaptive neural networks with domain shift robustness testing.

This module provides CIFAR-10 benchmarking capabilities including:
- Standard CIFAR-10 classification
- Corrupted CIFAR-10 for domain shift robustness testing
- Robustness metrics and evaluation
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from ...api import AdaptiveConfig, AdaptiveModel
from ...training.datasets import load_cifar10, load_cifar10_corrupted
from ...training.loops import evaluate_model, train_epoch

logger = logging.getLogger(__name__)


class CIFAR10Benchmark:
    """
    CIFAR-10 benchmark for adaptive neural networks.

    Supports both standard and corrupted CIFAR-10 evaluation for robustness testing.
    """

    def __init__(
        self,
        config: AdaptiveConfig,
        device: Optional[torch.device] = None,
        data_root: str = "./data",
    ):
        """
        Initialize CIFAR-10 benchmark.

        Args:
            config: Adaptive neural network configuration
            device: Device to run on (CPU/GPU)
            data_root: Root directory for CIFAR-10 data
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root

        # Update config for CIFAR-10 (32x32x3 = 3072 input features, 10 classes)
        self.config = self._update_config_for_cifar10(config)

        # Initialize model
        self.model = AdaptiveModel(self.config).to(self.device)

        logger.info(
            f"Initialized CIFAR-10 benchmark with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def _update_config_for_cifar10(self, config: AdaptiveConfig) -> AdaptiveConfig:
        """Update configuration for CIFAR-10 dataset."""
        # Create a copy to avoid modifying original
        new_config = AdaptiveConfig(**config.to_dict())
        new_config.input_dim = 3072  # 32x32x3 flattened
        new_config.output_dim = 10  # 10 classes
        return new_config

    def run_standard_benchmark(
        self,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        num_workers: int = 0,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run standard CIFAR-10 benchmark.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_workers: Number of data loading workers
            save_results: Whether to save results to file

        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Starting standard CIFAR-10 benchmark...")

        # Load data
        train_loader, test_loader = load_cifar10(
            batch_size=batch_size, root=self.data_root, num_workers=num_workers
        )

        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training metrics
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        training_time = 0.0

        # Training loop
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train epoch
            train_loss, train_acc = train_epoch(
                self.model, train_loader, optimizer, criterion, self.device
            )

            # Evaluate
            test_acc = evaluate_model(self.model, test_loader, self.device)

            epoch_time = time.time() - epoch_start
            training_time += epoch_time

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Test Acc: {test_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

        total_time = time.time() - start_time

        # Compile results
        results = {
            "benchmark_type": "cifar10_standard",
            "final_train_accuracy": train_accuracies[-1],
            "final_test_accuracy": test_accuracies[-1],
            "best_test_accuracy": max(test_accuracies),
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "training_time": training_time,
            "total_time": total_time,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "config": self.config.to_dict(),
        }

        if save_results:
            self._save_results(results, "cifar10_standard_results.json")

        logger.info(
            f"Standard CIFAR-10 benchmark completed. "
            f"Final test accuracy: {results['final_test_accuracy']:.4f}"
        )

        return results

    def run_robustness_benchmark(
        self,
        corruption_types: Optional[List[str]] = None,
        severities: Optional[List[int]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pretrained_model_path: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run domain shift robustness benchmark with corrupted CIFAR-10.

        Args:
            corruption_types: List of corruption types to test
            severities: List of severity levels to test
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            pretrained_model_path: Path to pretrained model weights
            save_results: Whether to save results to file

        Returns:
            Dictionary containing robustness benchmark results
        """
        logger.info("Starting CIFAR-10 robustness benchmark...")

        if corruption_types is None:
            corruption_types = [
                "gaussian_noise",
                "brightness",
                "contrast",
                "pixelate",
                "gaussian_blur",
            ]

        if severities is None:
            severities = [1, 2, 3, 4, 5]

        # Load pretrained model if provided
        if pretrained_model_path and Path(pretrained_model_path).exists():
            self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
            logger.info(f"Loaded pretrained model from {pretrained_model_path}")

        # Evaluate on clean test set first
        clean_loader = load_cifar10(
            batch_size=batch_size, root=self.data_root, num_workers=num_workers
        )[1]
        clean_accuracy = evaluate_model(self.model, clean_loader, self.device)

        # Robustness results
        robustness_results = {
            "clean_accuracy": clean_accuracy,
            "corruption_results": {},
            "mean_corruption_error": 0.0,
            "relative_robustness": 0.0,
        }

        total_error = 0.0
        num_evaluations = 0

        # Test each corruption type and severity
        for corruption_type in corruption_types:
            robustness_results["corruption_results"][corruption_type] = {}

            for severity in severities:
                logger.info(f"Testing {corruption_type} at severity {severity}")

                try:
                    # Load corrupted data
                    _, corrupted_loader = load_cifar10_corrupted(
                        corruption_type=corruption_type,
                        severity=severity,
                        batch_size=batch_size,
                        root=self.data_root,
                        num_workers=num_workers,
                    )

                    # Evaluate on corrupted data
                    corrupted_accuracy = evaluate_model(self.model, corrupted_loader, self.device)

                    # Calculate corruption error (difference from clean accuracy)
                    corruption_error = clean_accuracy - corrupted_accuracy
                    total_error += corruption_error
                    num_evaluations += 1

                    robustness_results["corruption_results"][corruption_type][severity] = {
                        "accuracy": corrupted_accuracy,
                        "corruption_error": corruption_error,
                        "relative_accuracy": (
                            corrupted_accuracy / clean_accuracy if clean_accuracy > 0 else 0.0
                        ),
                    }

                    logger.info(
                        f"  Accuracy: {corrupted_accuracy:.4f}, Error: {corruption_error:.4f}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to evaluate {corruption_type} severity {severity}: {e}")
                    robustness_results["corruption_results"][corruption_type][severity] = {
                        "accuracy": 0.0,
                        "corruption_error": clean_accuracy,
                        "relative_accuracy": 0.0,
                        "error": str(e),
                    }

        # Calculate overall robustness metrics
        if num_evaluations > 0:
            robustness_results["mean_corruption_error"] = total_error / num_evaluations
            robustness_results["relative_robustness"] = (
                1.0 - (robustness_results["mean_corruption_error"] / clean_accuracy)
                if clean_accuracy > 0
                else 0.0
            )

        # Compile full results
        results = {
            "benchmark_type": "cifar10_robustness",
            "robustness_results": robustness_results,
            "corruption_types": corruption_types,
            "severities": severities,
            "batch_size": batch_size,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "config": self.config.to_dict(),
        }

        if save_results:
            self._save_results(results, "cifar10_robustness_results.json")

        logger.info(
            f"Robustness benchmark completed. "
            f"Clean accuracy: {clean_accuracy:.4f}, "
            f"Mean corruption error: {robustness_results['mean_corruption_error']:.4f}, "
            f"Relative robustness: {robustness_results['relative_robustness']:.4f}"
        )

        return results

    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save benchmark results to JSON file."""
        import json

        output_path = Path(filename)
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")


def run_cifar10_benchmark(
    config: Optional[AdaptiveConfig] = None,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    test_robustness: bool = True,
    corruption_types: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run complete CIFAR-10 benchmark.

    Args:
        config: Model configuration (uses default if None)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        test_robustness: Whether to run robustness tests
        corruption_types: Corruption types for robustness testing
        device: Device to run on

    Returns:
        Dictionary containing all benchmark results
    """
    # Use default config if not provided
    if config is None:
        config = AdaptiveConfig(
            num_nodes=128, hidden_dim=64, num_epochs=epochs, learning_rate=learning_rate
        )

    # Initialize benchmark
    benchmark = CIFAR10Benchmark(config, device)

    # Run standard benchmark
    standard_results = benchmark.run_standard_benchmark(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, save_results=True
    )

    results = {"standard": standard_results}

    # Run robustness benchmark if requested
    if test_robustness:
        robustness_results = benchmark.run_robustness_benchmark(
            corruption_types=corruption_types, batch_size=batch_size, save_results=True
        )
        results["robustness"] = robustness_results

    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    config = AdaptiveConfig(num_nodes=100, hidden_dim=64, num_epochs=5, learning_rate=0.001)

    results = run_cifar10_benchmark(
        config=config,
        epochs=5,
        batch_size=32,
        test_robustness=True,
        corruption_types=["gaussian_noise", "brightness"],
    )

    print(f"CIFAR-10 Standard Accuracy: {results['standard']['final_test_accuracy']:.4f}")
    if "robustness" in results:
        print(
            f"CIFAR-10 Robustness: {results['robustness']['robustness_results']['relative_robustness']:.4f}"
        )
