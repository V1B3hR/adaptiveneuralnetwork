"""
MNIST benchmark for adaptive neural networks.

This module provides a complete MNIST benchmark including training,
evaluation, and metrics collection.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from ...api.config import AdaptiveConfig
from ...api.model import AdaptiveModel
from ...training.datasets import load_mnist, load_mnist_subset
from ...training.loops import TrainingLoop


def run_mnist_benchmark(
    config: Optional[AdaptiveConfig] = None,
    subset_size: Optional[int] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run complete MNIST benchmark.
    
    Args:
        config: Model configuration (uses default if None)
        subset_size: Use subset of MNIST for faster testing (None for full dataset)
        save_results: Whether to save results to file
        
    Returns:
        Benchmark results dictionary
    """
    # Use default config if none provided
    if config is None:
        config = AdaptiveConfig(
            num_nodes=100,
            hidden_dim=64,
            num_epochs=5,
            batch_size=128,
            learning_rate=0.001,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    print("=== MNIST Benchmark for Adaptive Neural Networks ===")
    print(f"Configuration: {config.num_nodes} nodes, {config.hidden_dim} hidden dim")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load dataset
    print("\nLoading MNIST dataset...")
    if subset_size is not None:
        train_loader, test_loader = load_mnist_subset(
            batch_size=config.batch_size,
            subset_size=subset_size
        )
        print(f"Using subset: {subset_size} training samples")
    else:
        train_loader, test_loader = load_mnist(batch_size=config.batch_size)
        print(f"Using full dataset: {len(train_loader.dataset)} training samples")
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing adaptive neural network...")
    model = AdaptiveModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create training loop
    trainer = TrainingLoop(model, config)
    
    # Record start time
    start_time = time.time()
    
    # Run training
    print("\nStarting training...")
    metrics_history = trainer.train(train_loader, test_loader)
    
    # Record end time
    total_time = time.time() - start_time
    
    # Get final results
    final_metrics = metrics_history[-1] if metrics_history else {}
    
    # Compile benchmark results
    results = {
        'benchmark_info': {
            'dataset': 'MNIST',
            'model_type': 'AdaptiveNeuralNetwork',
            'timestamp': time.time(),
            'total_training_time': total_time,
            'config': config.to_dict()
        },
        'final_metrics': final_metrics,
        'training_history': metrics_history,
        'model_summary': {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    # Print summary
    print(f"\n=== Benchmark Results ===")
    print(f"Total training time: {total_time:.2f} seconds")
    if final_metrics:
        print(f"Final train accuracy: {final_metrics.get('train_accuracy', 'N/A'):.2f}%")
        print(f"Final test accuracy: {final_metrics.get('val_accuracy', 'N/A'):.2f}%")
        print(f"Final active node ratio: {final_metrics.get('active_node_ratio', 'N/A'):.3f}")
        print(f"Final mean energy: {final_metrics.get('mean_energy', 'N/A'):.3f}")
    
    # Save results if requested
    if save_results:
        results_file = f"mnist_benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return results


def quick_mnist_test(num_epochs: int = 1, subset_size: int = 1000) -> Dict[str, Any]:
    """
    Quick MNIST test for development and CI.
    
    Args:
        num_epochs: Number of epochs to train
        subset_size: Size of dataset subset
        
    Returns:
        Test results
    """
    config = AdaptiveConfig(
        num_nodes=50,  # Smaller for quick test
        hidden_dim=32,
        num_epochs=num_epochs,
        batch_size=64,
        learning_rate=0.01,
        save_checkpoint=False,
        metrics_file="quick_test_metrics.json"
    )
    
    return run_mnist_benchmark(
        config=config,
        subset_size=subset_size,
        save_results=True
    )


if __name__ == "__main__":
    # Run full benchmark when script is executed directly
    results = run_mnist_benchmark()
    print("\nBenchmark completed successfully!")