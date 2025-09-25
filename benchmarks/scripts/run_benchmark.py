#!/usr/bin/env python3
"""
Main benchmark script for adaptive neural networks.

This script provides a command-line interface for running benchmarks
on various datasets with configurable parameters.
"""

import argparse
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.benchmarks.vision.mnist import run_mnist_benchmark, quick_mnist_test


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for adaptive neural networks"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=["mnist"],
        default="mnist",
        help="Dataset to use for benchmarking"
    )
    
    # Model configuration
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=100,
        help="Number of adaptive nodes"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension size"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for training"
    )
    
    # Testing options
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with subset of data"
    )
    
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Use subset of training data (for testing)"
    )
    
    # Output options
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main():
    """Main benchmark execution."""
    args = parse_arguments()
    
    # Determine device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"Running {args.dataset.upper()} benchmark on {device}")
    
    # Create configuration
    config = AdaptiveConfig(
        num_nodes=args.num_nodes,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=args.seed,
        save_checkpoint=not args.no_save,
        metrics_file=args.output_file
    )
    
    # Run appropriate benchmark
    if args.dataset == "mnist":
        if args.quick_test:
            print("Running quick MNIST test...")
            results = quick_mnist_test(
                num_epochs=min(args.epochs, 2),
                subset_size=args.subset_size or 1000
            )
        else:
            print("Running full MNIST benchmark...")
            results = run_mnist_benchmark(
                config=config,
                subset_size=args.subset_size,
                save_results=not args.no_save
            )
    
    # Print final summary
    final_metrics = results.get('final_metrics', {})
    print("\n" + "="*50)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("="*50)
    
    if final_metrics:
        print(f"Final Results:")
        print(f"  Train Accuracy: {final_metrics.get('train_accuracy', 'N/A'):.2f}%")
        print(f"  Test Accuracy:  {final_metrics.get('val_accuracy', 'N/A'):.2f}%")
        print(f"  Active Nodes:   {final_metrics.get('active_node_ratio', 'N/A'):.3f}")
        print(f"  Mean Energy:    {final_metrics.get('mean_energy', 'N/A'):.3f}")
        print(f"  Training Time:  {results.get('benchmark_info', {}).get('total_training_time', 'N/A'):.2f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())