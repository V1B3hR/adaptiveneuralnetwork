#!/usr/bin/env python3
"""
Training script for ANNOMI Motivational Interviewing Dataset.

This script trains the adaptive neural network on the ANNOMI dataset for 100 epochs
as specified in the problem statement.

Usage:
    python train_annomi.py --data-path /path/to/annomi/dataset

If you don't have the dataset, it will use synthetic data for demonstration.
"""

import argparse
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from adaptiveneuralnetwork.data import print_dataset_info
from run_essay_benchmark import main as run_benchmark


def main():
    """Main function for ANNOMI dataset training."""
    parser = argparse.ArgumentParser(
        description="Train Adaptive Neural Network on ANNOMI Motivational Interviewing Dataset"
    )

    parser.add_argument("--data-path", type=str, help="Path to ANNOMI dataset directory")
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of synthetic samples if no real data provided",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs (default: 100)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension size (default: 128)"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=100, help="Number of adaptive nodes (default: 100)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Print dataset information
    print("=" * 60)
    print("ANNOMI MOTIVATIONAL INTERVIEWING DATASET TRAINING")
    print("=" * 60)
    print_dataset_info()

    if not args.data_path:
        print("\nWARNING: No dataset path provided.")
        print("Using synthetic data for demonstration.")
        print("To use real ANNOMI data:")
        print(
            "1. Download from: https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing"
        )
        print("2. Run: python train_annomi.py --data-path /path/to/dataset")
        print()

    # Construct arguments for the main benchmark script
    benchmark_args = [
        "--dataset-type",
        "annomi",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-nodes",
        str(args.num_nodes),
        "--device",
        args.device,
    ]

    if args.data_path:
        benchmark_args.extend(["--data-path", args.data_path])
    else:
        benchmark_args.append("--synthetic")
        benchmark_args.extend(["--samples", str(args.samples)])

    if args.verbose:
        benchmark_args.append("--verbose")

    # Override sys.argv to pass arguments to the benchmark script
    original_argv = sys.argv
    sys.argv = ["run_essay_benchmark.py"] + benchmark_args

    try:
        run_benchmark()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
