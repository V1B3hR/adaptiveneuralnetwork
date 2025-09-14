#!/usr/bin/env python3
"""
Comprehensive training script for both Kaggle datasets specified in the problem statement.

This script can train on either:
1. ANNOMI Motivational Interviewing Dataset
2. Mental Health FAQs Dataset

Both datasets are trained for 100 epochs as specified in the problem statement.

Usage:
    python train_kaggle_datasets.py --dataset annomi --data-path /path/to/dataset
    python train_kaggle_datasets.py --dataset mental_health --data-path /path/to/dataset
    python train_kaggle_datasets.py --dataset both  # Train on both with synthetic data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from run_essay_benchmark import main as run_benchmark
from adaptiveneuralnetwork.data import print_dataset_info


def train_dataset(dataset_type, args):
    """Train on a specific dataset type."""
    print(f"\n{'='*60}")
    print(f"TRAINING {dataset_type.upper()} DATASET")
    print(f"{'='*60}")
    
    # Construct arguments for the main benchmark script
    benchmark_args = [
        "--dataset-type", dataset_type,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--hidden-dim", str(args.hidden_dim),
        "--num-nodes", str(args.num_nodes),
        "--device", args.device
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
        print(f"Starting {dataset_type} training...")
        run_benchmark()
        print(f"{dataset_type} training completed successfully!")
    except Exception as e:
        print(f"Error training {dataset_type}: {e}")
        raise
    finally:
        sys.argv = original_argv


def main():
    """Main function for comprehensive Kaggle dataset training."""
    parser = argparse.ArgumentParser(
        description="Train Adaptive Neural Network on Kaggle Datasets (100 epochs)"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["annomi", "mental_health", "both"],
        default="annomi",
        help="Which dataset(s) to train on"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to dataset directory (required for real data)"
    )
    parser.add_argument(
        "--epochs",
        type=int, 
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension size (default: 128)"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=100,
        help="Number of adaptive nodes (default: 100)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of synthetic samples if no real data provided"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Print dataset information
    print("=" * 80)
    print("KAGGLE DATASETS TRAINING - ADAPTIVE NEURAL NETWORK")
    print("=" * 80)
    print("Problem Statement Requirements:")
    print("- Train adaptive neural network for 100 epochs")
    print("- Support ANNOMI Motivational Interviewing Dataset")
    print("- Support Mental Health FAQs Dataset")
    print("=" * 80)
    
    print_dataset_info()
    
    if not args.data_path and args.dataset != "both":
        print(f"\nWARNING: No dataset path provided for {args.dataset}.")
        print("Using synthetic data for demonstration.")
        print("To use real data:")
        if args.dataset == "annomi":
            print("1. Download: https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing")
        elif args.dataset == "mental_health":
            print("2. Download: https://www.kaggle.com/datasets/ragishehab/mental-healthfaqs")
        print(f"3. Run: python train_kaggle_datasets.py --dataset {args.dataset} --data-path /path/to/dataset")
        print()
    
    # Train based on dataset selection
    if args.dataset == "both":
        print("\nTraining on both datasets with synthetic data...")
        train_dataset("annomi", args)
        print("\n" + "="*80)
        train_dataset("mental_health", args)
    else:
        train_dataset(args.dataset, args)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("All training sessions completed successfully!")
    print(f"Training configuration: {args.epochs} epochs, {args.batch_size} batch size, {args.learning_rate} learning rate")
    print("Results saved in benchmark_results/ directory")


if __name__ == "__main__":
    main()