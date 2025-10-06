#!/usr/bin/env python3
"""
Unified Training Script for Adaptive Neural Network

This script provides a ready-to-run, single-file training entry point for all supported datasets
in the Adaptive Neural Network project. Supports configuration for dataset, epochs, data path, 
synthetic sample count, output directory, and verbosity.

Usage (as script):
    python adaptiveneuralnetwork_full_training.py --dataset vr_driving --epochs 10
    python adaptiveneuralnetwork_full_training.py --dataset all --epochs 10

Usage (in Jupyter/Colab cell):
    !python adaptiveneuralnetwork_full_training.py --dataset all --epochs 5 --verbose

"""

import sys, argparse, logging
from pathlib import Path
import json

def get_args():
    # Detect CLI vs notebook
    if hasattr(sys, 'argv') and len(sys.argv) > 1 and sys.argv[0].endswith('.py'):
        parser = argparse.ArgumentParser(
            description="Train Adaptive Neural Network"
        )
        available_datasets = ["vr_driving", "autvi", "digakust"]
        parser.add_argument("--dataset", choices=available_datasets + ["all"], default="all", help="Dataset to train on")
        parser.add_argument("--data-path", type=str, help="Path to dataset file")
        parser.add_argument("--epochs", type=int, default=10, help="Epochs")
        parser.add_argument("--num-samples", type=int, default=1000, help="Synthetic samples")
        parser.add_argument("--output-dir", type=str, default="outputs", help="Output dir")
        parser.add_argument("--verbose", action="store_true", help="Verbose logging")
        return parser.parse_args()
    else:
        class Args:
            dataset = "all"
            data_path = None
            epochs = 10
            num_samples = 1000
            output_dir = "outputs"
            verbose = True
        return Args()

args = get_args()
logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
logger = logging.getLogger(__name__)

# --- Import project modules (edit these if your structure differs) ---
try:
    from training.scripts.train_new_datasets import (
        train_dataset,
        save_results,
        create_synthetic_dataset
    )
except ImportError as e:
    print("Could not import the core training components. Make sure you run this script in the root directory of adaptiveneuralnetwork, or adjust the import paths.")
    raise e

def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Unified training script for Adaptive Neural Network"
    )
    parser.add_argument(
        "--dataset",
        choices=["vr_driving", "autvi", "digakust", "all"],
        default="all",
        help="Dataset to train on (default: all)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to dataset file or directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    # --- Logging ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logger = logging.getLogger(__name__)

    # --- Print Header ---
    print("=" * 80)
    print("ADAPTIVE NEURAL NETWORK - FULL TRAINING SCRIPT")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Synthetic Samples: {args.num_samples}")
    print(f"  Output Directory: {args.output_dir}")
    print("=" * 80)
    print()

    # --- Training ---
    if args.dataset == "all":
        logger.info("Training on all supported datasets...")
        all_results = {}
        datasets = ["vr_driving", "autvi", "digakust"]

        for i, dataset in enumerate(datasets, 1):
            print(f"\n{'='*60}")
            print(f"Training on {dataset} ({i}/{len(datasets)})")
            print(f"{'='*60}")

            results = train_dataset(dataset, args)
            all_results[dataset] = results
            save_results(results, args.output_dir)

            if results.get("success"):
                logger.info(f"✓ {dataset} training completed successfully")
            else:
                logger.error(f"✗ {dataset} training failed: {results.get('error', 'Unknown error')}")

        # Save combined results
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        combined_file = output_path / "all_datasets_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\n✓ Combined results saved to {combined_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        for dataset, results in all_results.items():
            status = "✓ SUCCESS" if results.get("success") else "✗ FAILED"
            print(f"{dataset:20s} - {status}")
            if results.get("success"):
                print(f"  Best Accuracy: {results.get('best_accuracy', 0):.4f}")
                print(f"  Training Time: {results.get('training_time', 0):.2f}s")
        print("=" * 80)

    else:
        # Train single dataset
        logger.info(f"Training on {args.dataset} dataset...")
        results = train_dataset(args.dataset, args)
        save_results(results, args.output_dir)

        if results.get("success"):
            logger.info(f"✓ Training completed successfully")
            logger.info(f"  Best Accuracy: {results.get('best_accuracy', 0):.4f}")
            logger.info(f"  Training Time: {results.get('training_time', 0):.2f}s")
        else:
            logger.error(f"✗ Training failed: {results.get('error', 'Unknown error')}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
