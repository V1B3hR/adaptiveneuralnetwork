#!/usr/bin/env python3
"""
Core training script for adaptive neural network datasets.

This script provides a unified entry point for training on all supported datasets
from the core module.

Usage:
    python -m core.train --dataset vr_driving --epochs 10
    python -m core.train --dataset all --epochs 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.scripts.train_new_datasets import (
    train_dataset,
    save_results,
    create_synthetic_dataset
)
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Core training script for adaptive neural network datasets"
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
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print header
    print("=" * 80)
    print("CORE MODULE - ADAPTIVE NEURAL NETWORK TRAINING")
    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Output: {args.output_dir}")
    print("=" * 80)
    print()
    
    # Train based on dataset selection
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
                metrics = results.get("final_metrics", {})
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
