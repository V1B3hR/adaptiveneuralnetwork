"""
CLI entrypoint for Bitext dataset training with smoke and benchmark modes.

This module provides the main training entry point that:
1. Loads Bitext dataset with graceful Kaggle fallback
2. Trains sklearn baseline model
3. Supports smoke and benchmark modes
4. Outputs metrics.json for CI/CD integration
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    # Also set specific loggers
    logging.getLogger('adaptiveneuralnetwork').setLevel(level)


def check_dependencies():
    """Check if optional NLP dependencies are available."""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import tqdm
    except ImportError:
        missing_deps.append("tqdm")
    
    if missing_deps:
        logger.error(f"Missing required dependencies: {missing_deps}")
        logger.error("Install with: pip install 'adaptiveneuralnetwork[nlp]'")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train Bitext intent classification baseline model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke
  python -m adaptiveneuralnetwork.training.run_bitext_training --mode benchmark
  python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke --synthetic
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["smoke", "benchmark"],
        default="smoke",
        help="Training mode: 'smoke' for fast test, 'benchmark' for full training"
    )
    
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force use of synthetic dataset (ignore Kaggle credentials)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for synthetic dataset (default: 1000)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("BITEXT INTENT CLASSIFICATION TRAINING")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Synthetic dataset: {args.synthetic}")
    
    # Check dependencies
    check_dependencies()
    
    try:
        # Import our modules
        from .bitext_dataset import load_bitext_dataset, has_kaggle_credentials
        from .sklearn_baseline import run_baseline_training
        
        # Check Kaggle credentials
        has_kaggle = has_kaggle_credentials()
        logger.info(f"Kaggle credentials available: {has_kaggle}")
        
        if not has_kaggle and not args.synthetic:
            if args.mode == "benchmark":
                logger.warning("No Kaggle credentials found. Using synthetic dataset for benchmark mode.")
            logger.info("Will use synthetic dataset automatically.")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = load_bitext_dataset(
            use_synthetic=args.synthetic or not has_kaggle,
            num_samples=args.num_samples
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")
        logger.info(f"Classes: {dataset.class_names}")
        
        # Determine dataset type for logging
        dataset_type = "synthetic" if (args.synthetic or not has_kaggle) else "kaggle"
        logger.info(f"Dataset type: {dataset_type}")
        
        # Run training
        logger.info("Starting training...")
        results = run_baseline_training(dataset, mode=args.mode, output_dir=args.output_dir)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model: {results.model_type}")
        logger.info(f"Train accuracy: {results.train_accuracy:.4f}")
        logger.info(f"Test accuracy: {results.test_accuracy:.4f}")
        logger.info(f"Training time: {results.training_time:.2f}s")
        logger.info(f"Samples: {results.num_samples}")
        logger.info(f"Classes: {results.num_classes}")
        
        # Check if metrics.json was created
        metrics_path = Path(args.output_dir) / "metrics.json"
        if metrics_path.exists():
            logger.info(f"Metrics saved to: {metrics_path}")
        else:
            logger.warning("metrics.json was not created")
        
        logger.info("Training completed successfully!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure to install with: pip install 'adaptiveneuralnetwork[nlp]'")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()