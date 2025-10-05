#!/usr/bin/env python3
"""
CLI for running bitext training with Adaptive Neural Network.

This script provides a command-line interface for training text classification
baselines with smoke testing and benchmark modes.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import warnings
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def check_dependencies() -> Dict[str, bool]:
    """Check which optional dependencies are available."""
    deps = {}

    try:
        import pandas
        deps['pandas'] = True
    except ImportError:
        deps['pandas'] = False

    try:
        import sklearn
        deps['sklearn'] = True
    except ImportError:
        deps['sklearn'] = False

    try:
        import kagglehub
        deps['kagglehub'] = True
    except ImportError:
        deps['kagglehub'] = False

    try:
        import matplotlib
        deps['matplotlib'] = True
    except ImportError:
        deps['matplotlib'] = False

    return deps


def print_dependency_status():
    """Print status of optional dependencies."""
    deps = check_dependencies()

    print("Dependency Status:")
    print("-" * 20)
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {dep:12}: {status}")

    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        print("\nTo install missing dependencies:")
        print("  pip install 'adaptiveneuralnetwork[nlp]'")

    return deps


def run_smoke_test(
    dataset_name: Optional[str] = None,
    local_path: Optional[str] = None,
    subset_size: int = 100,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Run a smoke test with minimal data and quick training.

    Args:
        dataset_name: Kaggle dataset name
        local_path: Path to local CSV file
        subset_size: Size of subset for smoke test
        output_dir: Output directory for results

    Returns:
        Dictionary with test results
    """
    try:
        from adaptiveneuralnetwork.training.datasets.bitext_dataset import (
            BitextDatasetLoader,
            create_synthetic_bitext_data
        )
        from adaptiveneuralnetwork.training.models.text_baseline import TextClassificationBaseline

        logger.info("Starting smoke test...")
        start_time = time.time()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Try to load real data first
        train_df, val_df = None, None

        if dataset_name or local_path:
            loader = BitextDatasetLoader(
                dataset_name=dataset_name,
                local_path=local_path,
                sampling_fraction=min(subset_size / 10000, 1.0),  # Rough sampling
                normalize_text=True,
                random_seed=42
            )

            try:
                train_df, val_df = loader.load_dataset(val_split=0.2)
                if train_df is not None:
                    # Limit to subset size
                    if len(train_df) > subset_size:
                        train_df = train_df.head(subset_size)
                    if len(val_df) > subset_size // 4:
                        val_df = val_df.head(subset_size // 4)

                    logger.info(f"Loaded real data: {len(train_df)} train, {len(val_df)} val samples")
            except Exception as e:
                logger.warning(f"Failed to load real data: {e}")
                train_df, val_df = None, None

        # Fallback to synthetic data
        if train_df is None:
            logger.info("Using synthetic data for smoke test")
            train_df, val_df = create_synthetic_bitext_data(
                num_samples=subset_size,
                num_classes=2,
                random_seed=42
            )

        if train_df is None:
            raise RuntimeError("Failed to create any data for smoke test")

        # Create and train baseline model
        logger.info("Training baseline model...")
        baseline = TextClassificationBaseline(
            max_features=min(1000, subset_size),  # Adaptive feature count
            random_state=42,
            verbose=True
        )

        # Train the model
        train_metrics = baseline.fit(
            texts=train_df['text'].tolist(),
            labels=train_df['label'].tolist(),
            validation_texts=val_df['text'].tolist() if val_df is not None else None,
            validation_labels=val_df['label'].tolist() if val_df is not None else None
        )

        # Evaluate on validation set
        if val_df is not None:
            eval_metrics = baseline.evaluate(
                texts=val_df['text'].tolist(),
                labels=val_df['label'].tolist()
            )
        else:
            eval_metrics = {}

        # Get feature importance
        feature_importance = baseline.get_feature_importance(top_k=10)

        # Calculate runtime
        runtime = time.time() - start_time

        # Compile results
        results = {
            "mode": "smoke",
            "runtime_seconds": runtime,
            "dataset_info": {
                "train_samples": len(train_df),
                "val_samples": len(val_df) if val_df is not None else 0,
                "data_source": "real" if dataset_name or local_path else "synthetic"
            },
            "model_info": baseline.get_model_info(),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "feature_importance": feature_importance,
            "success": True,
            "warnings": []
        }

        # Save results
        results_file = output_path / "smoke_test_results.json"
        try:
            # Convert numpy types before saving
            results_converted = convert_numpy_types(results)
            with open(results_file, 'w') as f:
                json.dump(results_converted, f, indent=2, default=str)
        except Exception as save_error:
            logger.error(f"Failed to save results: {save_error}")
            # Try to identify which part is problematic
            for key, value in results.items():
                try:
                    json.dumps({key: value})
                    logger.debug(f"  {key}: OK")
                except Exception as e:
                    logger.error(f"  {key}: FAILED - {e}")
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            try:
                                json.dumps({subkey: subvalue})
                            except Exception as e2:
                                logger.error(f"    {subkey} (type {type(subkey).__name__}): FAILED - {e2}")
            raise

        # Save model
        model_file = output_path / "smoke_test_model.pkl"
        baseline.save_model(model_file)

        logger.info(f"Smoke test completed successfully in {runtime:.2f} seconds")
        logger.info(f"Results saved to: {results_file}")

        return results

    except Exception as e:
        error_msg = f"Smoke test failed: {e}"
        logger.error(error_msg)

        return {
            "mode": "smoke",
            "success": False,
            "error": str(e),
            "runtime_seconds": time.time() - start_time if 'start_time' in locals() else 0
        }


def run_benchmark(
    dataset_name: Optional[str] = None,
    local_path: Optional[str] = None,
    subset_size: Optional[int] = None,
    epochs: int = 1,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Run a benchmark with full data and detailed evaluation.

    Args:
        dataset_name: Kaggle dataset name
        local_path: Path to local CSV file
        subset_size: Optional size limit for dataset
        epochs: Number of training epochs (for future use)
        output_dir: Output directory for results

    Returns:
        Dictionary with benchmark results
    """
    try:
        from adaptiveneuralnetwork.training.datasets.bitext_dataset import (
            BitextDatasetLoader,
            create_synthetic_bitext_data
        )
        from adaptiveneuralnetwork.training.models.text_baseline import TextClassificationBaseline

        logger.info("Starting benchmark...")
        start_time = time.time()

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load data
        train_df, val_df = None, None

        if dataset_name or local_path:
            sampling_fraction = None
            if subset_size:
                # Rough estimate for sampling fraction
                sampling_fraction = min(subset_size / 100000, 1.0)

            loader = BitextDatasetLoader(
                dataset_name=dataset_name,
                local_path=local_path,
                sampling_fraction=sampling_fraction,
                normalize_text=True,
                random_seed=42
            )

            train_df, val_df = loader.load_dataset(val_split=0.2)

        # Fallback to synthetic data
        if train_df is None:
            logger.info("Using synthetic data for benchmark")
            num_samples = subset_size or 5000
            train_df, val_df = create_synthetic_bitext_data(
                num_samples=num_samples,
                num_classes=3,
                random_seed=42
            )

        if train_df is None:
            raise RuntimeError("Failed to create any data for benchmark")

        # Apply subset limit if specified
        if subset_size and len(train_df) > subset_size:
            train_df = train_df.head(subset_size)
            if val_df is not None and len(val_df) > subset_size // 4:
                val_df = val_df.head(subset_size // 4)

        # Create and train baseline model with full configuration
        logger.info("Training benchmark model...")
        baseline = TextClassificationBaseline(
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            C=1.0,
            random_state=42,
            verbose=True
        )

        # Train the model (epochs parameter for future use)
        train_metrics = baseline.fit(
            texts=train_df['text'].tolist(),
            labels=train_df['label'].tolist(),
            validation_texts=val_df['text'].tolist() if val_df is not None else None,
            validation_labels=val_df['label'].tolist() if val_df is not None else None
        )

        # Detailed evaluation
        if val_df is not None:
            eval_metrics = baseline.evaluate(
                texts=val_df['text'].tolist(),
                labels=val_df['label'].tolist()
            )
        else:
            eval_metrics = {}

        # Get feature importance
        feature_importance = baseline.get_feature_importance(top_k=20)

        # Calculate runtime
        runtime = time.time() - start_time

        # Compile results
        results = {
            "mode": "benchmark",
            "runtime_seconds": runtime,
            "dataset_info": {
                "train_samples": len(train_df),
                "val_samples": len(val_df) if val_df is not None else 0,
                "data_source": "real" if dataset_name or local_path else "synthetic",
                "subset_size": subset_size
            },
            "model_info": baseline.get_model_info(),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "feature_importance": feature_importance,
            "epochs": epochs,
            "success": True
        }

        # Save results
        results_file = output_path / "benchmark_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types before saving
            results_converted = convert_numpy_types(results)
            json.dump(results_converted, f, indent=2, default=str)

        # Save model
        model_file = output_path / "benchmark_model.pkl"
        baseline.save_model(model_file)

        # Create visualization if matplotlib available
        try:
            import matplotlib.pyplot as plt

            if 'confusion_matrix' in eval_metrics:
                plt.figure(figsize=(8, 6))
                plt.imshow(eval_metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')

                confusion_plot = output_path / "confusion_matrix.png"
                plt.savefig(confusion_plot, dpi=150, bbox_inches='tight')
                plt.close()

                logger.info(f"Confusion matrix saved to: {confusion_plot}")

        except ImportError:
            logger.info("matplotlib not available, skipping visualization")

        logger.info(f"Benchmark completed successfully in {runtime:.2f} seconds")
        logger.info(f"Results saved to: {results_file}")

        return results

    except Exception as e:
        error_msg = f"Benchmark failed: {e}"
        logger.error(error_msg)

        return {
            "mode": "benchmark",
            "success": False,
            "error": str(e),
            "runtime_seconds": time.time() - start_time if 'start_time' in locals() else 0
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run bitext training with Adaptive Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test with synthetic data
  python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke

  # Smoke test with Kaggle dataset
  python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke --dataset-name username/dataset

  # Benchmark with local CSV
  python -m adaptiveneuralnetwork.training.run_bitext_training --mode benchmark --local-path data.csv

  # Check dependencies
  python -m adaptiveneuralnetwork.training.run_bitext_training --check-deps
        """
    )

    parser.add_argument(
        "--mode",
        choices=["smoke", "benchmark"],
        default="smoke",
        help="Training mode (default: smoke)"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Kaggle dataset name (e.g., 'username/dataset-name')"
    )

    parser.add_argument(
        "--local-path",
        type=str,
        help="Path to local CSV file"
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        help="Maximum number of samples to use"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)"
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependency status and exit"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check dependencies if requested
    if args.check_deps:
        deps = print_dependency_status()
        missing = [dep for dep, available in deps.items() if not available]
        sys.exit(1 if missing else 0)

    # Print dependency status
    deps = check_dependencies()
    required_deps = ['pandas', 'sklearn']
    missing_required = [dep for dep in required_deps if not deps.get(dep, False)]

    if missing_required:
        logger.error(f"Required dependencies missing: {missing_required}")
        logger.error("Install with: pip install 'adaptiveneuralnetwork[nlp]'")
        sys.exit(1)

    # Check Kaggle credentials if using Kaggle dataset
    if args.dataset_name:
        import os
        if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
            kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
            if not kaggle_json.exists():
                logger.warning(
                    "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
                    "environment variables or place kaggle.json in ~/.kaggle/"
                )
                logger.warning("Falling back to local/synthetic data...")

    # Run training
    try:
        if args.mode == "smoke":
            results = run_smoke_test(
                dataset_name=args.dataset_name,
                local_path=args.local_path,
                subset_size=args.subset_size or 100,
                output_dir=args.output_dir
            )
        else:  # benchmark
            results = run_benchmark(
                dataset_name=args.dataset_name,
                local_path=args.local_path,
                subset_size=args.subset_size,
                epochs=args.epochs,
                output_dir=args.output_dir
            )

        # Print summary
        print("\n" + "=" * 50)
        print(f"Training Summary ({results['mode']} mode)")
        print("=" * 50)

        if results['success']:
            print("Status: ✓ SUCCESS")
            print(f"Runtime: {results['runtime_seconds']:.2f} seconds")

            if 'dataset_info' in results:
                info = results['dataset_info']
                print(f"Data: {info['train_samples']} train, {info['val_samples']} val samples")

            if 'train_metrics' in results:
                train_acc = results['train_metrics'].get('train_accuracy', 0)
                print(f"Train Accuracy: {train_acc:.4f}")

            if 'eval_metrics' in results:
                val_acc = results['eval_metrics'].get('accuracy', 0)
                print(f"Validation Accuracy: {val_acc:.4f}")
        else:
            print("Status: ✗ FAILED")
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)

        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
