#!/usr/bin/env python3
"""
Command-line interface for the Adaptive AutoML Engine.

This script provides a simple CLI for running AutoML experiments
with different configurations and data sources.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ..automl import AutoMLConfig, create_automl_engine
    from ..automl.config import (
        create_default_config,
        create_high_performance_config,
        create_minimal_config,
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent))
    from automl import create_automl_engine


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('automl_cli.log')
        ]
    )


def load_data(file_path: str, target_column: str = None) -> tuple:
    """Load data from file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load based on file extension
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.json', '.jsonl']:
        df = pd.read_json(file_path)
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    print(f"✓ Loaded data: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Handle target column
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        y = df[target_column]
        X = df.drop(columns=[target_column])
    else:
        # Try to auto-detect target column
        possible_targets = ['target', 'label', 'y', 'class', 'outcome']
        target_column = None

        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break

        if target_column:
            print(f"  Auto-detected target column: {target_column}")
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            print("  No target column specified or detected - using dummy target")
            X = df
            y = np.random.choice([0, 1], len(df))  # Dummy binary target

    print(f"  Features: {X.shape}")
    print(f"  Target distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y


def create_synthetic_data(n_samples: int = 1000, n_features: int = 10, complexity: str = 'medium') -> tuple:
    """Create synthetic dataset for testing."""
    print(f"Creating synthetic dataset: {n_samples} samples, {n_features} features ({complexity})")

    np.random.seed(42)

    # Generate features
    data = {}

    # Numeric features
    for i in range(n_features // 2):
        if i % 3 == 0:
            data[f'numeric_{i}'] = np.random.randn(n_samples)
        elif i % 3 == 1:
            data[f'numeric_{i}'] = np.random.exponential(2, n_samples)
        else:
            data[f'numeric_{i}'] = np.random.uniform(0, 100, n_samples)

    # Categorical features
    for i in range(n_features // 4):
        if complexity == 'simple':
            categories = ['A', 'B']
        elif complexity == 'medium':
            categories = ['A', 'B', 'C', 'D']
        else:
            categories = [f'Cat_{j}' for j in range(8)]

        data[f'categorical_{i}'] = np.random.choice(categories, n_samples)

    # Binary features
    for i in range(n_features // 4):
        data[f'binary_{i}'] = np.random.choice([0, 1], n_samples)

    X = pd.DataFrame(data)

    # Add missing values
    if complexity != 'simple':
        missing_ratio = 0.05 if complexity == 'medium' else 0.15
        n_missing = int(n_samples * missing_ratio)

        for col in X.select_dtypes(include=[np.number]).columns[:2]:
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
            X.loc[missing_indices, col] = None

    # Generate correlated target
    numeric_features = X.select_dtypes(include=[np.number]).columns
    y = 0

    for i, col in enumerate(numeric_features[:3]):
        y += 0.3 * X[col].fillna(0) * ((-1) ** i)

    # Add categorical influence
    cat_features = [col for col in X.columns if 'categorical' in col]
    if cat_features:
        y += 2 * (X[cat_features[0]] == X[cat_features[0]].mode().iloc[0]).astype(int)

    # Add noise and convert to binary
    y += np.random.normal(0, 0.5, n_samples)
    y = (y > y.median()).astype(int)

    print(f"✓ Synthetic data created: {X.shape}")
    return X, y


def run_automl(
    X: pd.DataFrame,
    y: pd.Series,
    config_type: str = 'default',
    output_dir: str = 'automl_results',
    hyperparameter_optimization: bool = False
) -> dict:
    """Run AutoML pipeline."""

    print(f"\n🚀 Running AutoML with {config_type} configuration...")

    # Create AutoML engine
    engine = create_automl_engine(config_type)

    # Run complete pipeline
    start_time = pd.Timestamp.now()

    X_processed, y_processed, results = engine.fit_transform_all(
        X, y,
        optimize_hyperparameters=hyperparameter_optimization
    )

    end_time = pd.Timestamp.now()
    total_time = (end_time - start_time).total_seconds()

    # Get energy report
    energy_report = engine.get_energy_report()

    # Compile final results
    final_results = {
        'input_shape': X.shape,
        'output_shape': X_processed.shape,
        'processing_time': total_time,
        'config_type': config_type,
        'pipeline_results': results,
        'energy_report': energy_report,
        'timestamp': start_time.isoformat()
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results_file = output_path / f'automl_results_{config_type}_{start_time.strftime("%Y%m%d_%H%M%S")}.json'

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"✅ Results saved to: {results_file}")

    return final_results


def print_summary(results: dict):
    """Print summary of results."""
    print("\n" + "="*60)
    print("📊 AUTOML RESULTS SUMMARY")
    print("="*60)

    print(f"Configuration: {results['config_type']}")
    print(f"Processing time: {results['processing_time']:.2f}s")
    print(f"Data transformation: {results['input_shape']} → {results['output_shape']}")

    energy = results['energy_report']
    print("\n⚡ Energy Report:")
    print(f"  Total consumption: {energy['total_energy']:.4f}")
    print(f"  Efficiency score: {energy['energy_efficiency_score']:.4f}")

    pipeline = results['pipeline_results']

    if 'preprocessing' in pipeline:
        prep = pipeline['preprocessing']
        print("\n🔧 Preprocessing:")
        print(f"  Energy: {prep.get('energy_consumption', 0):.4f}")
        print(f"  Missing values handled: {prep.get('missing_values_handled', 0)}")
        print(f"  Outliers detected: {prep.get('outliers_detected', 0)}")

    if 'feature_engineering' in pipeline:
        eng = pipeline['feature_engineering']
        print("\n🛠️  Feature Engineering:")
        print(f"  Energy: {eng.get('energy_consumption', 0):.4f}")
        print(f"  Features engineered: {eng.get('engineered_features', 0)}")

    if 'feature_selection' in pipeline:
        sel = pipeline['feature_selection']
        print("\n🎯 Feature Selection:")
        print(f"  Energy: {sel.get('energy_consumption', 0):.4f}")
        print(f"  Features selected: {sel.get('selected_features', 0)}")
        print(f"  Selection ratio: {sel.get('selection_ratio', 0):.2%}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Adaptive AutoML Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data
  python automl_cli.py --synthetic --samples 1000 --features 10
  
  # Run with CSV file
  python automl_cli.py --data data.csv --target target_column
  
  # Run with high performance configuration
  python automl_cli.py --synthetic --config high_performance
  
  # Run with hyperparameter optimization
  python automl_cli.py --data data.csv --target y --hyperopt
        """
    )

    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--data',
        help='Path to data file (CSV, JSON, Excel, Parquet)'
    )
    data_group.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing'
    )

    # Data options
    parser.add_argument(
        '--target',
        help='Target column name (for file input)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples for synthetic data (default: 1000)'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=10,
        help='Number of features for synthetic data (default: 10)'
    )
    parser.add_argument(
        '--complexity',
        choices=['simple', 'medium', 'complex'],
        default='medium',
        help='Complexity of synthetic data (default: medium)'
    )

    # AutoML configuration
    parser.add_argument(
        '--config',
        choices=['minimal', 'default', 'high_performance'],
        default='default',
        help='AutoML configuration type (default: default)'
    )

    # Processing options
    parser.add_argument(
        '--hyperopt',
        action='store_true',
        help='Enable hyperparameter optimization'
    )
    parser.add_argument(
        '--output',
        default='automl_results',
        help='Output directory for results (default: automl_results)'
    )

    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    try:
        # Load data
        if args.synthetic:
            X, y = create_synthetic_data(
                n_samples=args.samples,
                n_features=args.features,
                complexity=args.complexity
            )
        else:
            X, y = load_data(args.data, args.target)

        # Run AutoML
        results = run_automl(
            X, y,
            config_type=args.config,
            output_dir=args.output,
            hyperparameter_optimization=args.hyperopt
        )

        # Print summary
        print_summary(results)

        print("\n🎉 AutoML completed successfully!")
        return 0

    except Exception as e:
        logging.error(f"AutoML failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
