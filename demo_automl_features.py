#!/usr/bin/env python3
"""
Demonstration of Adaptive AutoML Engine capabilities.

This script showcases the neuromorphic-inspired AutoML features including:
- Energy-aware preprocessing
- Phase-based feature engineering  
- Adaptive feature selection
- Energy-guided hyperparameter optimization
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Import the AutoML components
from adaptiveneuralnetwork import (
    AdaptiveAutoMLEngine, 
    AutoMLConfig,
    create_automl_engine
)
from adaptiveneuralnetwork.automl.config import (
    create_default_config,
    create_minimal_config, 
    create_high_performance_config
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_demo_dataset(n_samples=500, complexity='medium'):
    """Create a synthetic dataset for demonstration."""
    logger.info(f"Creating synthetic dataset with {n_samples} samples (complexity: {complexity})")
    
    np.random.seed(42)
    
    if complexity == 'simple':
        n_features = 5
    elif complexity == 'medium':
        n_features = 10
    else:  # complex
        n_features = 20
    
    # Generate features with different characteristics
    data = {}
    
    # Numeric features with different distributions
    for i in range(n_features // 2):
        if i % 3 == 0:
            # Normal distribution
            data[f'numeric_{i}'] = np.random.randn(n_samples)
        elif i % 3 == 1:
            # Exponential distribution (skewed)
            data[f'numeric_{i}'] = np.random.exponential(2, n_samples)
        else:
            # Uniform distribution
            data[f'numeric_{i}'] = np.random.uniform(0, 100, n_samples)
    
    # Categorical features
    for i in range(n_features // 4):
        if i % 2 == 0:
            # Low cardinality
            data[f'categorical_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
        else:
            # Medium cardinality
            data[f'categorical_{i}'] = np.random.choice([f'Cat_{j}' for j in range(8)], n_samples)
    
    # Binary features
    for i in range(n_features // 4):
        data[f'binary_{i}'] = np.random.choice([0, 1], n_samples)
    
    X = pd.DataFrame(data)
    
    # Add missing values
    missing_ratio = 0.05 if complexity == 'simple' else 0.15
    n_missing = int(n_samples * missing_ratio)
    
    for col in X.select_dtypes(include=[np.number]).columns[:3]:
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        X.loc[missing_indices, col] = None
    
    # Add outliers
    for col in X.select_dtypes(include=[np.number]).columns[:2]:
        outlier_indices = np.random.choice(n_samples, 5, replace=False)
        outlier_values = np.random.choice([-1000, 1000], 5)
        X.loc[outlier_indices, col] = outlier_values
    
    # Generate correlated target
    numeric_features = X.select_dtypes(include=[np.number]).columns
    
    y = 0
    for i, col in enumerate(numeric_features[:3]):
        y += 0.3 * X[col].fillna(0) * ((-1) ** i)  # Alternating signs
    
    # Add categorical influence
    cat_features = [col for col in X.columns if 'categorical' in col]
    if cat_features:
        y += 2 * (X[cat_features[0]] == X[cat_features[0]].mode().iloc[0]).astype(int)
    
    # Add noise and convert to binary classification
    y += np.random.normal(0, 0.5, n_samples)
    y = (y > y.median()).astype(int)
    
    logger.info(f"Dataset created: {X.shape}, target distribution: {pd.Series(y).value_counts().to_dict()}")
    return X, y


def demo_basic_automl():
    """Demonstrate basic AutoML functionality."""
    print("\n" + "="*60)
    print("🚀 BASIC AUTOML DEMONSTRATION")
    print("="*60)
    
    # Create simple dataset
    X, y = create_demo_dataset(n_samples=200, complexity='simple')
    
    # Create minimal AutoML engine for speed
    engine = create_automl_engine('minimal')
    
    print(f"\n📊 Input Data:")
    print(f"  Shape: {X.shape}")
    print(f"  Data types: {X.dtypes.value_counts().to_dict()}")
    print(f"  Missing values: {X.isnull().sum().sum()}")
    
    # Run complete AutoML pipeline
    X_processed, y_processed, results = engine.fit_transform_all(
        X, y, optimize_hyperparameters=False
    )
    
    print(f"\n✨ AutoML Results:")
    print(f"  Final shape: {X_processed.shape}")
    print(f"  Feature expansion: {X.shape[1]} → {X_processed.shape[1]} features")
    
    # Print energy report
    energy_report = engine.get_energy_report()
    print(f"\n⚡ Energy Report:")
    print(f"  Total energy consumption: {energy_report['total_energy']:.4f}")
    print(f"  Energy efficiency score: {energy_report['energy_efficiency_score']:.4f}")
    
    return engine, results


def demo_advanced_automl():
    """Demonstrate advanced AutoML features."""
    print("\n" + "="*60)
    print("🔬 ADVANCED AUTOML DEMONSTRATION")
    print("="*60)
    
    # Create complex dataset
    X, y = create_demo_dataset(n_samples=300, complexity='complex')
    
    # Create high-performance AutoML engine
    engine = create_automl_engine('high_performance')
    
    print(f"\n📊 Input Data:")
    print(f"  Shape: {X.shape}")
    print(f"  Data types: {X.dtypes.value_counts().to_dict()}")
    print(f"  Missing values per column: {dict(X.isnull().sum()[X.isnull().sum() > 0])}")
    
    # Step-by-step demonstration
    print(f"\n🔧 Step 1: Preprocessing...")
    X_preprocessed, preprocessing_info = engine.auto_preprocess_pipeline(X, y)
    print(f"  Preprocessing: {X.shape} → {X_preprocessed.shape}")
    print(f"  Energy consumed: {preprocessing_info.get('energy_consumption', 0):.4f}")
    
    print(f"\n🛠️ Step 2: Feature Engineering...")
    X_engineered, engineering_info = engine.neuromorphic_feature_engineering(X_preprocessed, y)
    print(f"  Feature engineering: {X_preprocessed.shape} → {X_engineered.shape}")
    print(f"  New features created: {engineering_info.get('engineered_features', 0) - X_preprocessed.shape[1]}")
    print(f"  Energy consumed: {engineering_info.get('energy_consumption', 0):.4f}")
    
    print(f"\n🎯 Step 3: Feature Selection...")
    X_selected, selection_info = engine.adaptive_feature_selection(X_engineered, y)
    print(f"  Feature selection: {X_engineered.shape} → {X_selected.shape}")
    print(f"  Selection ratio: {selection_info.get('selection_ratio', 0):.2%}")
    print(f"  Energy consumed: {selection_info.get('energy_consumption', 0):.4f}")
    
    # Get detailed reports
    energy_report = engine.get_energy_report()
    print(f"\n📈 Performance Summary:")
    print(f"  Total pipeline energy: {energy_report['total_energy']:.4f}")
    print(f"  Energy efficiency: {energy_report['energy_efficiency_score']:.4f}")
    print(f"  Feature reduction: {X.shape[1]} → {X_selected.shape[1]} ({X_selected.shape[1]/X.shape[1]:.1%} retained)")
    
    return engine, X_selected


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION OPTIONS DEMONSTRATION")
    print("="*60)
    
    # Create test dataset
    X, y = create_demo_dataset(n_samples=150, complexity='medium')
    
    configs = {
        'Minimal (Fast)': create_minimal_config(),
        'Default (Balanced)': create_default_config(), 
        'High Performance (Thorough)': create_high_performance_config()
    }
    
    results_comparison = {}
    
    for config_name, config in configs.items():
        print(f"\n🔧 Testing {config_name} Configuration...")
        
        engine = AdaptiveAutoMLEngine(config=config)
        
        start_time = pd.Timestamp.now()
        X_processed, _, results = engine.fit_transform_all(X, y, optimize_hyperparameters=False)
        end_time = pd.Timestamp.now()
        
        processing_time = (end_time - start_time).total_seconds()
        energy_report = engine.get_energy_report()
        
        results_comparison[config_name] = {
            'processing_time': processing_time,
            'final_features': X_processed.shape[1],
            'energy_consumption': energy_report['total_energy'],
            'energy_efficiency': energy_report['energy_efficiency_score']
        }
        
        print(f"  Time: {processing_time:.2f}s")
        print(f"  Features: {X.shape[1]} → {X_processed.shape[1]}")
        print(f"  Energy: {energy_report['total_energy']:.4f}")
    
    # Print comparison table
    print(f"\n📊 Configuration Comparison:")
    print(f"{'Config':<20} {'Time (s)':<10} {'Features':<10} {'Energy':<10} {'Efficiency':<12}")
    print("-" * 70)
    
    for config_name, metrics in results_comparison.items():
        print(f"{config_name:<20} {metrics['processing_time']:<10.2f} "
              f"{metrics['final_features']:<10} {metrics['energy_consumption']:<10.4f} "
              f"{metrics['energy_efficiency']:<12.4f}")


def demo_pipeline_orchestration():
    """Demonstrate pipeline orchestration features."""
    print("\n" + "="*60)
    print("🔄 PIPELINE ORCHESTRATION DEMONSTRATION")
    print("="*60)
    
    from adaptiveneuralnetwork.automl.pipeline import AutoMLWorkflow
    
    # Create dataset
    X, y = create_demo_dataset(n_samples=250, complexity='medium')
    
    # Create workflow
    workflow = AutoMLWorkflow(neuromorphic_enabled=True)
    
    print(f"📊 Running complete AutoML workflow...")
    
    # Run complete workflow
    complete_results = workflow.run_complete_workflow(X, y)
    
    print(f"\n✅ Workflow completed successfully!")
    
    # Print workflow summary
    workflow_meta = complete_results['workflow_metadata']
    print(f"\n📈 Workflow Summary:")
    print(f"  Total time: {workflow_meta['total_time']:.2f}s")
    print(f"  Data shape: {workflow_meta['data_shape']} → {workflow_meta['processed_data_shape']}")
    print(f"  Neuromorphic features: {'✓' if workflow_meta['neuromorphic_enabled'] else '✗'}")
    
    # Print preprocessing summary
    preprocessing = complete_results['preprocessing_results']
    print(f"\n🔧 Preprocessing Summary:")
    print(f"  Steps completed: {preprocessing['total_steps']}")
    print(f"  Energy consumption: {preprocessing['total_energy_consumption']:.4f}")
    print(f"  Data leakage prevention: {'✓' if preprocessing['leakage_prevention_enabled'] else '✗'}")


def save_demo_results():
    """Demonstrate saving and loading AutoML results."""
    print("\n" + "="*60)
    print("💾 SAVE/LOAD DEMONSTRATION")
    print("="*60)
    
    # Create and run AutoML
    X, y = create_demo_dataset(n_samples=100, complexity='simple')
    engine = create_automl_engine('minimal')
    
    X_processed, _, results = engine.fit_transform_all(X, y, optimize_hyperparameters=False)
    
    # Save results
    results_file = Path("automl_demo_results.json")
    engine.save_results(results_file)
    print(f"✅ Results saved to {results_file}")
    
    # Create and save pipeline
    pipeline = engine.create_full_pipeline(X, y)
    pipeline_file = Path("automl_demo_pipeline.pkl")
    pipeline.save_pipeline(pipeline_file)
    print(f"✅ Pipeline saved to {pipeline_file}")
    
    # Clean up
    if results_file.exists():
        results_file.unlink()
        print(f"🗑️  Cleaned up {results_file}")
    
    if pipeline_file.exists():
        pipeline_file.unlink()
        print(f"🗑️  Cleaned up {pipeline_file}")


def main():
    """Run all AutoML demonstrations."""
    print("🧠 ADAPTIVE NEURAL NETWORK - AUTOML DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the neuromorphic-inspired AutoML capabilities:")
    print("• Energy-aware preprocessing with spike-based outlier detection")
    print("• Phase-based feature engineering using neuromorphic dynamics")
    print("• Adaptive feature selection guided by energy importance")
    print("• Pipeline orchestration with data leakage prevention")
    
    try:
        # Run demonstrations
        demo_basic_automl()
        demo_advanced_automl()
        demo_configuration_options()
        demo_pipeline_orchestration()
        save_demo_results()
        
        print("\n" + "="*80)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("The Adaptive AutoML Engine is ready for production use.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())