#!/usr/bin/env python3
"""
Comprehensive test script to verify all Quick Start features.

This script validates that all features listed in the problem statement work correctly:
1. Run smoke tests for quick validation
2. Run benchmarks for full evaluation
3. Use local CSV files or Kaggle datasets
4. Configure output directories and subset sizes
5. Access trained models and detailed metrics
"""

import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_1_smoke_test_default():
    """Test 1: Run smoke tests for quick validation (default settings)"""
    print("=" * 70)
    print("TEST 1: Smoke Test - Default Settings")
    print("=" * 70)
    
    from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_smoke_test(output_dir=tmpdir)
        
        assert result['success'], f"Smoke test failed: {result.get('error')}"
        assert result['mode'] == 'smoke'
        assert result['runtime_seconds'] > 0
        assert result['dataset_info']['train_samples'] > 0
        assert 'train_accuracy' in result['train_metrics']
        assert 'accuracy' in result['eval_metrics']
        
        # Verify output files exist
        assert Path(tmpdir, 'smoke_test_results.json').exists()
        assert Path(tmpdir, 'smoke_test_model.pkl').exists()
        
        print("✓ PASSED - Smoke test with default settings works")
        return True


def test_2_smoke_test_custom():
    """Test 2: Smoke test with custom subset size and output directory"""
    print("\n" + "=" * 70)
    print("TEST 2: Smoke Test - Custom Settings")
    print("=" * 70)
    
    from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test
    
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_dir = Path(tmpdir) / "custom_output"
        result = run_smoke_test(
            subset_size=50,
            output_dir=str(custom_dir)
        )
        
        assert result['success'], f"Custom smoke test failed: {result.get('error')}"
        assert result['dataset_info']['train_samples'] <= 50
        assert custom_dir.exists()
        assert (custom_dir / 'smoke_test_results.json').exists()
        assert (custom_dir / 'smoke_test_model.pkl').exists()
        
        print("✓ PASSED - Smoke test with custom settings works")
        return True


def test_3_benchmark_mode():
    """Test 3: Run benchmarks for full evaluation"""
    print("\n" + "=" * 70)
    print("TEST 3: Benchmark Mode - Full Evaluation")
    print("=" * 70)
    
    from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_benchmark
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_benchmark(
            subset_size=200,
            output_dir=tmpdir
        )
        
        assert result['success'], f"Benchmark failed: {result.get('error')}"
        assert result['mode'] == 'benchmark'
        assert 'eval_metrics' in result
        assert 'accuracy' in result['eval_metrics']
        assert 'precision' in result['eval_metrics']
        assert 'recall' in result['eval_metrics']
        assert 'f1_score' in result['eval_metrics']
        
        # Verify detailed metrics
        assert 'feature_importance' in result
        assert 'classification_report' in result['eval_metrics']
        
        # Verify output files
        assert Path(tmpdir, 'benchmark_results.json').exists()
        assert Path(tmpdir, 'benchmark_model.pkl').exists()
        
        print("✓ PASSED - Benchmark mode works with detailed metrics")
        return True


def test_4_local_csv():
    """Test 4: Use local CSV files"""
    print("\n" + "=" * 70)
    print("TEST 4: Local CSV File Support")
    print("=" * 70)
    
    import pandas as pd
    from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test
    
    # Create test CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_data.csv"
        
        # Generate test data
        data = {
            'text': [
                f"positive sample {i} with good quality" for i in range(50)
            ] + [
                f"negative sample {i} with poor quality" for i in range(50)
            ],
            'label': [1] * 50 + [0] * 50
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        # Run training with local CSV
        output_dir = Path(tmpdir) / "output"
        result = run_smoke_test(
            local_path=str(csv_path),
            output_dir=str(output_dir)
        )
        
        assert result['success'], f"Local CSV test failed: {result.get('error')}"
        assert result['dataset_info']['data_source'] == 'real'
        assert result['dataset_info']['train_samples'] > 0
        
        print("✓ PASSED - Local CSV file loading works")
        return True


def test_5_access_trained_model():
    """Test 5: Access trained models and make predictions"""
    print("\n" + "=" * 70)
    print("TEST 5: Access Trained Models")
    print("=" * 70)
    
    from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test
    from adaptiveneuralnetwork.training.models.text_baseline import TextClassificationBaseline
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train model
        result = run_smoke_test(output_dir=tmpdir)
        assert result['success']
        
        model_path = Path(tmpdir) / 'smoke_test_model.pkl'
        assert model_path.exists()
        
        # Load model
        model = TextClassificationBaseline()
        model.load_model(str(model_path))
        
        # Make predictions
        test_texts = [
            "machine learning is great",
            "artificial intelligence example",
            "hello world test"
        ]
        predictions = model.predict(test_texts)
        probabilities = model.predict_proba(test_texts)
        
        assert len(predictions) == len(test_texts)
        assert len(probabilities) == len(test_texts)
        assert all(len(p) == 2 for p in probabilities)  # Binary classification
        
        print("✓ PASSED - Trained model can be loaded and used for predictions")
        return True


def test_6_access_detailed_metrics():
    """Test 6: Access detailed metrics from results JSON"""
    print("\n" + "=" * 70)
    print("TEST 6: Access Detailed Metrics")
    print("=" * 70)
    
    from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_benchmark
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run benchmark
        result = run_benchmark(
            subset_size=200,
            output_dir=tmpdir
        )
        assert result['success']
        
        # Load JSON results
        results_path = Path(tmpdir) / 'benchmark_results.json'
        assert results_path.exists()
        
        with open(results_path) as f:
            metrics = json.load(f)
        
        # Verify all required metrics are present
        required_fields = [
            'mode', 'runtime_seconds', 'dataset_info', 
            'model_info', 'train_metrics', 'eval_metrics'
        ]
        for field in required_fields:
            assert field in metrics, f"Missing field: {field}"
        
        # Verify detailed evaluation metrics
        eval_metrics = metrics['eval_metrics']
        assert 'accuracy' in eval_metrics
        assert 'precision' in eval_metrics
        assert 'recall' in eval_metrics
        assert 'f1_score' in eval_metrics
        assert 'confusion_matrix' in eval_metrics
        assert 'classification_report' in eval_metrics
        
        # Verify model info
        model_info = metrics['model_info']
        assert 'num_features' in model_info
        assert 'num_classes' in model_info
        
        # Verify feature importance
        assert 'feature_importance' in metrics
        
        print("✓ PASSED - All detailed metrics are accessible")
        return True


def test_7_configure_subset_sizes():
    """Test 7: Configure different subset sizes"""
    print("\n" + "=" * 70)
    print("TEST 7: Configure Subset Sizes")
    print("=" * 70)
    
    from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test
    
    subset_sizes = [50, 100, 200]
    
    for size in subset_sizes:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_smoke_test(
                subset_size=size,
                output_dir=tmpdir
            )
            
            assert result['success'], f"Failed with subset_size={size}"
            total_samples = (
                result['dataset_info']['train_samples'] + 
                result['dataset_info']['val_samples']
            )
            assert total_samples <= size, f"Generated {total_samples} samples, expected <= {size}"
    
    print("✓ PASSED - Subset size configuration works for multiple sizes")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE QUICK START FEATURE VALIDATION")
    print("=" * 70)
    print("\nValidating all features from problem statement:")
    print("1. Run smoke tests for quick validation")
    print("2. Run benchmarks for full evaluation")
    print("3. Use local CSV files or Kaggle datasets")
    print("4. Configure output directories and subset sizes")
    print("5. Access trained models and detailed metrics")
    print("\n")
    
    tests = [
        ("Smoke Test - Default Settings", test_1_smoke_test_default),
        ("Smoke Test - Custom Settings", test_2_smoke_test_custom),
        ("Benchmark Mode", test_3_benchmark_mode),
        ("Local CSV Files", test_4_local_csv),
        ("Access Trained Models", test_5_access_trained_model),
        ("Access Detailed Metrics", test_6_access_detailed_metrics),
        ("Configure Subset Sizes", test_7_configure_subset_sizes),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED - {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - All Quick Start features work correctly!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
