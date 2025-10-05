#!/usr/bin/env python3
"""
Quick Start Example: Train and Use Text Classification Models

This example demonstrates the core features:
1. Training models in smoke/benchmark mode
2. Using local CSV files
3. Accessing trained models and metrics
4. Making predictions with trained models
"""

import json
from pathlib import Path
from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test, run_benchmark
from adaptiveneuralnetwork.training.models.text_baseline import TextClassificationBaseline


def example_1_smoke_test():
    """Example 1: Quick smoke test with default settings"""
    print("=" * 60)
    print("Example 1: Smoke Test (Quick Validation)")
    print("=" * 60)
    
    results = run_smoke_test(
        subset_size=100,
        output_dir="examples/outputs"
    )
    
    if results['success']:
        print(f"✓ Success! Runtime: {results['runtime_seconds']:.2f}s")
        print(f"✓ Train Accuracy: {results['train_metrics']['train_accuracy']:.4f}")
        print(f"✓ Val Accuracy: {results['eval_metrics']['accuracy']:.4f}")
    else:
        print(f"✗ Failed: {results.get('error', 'Unknown error')}")
    
    return results


def example_2_custom_smoke_test():
    """Example 2: Custom smoke test with specific parameters"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Smoke Test")
    print("=" * 60)
    
    results = run_smoke_test(
        subset_size=50,
        output_dir="examples/outputs/custom"
    )
    
    if results['success']:
        print(f"✓ Custom test completed in {results['runtime_seconds']:.2f}s")
        print(f"✓ Used {results['dataset_info']['train_samples']} training samples")
        print(f"✓ Model saved to: examples/outputs/custom/smoke_test_model.pkl")
    
    return results


def example_3_benchmark():
    """Example 3: Full benchmark evaluation"""
    print("\n" + "=" * 60)
    print("Example 3: Benchmark Mode (Full Evaluation)")
    print("=" * 60)
    
    results = run_benchmark(
        subset_size=500,
        output_dir="examples/outputs/benchmark"
    )
    
    if results['success']:
        print(f"✓ Benchmark completed in {results['runtime_seconds']:.2f}s")
        print(f"✓ Accuracy: {results['eval_metrics']['accuracy']:.4f}")
        print(f"✓ F1 Score: {results['eval_metrics']['f1_score']:.4f}")
        print(f"✓ Precision: {results['eval_metrics']['precision']:.4f}")
        print(f"✓ Recall: {results['eval_metrics']['recall']:.4f}")
        
        # Show top features
        print("\n✓ Top Features per Class:")
        for class_name, features in list(results['feature_importance'].items())[:2]:
            print(f"  Class {class_name}:")
            for feature, weight in features[:3]:
                print(f"    - {feature}: {weight:.4f}")
    
    return results


def example_4_load_and_predict():
    """Example 4: Load trained model and make predictions"""
    print("\n" + "=" * 60)
    print("Example 4: Load Model and Make Predictions")
    print("=" * 60)
    
    # First, ensure we have a trained model
    model_path = Path("examples/outputs/smoke_test_model.pkl")
    if not model_path.exists():
        print("Training a model first...")
        run_smoke_test(subset_size=100, output_dir="examples/outputs")
    
    # Load the model
    model = TextClassificationBaseline()
    model.load_model(str(model_path))
    print(f"✓ Model loaded from {model_path}")
    
    # Make predictions
    test_texts = [
        "machine learning is amazing",
        "the quick brown fox jumps",
        "artificial intelligence dataset",
        "hello world example"
    ]
    
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    print("\n✓ Predictions:")
    for i, text in enumerate(test_texts):
        pred = predictions[i]
        prob = probabilities[i]
        confidence = max(prob)
        print(f"  Text: '{text}'")
        print(f"    → Prediction: {pred} (confidence: {confidence:.2%})")
    
    return predictions, probabilities


def example_5_analyze_results():
    """Example 5: Analyze results JSON file"""
    print("\n" + "=" * 60)
    print("Example 5: Analyze Results from JSON")
    print("=" * 60)
    
    results_path = Path("examples/outputs/smoke_test_results.json")
    
    if not results_path.exists():
        print("No results file found. Running smoke test first...")
        run_smoke_test(subset_size=100, output_dir="examples/outputs")
    
    # Load and analyze results
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"✓ Results loaded from {results_path}")
    print(f"\n📊 Training Summary:")
    print(f"  Mode: {results['mode']}")
    print(f"  Runtime: {results['runtime_seconds']:.2f}s")
    print(f"  Data Source: {results['dataset_info']['data_source']}")
    print(f"  Train Samples: {results['dataset_info']['train_samples']}")
    print(f"  Val Samples: {results['dataset_info']['val_samples']}")
    
    print(f"\n📈 Model Performance:")
    print(f"  Accuracy: {results['eval_metrics']['accuracy']:.4f}")
    print(f"  Precision: {results['eval_metrics']['precision']:.4f}")
    print(f"  Recall: {results['eval_metrics']['recall']:.4f}")
    print(f"  F1 Score: {results['eval_metrics']['f1_score']:.4f}")
    
    print(f"\n🔍 Model Info:")
    print(f"  Features: {results['model_info']['num_features']}")
    print(f"  Classes: {results['model_info']['num_classes']}")
    print(f"  Class Names: {results['model_info']['class_names']}")
    
    # Show confusion matrix
    cm = results['eval_metrics']['confusion_matrix']
    print(f"\n📊 Confusion Matrix:")
    for i, row in enumerate(cm):
        print(f"  Class {i}: {row}")
    
    return results


def main():
    """Run all examples"""
    print("\n🚀 Adaptive Neural Network - Quick Start Examples")
    print("=" * 60)
    
    # Create output directory
    Path("examples/outputs").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    try:
        example_1_smoke_test()
        example_2_custom_smoke_test()
        example_3_benchmark()
        example_4_load_and_predict()
        example_5_analyze_results()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        print("\n📁 Output files created in examples/outputs/")
        print("📖 See QUICKSTART.md for more information")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
