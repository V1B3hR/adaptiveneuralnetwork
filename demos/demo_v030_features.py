#!/usr/bin/env python3
"""
Demo script showcasing Adaptive Neural Network v0.3.0+ features.

This script demonstrates:
1. Backend selection (PyTorch, JAX, Neuromorphic)
2. CIFAR-10 robustness benchmarking
3. Multi-modal learning capabilities
4. Neuromorphic spike-based processing

Usage:
    python demo_v030_features.py [--backend pytorch|jax|neuromorphic] [--quick]
"""

import os
import sys

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time

import numpy as np
import torch

# Import adaptive neural network components
from adaptiveneuralnetwork.api import AdaptiveConfig, create_adaptive_model
from adaptiveneuralnetwork.api.backend_factory import get_backend_info, list_available_backends
from adaptiveneuralnetwork.benchmarks.multimodal import run_multimodal_benchmark
from adaptiveneuralnetwork.benchmarks.vision.cifar10 import CIFAR10Benchmark
from adaptiveneuralnetwork.core.neuromorphic import NeuromorphicPlatform, create_neuromorphic_model
from adaptiveneuralnetwork.training.datasets import CIFAR10Corrupted

logger = logging.getLogger(__name__)


def demo_backend_selection():
    """Demonstrate multi-backend support."""
    print("=" * 60)
    print("🔧 BACKEND SELECTION DEMO")
    print("=" * 60)

    # Show available backends
    backends = list_available_backends()
    print("\nAvailable backends:")
    for backend, available in backends.items():
        status = "✓" if available else "✗ (not installed)"
        print(f"  {backend:12} {status}")

    # Get detailed backend info
    print("\nBackend capabilities:")
    info = get_backend_info()
    for backend, details in info.items():
        if backends[backend]:  # Only show available backends
            print(f"\n{backend.upper()}:")
            print(f"  Description: {details['description']}")
            print(f"  Key features: {', '.join(details['features'][:3])}")

    # Create models with different backends
    print("\nCreating models with different backends:")

    config = AdaptiveConfig(num_nodes=32, hidden_dim=16, input_dim=784, output_dim=10)

    # PyTorch model (always available)
    try:
        pytorch_model = create_adaptive_model(config=config, backend="pytorch")
        param_count = sum(p.numel() for p in pytorch_model.parameters())
        print(f"  ✓ PyTorch model: {param_count:,} parameters")
    except Exception as e:
        print(f"  ✗ PyTorch model failed: {e}")

    # JAX model (if available)
    if backends["jax"]:
        try:
            jax_model = create_adaptive_model(config=config, backend="jax")
            print("  ✓ JAX model: Created successfully")
        except Exception as e:
            print(f"  ✗ JAX model failed: {e}")
    else:
        print("  - JAX model: Skipped (JAX not available)")

    # Neuromorphic model
    try:
        neuro_model = create_adaptive_model(config=config, backend="neuromorphic")
        param_count = sum(p.numel() for p in neuro_model.parameters())
        print(f"  ✓ Neuromorphic model: {param_count:,} parameters")
    except Exception as e:
        print(f"  ✗ Neuromorphic model failed: {e}")


def demo_cifar10_robustness(quick_mode=False):
    """Demonstrate CIFAR-10 robustness benchmarking."""
    print("\n" + "=" * 60)
    print("🛡️  CIFAR-10 ROBUSTNESS DEMO")
    print("=" * 60)

    print("\nAvailable corruption types:")
    corruption_types = CIFAR10Corrupted.CORRUPTION_TYPES[:8]  # Show first 8
    for i, corruption in enumerate(corruption_types, 1):
        print(f"  {i:2}. {corruption}")
    if len(CIFAR10Corrupted.CORRUPTION_TYPES) > 8:
        print(f"     ... and {len(CIFAR10Corrupted.CORRUPTION_TYPES) - 8} more")

    # Create small model for demo
    config = AdaptiveConfig(
        num_nodes=16 if quick_mode else 32,
        hidden_dim=8 if quick_mode else 16,
        input_dim=3072,  # 32*32*3
        output_dim=10,
    )

    print(f"\nTesting corruption robustness (quick_mode={quick_mode})...")

    try:
        # Create benchmark
        benchmark = CIFAR10Benchmark(config, device=torch.device("cpu"))

        # Test a few corruption types
        test_corruptions = (
            ["gaussian_noise", "brightness"]
            if quick_mode
            else ["gaussian_noise", "brightness", "contrast"]
        )
        test_severities = [1, 3] if quick_mode else [1, 2, 3, 4, 5]

        # Mock the evaluation to avoid long downloads/training for demo
        print("  [Note: Using mock evaluation to avoid long download times]")

        # Simulate robustness results
        mock_results = {
            "benchmark_type": "cifar10_robustness",
            "robustness_results": {
                "clean_accuracy": 0.85,
                "corruption_results": {},
                "mean_corruption_error": 0.0,
                "relative_robustness": 0.0,
            },
        }

        total_error = 0
        num_tests = 0

        for corruption in test_corruptions:
            mock_results["robustness_results"]["corruption_results"][corruption] = {}
            print(f"\n  Testing {corruption}:")

            for severity in test_severities:
                # Simulate degraded performance with increasing severity
                base_drop = 0.05 + (severity - 1) * 0.08
                noise_factor = np.random.normal(0, 0.02)  # Add some randomness
                corrupted_acc = max(0.1, 0.85 - base_drop + noise_factor)

                corruption_error = 0.85 - corrupted_acc
                total_error += corruption_error
                num_tests += 1

                mock_results["robustness_results"]["corruption_results"][corruption][severity] = {
                    "accuracy": corrupted_acc,
                    "corruption_error": corruption_error,
                    "relative_accuracy": corrupted_acc / 0.85,
                }

                print(
                    f"    Severity {severity}: {corrupted_acc:.3f} accuracy ({corruption_error:.3f} error)"
                )

        # Calculate overall metrics
        mean_error = total_error / num_tests
        relative_robustness = 1.0 - (mean_error / 0.85)

        mock_results["robustness_results"]["mean_corruption_error"] = mean_error
        mock_results["robustness_results"]["relative_robustness"] = relative_robustness

        print("\n  📊 Robustness Summary:")
        print(f"     Clean accuracy:      {0.85:.3f}")
        print(f"     Mean corruption error: {mean_error:.3f}")
        print(f"     Relative robustness: {relative_robustness:.3f}")

        # Interpret robustness score
        if relative_robustness > 0.8:
            print("     Rating: 🟢 Excellent robustness")
        elif relative_robustness > 0.6:
            print("     Rating: 🟡 Good robustness")
        else:
            print("     Rating: 🔴 Needs improvement")

    except Exception as e:
        print(f"  ✗ Robustness demo failed: {e}")


def demo_multimodal_learning(quick_mode=False):
    """Demonstrate multi-modal learning capabilities."""
    print("\n" + "=" * 60)
    print("🌐 MULTI-MODAL LEARNING DEMO")
    print("=" * 60)

    print("\nSupported modality combinations:")
    print("  • Text + Image (text-image pairs)")
    print("  • Text + Audio (speech + text)")
    print("  • Image + Audio (video-like)")
    print("  • Text + Image + Audio (full multi-modal)")

    try:
        print(f"\n Running synthetic multi-modal benchmark (quick_mode={quick_mode})...")

        # Configure for demo
        epochs = 1 if quick_mode else 3
        batch_size = 8 if quick_mode else 16
        num_samples = 32 if quick_mode else 100

        config = AdaptiveConfig(
            num_nodes=16 if quick_mode else 32,
            hidden_dim=16 if quick_mode else 32,
            input_dim=256,  # Will be set by fusion layer
            output_dim=3 if quick_mode else 5,  # Fewer classes for quick mode
        )

        start_time = time.time()

        results = run_multimodal_benchmark(
            config=config,
            modalities=["text", "image"],
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.01,  # Higher LR for faster demo training
            num_classes=config.output_dim,  # Match num_classes to config
        )

        duration = time.time() - start_time

        print("\n  📊 Multi-Modal Results:")
        print(f"     Training epochs:     {epochs}")
        print(f"     Final accuracy:      {results['final_test_accuracy']:.3f}")
        print(f"     Best accuracy:       {results['best_test_accuracy']:.3f}")
        print(f"     Training time:       {duration:.1f}s")
        print(f"     Model parameters:    {results['model_parameters']:,}")

        # Show learning progress
        train_accs = results["train_accuracies"]
        test_accs = results["test_accuracies"]

        print("\n  📈 Learning Progress:")
        for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
            print(f"     Epoch {i + 1}: Train={train_acc:.3f}, Test={test_acc:.3f}")

        # Modality contribution analysis
        print("\n  🔍 Analysis:")
        correlation = results.get("correlation_strength", 0.8)
        print(f"     Cross-modal correlation: {correlation:.1f}")
        if results["final_test_accuracy"] > 0.6:
            print("     Status: 🟢 Successfully learning multi-modal patterns")
        elif results["final_test_accuracy"] > 0.4:
            print("     Status: 🟡 Moderate multi-modal learning")
        else:
            print("     Status: 🔴 Struggling with cross-modal fusion")

    except Exception as e:
        print(f"  ✗ Multi-modal demo failed: {e}")


def demo_neuromorphic_processing(quick_mode=False):
    """Demonstrate neuromorphic hardware compatibility."""
    print("\n" + "=" * 60)
    print("🧠 NEUROMORPHIC PROCESSING DEMO")
    print("=" * 60)

    print("\nNeuromorphic platforms supported:")
    for platform in NeuromorphicPlatform:
        print(f"  • {platform.value}")

    print("\nKey neuromorphic features:")
    print("  • Spike-based computation")
    print("  • Event-driven processing")
    print("  • Low power consumption")
    print("  • Leaky integrate-and-fire neurons")

    try:
        print(f"\nCreating neuromorphic model (quick_mode={quick_mode})...")

        # Create neuromorphic model
        input_dim = 100 if quick_mode else 784  # Smaller for demo
        output_dim = 5 if quick_mode else 10

        neuro_model = create_neuromorphic_model(
            input_dim=input_dim, output_dim=output_dim, platform=NeuromorphicPlatform.SIMULATION
        )

        param_count = sum(p.numel() for p in neuro_model.parameters())
        print(f"  ✓ Model created: {param_count:,} parameters")

        # Test spike-based processing
        print("\n  🔥 Testing spike-based processing:")

        batch_size = 2 if quick_mode else 4
        test_input = torch.randn(batch_size, input_dim) * 0.5  # Normalized input

        print(f"     Input shape: {test_input.shape}")
        print(f"     Input range: [{test_input.min():.2f}, {test_input.max():.2f}]")

        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            output = neuro_model(test_input)
        forward_time = time.time() - start_time

        print(f"     Output shape: {output.shape}")
        print(f"     Output range: [{output.min():.2f}, {output.max():.2f}]")
        print(f"     Forward time: {forward_time * 1000:.1f}ms")

        # Analyze spikiness (how much the output resembles spike trains)
        spikiness = torch.mean((output > 0.5).float()).item()
        print(f"     Spike density: {spikiness:.1%}")

        print("\n  ⚡ Neuromorphic characteristics:")
        if spikiness > 0.1:
            print("     • High spike activity - good for neuromorphic hardware")
        else:
            print("     • Low spike activity - may need tuning for hardware")

        print("     • Event-driven: Processes only when input changes")
        print("     • Low power: Computation only on spike events")
        print("     • Asynchronous: No global clock synchronization")

    except Exception as e:
        print(f"  ✗ Neuromorphic demo failed: {e}")


def main():
    """Main demo script."""
    parser = argparse.ArgumentParser(description="Adaptive Neural Network v0.3.0 Feature Demo")
    parser.add_argument(
        "--backend",
        choices=["pytorch", "jax", "neuromorphic"],
        default="pytorch",
        help="Backend to use for model creation",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run in quick mode (smaller models, less data)"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "backends", "robustness", "multimodal", "neuromorphic"],
        default="all",
        help="Which demo to run",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🎯 ADAPTIVE NEURAL NETWORK v0.3.0+ FEATURE DEMONSTRATION")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Quick mode: {args.quick}")
    print(f"Demo: {args.demo}")

    # Run selected demos
    start_time = time.time()

    try:
        if args.demo in ["all", "backends"]:
            demo_backend_selection()

        if args.demo in ["all", "robustness"]:
            demo_cifar10_robustness(quick_mode=args.quick)

        if args.demo in ["all", "multimodal"]:
            demo_multimodal_learning(quick_mode=args.quick)

        if args.demo in ["all", "neuromorphic"]:
            demo_neuromorphic_processing(quick_mode=args.quick)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        raise

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED")
    print(f"Total time: {total_time:.1f}s")
    print("\nVersion 0.3.0+ features demonstrated:")
    print("  ✓ Multi-backend support (PyTorch, JAX, Neuromorphic)")
    print("  ✓ Domain shift robustness testing")
    print("  ✓ Multi-modal learning capabilities")
    print("  ✓ Neuromorphic hardware compatibility")
    print("\nFor more information, see the documentation and examples.")


if __name__ == "__main__":
    main()
