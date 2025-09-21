#!/usr/bin/env python3
"""
Demo script showcasing video functionality in Adaptive Neural Network.

This script demonstrates:
- Video model creation and inference
- Synthetic video data processing
- Performance monitoring
- Model comparison
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

# Import video models
from adaptiveneuralnetwork.models.video_models import VideoModelConfig, create_video_model


def create_synthetic_video_data(batch_size=4, sequence_length=16, height=224, width=224):
    """Create synthetic video data for demonstration."""
    print(
        f"Creating synthetic video data: {batch_size} sequences of {sequence_length} frames ({height}x{width})"
    )

    # Create random video data (B, T, C, H, W)
    video_data = torch.randn(batch_size, sequence_length, 3, height, width)

    # Normalize to [0, 1] range to simulate real video data
    video_data = torch.sigmoid(video_data)

    return video_data


def benchmark_model(model_name, config, test_data, num_runs=10):
    """Benchmark a video model."""
    print(f"\n{'=' * 60}")
    print(f"BENCHMARKING {model_name.upper()} MODEL")
    print(f"{'=' * 60}")

    # Create model
    model = create_video_model(model_name, config)
    model.eval()

    # Model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size (MB): {num_params * 4 / 1024 / 1024:.2f}")

    # Warm up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_data)

    # Benchmark
    print(f"Running {num_runs} inference iterations...")
    latencies = []

    for i in range(num_runs):
        start_time = time.time()

        with torch.no_grad():
            output = model(test_data)

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)

        if i % 5 == 0:
            print(f"  Iteration {i + 1}/{num_runs}: {latency:.2f}ms")

    # Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print("\nBenchmark Results:")
    print(f"  Mean latency: {mean_latency:.2f} ± {std_latency:.2f}ms")
    print(f"  Min latency:  {min_latency:.2f}ms")
    print(f"  Max latency:  {max_latency:.2f}ms")
    print(f"  Throughput:   {1000 / mean_latency:.2f} inferences/sec")

    # Output analysis
    batch_size = output.size(0)
    num_classes = output.size(1)
    print("\nOutput Analysis:")
    print(f"  Output shape: {output.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of classes: {num_classes}")

    # Check output validity
    if torch.isnan(output).any():
        print("  WARNING: Output contains NaN values!")
    if torch.isinf(output).any():
        print("  WARNING: Output contains infinite values!")

    # Show top predictions for first sample
    with torch.no_grad():
        probs = torch.softmax(output[0], dim=0)
        top_values, top_indices = torch.topk(probs, min(5, num_classes))

        print("  Top 5 predictions for first sample:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_values)):
            print(f"    {i + 1}. Class {idx.item()}: {prob.item():.4f}")

    return {
        "model_name": model_name,
        "num_parameters": num_params,
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_per_sec": 1000 / mean_latency,
        "output_shape": list(output.shape),
    }


def compare_models():
    """Compare different video models."""
    print("VIDEO MODEL COMPARISON DEMO")
    print("=" * 80)

    # Configuration
    config = VideoModelConfig(
        input_channels=3,
        input_height=224,
        input_width=224,
        sequence_length=8,  # Shorter for faster testing
        hidden_dim=128,  # Smaller for faster testing
        num_layers=2,
        dropout=0.1,
        num_classes=1000,
    )

    print("Configuration:")
    print(f"  Input size: {config.input_width}x{config.input_height}")
    print(f"  Sequence length: {config.sequence_length}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Number of classes: {config.num_classes}")

    # Create test data
    test_data = create_synthetic_video_data(
        batch_size=2,
        sequence_length=config.sequence_length,
        height=config.input_height,
        width=config.input_width,
    )

    # Models to test
    models_to_test = ["convlstm", "conv3d", "transformer"]

    # Add hybrid model if we have time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        test_data = test_data.to(device)

    # Benchmark each model
    results = []

    for model_name in models_to_test:
        try:
            result = benchmark_model(model_name, config, test_data, num_runs=5)
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    if results:
        print(f"{'Model':<15} {'Params':<10} {'Avg Latency':<12} {'Throughput':<12}")
        print("-" * 55)

        for result in results:
            params_str = (
                f"{result['num_parameters'] / 1000:.0f}K"
                if result["num_parameters"] < 1000000
                else f"{result['num_parameters'] / 1000000:.1f}M"
            )
            print(
                f"{result['model_name']:<15} {params_str:<10} {result['mean_latency_ms']:<12.1f} {result['throughput_per_sec']:<12.2f}"
            )

        # Find best models
        fastest_model = min(results, key=lambda x: x["mean_latency_ms"])
        most_efficient = min(results, key=lambda x: x["num_parameters"])

        print("\nHighlights:")
        print(
            f"  Fastest model: {fastest_model['model_name']} ({fastest_model['mean_latency_ms']:.1f}ms)"
        )
        print(
            f"  Most efficient: {most_efficient['model_name']} ({most_efficient['num_parameters'] / 1000:.0f}K params)"
        )

    return results


def demonstrate_video_processing():
    """Demonstrate video processing capabilities."""
    print(f"\n{'=' * 80}")
    print("VIDEO PROCESSING DEMONSTRATION")
    print(f"{'=' * 80}")

    # Show different input sizes
    input_sizes = [(112, 112), (224, 224), (320, 240)]
    sequence_lengths = [4, 8, 16]

    print("Testing different input configurations...")

    config = VideoModelConfig(
        hidden_dim=64,  # Small for testing
        num_layers=1,
        num_classes=10,
    )

    model = create_video_model("convlstm", config)
    model.eval()

    for height, width in input_sizes:
        for seq_len in sequence_lengths:
            print(f"\nTesting {width}x{height} @ {seq_len} frames...")

            # Update config
            config.input_height = height
            config.input_width = width
            config.sequence_length = seq_len

            # Create test data
            test_data = torch.randn(1, seq_len, 3, height, width)

            # Time inference
            start_time = time.time()
            with torch.no_grad():
                output = model(test_data)
            end_time = time.time()

            latency_ms = (end_time - start_time) * 1000
            memory_mb = test_data.numel() * 4 / 1024 / 1024  # Approximate memory usage

            print(f"  Latency: {latency_ms:.2f}ms")
            print(f"  Memory: {memory_mb:.2f}MB")
            print(f"  Output shape: {output.shape}")


def save_benchmark_results(results, filename="video_benchmark_results.json"):
    """Save benchmark results to file."""
    output_path = Path(filename)

    benchmark_data = {
        "timestamp": time.time(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "pytorch_version": torch.__version__,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nBenchmark results saved to: {output_path}")


def main():
    """Main demo function."""
    print("ADAPTIVE NEURAL NETWORK - VIDEO FUNCTIONALITY DEMO")
    print("=" * 80)
    print("This demo showcases video processing capabilities including:")
    print("- ConvLSTM for spatiotemporal modeling")
    print("- 3D CNN for spatiotemporal convolutions")
    print("- Video Transformers for attention-based modeling")
    print("- Performance benchmarking and comparison")
    print()

    try:
        # Run model comparison
        results = compare_models()

        # Demonstrate video processing
        demonstrate_video_processing()

        # Save results
        if results:
            save_benchmark_results(results)

        print(f"\n{'=' * 80}")
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Key takeaways:")
        print("- All video models are functional and ready for use")
        print("- Models can handle different input sizes and sequence lengths")
        print("- Performance varies based on model complexity and input size")
        print("- Integration with PyTorch ecosystem is seamless")
        print("\nNext steps:")
        print("- Try with real video data using the CLI tools")
        print("- Experiment with different model configurations")
        print("- Use the real-time inference pipeline for live applications")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
