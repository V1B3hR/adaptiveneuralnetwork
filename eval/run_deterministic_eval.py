#!/usr/bin/env python3
"""
Deterministic evaluation script for reproducible model assessment.

This script ensures reproducible evaluation by:
- Setting all random seeds
- Using deterministic algorithms
- Capturing environment state
- Producing versioned benchmark results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptiveneuralnetwork.utils.reproducibility import (
    EnvironmentSnapshot,
    ReproducibilityHarness,
)
from eval.metrics import compute_metrics, StandardMetrics
from eval.microbenchmark import run_microbenchmark
from eval.drift_detection import detect_drift
from eval.comparison import compare_metrics, MetricsComparator


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Run deterministic model evaluation"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to evaluate on",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/history",
        help="Directory to save results",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (cpu/cuda)",
    )
    
    parser.add_argument(
        "--microbenchmark",
        action="store_true",
        help="Run microbenchmarks",
    )
    
    parser.add_argument(
        "--drift-detection",
        action="store_true",
        help="Run drift detection",
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with previous run",
    )
    
    return parser


def load_model_and_data(
    model_path: str,
    dataset: str,
    batch_size: int,
    device: torch.device,
) -> tuple[nn.Module, DataLoader]:
    """
    Load model and dataset.
    
    Note: This is a placeholder that should be extended based on your models.
    """
    # Import here to avoid circular dependencies
    from adaptiveneuralnetwork.api.model import AdaptiveModel
    from adaptiveneuralnetwork.api.config import AdaptiveConfig
    
    # Load model
    if Path(model_path).exists():
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            # Default config
            config = AdaptiveConfig()
        
        model = AdaptiveModel(config)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Create new model for testing
        config = AdaptiveConfig()
        model = AdaptiveModel(config)
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    if dataset.lower() == "mnist":
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return model, test_loader


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with all results
    """
    # Setup reproducibility
    harness = ReproducibilityHarness(master_seed=args.seed, strict_mode=True)
    harness.set_seed()
    
    # Capture environment
    env_snapshot = EnvironmentSnapshot.capture()
    
    # Setup device
    device = torch.device(args.device)
    
    # Load model and data
    print(f"Loading model from {args.model}...")
    model, test_loader = load_model_and_data(
        args.model,
        args.dataset,
        args.batch_size,
        device,
    )
    
    print(f"Running evaluation on {args.dataset} dataset...")
    
    # Run standard evaluation
    loss_fn = nn.CrossEntropyLoss()
    metrics = compute_metrics(
        model=model,
        data_loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        compute_detailed=True,
    )
    
    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {metrics.accuracy:.2f}%")
    print(f"  Loss: {metrics.loss:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall: {metrics.recall:.4f}")
    print(f"  F1 Score: {metrics.f1_score:.4f}")
    print(f"  Throughput: {metrics.throughput:.2f} samples/sec")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "model_path": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": metrics.to_dict(),
        "environment": {
            "python_version": env_snapshot.python_version.split()[0],
            "torch_version": env_snapshot.torch_version,
            "cuda_available": env_snapshot.cuda_available,
        },
    }
    
    # Run microbenchmarks if requested
    if args.microbenchmark:
        print("\nRunning microbenchmarks...")
        microbenchmark_results = run_microbenchmark(
            model=model,
            data_loader=test_loader,
            device=device,
            num_iterations=100,
            warmup_iterations=10,
        )
        
        print(f"  Forward latency: {microbenchmark_results.forward_latency_mean:.3f} ± "
              f"{microbenchmark_results.forward_latency_std:.3f} ms")
        print(f"  Data loader throughput: {microbenchmark_results.data_loader_throughput:.2f} samples/sec")
        print(f"  Peak memory (GPU): {microbenchmark_results.peak_gpu_memory_mb:.2f} MB")
        print(f"  Peak memory (CPU): {microbenchmark_results.peak_cpu_memory_mb:.2f} MB")
        
        results["microbenchmark"] = microbenchmark_results.to_dict()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_file = output_dir / f"{timestamp_str}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Run drift detection if requested
    if args.drift_detection:
        print("\nRunning drift detection...")
        
        drift_metrics = {
            "accuracy": metrics.accuracy,
            "loss": metrics.loss,
            "latency_ms": metrics.latency_ms,
        }
        
        drift_results = detect_drift(
            current_metrics=drift_metrics,
            history_path=args.output_dir,
            lookback_n=5,
            threshold_std=2.0,
            metric_directions={
                "accuracy": True,
                "loss": False,
                "latency_ms": False,
            },
        )
        
        for drift_result in drift_results:
            if drift_result.drift_detected:
                print(f"  ⚠️  Drift detected in {drift_result.metric_name}:")
                print(f"      Current: {drift_result.current_value:.4f}")
                print(f"      Baseline: {drift_result.baseline_median:.4f}")
                print(f"      Direction: {drift_result.drift_direction}")
                print(f"      Change: {drift_result.drift_percentage:+.2f}%")
            else:
                print(f"  ✓  {drift_result.metric_name}: stable")
        
        results["drift_detection"] = [d.to_dict() for d in drift_results]
    
    # Run comparison if requested
    if args.compare:
        print("\nComparing with previous run...")
        
        comparator = MetricsComparator(args.output_dir)
        comparisons = compare_metrics(
            history_path=args.output_dir,
            metric_directions={
                "accuracy": True,
                "loss": False,
                "latency_ms": False,
            },
        )
        
        if comparisons:
            for comp in comparisons[:5]:  # Show top 5
                status = "↑" if comp.is_improvement else "↓"
                print(f"  {status} {comp.metric_name}: {comp.change_percentage:+.2f}%")
            
            results["comparison"] = [c.to_dict() for c in comparisons]
    
    return results


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        results = run_evaluation(args)
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return 0
    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
