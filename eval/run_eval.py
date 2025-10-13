#!/usr/bin/env python3
"""
One-command evaluation and benchmark runner.

This script provides a single command to produce all evaluation and benchmark artifacts.

Usage:
    python eval/run_eval.py --model checkpoints/model.pt --dataset mnist
    python eval/run_eval.py --model checkpoints/model.pt --dataset mnist --full
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_deterministic_eval import run_evaluation as run_deterministic_evaluation


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Run complete evaluation and benchmarking"
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
        "--full",
        action="store_true",
        help="Run full evaluation suite (includes microbenchmarks, drift detection, comparison)",
    )
    
    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Enable all features if --full is specified
    if args.full:
        args.microbenchmark = True
        args.drift_detection = True
        args.compare = True
    else:
        args.microbenchmark = False
        args.drift_detection = False
        args.compare = False
    
    print("=" * 80)
    print("ADAPTIVE NEURAL NETWORK - EVALUATION & BENCHMARKING")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    
    if args.full:
        print("\nRunning FULL evaluation suite:")
        print("  ✓ Standard metrics")
        print("  ✓ Microbenchmarks")
        print("  ✓ Drift detection")
        print("  ✓ Metrics comparison")
    else:
        print("\nRunning STANDARD evaluation")
        print("  (Use --full for complete benchmarking)")
    
    print("\n" + "-" * 80)
    
    try:
        results = run_deterministic_evaluation(args)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Print summary
        if "metrics" in results:
            metrics = results["metrics"]
            print("\nKey Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Throughput: {metrics['throughput']:.2f} samples/sec")
        
        if "microbenchmark" in results:
            mb = results["microbenchmark"]
            print("\nMicrobenchmark Results:")
            print(f"  Forward latency: {mb['forward_latency_ms']['mean']:.3f} ms")
            print(f"  Latency std dev: {mb['forward_latency_ms']['std']:.3f} ms")
            
            # Calculate reproducibility variance percentage
            if mb['forward_latency_ms']['mean'] > 0:
                variance_pct = 100.0 * mb['forward_latency_ms']['std'] / mb['forward_latency_ms']['mean']
                print(f"  Repro variance: {variance_pct:.2f}%")
                
                if variance_pct < 5.0:
                    print("  ✓ Excellent reproducibility (< 5%)")
                elif variance_pct < 10.0:
                    print("  ✓ Good reproducibility (< 10%)")
                else:
                    print("  ⚠️ High variance (> 10%)")
        
        if "drift_detection" in results:
            drift_count = sum(1 for d in results["drift_detection"] if d["drift_detected"])
            print(f"\nDrift Detection: {drift_count} metric(s) showing drift")
        
        if "comparison" in results:
            improvements = sum(1 for c in results["comparison"] if c["is_improvement"])
            degradations = len(results["comparison"]) - improvements
            print(f"\nComparison: {improvements} improved, {degradations} degraded")
        
        print("\n" + "=" * 80)
        
        return 0
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
