#!/usr/bin/env python3
"""
Profiling script for adaptive neural networks.

This script provides performance profiling and analysis tools to help
optimize the adaptive neural network implementation.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.utils.profiling import PerformanceProfiler, run_torch_profiler


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Profile adaptive neural network performance")

    # Profiling options
    parser.add_argument(
        "--profile-type",
        choices=["comprehensive", "torch", "both"],
        default="comprehensive",
        help="Type of profiling to run",
    )

    # Model configuration
    parser.add_argument("--num-nodes", type=int, default=100, help="Number of adaptive nodes")

    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for profiling")

    # Hardware configuration
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for profiling",
    )

    # Profiling parameters
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of iterations for timing"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_results",
        help="Output directory for profiling results",
    )

    # Output options
    parser.add_argument("--save-json", action="store_true", help="Save profiling results as JSON")

    parser.add_argument(
        "--json-file", type=str, default="profile_report.json", help="JSON output file name"
    )

    return parser.parse_args()


def main():
    """Main profiling execution."""
    args = parse_arguments()

    # Determine device
    if args.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Running performance profiling on {device}")

    # Create configuration
    config = AdaptiveConfig(
        num_nodes=args.num_nodes,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"Model configuration: {args.num_nodes} nodes, {args.hidden_dim} hidden dim")
    print(f"Batch size: {args.batch_size}")

    # Create profiler
    profiler = PerformanceProfiler(output_dir=args.output_dir)

    results = {}

    # Run comprehensive profiling
    if args.profile_type in ["comprehensive", "both"]:
        print("\n" + "=" * 50)
        print("RUNNING COMPREHENSIVE PROFILING")
        print("=" * 50)

        comprehensive_results = profiler.run_comprehensive_profile(
            config=config,
            save_results=False,  # We'll save separately
        )
        results["comprehensive"] = comprehensive_results

    # Run PyTorch profiler
    if args.profile_type in ["torch", "both"]:
        print("\n" + "=" * 50)
        print("RUNNING PYTORCH PROFILER")
        print("=" * 50)

        trace_file = run_torch_profiler(
            config=config, num_steps=args.num_iterations, output_dir=args.output_dir
        )
        results["torch_profiler"] = {"trace_file": trace_file, "config": config.to_dict()}

    # Save JSON report if requested
    if args.save_json:
        json_path = Path(args.output_dir) / args.json_file
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nProfiling report saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("PROFILING COMPLETED")
    print("=" * 50)

    if "comprehensive" in results:
        comp_results = results["comprehensive"]
        if "forward_pass" in comp_results:
            fp_time = comp_results["forward_pass"]["avg_forward_time"]
            throughput = comp_results["forward_pass"]["throughput_samples_per_sec"]
            print(f"Forward pass time: {fp_time * 1000:.2f} ms")
            print(f"Throughput: {throughput:.1f} samples/sec")

        if "memory_usage" in comp_results and "peak_memory_mb" in comp_results["memory_usage"]:
            memory_mb = comp_results["memory_usage"]["peak_memory_mb"]
            print(f"Peak memory usage: {memory_mb:.1f} MB")

    if "torch_profiler" in results:
        print(f"PyTorch trace available at: {results['torch_profiler']['trace_file']}")
        print("  View with: chrome://tracing/ or tensorboard --logdir profiling_results")

    return 0


if __name__ == "__main__":
    sys.exit(main())
