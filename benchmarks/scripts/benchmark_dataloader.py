#!/usr/bin/env python3
"""
Phase 1 Data Loader Benchmark Script

This script benchmarks data loading performance to validate Phase 1 improvements.
It measures:
- Data throughput (samples/sec)
- Batch loading time
- Prefetch efficiency
- CPU time share

Usage:
    python benchmarks/scripts/benchmark_dataloader.py
    python benchmarks/scripts/benchmark_dataloader.py --baseline  # Use baseline implementation
    python benchmarks/scripts/benchmark_dataloader.py --optimized  # Use Phase 1 optimized implementation
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Direct imports to avoid circular dependencies
from adaptiveneuralnetwork.training.datasets.datasets import SyntheticDataset

# Import the optimized dataset module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "optimized_datasets",
    project_root / "adaptiveneuralnetwork" / "data" / "optimized_datasets.py"
)
optimized_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimized_datasets)

# Use functions from the module
VectorizedDataset = optimized_datasets.VectorizedDataset
create_optimized_loader = optimized_datasets.create_optimized_loader
optimize_dataset = optimized_datasets.optimize_dataset
vectorized_collate_fn = optimized_datasets.vectorized_collate_fn

from torch.utils.data import DataLoader


class BenchmarkTimer:
    """Simple timer for benchmarking."""
    
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed
    
    def get_elapsed_ms(self) -> float:
        return self.elapsed * 1000


def create_baseline_loader(dataset, batch_size: int, num_workers: int = 0) -> DataLoader:
    """Create baseline DataLoader without optimizations."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        # Default collate_fn (per-sample Python loops)
    )


def benchmark_loader(
    loader: DataLoader,
    num_batches: int = 100,
    warmup_batches: int = 10,
    name: str = "loader"
) -> Dict[str, Any]:
    """
    Benchmark a data loader.
    
    Args:
        loader: DataLoader to benchmark
        num_batches: Number of batches to measure
        warmup_batches: Number of warmup batches (not measured)
        name: Name for this benchmark
        
    Returns:
        Dictionary of benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    timer = BenchmarkTimer()
    batch_times = []
    total_samples = 0
    
    # Warmup
    print(f"Warming up ({warmup_batches} batches)...")
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
    
    # Actual benchmark
    print(f"Benchmarking ({num_batches} batches)...")
    iter_loader = iter(loader)
    
    for i in range(num_batches):
        try:
            timer.start()
            batch = next(iter_loader)
            elapsed_ms = timer.stop() * 1000
            
            batch_times.append(elapsed_ms)
            
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                batch_size = len(batch[0])
            else:
                batch_size = len(batch)
            
            total_samples += batch_size
            
            if (i + 1) % 20 == 0:
                avg_time = np.mean(batch_times[-20:])
                throughput = (20 * batch_size) / (avg_time / 1000 * 20)
                print(f"  Batch {i+1}/{num_batches}: {avg_time:.3f}ms/batch, {throughput:.0f} samples/sec")
        
        except StopIteration:
            print(f"  Loader exhausted after {i} batches")
            break
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    avg_batch_time_ms = np.mean(batch_times)
    min_batch_time_ms = np.min(batch_times)
    max_batch_time_ms = np.max(batch_times)
    std_batch_time_ms = np.std(batch_times)
    
    total_time_sec = np.sum(batch_times) / 1000
    throughput_samples_per_sec = total_samples / total_time_sec
    
    results = {
        'name': name,
        'num_batches_measured': len(batch_times),
        'total_samples': total_samples,
        'total_time_sec': total_time_sec,
        'avg_batch_time_ms': float(avg_batch_time_ms),
        'min_batch_time_ms': float(min_batch_time_ms),
        'max_batch_time_ms': float(max_batch_time_ms),
        'std_batch_time_ms': float(std_batch_time_ms),
        'throughput_samples_per_sec': float(throughput_samples_per_sec),
        'throughput_batches_per_sec': float(len(batch_times) / total_time_sec),
    }
    
    print(f"\nResults for {name}:")
    print(f"  Average batch time: {avg_batch_time_ms:.3f} ± {std_batch_time_ms:.3f} ms")
    print(f"  Min/Max batch time: {min_batch_time_ms:.3f} / {max_batch_time_ms:.3f} ms")
    print(f"  Throughput: {throughput_samples_per_sec:.0f} samples/sec")
    print(f"  Batches/sec: {results['throughput_batches_per_sec']:.1f}")
    
    return results


def run_comparison_benchmark(
    num_samples: int = 10000,
    batch_size: int = 32,
    input_dim: int = 784,
    num_classes: int = 10,
    num_batches: int = 100,
    num_workers: int = 0
) -> Dict[str, Any]:
    """
    Run comparison between baseline and optimized loaders.
    
    Args:
        num_samples: Number of samples in synthetic dataset
        batch_size: Batch size
        input_dim: Input dimension
        num_classes: Number of classes
        num_batches: Number of batches to benchmark
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with all benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Phase 1 Data Loader Benchmark")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Dataset size: {num_samples} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Input dim: {input_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Num workers: {num_workers}")
    print(f"  Batches to measure: {num_batches}")
    
    # Create base dataset
    base_dataset = SyntheticDataset(
        num_samples=num_samples,
        input_dim=input_dim,
        num_classes=num_classes
    )
    
    # Benchmark 1: Baseline loader (no optimizations)
    baseline_loader = create_baseline_loader(base_dataset, batch_size, num_workers=num_workers)
    baseline_results = benchmark_loader(
        baseline_loader,
        num_batches=num_batches,
        warmup_batches=10,
        name="Baseline (no optimizations)"
    )
    
    # Benchmark 2: Optimized loader with vectorized collation
    optimized_loader_1 = create_optimized_loader(
        base_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    optimized_results_1 = benchmark_loader(
        optimized_loader_1,
        num_batches=num_batches,
        warmup_batches=10,
        name="Optimized (vectorized collation + pinned memory)"
    )
    
    # Benchmark 3: Pre-loaded optimized dataset
    preloaded_dataset = optimize_dataset(base_dataset, preload=True, pin_memory=True)
    optimized_loader_2 = create_optimized_loader(
        preloaded_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No workers needed for preloaded data
        pin_memory=True,
        prefetch_factor=None
    )
    optimized_results_2 = benchmark_loader(
        optimized_loader_2,
        num_batches=num_batches,
        warmup_batches=10,
        name="Optimized (pre-loaded + vectorized)"
    )
    
    # Calculate improvements
    print(f"\n{'='*60}")
    print(f"Improvement Analysis")
    print(f"{'='*60}")
    
    baseline_throughput = baseline_results['throughput_samples_per_sec']
    opt1_throughput = optimized_results_1['throughput_samples_per_sec']
    opt2_throughput = optimized_results_2['throughput_samples_per_sec']
    
    improvement_1 = ((opt1_throughput - baseline_throughput) / baseline_throughput) * 100
    improvement_2 = ((opt2_throughput - baseline_throughput) / baseline_throughput) * 100
    
    print(f"Baseline throughput: {baseline_throughput:.0f} samples/sec")
    print(f"Optimized (collation) throughput: {opt1_throughput:.0f} samples/sec ({improvement_1:+.1f}%)")
    print(f"Optimized (preloaded) throughput: {opt2_throughput:.0f} samples/sec ({improvement_2:+.1f}%)")
    
    baseline_time = baseline_results['avg_batch_time_ms']
    opt1_time = optimized_results_1['avg_batch_time_ms']
    opt2_time = optimized_results_2['avg_batch_time_ms']
    
    time_reduction_1 = ((baseline_time - opt1_time) / baseline_time) * 100
    time_reduction_2 = ((baseline_time - opt2_time) / baseline_time) * 100
    
    print(f"\nBaseline batch time: {baseline_time:.3f} ms")
    print(f"Optimized (collation) batch time: {opt1_time:.3f} ms ({time_reduction_1:+.1f}%)")
    print(f"Optimized (preloaded) batch time: {opt2_time:.3f} ms ({time_reduction_2:+.1f}%)")
    
    # Compare with Phase 0 baseline
    phase0_throughput = 20240.0  # From baseline.json
    phase0_data_time = 0.067  # ms
    
    print(f"\n{'='*60}")
    print(f"Comparison with Phase 0 Baseline")
    print(f"{'='*60}")
    print(f"Phase 0 baseline throughput: {phase0_throughput:.0f} samples/sec")
    print(f"Phase 0 data loading time: {phase0_data_time:.3f} ms/batch")
    
    phase1_improvement = ((opt2_throughput - phase0_throughput) / phase0_throughput) * 100
    print(f"Phase 1 best improvement: {phase1_improvement:+.1f}%")
    print(f"Phase 1 target (+30%): {'✓ ACHIEVED' if phase1_improvement >= 30 else '✗ NOT YET'}")
    
    # Compile results
    results = {
        'configuration': {
            'num_samples': num_samples,
            'batch_size': batch_size,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'num_batches': num_batches,
            'num_workers': num_workers
        },
        'baseline': baseline_results,
        'optimized_collation': optimized_results_1,
        'optimized_preloaded': optimized_results_2,
        'improvements': {
            'optimized_collation_percent': float(improvement_1),
            'optimized_preloaded_percent': float(improvement_2),
            'vs_phase0_percent': float(phase1_improvement),
            'phase1_target_achieved': phase1_improvement >= 30
        },
        'phase0_baseline': {
            'throughput_samples_per_sec': phase0_throughput,
            'data_loading_time_ms': phase0_data_time
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Phase 1 data loader optimizations')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--input-dim', type=int, default=784, help='Input dimension')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--num-batches', type=int, default=100, help='Batches to benchmark')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of worker processes')
    parser.add_argument('--output', type=str, default='benchmarks/phase1_metrics.json', 
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_comparison_benchmark(
        num_samples=args.samples,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        num_batches=args.num_batches,
        num_workers=args.num_workers
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
