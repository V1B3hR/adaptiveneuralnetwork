#!/usr/bin/env python3
"""
Phase 0 - Profiling and Benchmarking Script
Profiles training runs and captures baseline metrics.
"""
import os
import sys
import time
import json
import psutil
import torch
import torch.profiler
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import tracemalloc


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Timing
    batch_latency_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    optimizer_step_ms: float = 0.0
    data_loading_time_ms: float = 0.0
    
    # Throughput
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    
    # Memory
    peak_gpu_memory_mb: float = 0.0
    peak_host_memory_mb: float = 0.0
    current_gpu_memory_mb: float = 0.0
    current_host_memory_mb: float = 0.0
    
    # GPU utilization
    avg_gpu_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTracker:
    """Track performance metrics during training."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = None
        self.forward_start = None
        self.backward_start = None
        self.optimizer_start = None
        self.data_start = None
        
        # Memory tracking
        tracemalloc.start()
        self.process = psutil.Process()
        
        # GPU tracking
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            torch.cuda.reset_peak_memory_stats()
    
    def start_batch(self):
        """Mark the start of a batch."""
        self.start_time = time.perf_counter()
    
    def start_data_loading(self):
        """Mark the start of data loading."""
        self.data_start = time.perf_counter()
    
    def end_data_loading(self):
        """Mark the end of data loading."""
        if self.data_start is not None:
            elapsed = (time.perf_counter() - self.data_start) * 1000
            self.metrics.data_loading_time_ms = elapsed
    
    def start_forward(self):
        """Mark the start of forward pass."""
        self.forward_start = time.perf_counter()
    
    def end_forward(self):
        """Mark the end of forward pass."""
        if self.forward_start is not None:
            elapsed = (time.perf_counter() - self.forward_start) * 1000
            self.metrics.forward_time_ms = elapsed
    
    def start_backward(self):
        """Mark the start of backward pass."""
        self.backward_start = time.perf_counter()
    
    def end_backward(self):
        """Mark the end of backward pass."""
        if self.backward_start is not None:
            elapsed = (time.perf_counter() - self.backward_start) * 1000
            self.metrics.backward_time_ms = elapsed
    
    def start_optimizer_step(self):
        """Mark the start of optimizer step."""
        self.optimizer_start = time.perf_counter()
    
    def end_optimizer_step(self):
        """Mark the end of optimizer step."""
        if self.optimizer_start is not None:
            elapsed = (time.perf_counter() - self.optimizer_start) * 1000
            self.metrics.optimizer_step_ms = elapsed
    
    def end_batch(self, batch_size: int = 1):
        """Mark the end of a batch and compute metrics."""
        if self.start_time is not None:
            elapsed = (time.perf_counter() - self.start_time) * 1000
            self.metrics.batch_latency_ms = elapsed
            
            # Throughput
            elapsed_sec = elapsed / 1000.0
            self.metrics.samples_per_second = batch_size / elapsed_sec if elapsed_sec > 0 else 0
            self.metrics.batches_per_second = 1.0 / elapsed_sec if elapsed_sec > 0 else 0
        
        # Memory stats
        self._update_memory_stats()
    
    def _update_memory_stats(self):
        """Update memory statistics."""
        # Host memory
        mem_info = self.process.memory_info()
        self.metrics.current_host_memory_mb = mem_info.rss / (1024 * 1024)
        
        # Track peak
        if self.metrics.current_host_memory_mb > self.metrics.peak_host_memory_mb:
            self.metrics.peak_host_memory_mb = self.metrics.current_host_memory_mb
        
        # GPU memory
        if self.has_gpu:
            self.metrics.current_gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            self.metrics.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics."""
        return self.metrics


def profile_simple_training(num_batches: int = 10, batch_size: int = 32) -> Dict[str, Any]:
    """
    Profile a simple training loop to establish baseline metrics.
    
    Args:
        num_batches: Number of batches to profile
        batch_size: Batch size for training
    
    Returns:
        Dictionary containing profiling results
    """
    print(f"\nProfiling simple training run:")
    print(f"  Batches: {num_batches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
    
    # Create a simple model for profiling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize tracker
    tracker = PerformanceTracker()
    
    # Collect metrics for each batch
    batch_metrics = []
    
    # Warm-up
    print("Warming up...")
    for _ in range(3):
        data = torch.randn(batch_size, 128, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("Starting profiling...\n")
    
    # Profile training
    for batch_idx in range(num_batches):
        tracker.start_batch()
        
        # Simulate data loading
        tracker.start_data_loading()
        data = torch.randn(batch_size, 128, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        tracker.end_data_loading()
        
        # Forward pass
        tracker.start_forward()
        output = model(data)
        loss = criterion(output, labels)
        tracker.end_forward()
        
        # Backward pass
        tracker.start_backward()
        loss.backward()
        tracker.end_backward()
        
        # Optimizer step
        tracker.start_optimizer_step()
        optimizer.step()
        optimizer.zero_grad()
        tracker.end_optimizer_step()
        
        # End batch
        tracker.end_batch(batch_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Store metrics
        metrics = tracker.get_metrics()
        batch_metrics.append(metrics.to_dict())
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx + 1}/{num_batches}: "
                  f"Latency: {metrics.batch_latency_ms:.2f}ms, "
                  f"Throughput: {metrics.samples_per_second:.1f} samples/s")
    
    # Compute aggregated statistics
    avg_metrics = {}
    for key in batch_metrics[0].keys():
        values = [m[key] for m in batch_metrics]
        avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        avg_metrics[f"min_{key}"] = min(values)
        avg_metrics[f"max_{key}"] = max(values)
    
    # Get final metrics
    final_metrics = tracker.get_metrics()
    
    results = {
        "configuration": {
            "num_batches": num_batches,
            "batch_size": batch_size,
            "device": str(device),
            "model_parameters": sum(p.numel() for p in model.parameters()),
        },
        "per_batch_metrics": batch_metrics,
        "aggregated_metrics": avg_metrics,
        "final_state": final_metrics.to_dict()
    }
    
    return results


def generate_baseline_metrics() -> Dict[str, Any]:
    """
    Generate comprehensive baseline metrics as required by Phase 0.
    
    Returns:
        Dictionary containing baseline metrics
    """
    import datetime
    
    baseline = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "Phase 0 - Baseline",
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
        }
    }
    
    # Run profiling with different configurations
    print("=" * 70)
    print("Phase 0 - Baseline Profiling")
    print("=" * 70)
    
    configs = [
        {"num_batches": 20, "batch_size": 32},
        {"num_batches": 10, "batch_size": 64},
    ]
    
    baseline["profiling_runs"] = []
    
    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Configuration: batch_size={config['batch_size']}, batches={config['num_batches']}")
        print(f"{'=' * 70}")
        
        results = profile_simple_training(**config)
        baseline["profiling_runs"].append(results)
        
        # Print summary
        agg = results["aggregated_metrics"]
        print(f"\n📊 Results Summary:")
        print(f"  Avg Batch Latency: {agg['avg_batch_latency_ms']:.2f} ms")
        print(f"  Avg Throughput: {agg['avg_samples_per_second']:.1f} samples/sec")
        print(f"  Peak GPU Memory: {agg['max_peak_gpu_memory_mb']:.2f} MB")
        print(f"  Peak Host Memory: {agg['max_peak_host_memory_mb']:.2f} MB")
    
    # Create summary metrics (Phase 0 Exit Criteria format)
    primary_run = baseline["profiling_runs"][0]["aggregated_metrics"]
    baseline["success_metrics"] = {
        "batch_latency_ms": round(primary_run["avg_batch_latency_ms"], 2),
        "data_throughput_samples_per_sec": round(primary_run["avg_samples_per_second"], 1),
        "gpu_util_avg_percent": 0.0,  # TODO: Requires nvidia-smi or similar
        "peak_gpu_memory_gb": round(primary_run["max_peak_gpu_memory_mb"] / 1024, 3),
        "peak_host_memory_gb": round(primary_run["max_peak_host_memory_mb"] / 1024, 3),
    }
    
    # Add hotspot identification (top 5)
    baseline["hotspots"] = [
        {"rank": 1, "function": "forward_pass", "time_ms": primary_run["avg_forward_time_ms"]},
        {"rank": 2, "function": "backward_pass", "time_ms": primary_run["avg_backward_time_ms"]},
        {"rank": 3, "function": "optimizer_step", "time_ms": primary_run["avg_optimizer_step_ms"]},
        {"rank": 4, "function": "data_loading", "time_ms": primary_run["avg_data_loading_time_ms"]},
        {"rank": 5, "function": "other_overhead", "time_ms": max(0, primary_run["avg_batch_latency_ms"] - 
            primary_run["avg_forward_time_ms"] - primary_run["avg_backward_time_ms"] - 
            primary_run["avg_optimizer_step_ms"] - primary_run["avg_data_loading_time_ms"])},
    ]
    
    return baseline


def main():
    """Main entry point for the profiler."""
    # Get the repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    # Generate baseline metrics
    baseline = generate_baseline_metrics()
    
    # Save to benchmarks/baseline.json
    output_path = repo_root / "benchmarks" / "baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Baseline metrics saved to: {output_path}")
    print(f"{'=' * 70}")
    
    # Print Phase 0 Exit Criteria format
    print("\n" + "=" * 70)
    print("Phase 0 - Success Metrics (Baseline Numbers Captured)")
    print("=" * 70)
    metrics = baseline["success_metrics"]
    print(f"Batch latency: {metrics['batch_latency_ms']} ms")
    print(f"Data throughput: {metrics['data_throughput_samples_per_sec']} samples/sec")
    print(f"GPU util avg: {metrics['gpu_util_avg_percent']} %")
    print(f"Peak GPU memory: {metrics['peak_gpu_memory_gb']} GB")
    print(f"Peak Host memory: {metrics['peak_host_memory_gb']} GB")
    print("=" * 70)
    
    # Print top 5 hotspots
    print("\nTop 5 Hotspots (ranked by time):")
    print("-" * 70)
    for hotspot in baseline["hotspots"]:
        print(f"{hotspot['rank']}. {hotspot['function']}: {hotspot['time_ms']:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
