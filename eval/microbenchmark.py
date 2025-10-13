"""
Microbenchmarking utilities for performance measurement.

Provides fine-grained benchmarks for:
- Forward pass latency
- Data loader throughput
- Memory usage
"""

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class MicroBenchmarkResults:
    """Container for microbenchmark results."""
    
    # Latency metrics (in milliseconds)
    forward_latency_mean: float = 0.0
    forward_latency_std: float = 0.0
    forward_latency_min: float = 0.0
    forward_latency_max: float = 0.0
    
    # Throughput metrics
    data_loader_throughput: float = 0.0  # samples/second
    batches_per_second: float = 0.0
    
    # Memory metrics (in MB)
    peak_gpu_memory_mb: float = 0.0
    current_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0
    
    # Additional info
    num_iterations: int = 0
    batch_size: int = 0
    device: str = "cpu"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "forward_latency_ms": {
                "mean": self.forward_latency_mean,
                "std": self.forward_latency_std,
                "min": self.forward_latency_min,
                "max": self.forward_latency_max,
            },
            "throughput": {
                "data_loader_samples_per_sec": self.data_loader_throughput,
                "batches_per_second": self.batches_per_second,
            },
            "memory_mb": {
                "peak_gpu": self.peak_gpu_memory_mb,
                "current_gpu": self.current_gpu_memory_mb,
                "peak_cpu": self.peak_cpu_memory_mb,
            },
            "config": {
                "num_iterations": self.num_iterations,
                "batch_size": self.batch_size,
                "device": self.device,
            },
        }


class MicroBenchmark:
    """Microbenchmarking utility for neural networks."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize microbenchmark.
        
        Args:
            model: Model to benchmark
            device: Device to run benchmark on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def benchmark_forward_latency(
        self,
        data_loader: DataLoader,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark forward pass latency.
        
        Args:
            data_loader: Data loader for benchmark data
            num_iterations: Number of iterations to benchmark
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        # Get a single batch for repeated benchmarking
        data_iter = iter(data_loader)
        batch = next(data_iter)
        
        if isinstance(batch, (tuple, list)):
            inputs = batch[0].to(self.device)
        else:
            inputs = batch.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(inputs)
        
        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = self.model(inputs)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        import numpy as np
        return {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }
    
    def benchmark_data_loader(
        self,
        data_loader: DataLoader,
        num_batches: int = 100,
    ) -> dict[str, float]:
        """
        Benchmark data loader throughput.
        
        Args:
            data_loader: Data loader to benchmark
            num_batches: Number of batches to process
            
        Returns:
            Dictionary with throughput metrics
        """
        start_time = time.perf_counter()
        total_samples = 0
        batches_processed = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            if isinstance(batch, (tuple, list)):
                batch_size = batch[0].size(0)
            else:
                batch_size = batch.size(0)
            
            total_samples += batch_size
            batches_processed += 1
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        return {
            "samples_per_second": total_samples / elapsed_time if elapsed_time > 0 else 0.0,
            "batches_per_second": batches_processed / elapsed_time if elapsed_time > 0 else 0.0,
        }
    
    def benchmark_memory(
        self,
        data_loader: DataLoader,
        num_iterations: int = 10,
    ) -> dict[str, float]:
        """
        Benchmark memory usage.
        
        Args:
            data_loader: Data loader for benchmark data
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary with memory metrics
        """
        import psutil
        
        process = psutil.Process()
        
        # Initial memory
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        initial_cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run iterations
        data_iter = iter(data_loader)
        with torch.no_grad():
            for i in range(num_iterations):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    batch = next(data_iter)
                
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                _ = self.model(inputs)
        
        # Measure peak memory
        peak_cpu_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_metrics = {
            "peak_cpu_mb": peak_cpu_memory - initial_cpu_memory,
            "peak_gpu_mb": 0.0,
            "current_gpu_mb": 0.0,
        }
        
        if self.device.type == "cuda":
            memory_metrics["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            memory_metrics["current_gpu_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        
        return memory_metrics
    
    def run_full_benchmark(
        self,
        data_loader: DataLoader,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> MicroBenchmarkResults:
        """
        Run full microbenchmark suite.
        
        Args:
            data_loader: Data loader for benchmark data
            num_iterations: Number of iterations for latency benchmark
            warmup_iterations: Number of warmup iterations
            
        Returns:
            MicroBenchmarkResults with all metrics
        """
        # Benchmark forward latency
        latency_metrics = self.benchmark_forward_latency(
            data_loader, num_iterations, warmup_iterations
        )
        
        # Benchmark data loader
        throughput_metrics = self.benchmark_data_loader(data_loader, num_batches=100)
        
        # Benchmark memory
        memory_metrics = self.benchmark_memory(data_loader, num_iterations=10)
        
        # Create results
        results = MicroBenchmarkResults(
            forward_latency_mean=latency_metrics["mean"],
            forward_latency_std=latency_metrics["std"],
            forward_latency_min=latency_metrics["min"],
            forward_latency_max=latency_metrics["max"],
            data_loader_throughput=throughput_metrics["samples_per_second"],
            batches_per_second=throughput_metrics["batches_per_second"],
            peak_gpu_memory_mb=memory_metrics["peak_gpu_mb"],
            current_gpu_memory_mb=memory_metrics["current_gpu_mb"],
            peak_cpu_memory_mb=memory_metrics["peak_cpu_mb"],
            num_iterations=num_iterations,
            batch_size=data_loader.batch_size if hasattr(data_loader, "batch_size") else 0,
            device=str(self.device),
        )
        
        return results


def run_microbenchmark(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> MicroBenchmarkResults:
    """
    Convenience function to run microbenchmark.
    
    Args:
        model: Model to benchmark
        data_loader: Data loader for benchmark data
        device: Device to run benchmark on
        num_iterations: Number of iterations for latency benchmark
        warmup_iterations: Number of warmup iterations
        
    Returns:
        MicroBenchmarkResults with all metrics
    """
    benchmark = MicroBenchmark(model, device)
    return benchmark.run_full_benchmark(data_loader, num_iterations, warmup_iterations)
