"""
Phase 2 Profiler - Core Tensor Path Optimization

This script profiles tensor operations, device transfers, allocations,
and kernel launches to measure Phase 2 optimization impact.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel
from adaptiveneuralnetwork.training.datasets import create_synthetic_loaders


@dataclass
class Phase2Metrics:
    """Metrics for Phase 2 tensor path optimization."""

    mean_step_latency_ms: float
    forward_latency_ms: float
    backward_latency_ms: float
    optimizer_latency_ms: float
    allocations_per_step: int
    peak_memory_mb: float
    device_to_device_transfers: int
    kernel_launches: int
    throughput_samples_per_sec: float


class Phase2Profiler:
    """Profiler for Phase 2 tensor path optimization."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.results = {}

    def count_allocations(self, fn, *args, **kwargs):
        """Count tensor allocations during function execution."""
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            start_mem = torch.cuda.memory_allocated(self.device)

            result = fn(*args, **kwargs)

            torch.cuda.synchronize(self.device)
            end_mem = torch.cuda.memory_allocated(self.device)
            peak_mem = torch.cuda.max_memory_allocated(self.device)

            allocations = (peak_mem - start_mem) // (1024 * 1024)  # MB
            return result, allocations, peak_mem / (1024 * 1024)
        else:
            result = fn(*args, **kwargs)
            return result, 0, 0

    def profile_training_step(
        self,
        model: AdaptiveModel,
        data: torch.Tensor,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        warmup: bool = False
    ) -> dict[str, float]:
        """Profile a single training step."""

        model.train()

        # Clear any gradients
        optimizer.zero_grad()

        # Profile forward pass
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize(self.device)

        forward_start = time.perf_counter()
        output = model(data)

        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize(self.device)
        forward_time = (time.perf_counter() - forward_start) * 1000

        # Profile loss computation and backward pass
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize(self.device)

        backward_start = time.perf_counter()
        loss = criterion(output, target)
        loss.backward()

        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize(self.device)
        backward_time = (time.perf_counter() - backward_start) * 1000

        # Profile optimizer step
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize(self.device)

        optimizer_start = time.perf_counter()
        optimizer.step()

        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize(self.device)
        optimizer_time = (time.perf_counter() - optimizer_start) * 1000

        total_step_time = forward_time + backward_time + optimizer_time

        return {
            "forward_ms": forward_time,
            "backward_ms": backward_time,
            "optimizer_ms": optimizer_time,
            "total_step_ms": total_step_time
        }

    def profile_model(
        self,
        config: AdaptiveConfig,
        num_batches: int = 50
    ) -> Phase2Metrics:
        """Run comprehensive Phase 2 profiling."""

        print("=" * 70)
        print("Phase 2 - Tensor Path Profiling")
        print("=" * 70)
        print(f"Device: {config.device}")
        print(f"Batch size: {config.batch_size}")
        print(f"Model: {config.num_nodes} nodes, {config.hidden_dim} hidden dim")
        print()

        # Create model and data
        model = AdaptiveModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_loader, _ = create_synthetic_loaders(
            num_samples=(num_batches + 10) * config.batch_size,  # Extra for warmup
            batch_size=config.batch_size
        )

        # Warmup runs
        print("Warming up...")
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 3:
                break
            data = data.to(config.device)
            target = target.to(config.device)
            if data.dim() == 4:
                data = data.view(data.size(0), -1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Profile training steps
        print("Profiling training steps...")
        step_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []

        # Reset memory stats
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)

        batch_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_count >= num_batches:
                break
            # Skip warmup batches
            if batch_idx < 3:
                continue

            data = data.to(config.device)
            target = target.to(config.device)

            if data.dim() == 4:
                data = data.view(data.size(0), -1)

            metrics = self.profile_training_step(
                model, data, target, optimizer, criterion
            )

            step_times.append(metrics["total_step_ms"])
            forward_times.append(metrics["forward_ms"])
            backward_times.append(metrics["backward_ms"])
            optimizer_times.append(metrics["optimizer_ms"])
            batch_count += 1

        # Calculate averages
        mean_step_latency = sum(step_times) / len(step_times)
        mean_forward = sum(forward_times) / len(forward_times)
        mean_backward = sum(backward_times) / len(backward_times)
        mean_optimizer = sum(optimizer_times) / len(optimizer_times)

        # Memory metrics
        if torch.cuda.is_available() and self.device != "cpu":
            peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        else:
            peak_memory = 0.0

        # Calculate throughput
        total_time_sec = sum(step_times) / 1000
        total_samples = len(step_times) * config.batch_size
        throughput = total_samples / total_time_sec

        metrics = Phase2Metrics(
            mean_step_latency_ms=mean_step_latency,
            forward_latency_ms=mean_forward,
            backward_latency_ms=mean_backward,
            optimizer_latency_ms=mean_optimizer,
            allocations_per_step=0,  # Will be calculated with detailed profiling
            peak_memory_mb=peak_memory,
            device_to_device_transfers=0,  # Will be measured with hooks
            kernel_launches=0,  # Will be measured with profiler
            throughput_samples_per_sec=throughput
        )

        # Print results
        print("Phase 2 Baseline Metrics:")
        print("-" * 70)
        print(f"Mean step latency: {metrics.mean_step_latency_ms:.2f} ms")
        print(f"  - Forward: {metrics.forward_latency_ms:.2f} ms")
        print(f"  - Backward: {metrics.backward_latency_ms:.2f} ms")
        print(f"  - Optimizer: {metrics.optimizer_latency_ms:.2f} ms")
        print(f"Throughput: {metrics.throughput_samples_per_sec:.1f} samples/sec")
        print(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")
        print("=" * 70)

        return metrics


def save_baseline(metrics: Phase2Metrics, output_file: str = "benchmarks/phase2_baseline.json"):
    """Save baseline metrics to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = {
        "phase": "Phase 2 - Baseline",
        "timestamp": time.time(),
        "metrics": {
            "mean_step_latency_ms": metrics.mean_step_latency_ms,
            "forward_latency_ms": metrics.forward_latency_ms,
            "backward_latency_ms": metrics.backward_latency_ms,
            "optimizer_latency_ms": metrics.optimizer_latency_ms,
            "peak_memory_mb": metrics.peak_memory_mb,
            "throughput_samples_per_sec": metrics.throughput_samples_per_sec,
        }
    }

    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nBaseline saved to: {output_path}")


def main():
    """Main entry point."""

    # Configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = AdaptiveConfig(
        num_nodes=100,
        hidden_dim=64,
        batch_size=32,
        input_dim=784,
        output_dim=10,
        device=device,
        learning_rate=0.001
    )

    profiler = Phase2Profiler(device=device)
    metrics = profiler.profile_model(config, num_batches=50)

    # Save baseline
    save_baseline(metrics)


if __name__ == "__main__":
    main()
