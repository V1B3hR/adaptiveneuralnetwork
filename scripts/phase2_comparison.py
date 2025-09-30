"""
Phase 2 Comparison Script - Compare optimizations impact

This script compares the performance before and after Phase 2 optimizations.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel
from adaptiveneuralnetwork.training.datasets import create_synthetic_loaders
from adaptiveneuralnetwork.utils.phase2_optimizations import optimize_model_phase2


def profile_model(model, data_loader, criterion, optimizer, num_batches=50, device="cpu"):
    """Profile a model."""
    model.train()
    
    # Warmup
    print("  Warming up...")
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx >= 3:
            break
        data, target = data.to(device), target.to(device)
        if data.dim() == 4:
            data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # Profile
    print("  Profiling...")
    step_times = []
    forward_times = []
    
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    
    batch_count = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_count >= num_batches:
            break
        if batch_idx < 3:
            continue
        
        data, target = data.to(device), target.to(device)
        if data.dim() == 4:
            data = data.view(data.size(0), -1)
        
        # Time step
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize(device)
        
        step_start = time.perf_counter()
        
        # Forward
        forward_start = time.perf_counter()
        optimizer.zero_grad()
        output = model(data)
        
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize(device)
        forward_time = time.perf_counter() - forward_start
        
        # Backward + optimizer
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize(device)
        
        step_time = time.perf_counter() - step_start
        
        step_times.append(step_time * 1000)
        forward_times.append(forward_time * 1000)
        batch_count += 1
    
    mean_step = sum(step_times) / len(step_times)
    mean_forward = sum(forward_times) / len(forward_times)
    
    if torch.cuda.is_available() and device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_mem = 0.0
    
    return {
        "mean_step_ms": mean_step,
        "mean_forward_ms": mean_forward,
        "peak_memory_mb": peak_mem
    }


def main():
    """Main comparison."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("Phase 2 - Before/After Comparison")
    print("=" * 70)
    print(f"Device: {device}")
    print()
    
    config = AdaptiveConfig(
        num_nodes=100,
        hidden_dim=64,
        batch_size=32,
        input_dim=784,
        output_dim=10,
        device=device,
        learning_rate=0.001
    )
    
    # Create data
    train_loader, _ = create_synthetic_loaders(
        num_samples=60 * config.batch_size,
        batch_size=config.batch_size
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Test baseline model
    print("Testing baseline model...")
    model_baseline = AdaptiveModel(config)
    optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=config.learning_rate)
    
    baseline_metrics = profile_model(
        model_baseline, train_loader, criterion, optimizer_baseline,
        num_batches=50, device=device
    )
    
    print(f"  Mean step: {baseline_metrics['mean_step_ms']:.2f} ms")
    print(f"  Mean forward: {baseline_metrics['mean_forward_ms']:.2f} ms")
    print()
    
    # Test with torch.compile (if available)
    if hasattr(torch, 'compile'):
        print("Testing with torch.compile...")
        model_compiled = AdaptiveModel(config)
        model_compiled = optimize_model_phase2(
            model_compiled,
            enable_compile=True,
            enable_amp=False
        )
        optimizer_compiled = torch.optim.Adam(model_compiled.parameters(), lr=config.learning_rate)
        
        try:
            compiled_metrics = profile_model(
                model_compiled, train_loader, criterion, optimizer_compiled,
                num_batches=50, device=device
            )
            
            print(f"  Mean step: {compiled_metrics['mean_step_ms']:.2f} ms")
            print(f"  Mean forward: {compiled_metrics['mean_forward_ms']:.2f} ms")
            print(f"  Improvement: {(1 - compiled_metrics['mean_step_ms']/baseline_metrics['mean_step_ms'])*100:.1f}%")
            print()
        except Exception as e:
            print(f"  Compilation failed: {e}")
            print()
    
    # Test with mixed precision
    if device != "cpu":
        print("Testing with mixed precision (AMP)...")
        model_amp = AdaptiveModel(config)
        model_amp = optimize_model_phase2(
            model_amp,
            enable_compile=False,
            enable_amp=True
        )
        optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=config.learning_rate)
        
        amp_metrics = profile_model(
            model_amp, train_loader, criterion, optimizer_amp,
            num_batches=50, device=device
        )
        
        print(f"  Mean step: {amp_metrics['mean_step_ms']:.2f} ms")
        print(f"  Mean forward: {amp_metrics['mean_forward_ms']:.2f} ms")
        print(f"  Improvement: {(1 - amp_metrics['mean_step_ms']/baseline_metrics['mean_step_ms'])*100:.1f}%")
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Baseline step time: {baseline_metrics['mean_step_ms']:.2f} ms")
    
    # Save results
    results = {
        "timestamp": time.time(),
        "device": device,
        "baseline": baseline_metrics
    }
    
    if hasattr(torch, 'compile'):
        try:
            results["compiled"] = compiled_metrics
        except:
            pass
    
    if device != "cpu":
        results["amp"] = amp_metrics
    
    output_path = Path("benchmarks/phase2_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
