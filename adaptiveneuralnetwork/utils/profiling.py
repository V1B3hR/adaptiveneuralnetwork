"""
Profiling utilities for adaptive neural networks.

This module provides profiling tools to analyze performance and
identify bottlenecks in the adaptive neural network implementation.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

import torch
import torch.profiler as profiler

from ..api.model import AdaptiveModel
from ..api.config import AdaptiveConfig
from ..training.datasets import create_synthetic_loaders


class PerformanceProfiler:
    """Performance profiler for adaptive neural networks."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    @contextmanager
    def time_block(self, name: str):
        """Context manager for timing code blocks."""
        start_time = time.time()
        yield
        end_time = time.time()
        self.results[f"{name}_time"] = end_time - start_time
        
    def profile_model_forward(
        self,
        model: AdaptiveModel,
        input_tensor: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Profile model forward pass performance."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
                
        # Time forward passes
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        with self.time_block("forward_pass"):
            with torch.no_grad():
                for _ in range(num_iterations):
                    output = model(input_tensor)
                    
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        avg_time = self.results["forward_pass_time"] / num_iterations
        throughput = input_tensor.shape[0] * num_iterations / self.results["forward_pass_time"]
        
        return {
            'avg_forward_time': avg_time,
            'throughput_samples_per_sec': throughput,
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
        
    def profile_training_step(
        self,
        model: AdaptiveModel,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        num_iterations: int = 50
    ) -> Dict[str, float]:
        """Profile training step performance."""
        model.train()
        
        with self.time_block("training_step"):
            for _ in range(num_iterations):
                optimizer.zero_grad()
                output = model(input_tensor)
                loss = torch.nn.functional.cross_entropy(output, target_tensor)
                loss.backward()
                optimizer.step()
                
        avg_time = self.results["training_step_time"] / num_iterations
        
        return {
            'avg_training_step_time': avg_time,
            'training_throughput': input_tensor.shape[0] * num_iterations / self.results["training_step_time"]
        }
        
    def profile_memory_usage(self, model: AdaptiveModel) -> Dict[str, Any]:
        """Profile memory usage of the model."""
        if not torch.cuda.is_available():
            return {'memory_profiling': 'CUDA not available'}
            
        torch.cuda.reset_peak_memory_stats()
        
        # Create sample input
        sample_input = torch.randn(32, 784, device=model.config.device)
        
        # Forward pass
        output = model(sample_input)
        
        # Backward pass  
        loss = output.sum()
        loss.backward()
        
        return {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'model_parameters_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
        
    def run_comprehensive_profile(
        self,
        config: Optional[AdaptiveConfig] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive performance profile."""
        if config is None:
            config = AdaptiveConfig(
                num_nodes=100,
                hidden_dim=64,
                batch_size=32,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
        print("=== Comprehensive Performance Profiling ===")
        print(f"Device: {config.device}")
        print(f"Model: {config.num_nodes} nodes, {config.hidden_dim} hidden dim")
        
        # Create model
        model = AdaptiveModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Create sample data
        sample_input = torch.randn(config.batch_size, 784, device=config.device)
        sample_target = torch.randint(0, 10, (config.batch_size,), device=config.device)
        
        profile_results = {
            'config': config.to_dict(),
            'timestamp': time.time(),
            'device_info': {
                'device_type': str(config.device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # Profile forward pass
        print("\nProfiling forward pass...")
        forward_results = self.profile_model_forward(model, sample_input)
        profile_results['forward_pass'] = forward_results
        
        # Profile training step
        print("Profiling training step...")
        training_results = self.profile_training_step(model, sample_input, sample_target, optimizer)
        profile_results['training_step'] = training_results
        
        # Profile memory usage
        print("Profiling memory usage...")
        memory_results = self.profile_memory_usage(model)
        profile_results['memory_usage'] = memory_results
        
        # Print summary
        print(f"\n=== Profiling Results ===")
        print(f"Forward pass time: {forward_results['avg_forward_time']*1000:.2f} ms")
        print(f"Training step time: {training_results['avg_training_step_time']*1000:.2f} ms")
        print(f"Throughput: {forward_results['throughput_samples_per_sec']:.1f} samples/sec")
        if torch.cuda.is_available():
            print(f"Peak memory: {memory_results['peak_memory_mb']:.1f} MB")
        
        # Save results
        if save_results:
            results_file = self.output_dir / f"profile_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(profile_results, f, indent=2)
            print(f"\nResults saved to: {results_file}")
            
        return profile_results


def run_torch_profiler(
    config: Optional[AdaptiveConfig] = None,
    num_steps: int = 100,
    output_dir: str = "torch_profiler_results"
) -> str:
    """
    Run PyTorch profiler on the adaptive neural network.
    
    Args:
        config: Model configuration
        num_steps: Number of steps to profile
        output_dir: Directory to save profiler results
        
    Returns:
        Path to profiler trace file
    """
    if config is None:
        config = AdaptiveConfig(
            num_nodes=50,
            hidden_dim=32,
            batch_size=16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create model and data
    model = AdaptiveModel(config)
    train_loader, _ = create_synthetic_loaders(
        num_samples=500,
        batch_size=config.batch_size
    )
    optimizer = torch.optim.Adam(model.parameters())
    
    trace_file = output_path / f"trace_{int(time.time())}.json"
    
    # Run profiling
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else profiler.ProfilerActivity.CPU
        ],
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=profiler.tensorboard_trace_handler(str(output_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        model.train()
        for step, (data, target) in enumerate(train_loader):
            if step >= num_steps:
                break
                
            data = data.to(config.device)
            target = target.to(config.device)
            
            if data.dim() == 4:
                data = data.view(data.size(0), -1)
                
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            prof.step()
    
    # Export trace
    prof.export_chrome_trace(str(trace_file))
    print(f"PyTorch profiler trace saved to: {trace_file}")
    
    return str(trace_file)