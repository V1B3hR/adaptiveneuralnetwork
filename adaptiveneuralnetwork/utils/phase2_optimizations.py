"""
Phase 2 Optimizations: Mixed Precision and torch.compile support.

This module provides wrapper utilities for enabling AMP (Automatic Mixed Precision)
and torch.compile optimizations for adaptive neural networks.
"""

import functools
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def supports_amp(device: str) -> bool:
    """Check if device supports AMP."""
    if device == "cpu":
        # CPU supports bfloat16 on some architectures
        return hasattr(torch, 'bfloat16')
    elif "cuda" in device:
        # CUDA supports float16 AMP
        return torch.cuda.is_available()
    return False


def get_amp_dtype(device: str) -> torch.dtype | None:
    """Get appropriate AMP dtype for device."""
    if device == "cpu":
        # Use bfloat16 for CPU if available
        return torch.bfloat16 if hasattr(torch, 'bfloat16') else None
    elif "cuda" in device:
        # Use float16 for CUDA
        return torch.float16
    return None


class AMPContext:
    """Context manager for Automatic Mixed Precision."""

    def __init__(self, enabled: bool = True, device: str = "cpu"):
        self.enabled = enabled and supports_amp(device)
        self.device = device
        self.dtype = get_amp_dtype(device) if self.enabled else None

    def __enter__(self):
        if self.enabled and self.dtype:
            return torch.autocast(device_type=self.device.split(':')[0], dtype=self.dtype)
        return torch.no_grad() if not torch.is_grad_enabled() else self._dummy_context()

    def __exit__(self, *args):
        pass

    class _dummy_context:
        def __enter__(self): return self
        def __exit__(self, *args): pass


def try_compile(model: nn.Module, mode: str = "default") -> nn.Module:
    """
    Try to compile model with torch.compile.
    
    Falls back gracefully if compilation fails or is not supported.
    
    Args:
        model: Model to compile
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        
    Returns:
        Compiled model or original model if compilation not possible
    """
    # Check if torch.compile is available (PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model

    try:
        print(f"Attempting to compile model with mode='{mode}'...")
        compiled_model = torch.compile(model, mode=mode)
        print("Model compilation successful!")
        return compiled_model
    except Exception as e:
        print(f"torch.compile failed: {e}")
        print("Falling back to eager mode")
        return model


def mixed_precision_wrapper(
    forward_fn: Callable,
    enabled: bool = True,
    device: str = "cpu"
) -> Callable:
    """
    Decorator to enable mixed precision for forward pass.
    
    Args:
        forward_fn: Forward function to wrap
        enabled: Whether to enable mixed precision
        device: Device type for autocast
        
    Returns:
        Wrapped function with mixed precision support
    """
    if not enabled or not supports_amp(device):
        return forward_fn

    dtype = get_amp_dtype(device)
    device_type = device.split(':')[0]

    @functools.wraps(forward_fn)
    def wrapped(*args, **kwargs):
        with torch.autocast(device_type=device_type, dtype=dtype):
            return forward_fn(*args, **kwargs)

    return wrapped


class Phase2OptimizedModel(nn.Module):
    """
    Wrapper for models with Phase 2 optimizations enabled.
    
    Features:
    - Optional torch.compile
    - Optional mixed precision (AMP)
    - Optimized tensor operations
    """

    def __init__(
        self,
        base_model: nn.Module,
        enable_compile: bool = False,
        enable_amp: bool = False,
        compile_mode: str = "default"
    ):
        super().__init__()

        self.base_model = base_model
        self.enable_amp = enable_amp
        self.device = str(base_model.config.device) if hasattr(base_model, 'config') else 'cpu'

        # Apply torch.compile if requested
        if enable_compile:
            self.base_model = try_compile(self.base_model, mode=compile_mode)

        # Check AMP support
        self.amp_enabled = enable_amp and supports_amp(self.device)
        if enable_amp and not self.amp_enabled:
            print(f"AMP not supported on device {self.device}, disabling")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional mixed precision."""
        if self.amp_enabled:
            dtype = get_amp_dtype(self.device)
            device_type = self.device.split(':')[0]
            with torch.autocast(device_type=device_type, dtype=dtype):
                return self.base_model(x)
        return self.base_model(x)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def optimize_model_phase2(
    model: nn.Module,
    enable_compile: bool = False,
    enable_amp: bool = False,
    compile_mode: str = "default"
) -> nn.Module:
    """
    Apply Phase 2 optimizations to a model.
    
    Args:
        model: Model to optimize
        enable_compile: Whether to enable torch.compile
        enable_amp: Whether to enable mixed precision
        compile_mode: torch.compile mode
        
    Returns:
        Optimized model
    """
    return Phase2OptimizedModel(
        model,
        enable_compile=enable_compile,
        enable_amp=enable_amp,
        compile_mode=compile_mode
    )
