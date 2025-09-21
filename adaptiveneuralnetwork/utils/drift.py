"""
Drift utility functions for injecting distribution shifts in data.

This module provides utilities for creating synthetic concept drift and
distribution shifts in data streams for testing continual learning systems.
"""

from typing import Literal

import torch


def apply_gaussian_drift(x: torch.Tensor, sigma: float = 0.3) -> torch.Tensor:
    """
    Apply Gaussian noise drift to input tensor.

    Args:
        x: Input tensor to apply drift to
        sigma: Standard deviation of Gaussian noise

    Returns:
        Tensor with Gaussian noise added
    """
    return x + torch.randn_like(x) * sigma


def apply_shift(x: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    """
    Apply constant additive shift to input tensor.

    Args:
        x: Input tensor to apply shift to
        delta: Constant shift value

    Returns:
        Tensor with constant shift added
    """
    return x + delta


def alternating_drift(
    x: torch.Tensor,
    step: int,
    period: int = 5,
    mode: Literal["gaussian", "shift"] = "gaussian",
    **kwargs,
) -> torch.Tensor:
    """
    Apply alternating drift pattern to input tensor.

    Args:
        x: Input tensor to apply drift to
        step: Current step/time in the sequence
        period: Number of steps before alternating drift pattern
        mode: Type of drift to apply ("gaussian" or "shift")
        **kwargs: Additional arguments passed to drift functions

    Returns:
        Tensor with alternating drift applied
    """
    # Apply drift only during odd periods
    if (step // period) % 2 == 1:
        if mode == "gaussian":
            sigma = kwargs.get("sigma", 0.3)
            return apply_gaussian_drift(x, sigma)
        elif mode == "shift":
            delta = kwargs.get("delta", 0.5)
            return apply_shift(x, delta)

    return x
