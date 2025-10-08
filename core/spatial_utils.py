"""
Spatial utilities for dimension-agnostic operations.

This module provides helper functions for working with spatial dimensions
in the adaptive neural network, supporting arbitrary dimensional spaces.
"""

import numpy as np
from typing import Union, Sequence, Tuple, List


def zero_vector(dim: int) -> np.ndarray:
    """Create a zero vector of specified dimension.
    
    Args:
        dim: Number of spatial dimensions
        
    Returns:
        Zero vector of shape (dim,)
    """
    if dim < 1:
        raise ValueError(f"Dimension must be positive, got {dim}")
    return np.zeros(dim, dtype=float)


def rand_vector(dim: int, ranges: Union[Tuple[float, float], Sequence[Tuple[float, float]]]) -> np.ndarray:
    """Create a random vector within specified ranges.
    
    Args:
        dim: Number of spatial dimensions
        ranges: Either a single (min, max) tuple for all dimensions,
                or a sequence of (min, max) tuples for each dimension
                
    Returns:
        Random vector of shape (dim,)
    """
    if dim < 1:
        raise ValueError(f"Dimension must be positive, got {dim}")
    
    # Handle single range for all dimensions
    if isinstance(ranges, tuple) and len(ranges) == 2:
        min_val, max_val = ranges
        return np.random.uniform(min_val, max_val, size=dim)
    
    # Handle per-dimension ranges
    if len(ranges) != dim:
        raise ValueError(f"Range count {len(ranges)} doesn't match dimension {dim}")
    
    result = np.zeros(dim, dtype=float)
    for i, (min_val, max_val) in enumerate(ranges):
        result[i] = np.random.uniform(min_val, max_val)
    
    return result


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        a: First point as numpy array
        b: Second point as numpy array
        
    Returns:
        Euclidean distance
        
    Raises:
        ValueError: If points have different dimensions
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    if a.shape != b.shape:
        raise ValueError(f"Point dimensions don't match: {a.shape} vs {b.shape}")
    
    return float(np.linalg.norm(a - b))


def validate_spatial_dimensions(arrays: List[np.ndarray], expected_dims: int) -> None:
    """Validate that all arrays have the expected spatial dimensions.
    
    Args:
        arrays: List of numpy arrays to validate
        expected_dims: Expected number of spatial dimensions
        
    Raises:
        ValueError: If any array doesn't match expected dimensions
    """
    if expected_dims < 1:
        raise ValueError(f"Expected dimensions must be positive, got {expected_dims}")
    
    for i, array in enumerate(arrays):
        array = np.asarray(array)
        if array.ndim != 1:
            raise ValueError(f"Array {i} must be 1D, got shape {array.shape}")
        if array.shape[0] != expected_dims:
            raise ValueError(f"Array {i} has {array.shape[0]} dimensions, expected {expected_dims}")


def expand_bounds_to_dimensions(bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]], 
                                 dim: int) -> List[Tuple[float, float]]:
    """Expand bounds specification to match spatial dimensions.
    
    Args:
        bounds: Either a single (min, max) tuple for all dimensions,
                or a sequence of (min, max) tuples for each dimension
        dim: Number of spatial dimensions
        
    Returns:
        List of (min, max) tuples, one for each dimension
    """
    if dim < 1:
        raise ValueError(f"Dimension must be positive, got {dim}")
    
    # Handle single bounds for all dimensions
    if isinstance(bounds, tuple) and len(bounds) == 2:
        min_val, max_val = bounds
        if min_val >= max_val:
            raise ValueError(f"Invalid bounds: min {min_val} >= max {max_val}")
        return [(min_val, max_val)] * dim
    
    # Handle per-dimension bounds
    bounds_list = list(bounds)
    if len(bounds_list) != dim:
        raise ValueError(f"Bounds count {len(bounds_list)} doesn't match dimension {dim}")
    
    # Validate each bound
    for i, (min_val, max_val) in enumerate(bounds_list):
        if min_val >= max_val:
            raise ValueError(f"Invalid bounds for dimension {i}: min {min_val} >= max {max_val}")
    
    return bounds_list


def validate_position_in_bounds(position: np.ndarray, 
                                bounds: Sequence[Tuple[float, float]]) -> None:
    """Validate that a position is within specified bounds.
    
    Args:
        position: Position array to validate
        bounds: Sequence of (min, max) tuples for each dimension
        
    Raises:
        ValueError: If position is outside bounds
    """
    position = np.asarray(position)
    
    if position.ndim != 1:
        raise ValueError(f"Position must be 1D, got shape {position.shape}")
    
    if len(bounds) != position.shape[0]:
        raise ValueError(f"Bounds count {len(bounds)} doesn't match position dimensions {position.shape[0]}")
    
    for i, ((min_val, max_val), coord) in enumerate(zip(bounds, position, strict=False)):
        if not (min_val <= coord <= max_val):
            raise ValueError(f"Position component {i}={coord} outside bounds [{min_val}, {max_val}]")


def create_random_positions(count: int, dim: int, 
                           bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]]) -> np.ndarray:
    """Create multiple random positions within bounds.
    
    Args:
        count: Number of positions to generate
        dim: Spatial dimensions
        bounds: Bounds for position generation
        
    Returns:
        Array of shape (count, dim) with random positions
    """
    if count < 0:
        raise ValueError(f"Count must be non-negative, got {count}")
    
    expanded_bounds = expand_bounds_to_dimensions(bounds, dim)
    positions = np.zeros((count, dim), dtype=float)
    
    for i in range(count):
        positions[i] = rand_vector(dim, expanded_bounds)
    
    return positions