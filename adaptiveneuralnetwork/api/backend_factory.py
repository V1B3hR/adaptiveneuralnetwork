"""
Backend factory for adaptive neural networks.

This module provides factory functions to create models with different backends
(PyTorch, JAX, Neuromorphic) based on configuration.
"""

import logging
from typing import Optional, Union

from ..core.jax_backend import (
    JAXAdaptiveModel,
    convert_pytorch_to_jax_config,
    is_jax_available,
)
from ..core.neuromorphic import NeuromorphicAdaptiveModel, NeuromorphicConfig, NeuromorphicPlatform
from .config import AdaptiveConfig
from .model import AdaptiveModel

logger = logging.getLogger(__name__)


class BackendFactory:
    """Factory class for creating models with different backends."""

    @staticmethod
    def create_model(
        config: AdaptiveConfig,
    ) -> Union[AdaptiveModel, "JAXAdaptiveModel", NeuromorphicAdaptiveModel]:
        """
        Create adaptive model with specified backend.

        Args:
            config: Adaptive configuration with backend specification

        Returns:
            Model instance with appropriate backend

        Raises:
            ValueError: If backend is not supported
            ImportError: If required dependencies are not available
        """
        backend = config.backend.lower()

        if backend == "pytorch":
            return BackendFactory._create_pytorch_model(config)
        elif backend == "jax":
            return BackendFactory._create_jax_model(config)
        elif backend == "neuromorphic":
            return BackendFactory._create_neuromorphic_model(config)
        else:
            raise ValueError(
                f"Unsupported backend: {backend}. Supported backends: pytorch, jax, neuromorphic"
            )

    @staticmethod
    def _create_pytorch_model(config: AdaptiveConfig) -> AdaptiveModel:
        """Create PyTorch-based adaptive model."""
        logger.info("Creating PyTorch-based adaptive model")
        return AdaptiveModel(config)

    @staticmethod
    def _create_jax_model(config: AdaptiveConfig) -> "JAXAdaptiveModel":
        """Create JAX-based adaptive model."""
        if not is_jax_available():
            raise ImportError(
                "JAX backend requested but JAX is not available. "
                "Please install JAX with: pip install jax jaxlib flax optax"
            )

        logger.info("Creating JAX-based adaptive model")

        # Convert config to JAX format
        jax_config = convert_pytorch_to_jax_config(config)

        # Import JAX model (done here to avoid import errors if JAX not available)
        from ..core.jax_backend import JAXAdaptiveModel

        model = JAXAdaptiveModel(
            config=jax_config, input_dim=config.input_dim, output_dim=config.output_dim
        )

        return model

    @staticmethod
    def _create_neuromorphic_model(config: AdaptiveConfig) -> NeuromorphicAdaptiveModel:
        """Create neuromorphic-compatible adaptive model."""
        logger.info("Creating neuromorphic-compatible adaptive model")

        # Create neuromorphic config
        neuromorphic_config = NeuromorphicConfig(
            platform=NeuromorphicPlatform.SIMULATION,  # Default to simulation
            dt=0.001,
            v_threshold=1.0,
            tau_mem=0.01,
        )

        model = NeuromorphicAdaptiveModel(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            config=neuromorphic_config,
        )

        return model


def create_adaptive_model(
    config: Optional[AdaptiveConfig] = None, backend: str = "pytorch", **kwargs
) -> Union[AdaptiveModel, "JAXAdaptiveModel", NeuromorphicAdaptiveModel]:
    """
    Convenience function to create adaptive model with specified backend.

    Args:
        config: Adaptive configuration (creates default if None)
        backend: Backend to use ("pytorch", "jax", "neuromorphic")
        **kwargs: Additional configuration parameters

    Returns:
        Model instance with specified backend

    Example:
        >>> # PyTorch model (default)
        >>> model = create_adaptive_model()

        >>> # JAX model
        >>> model = create_adaptive_model(backend="jax", num_nodes=128)

        >>> # Neuromorphic model
        >>> model = create_adaptive_model(backend="neuromorphic", hidden_dim=64)
    """
    if config is None:
        config = AdaptiveConfig(backend=backend, **kwargs)
    else:
        # Update backend if specified
        if backend != "pytorch":  # Only update if not default
            config.backend = backend

        # Update any additional parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return BackendFactory.create_model(config)


def list_available_backends() -> dict[str, bool]:
    """
    List available backends and their availability status.

    Returns:
        Dictionary mapping backend names to availability status
    """
    backends = {
        "pytorch": True,  # Always available since it's a core dependency
        "jax": is_jax_available(),
        "neuromorphic": True,  # Always available (simulation mode)
    }

    return backends


def get_backend_info() -> dict[str, dict]:
    """
    Get detailed information about available backends.

    Returns:
        Dictionary with backend information
    """
    info = {
        "pytorch": {
            "available": True,
            "description": "PyTorch-based backend for standard GPU/CPU computation",
            "features": [
                "Dynamic computation graphs",
                "CUDA acceleration",
                "Automatic differentiation",
            ],
            "dependencies": ["torch", "torchvision"],
        },
        "jax": {
            "available": is_jax_available(),
            "description": "JAX-based backend for functional programming and advanced acceleration",
            "features": [
                "JIT compilation",
                "Functional programming",
                "Auto-vectorization",
                "TPU support",
            ],
            "dependencies": ["jax", "jaxlib", "flax", "optax"],
        },
        "neuromorphic": {
            "available": True,
            "description": "Neuromorphic hardware compatibility layer with spike-based processing",
            "features": [
                "Spike-based computation",
                "Event-driven processing",
                "Hardware simulation",
            ],
            "dependencies": ["scipy", "numpy"],
        },
    }

    return info


if __name__ == "__main__":
    # Example usage and testing
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Available backends:")
    backends = list_available_backends()
    for backend, available in backends.items():
        status = "✓" if available else "✗"
        print(f"  {status} {backend}")

    print("\nTesting backend creation:")

    # Test PyTorch backend
    try:
        pytorch_model = create_adaptive_model(backend="pytorch", num_nodes=50)
        print("✓ PyTorch model created successfully")
    except Exception as e:
        print(f"✗ PyTorch model failed: {e}")

    # Test JAX backend
    try:
        jax_model = create_adaptive_model(backend="jax", num_nodes=50)
        print("✓ JAX model created successfully")
    except Exception as e:
        print(f"✗ JAX model failed: {e}")

    # Test Neuromorphic backend
    try:
        neuro_model = create_adaptive_model(backend="neuromorphic", num_nodes=50)
        print("✓ Neuromorphic model created successfully")
    except Exception as e:
        print(f"✗ Neuromorphic model failed: {e}")
