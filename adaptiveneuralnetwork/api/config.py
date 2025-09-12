"""
Configuration management for adaptive neural networks.

This module provides configuration classes and utilities for reproducible
experiments and model training.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
import yaml

from ..core.nodes import NodeConfig


@dataclass
class AdaptiveConfig:
    """Main configuration class for adaptive neural network models."""

    # Model architecture
    num_nodes: int = 100
    hidden_dim: int = 64
    input_dim: int = 28 * 28  # Default for MNIST
    output_dim: int = 10  # Default for MNIST
    spatial_dim: int = 2

    # Node configuration
    energy_decay: float = 0.01
    activity_threshold: float = 0.5
    connection_radius: float = 1.0

    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 10
    adaptation_rate: float = 0.1

    # Phase scheduling
    circadian_period: int = 100
    phase_weights: dict[str, float] = field(
        default_factory=lambda: {"active": 0.6, "interactive": 0.25, "sleep": 0.1, "inspired": 0.05}
    )

    # Backend configuration
    backend: str = "pytorch"  # "pytorch", "jax", "neuromorphic"
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 42

    # Logging and metrics
    log_interval: int = 100
    save_checkpoint: bool = True
    checkpoint_dir: str = "checkpoints"
    metrics_file: str | None = "metrics.json"

    # Experimental features (stubs for future)
    enable_continual_learning: bool = False
    enable_ablation_study: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def to_node_config(self) -> NodeConfig:
        """Convert to NodeConfig for core modules."""
        dtype_map = {"float32": torch.float32, "float64": torch.float64, "float16": torch.float16}

        return NodeConfig(
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            energy_dim=1,
            activity_dim=1,
            spatial_dim=self.spatial_dim,
            device=self.device,
            dtype=dtype_map.get(self.dtype, torch.float32),
            energy_decay=self.energy_decay,
            activity_threshold=self.activity_threshold,
            connection_radius=self.connection_radius,
            learning_rate=self.learning_rate,
            adaptation_rate=self.adaptation_rate,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AdaptiveConfig":
        """Load configuration from YAML file."""
        try:
            with open(yaml_path) as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.__dict__.copy()
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()

    def update(self, **kwargs) -> "AdaptiveConfig":
        """Create new config with updated parameters."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__(**config_dict)
