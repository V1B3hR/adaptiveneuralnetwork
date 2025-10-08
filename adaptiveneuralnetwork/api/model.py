"""
Main model interface for adaptive neural networks.

This module provides the high-level API for creating and using adaptive
neural network models.
"""

from typing import Any

import torch
import torch.nn as nn

from ..core import AdaptiveDynamics, NodeState, PhaseScheduler
from .config import AdaptiveConfig


class AdaptiveModel(nn.Module):
    """
    Main adaptive neural network model.

    This model wraps the core components and provides a standard PyTorch
    interface with forward() method for training and inference.
    """

    def __init__(self, config: AdaptiveConfig):
        super().__init__()
        self.config = config

        # Initialize core components
        node_config = config.to_node_config()
        self.node_state = NodeState(node_config)

        self.phase_scheduler = PhaseScheduler(
            num_nodes=config.num_nodes,
            device=config.device,
            circadian_period=config.circadian_period,
        )

        self.dynamics = AdaptiveDynamics(hidden_dim=config.hidden_dim, device=config.device)

        # Input projection layer
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim, device=config.device)

        # Output projection layer
        self.output_projection = nn.Linear(
            config.num_nodes,  # Use node activities as features
            config.output_dim,
            device=config.device,
        )

        # Move to specified device
        self.to(config.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adaptive neural network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        batch_size = x.shape[0]

        # Detach node state from previous computational graph (Phase 2 optimization)
        # This prevents accumulation of gradients across batches
        if self.training:
            self.node_state.detach()

        # Expand node state for current batch size
        self.node_state.expand_batch(batch_size)

        # Project input to hidden dimension
        projected_input = self.input_projection(x)  # [batch_size, hidden_dim]

        # Expand input for all nodes
        node_input = projected_input.unsqueeze(1).expand(-1, self.config.num_nodes, -1)

        # Update node dynamics
        self.node_state = self.dynamics(self.node_state, node_input, self.phase_scheduler)

        # Use node activities as features for output
        node_activities = self.node_state.activity.squeeze(-1)  # [batch_size, num_nodes]

        # Project to output dimension
        output = self.output_projection(node_activities)

        return output

    def reset_state(self) -> None:
        """Reset the model to initial state."""
        self.node_state = NodeState(self.config.to_node_config())
        self.phase_scheduler.reset()

    def get_state_dict_full(self) -> dict[str, Any]:
        """Get full state including node states and scheduler."""
        state_dict = self.state_dict()
        state_dict["node_state"] = {
            "hidden_state": self.node_state.hidden_state,
            "energy": self.node_state.energy,
            "activity": self.node_state.activity,
            "position": self.node_state.position,
            "phase_mask": self.node_state.phase_mask,
        }
        state_dict["scheduler_state"] = {
            "current_step": self.phase_scheduler.current_step,
            "node_phases": self.phase_scheduler.node_phases,
        }
        return state_dict

    def load_state_dict_full(self, state_dict: dict[str, Any]) -> None:
        """Load full state including node states and scheduler."""
        # Load regular parameters
        regular_state = {
            k: v for k, v in state_dict.items() if k not in ["node_state", "scheduler_state"]
        }
        self.load_state_dict(regular_state, strict=False)

        # Load node state
        if "node_state" in state_dict:
            node_state = state_dict["node_state"]
            self.node_state.hidden_state = node_state["hidden_state"]
            self.node_state.energy = node_state["energy"]
            self.node_state.activity = node_state["activity"]
            self.node_state.position = node_state["position"]
            self.node_state.phase_mask = node_state["phase_mask"]

        # Load scheduler state
        if "scheduler_state" in state_dict:
            scheduler_state = state_dict["scheduler_state"]
            self.phase_scheduler.current_step = scheduler_state["current_step"]
            self.phase_scheduler.node_phases = scheduler_state["node_phases"]

    def get_metrics(self) -> dict[str, float]:
        """Get current model metrics for monitoring."""
        active_nodes = self.node_state.get_active_nodes()
        phase_stats = self.phase_scheduler.get_phase_stats(
            self.phase_scheduler.node_phases.unsqueeze(0)
        )

        metrics = {
            "active_node_ratio": active_nodes.float().mean().item(),
            "mean_energy": self.node_state.energy.mean().item(),
            "mean_activity": self.node_state.activity.mean().item(),
            **phase_stats,
        }

        # Add anxiety metrics if available
        if hasattr(self.phase_scheduler, 'get_anxiety_stats'):
            metrics.update(self.phase_scheduler.get_anxiety_stats())

        # Add sparsity metrics
        if hasattr(self.phase_scheduler, 'get_sparsity_metrics'):
            sparsity_metrics = self.phase_scheduler.get_sparsity_metrics(
                self.node_state.energy, self.node_state.activity
            )
            metrics.update(sparsity_metrics)

        return metrics

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for training."""
        return self.dynamics.compute_loss(self.node_state, targets)

    def set_training_mode(self, training: bool = True) -> "AdaptiveModel":
        """Set training mode and update internal state accordingly."""
        self.train(training)

        # Could add training-specific behavior here in the future
        # e.g., different phase scheduling during training vs inference

        return self
