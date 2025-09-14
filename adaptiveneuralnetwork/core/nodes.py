"""
Vectorized node state management for adaptive neural networks.

This module provides tensor-based node state abstractions that enable
efficient batch processing during training.
"""

from dataclasses import dataclass

import torch


@dataclass
class NodeConfig:
    """Configuration for adaptive neural network nodes."""

    num_nodes: int = 100
    hidden_dim: int = 64
    energy_dim: int = 1
    activity_dim: int = 1
    spatial_dim: int = 2
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    # Node dynamics parameters
    energy_decay: float = 0.01
    activity_threshold: float = 0.5
    connection_radius: float = 1.0

    # Learning parameters
    learning_rate: float = 0.001
    adaptation_rate: float = 0.1


class NodeState:
    """Vectorized state representation for multiple adaptive nodes."""

    def __init__(self, config: NodeConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Core state tensors [batch_size, num_nodes, dim]
        self.hidden_state = torch.zeros(
            1, config.num_nodes, config.hidden_dim, dtype=config.dtype, device=self.device
        )

        self.energy = (
            torch.ones(
                1, config.num_nodes, config.energy_dim, dtype=config.dtype, device=self.device
            )
            * 10.0
        )  # Initial energy

        self.activity = torch.zeros(
            1, config.num_nodes, config.activity_dim, dtype=config.dtype, device=self.device
        )

        self.position = torch.randn(
            1, config.num_nodes, config.spatial_dim, dtype=config.dtype, device=self.device
        )

        # Phase-specific state (will be set by PhaseScheduler)
        self.phase_mask = torch.zeros(1, config.num_nodes, 1, dtype=torch.bool, device=self.device)

    def clamp_energy(self, min_val: float = 0.0, max_val: float = 50.0) -> None:
        """Clamp energy values to valid range."""
        self.energy = torch.clamp(self.energy, min_val, max_val)

    def clamp_activity(self, min_val: float = 0.0, max_val: float = 1.0) -> None:
        """Clamp activity values to valid range."""
        self.activity = torch.clamp(self.activity, min_val, max_val)

    def get_active_nodes(self) -> torch.Tensor:
        """Get mask of currently active nodes."""
        return (self.energy > 1.0) & (self.activity > self.config.activity_threshold)

    def update_energy(self, delta: torch.Tensor) -> None:
        """Update energy with decay and external changes."""
        # Apply energy decay
        self.energy = self.energy * (1.0 - self.config.energy_decay)
        # Add external energy changes
        self.energy = self.energy + delta
        self.clamp_energy()

    def update_activity(self, input_stimulation: torch.Tensor) -> None:
        """Update activity based on input and current state."""
        # Simple activity update - can be made more complex later
        self.activity = torch.sigmoid(
            input_stimulation + self.hidden_state.mean(dim=-1, keepdim=True)
        )
        self.clamp_activity()

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.hidden_state.shape[0]

    def expand_batch(self, new_batch_size: int) -> None:
        """Expand or contract state tensors to accommodate different batch size."""
        current_batch_size = self.get_batch_size()
        if new_batch_size == current_batch_size:
            return
        
        if new_batch_size > current_batch_size:
            # Expand by repeating the first sample for additional batch entries
            num_repeats = new_batch_size - current_batch_size
            
            # Take the first sample and repeat it
            extra_hidden = self.hidden_state[:1].expand(num_repeats, -1, -1)
            extra_energy = self.energy[:1].expand(num_repeats, -1, -1)
            extra_activity = self.activity[:1].expand(num_repeats, -1, -1)
            extra_position = self.position[:1].expand(num_repeats, -1, -1)
            extra_phase_mask = self.phase_mask[:1].expand(num_repeats, -1, -1)
            
            # Concatenate with existing tensors
            self.hidden_state = torch.cat([self.hidden_state, extra_hidden], dim=0).contiguous()
            self.energy = torch.cat([self.energy, extra_energy], dim=0).contiguous()
            self.activity = torch.cat([self.activity, extra_activity], dim=0).contiguous()
            self.position = torch.cat([self.position, extra_position], dim=0).contiguous()
            self.phase_mask = torch.cat([self.phase_mask, extra_phase_mask], dim=0).contiguous()
        else:
            # Contract to smaller batch size by taking first n entries
            self.hidden_state = self.hidden_state[:new_batch_size].contiguous()
            self.energy = self.energy[:new_batch_size].contiguous()
            self.activity = self.activity[:new_batch_size].contiguous()
            self.position = self.position[:new_batch_size].contiguous()
            self.phase_mask = self.phase_mask[:new_batch_size].contiguous()

    def to(self, device: torch.device) -> "NodeState":
        """Move state to specified device."""
        self.device = device
        self.hidden_state = self.hidden_state.to(device)
        self.energy = self.energy.to(device)
        self.activity = self.activity.to(device)
        self.position = self.position.to(device)
        self.phase_mask = self.phase_mask.to(device)
        return self
