"""
Core dynamics for adaptive neural networks.

This module implements the basic dynamics update functions that drive
the adaptive behavior of the neural network nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nodes import NodeState
from .phases import PhaseScheduler


class AdaptiveDynamics(nn.Module):
    """Core dynamics engine for adaptive neural networks."""

    def __init__(self, hidden_dim: int, device: str = "cpu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        # Linear transformation for state updates
        self.state_update = nn.Linear(hidden_dim, hidden_dim, device=self.device)

        # Energy dynamics
        self.energy_update = nn.Linear(hidden_dim, 1, device=self.device)

        # Activity computation
        self.activity_update = nn.Linear(2, 1, device=self.device)  # mean(hidden) + mean(input)

        # Phase-dependent scaling
        self.phase_scales = nn.Parameter(torch.ones(4, device=self.device))  # 4 phases

    def _calculate_anxiety_levels(self, node_state: NodeState) -> torch.Tensor:
        """
        Calculate anxiety levels based on node state.
        
        Args:
            node_state: Current node state
            
        Returns:
            Anxiety levels tensor [batch_size, num_nodes, 1]
        """
        energy_stress = torch.clamp(5.0 - node_state.energy, min=0.0)
        activity_stress = node_state.activity * 2.0
        hidden_variance = node_state.hidden_state.var(dim=-1, keepdim=True)
        
        return energy_stress + activity_stress + hidden_variance

    def _apply_phase_dependent_scaling(self, hidden_delta: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Apply phase-dependent scaling to hidden state updates.
        
        Args:
            hidden_delta: Hidden state update tensor
            phases: Phase assignments for each node
            
        Returns:
            Scaled hidden state update
        """
        for phase_id in range(4):
            phase_mask = (phases == phase_id).unsqueeze(-1).float()
            hidden_delta = hidden_delta * (1.0 + self.phase_scales[phase_id] * phase_mask)
        return hidden_delta

    def _project_external_input(self, external_input: torch.Tensor) -> torch.Tensor:
        """
        Project external input to hidden dimension if needed.
        
        Args:
            external_input: External input tensor
            
        Returns:
            Projected input tensor
        """
        if external_input.shape[-1] != self.hidden_dim:
            return F.linear(
                external_input, self.state_update.weight[:, : external_input.shape[-1]]
            )
        return external_input

    def _update_hidden_state(
        self, node_state: NodeState, hidden_delta: torch.Tensor, 
        input_proj: torch.Tensor, active_mask: torch.Tensor
    ) -> None:
        """
        Update hidden state with active node masking.
        
        Args:
            node_state: Node state to update
            hidden_delta: Hidden state change
            input_proj: Projected external input
            active_mask: Mask for active nodes
        """
        node_state.hidden_state = node_state.hidden_state + active_mask * (
            hidden_delta + 0.1 * input_proj
        )
        node_state.hidden_state = torch.tanh(node_state.hidden_state)

    def _update_energy_with_anxiety(
        self, node_state: NodeState, anxiety_levels: torch.Tensor, active_mask: torch.Tensor
    ) -> None:
        """
        Update energy levels considering anxiety effects.
        
        Args:
            node_state: Node state to update
            anxiety_levels: Current anxiety levels
            active_mask: Mask for active nodes
        """
        energy_delta = self.energy_update(node_state.hidden_state)
        anxiety_factor = 1.0 - 0.1 * torch.clamp(anxiety_levels / 10.0, 0.0, 1.0)
        energy_delta = energy_delta * anxiety_factor
        node_state.update_energy(energy_delta * active_mask)

    def _update_activity_levels(self, node_state: NodeState, external_input: torch.Tensor) -> None:
        """
        Update activity levels based on hidden state and external input.
        
        Args:
            node_state: Node state to update  
            external_input: External input tensor
        """
        activity_input = torch.cat(
            [
                node_state.hidden_state.mean(dim=-1, keepdim=True),
                (
                    external_input.mean(dim=-1, keepdim=True)
                    if external_input.numel() > 0
                    else torch.zeros_like(node_state.energy)
                ),
            ],
            dim=-1,
        )

        batch_size, num_nodes, _ = activity_input.shape
        activity_input_flat = activity_input.view(-1, activity_input.shape[-1])
        activity_delta_flat = self.activity_update(activity_input_flat)
        activity_delta = activity_delta_flat.view(batch_size, num_nodes, -1)
        
        node_state.activity = torch.sigmoid(activity_delta)
        node_state.clamp_activity()

    def forward(
        self, node_state: NodeState, external_input: torch.Tensor, phase_scheduler: PhaseScheduler
    ) -> NodeState:
        """
        Update node state based on current state and external input.

        Args:
            node_state: Current node state
            external_input: External input [batch_size, num_nodes, input_dim]
            phase_scheduler: Phase scheduler for coordinating updates

        Returns:
            Updated node state
        """
        # Calculate anxiety levels
        anxiety_levels = self._calculate_anxiety_levels(node_state)

        # Get current phases with anxiety information
        phases = phase_scheduler.step(node_state.energy, node_state.activity, anxiety_levels)

        # Update hidden state with phase-dependent scaling
        hidden_delta = self.state_update(node_state.hidden_state)
        hidden_delta = self._apply_phase_dependent_scaling(hidden_delta, phases)

        # Project external input and get active nodes mask
        input_proj = self._project_external_input(external_input)
        active_mask = phase_scheduler.get_active_mask(phases).unsqueeze(-1).float()

        # Update hidden state, energy, and activity
        self._update_hidden_state(node_state, hidden_delta, input_proj, active_mask)
        self._update_energy_with_anxiety(node_state, anxiety_levels, active_mask)
        self._update_activity_levels(node_state, external_input)

        # Store phase information
        node_state.phase_mask = active_mask.bool()

        return node_state

    def compute_loss(
        self, node_state: NodeState, target: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute loss for training the adaptive dynamics.

        Args:
            node_state: Current node state
            target: Target output [batch_size, output_dim]
            reduction: Loss reduction method

        Returns:
            Computed loss
        """
        # Use node activity as prediction
        prediction = node_state.activity.mean(dim=1)  # [batch_size, 1]

        # Ensure target has the same shape
        if target.dim() == 1:
            target = target.unsqueeze(-1)

        # MSE loss
        loss = F.mse_loss(prediction, target.float(), reduction=reduction)

        # Add regularization terms
        energy_reg = 0.01 * torch.mean(torch.clamp(node_state.energy - 10.0, min=0.0) ** 2)
        activity_reg = 0.001 * torch.mean(node_state.activity**2)

        total_loss = loss + energy_reg + activity_reg

        return total_loss
