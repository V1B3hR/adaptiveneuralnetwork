"""
Phase scheduling system for adaptive neural networks.

This module provides phase management that coordinates different operational
modes (active, sleep, interactive, inspired) for the neural network.
"""

from enum import Enum

import numpy as np
import torch


class Phase(Enum):
    """Neural network operational phases."""

    ACTIVE = 0
    SLEEP = 1
    INTERACTIVE = 2
    INSPIRED = 3


class PhaseScheduler:
    """Manages phase transitions for adaptive neural networks."""

    def __init__(
        self,
        num_nodes: int,
        device: str = "cpu",
        circadian_period: int = 100,
        phase_weights: dict[Phase, float] | None = None,
    ):
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        self.circadian_period = circadian_period
        self.current_step = 0

        # Default phase weights (probability of being in each phase)
        self.phase_weights = phase_weights or {
            Phase.ACTIVE: 0.6,
            Phase.INTERACTIVE: 0.25,
            Phase.SLEEP: 0.1,
            Phase.INSPIRED: 0.05,
        }

        # Current phase for each node [num_nodes]
        self.node_phases = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        # Phase transition probabilities based on energy/activity
        self.transition_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self) -> torch.Tensor:
        """Build phase transition probability matrix."""
        # Simplified transition logic - can be made more sophisticated
        num_phases = len(Phase)
        matrix = torch.zeros(num_phases, num_phases, device=self.device)

        # Stay in same phase (diagonal)
        for phase in Phase:
            matrix[phase.value, phase.value] = 0.8

        # Specific transitions
        matrix[Phase.ACTIVE.value, Phase.INTERACTIVE.value] = 0.15
        matrix[Phase.ACTIVE.value, Phase.SLEEP.value] = 0.04
        matrix[Phase.ACTIVE.value, Phase.INSPIRED.value] = 0.01

        matrix[Phase.SLEEP.value, Phase.ACTIVE.value] = 0.15
        matrix[Phase.SLEEP.value, Phase.INTERACTIVE.value] = 0.04
        matrix[Phase.SLEEP.value, Phase.INSPIRED.value] = 0.01

        matrix[Phase.INTERACTIVE.value, Phase.ACTIVE.value] = 0.15
        matrix[Phase.INTERACTIVE.value, Phase.SLEEP.value] = 0.04
        matrix[Phase.INTERACTIVE.value, Phase.INSPIRED.value] = 0.01

        matrix[Phase.INSPIRED.value, Phase.ACTIVE.value] = 0.1
        matrix[Phase.INSPIRED.value, Phase.INTERACTIVE.value] = 0.05
        matrix[Phase.INSPIRED.value, Phase.SLEEP.value] = 0.05

        return matrix

    def step(self, energy_levels: torch.Tensor, activity_levels: torch.Tensor) -> torch.Tensor:
        """
        Update phases for all nodes based on current state.

        Args:
            energy_levels: Current energy for each node [batch_size, num_nodes, 1]
            activity_levels: Current activity for each node [batch_size, num_nodes, 1]

        Returns:
            Phase IDs for each node [batch_size, num_nodes]
        """
        self.current_step += 1
        batch_size = energy_levels.shape[0]

        # Circadian rhythm influence
        circadian_phase = (self.current_step % self.circadian_period) / self.circadian_period
        circadian_factor = np.sin(2 * np.pi * circadian_phase)

        # Expand node_phases for batch processing
        batch_phases = self.node_phases.unsqueeze(0).expand(batch_size, -1)

        for b in range(batch_size):
            for node in range(self.num_nodes):
                current_phase = batch_phases[b, node].item()
                energy = energy_levels[b, node, 0].item()
                activity = activity_levels[b, node, 0].item()

                # Phase transition logic based on energy, activity, and circadian rhythm
                if energy < 2.0:
                    # Low energy -> sleep
                    new_phase = Phase.SLEEP.value
                elif energy > 20.0 and activity < 0.3 and circadian_factor > 0.5:
                    # High energy, low activity, favorable circadian -> inspired
                    new_phase = Phase.INSPIRED.value
                elif activity > 0.7:
                    # High activity -> active or interactive
                    new_phase = (
                        Phase.ACTIVE.value if np.random.random() > 0.3 else Phase.INTERACTIVE.value
                    )
                else:
                    # Use transition probabilities
                    probs = self.transition_matrix[current_phase]
                    new_phase = torch.multinomial(probs, 1).item()

                batch_phases[b, node] = new_phase

        # Update stored phases with last batch (for stateful behavior)
        self.node_phases = batch_phases[-1].clone()

        return batch_phases

    def get_phase_mask(self, phases: torch.Tensor, target_phase: Phase) -> torch.Tensor:
        """Get boolean mask for nodes in specific phase."""
        return phases == target_phase.value

    def get_active_mask(self, phases: torch.Tensor) -> torch.Tensor:
        """Get mask for nodes that should be actively processing."""
        active_phases = {Phase.ACTIVE.value, Phase.INTERACTIVE.value, Phase.INSPIRED.value}
        mask = torch.zeros_like(phases, dtype=torch.bool)
        for phase_val in active_phases:
            mask |= phases == phase_val
        return mask

    def get_phase_stats(self, phases: torch.Tensor) -> dict[str, float]:
        """Get statistics about current phase distribution."""
        batch_size, num_nodes = phases.shape
        total_nodes = batch_size * num_nodes

        stats = {}
        for phase in Phase:
            count = (phases == phase.value).sum().item()
            stats[f"{phase.name.lower()}_ratio"] = count / total_nodes

        return stats

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_step = 0
        self.node_phases.zero_()
