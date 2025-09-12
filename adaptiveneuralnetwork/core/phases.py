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
        anxiety_threshold: float = 5.0,
        restorative_strength: float = 0.1,
    ):
        self.num_nodes = num_nodes
        self.device = torch.device(device)
        self.circadian_period = circadian_period
        self.current_step = 0
        self.anxiety_threshold = anxiety_threshold
        self.restorative_strength = restorative_strength

        # Default phase weights (probability of being in each phase)
        self.phase_weights = phase_weights or {
            Phase.ACTIVE: 0.6,
            Phase.INTERACTIVE: 0.25,
            Phase.SLEEP: 0.1,
            Phase.INSPIRED: 0.05,
        }

        # Current phase for each node [num_nodes]
        self.node_phases = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        
        # Anxiety tracking for enhanced phase control
        self.node_anxiety = torch.zeros(num_nodes, device=self.device)
        self.anxiety_history = torch.zeros(num_nodes, 10, device=self.device)  # Track last 10 steps
        
        # Restorative state tracking
        self.restorative_needs = torch.zeros(num_nodes, device=self.device)
        self.sleep_quality = torch.ones(num_nodes, device=self.device)  # 0-1, quality of sleep phases

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

    def step(self, energy_levels: torch.Tensor, activity_levels: torch.Tensor, anxiety_levels: torch.Tensor | None = None) -> torch.Tensor:
        """
        Update phases for all nodes based on current state.

        Args:
            energy_levels: Current energy for each node [batch_size, num_nodes, 1]
            activity_levels: Current activity for each node [batch_size, num_nodes, 1]
            anxiety_levels: Optional anxiety levels for each node [batch_size, num_nodes, 1]

        Returns:
            Phase IDs for each node [batch_size, num_nodes]
        """
        self.current_step += 1
        batch_size = energy_levels.shape[0]

        # Update anxiety tracking if provided
        if anxiety_levels is not None:
            # Use last batch for anxiety tracking
            last_batch_anxiety = anxiety_levels[-1].squeeze(-1)  # [num_nodes]
            self._update_anxiety_tracking(last_batch_anxiety)

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
                
                # Get anxiety and restorative factors for this node
                node_anxiety = self.node_anxiety[node].item()
                restorative_need = self.restorative_needs[node].item()
                sleep_qual = self.sleep_quality[node].item()

                # Enhanced phase transition logic with anxiety/restorative mechanics
                new_phase = self._determine_phase_transition(
                    current_phase, energy, activity, node_anxiety, 
                    restorative_need, sleep_qual, circadian_factor, node
                )

                batch_phases[b, node] = new_phase

        # Update stored phases with last batch (for stateful behavior)
        self.node_phases = batch_phases[-1].clone()
        
        # Update restorative needs based on current phases
        self._update_restorative_state()

        return batch_phases

    def _update_anxiety_tracking(self, anxiety_levels: torch.Tensor) -> None:
        """Update anxiety history and current levels."""
        # Shift anxiety history
        self.anxiety_history[:, 1:] = self.anxiety_history[:, :-1]
        self.anxiety_history[:, 0] = anxiety_levels
        
        # Update current anxiety (exponential moving average)
        alpha = 0.3
        self.node_anxiety = alpha * anxiety_levels + (1 - alpha) * self.node_anxiety
        
    def _determine_phase_transition(
        self, current_phase: int, energy: float, activity: float, 
        anxiety: float, restorative_need: float, sleep_quality: float,
        circadian_factor: float, node_idx: int
    ) -> int:
        """Determine phase transition with enhanced anxiety/restorative mechanics."""
        
        # High anxiety override - force restorative phases
        if anxiety > self.anxiety_threshold:
            anxiety_severity = min(1.0, anxiety / (self.anxiety_threshold * 2))
            
            # More severe anxiety leads to deeper restorative needs
            if anxiety_severity > 0.7:
                return Phase.SLEEP.value  # Deep restoration needed
            elif anxiety_severity > 0.4:
                return Phase.INTERACTIVE.value  # Social support seeking
            else:
                # Reduce to lower activity phase
                if current_phase == Phase.ACTIVE.value:
                    return Phase.INTERACTIVE.value
        
        # Restorative need override
        if restorative_need > 0.6:
            # Check if we've been in restorative phases recently
            recent_sleep = (self.node_phases[node_idx] == Phase.SLEEP.value).float()
            if recent_sleep < 0.1:  # Haven't had enough sleep
                return Phase.SLEEP.value
        
        # Low energy with anxiety consideration
        if energy < 2.0:
            # Anxiety affects sleep quality needs
            if anxiety > self.anxiety_threshold * 0.5:
                return Phase.SLEEP.value  # Need deeper restoration
            else:
                return Phase.SLEEP.value
                
        # High energy transitions with anxiety modulation
        elif energy > 20.0 and activity < 0.3:
            if anxiety < self.anxiety_threshold * 0.3 and circadian_factor > 0.5:
                # Low anxiety + good circadian timing = inspiration possible
                return Phase.INSPIRED.value
            else:
                # High energy but anxious = social interaction
                return Phase.INTERACTIVE.value
                
        elif activity > 0.7:
            # High activity - anxiety affects choice between active/interactive
            if anxiety > self.anxiety_threshold * 0.4:
                # Anxious nodes seek interaction over pure activity
                return Phase.INTERACTIVE.value
            else:
                # Normal high activity transition
                return (
                    Phase.ACTIVE.value if np.random.random() > 0.3 else Phase.INTERACTIVE.value
                )
        else:
            # Use transition probabilities with anxiety modulation
            probs = self.transition_matrix[current_phase].clone()
            
            # Modify probabilities based on anxiety and restorative needs
            if anxiety > self.anxiety_threshold * 0.5:
                probs[Phase.SLEEP.value] *= 1.5  # Increase sleep probability
                probs[Phase.INTERACTIVE.value] *= 1.3  # Increase social interaction
                probs[Phase.ACTIVE.value] *= 0.7  # Decrease pure activity
            
            if restorative_need > 0.4:
                probs[Phase.SLEEP.value] *= 1.4
                probs[Phase.INSPIRED.value] *= 0.6  # Less likely to be inspired when tired
            
            # Normalize probabilities
            probs = probs / probs.sum()
            
            return torch.multinomial(probs, 1).item()
    
    def _update_restorative_state(self) -> None:
        """Update restorative needs and sleep quality based on current phases."""
        # Increase restorative need for active phases
        active_mask = (self.node_phases == Phase.ACTIVE.value)
        interactive_mask = (self.node_phases == Phase.INTERACTIVE.value)
        sleep_mask = (self.node_phases == Phase.SLEEP.value)
        
        # Active phases increase restorative need
        self.restorative_needs[active_mask] += 0.05
        self.restorative_needs[interactive_mask] += 0.03
        
        # Sleep phases reduce restorative need and improve sleep quality
        sleep_effectiveness = torch.where(
            self.node_anxiety[sleep_mask] > self.anxiety_threshold,
            0.7,  # Reduced effectiveness when anxious
            1.0   # Full effectiveness when calm
        )
        
        self.restorative_needs[sleep_mask] -= self.restorative_strength * sleep_effectiveness
        self.sleep_quality[sleep_mask] = torch.clamp(
            self.sleep_quality[sleep_mask] + 0.02 * sleep_effectiveness, 0.0, 1.0
        )
        
        # Clamp restorative needs
        self.restorative_needs = torch.clamp(self.restorative_needs, 0.0, 1.0)
        
        # Gradual sleep quality decay when not sleeping
        non_sleep_mask = ~sleep_mask
        self.sleep_quality[non_sleep_mask] *= 0.995

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
    
    def get_anxiety_stats(self) -> dict[str, float]:
        """Get anxiety-related statistics."""
        return {
            'mean_anxiety': self.node_anxiety.mean().item(),
            'max_anxiety': self.node_anxiety.max().item(),
            'anxious_nodes_ratio': (self.node_anxiety > self.anxiety_threshold).float().mean().item(),
            'mean_restorative_need': self.restorative_needs.mean().item(),
            'mean_sleep_quality': self.sleep_quality.mean().item()
        }
    
    def get_sparsity_metrics(self, energy_levels: torch.Tensor, activity_levels: torch.Tensor) -> dict[str, float]:
        """
        Calculate energy and activity sparsity metrics.
        
        Args:
            energy_levels: [batch_size, num_nodes, 1]
            activity_levels: [batch_size, num_nodes, 1]
            
        Returns:
            Dictionary with sparsity metrics
        """
        # Flatten to [batch_size * num_nodes]
        energy_flat = energy_levels.flatten()
        activity_flat = activity_levels.flatten()
        
        # Energy sparsity metrics
        energy_sparsity = (energy_flat < 0.1).float().mean().item()  # Fraction with very low energy
        energy_l0_norm = (energy_flat > 0.01).float().sum().item()  # Count of non-zero energies
        energy_l1_norm = energy_flat.abs().sum().item()
        energy_l2_norm = torch.sqrt((energy_flat ** 2).sum()).item()
        
        # Activity sparsity metrics  
        activity_sparsity = (activity_flat < 0.1).float().mean().item()  # Fraction with very low activity
        activity_l0_norm = (activity_flat > 0.01).float().sum().item()  # Count of active nodes
        activity_l1_norm = activity_flat.abs().sum().item()
        activity_l2_norm = torch.sqrt((activity_flat ** 2).sum()).item()
        
        # Combined sparsity (nodes with both low energy and activity)
        combined_sparse = ((energy_flat < 0.1) & (activity_flat < 0.1)).float().mean().item()
        
        # Phase-based sparsity
        active_nodes = self.get_active_mask(self.node_phases.unsqueeze(0)).flatten()
        active_ratio = active_nodes.float().mean().item()
        
        return {
            'energy_sparsity': energy_sparsity,
            'energy_l0_ratio': energy_l0_norm / len(energy_flat),
            'energy_l1_norm': energy_l1_norm,
            'energy_l2_norm': energy_l2_norm,
            'activity_sparsity': activity_sparsity,
            'activity_l0_ratio': activity_l0_norm / len(activity_flat),
            'activity_l1_norm': activity_l1_norm,
            'activity_l2_norm': activity_l2_norm,
            'combined_sparsity': combined_sparse,
            'active_phase_ratio': active_ratio,
            'mean_energy': energy_flat.mean().item(),
            'mean_activity': activity_flat.mean().item()
        }

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_step = 0
        self.node_phases.zero_()
        self.node_anxiety.zero_()
        self.anxiety_history.zero_()
        self.restorative_needs.zero_()
        self.sleep_quality.fill_(1.0)
