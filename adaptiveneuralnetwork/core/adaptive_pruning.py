"""
Adaptive pruning and self-healing node lifecycle management.

This module implements intelligent pruning that can identify underperforming
nodes and adaptively restructure the network, with self-healing capabilities
to recover from over-pruning or network damage.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from .nodes import NodeState
from .phases import PhaseScheduler


class NodeHealth(Enum):
    """Node health states."""
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DORMANT = "dormant"
    FAILING = "failing"
    DEAD = "dead"


@dataclass
class PruningConfig:
    """Configuration for adaptive pruning."""
    # Pruning thresholds
    activity_threshold: float = 0.05  # Below this, node is candidate for pruning
    energy_threshold: float = 0.1    # Below this, node is low energy
    contribution_threshold: float = 0.01  # Minimum contribution to network output

    # Pruning rates
    max_prune_rate: float = 0.1      # Maximum fraction to prune per cycle
    min_nodes: int = 10              # Minimum nodes to maintain

    # Healing parameters
    healing_enabled: bool = True
    healing_threshold: float = 0.05   # Performance drop to trigger healing
    max_heal_rate: float = 0.2       # Maximum fraction to heal per cycle

    # Evaluation periods
    evaluation_window: int = 100     # Steps to evaluate node performance
    pruning_frequency: int = 500     # Steps between pruning cycles
    healing_frequency: int = 200     # Steps between healing evaluations


class NodeLifecycleManager:
    """Manages adaptive pruning and self-healing of nodes."""

    def __init__(
        self,
        node_state: NodeState,
        phase_scheduler: PhaseScheduler | None = None,
        config: PruningConfig | None = None
    ):
        self.node_state = node_state
        self.phase_scheduler = phase_scheduler
        self.config = config or PruningConfig()

        self.num_nodes = node_state.config.num_nodes
        self.device = node_state.device

        # Node health tracking
        self.node_health = torch.full(
            (self.num_nodes,),
            0,  # 0 = healthy
            dtype=torch.int8,
            device=self.device
        )

        # Performance history
        self.activity_history = torch.zeros(
            (self.num_nodes, self.config.evaluation_window),
            device=self.device
        )
        self.energy_history = torch.zeros(
            (self.num_nodes, self.config.evaluation_window),
            device=self.device
        )
        self.contribution_history = torch.zeros(
            (self.num_nodes, self.config.evaluation_window),
            device=self.device
        )

        # Tracking variables
        self.step_count = 0
        self.history_index = 0
        self.active_nodes = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
        self.pruned_nodes = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)

        # Performance baseline for healing decisions
        self.performance_baseline = 0.0
        self.performance_history = []

        # Statistics
        self.pruning_stats = {
            "total_pruned": 0,
            "total_healed": 0,
            "pruning_cycles": 0,
            "healing_cycles": 0,
        }

    def update_node_metrics(self, contribution_scores: torch.Tensor | None = None) -> None:
        """Update node performance metrics."""

        # Update activity history - flatten to 1D if needed and match expected size
        activity = self.node_state.activity.flatten()
        if activity.shape[0] != self.num_nodes:
            # Pad or truncate to match expected size
            if activity.shape[0] < self.num_nodes:
                activity = torch.cat([activity, torch.zeros(self.num_nodes - activity.shape[0], device=activity.device)])
            else:
                activity = activity[:self.num_nodes]

        self.activity_history[:, self.history_index] = activity

        # Update energy history - flatten to 1D if needed and match expected size
        energy = self.node_state.energy.flatten()
        if energy.shape[0] != self.num_nodes:
            # Pad or truncate to match expected size
            if energy.shape[0] < self.num_nodes:
                energy = torch.cat([energy, torch.zeros(self.num_nodes - energy.shape[0], device=energy.device)])
            else:
                energy = energy[:self.num_nodes]

        self.energy_history[:, self.history_index] = energy

        # Update contribution history
        if contribution_scores is not None:
            contrib = contribution_scores.flatten()
            if contrib.shape[0] != self.num_nodes:
                if contrib.shape[0] < self.num_nodes:
                    contrib = torch.cat([contrib, torch.zeros(self.num_nodes - contrib.shape[0], device=contrib.device)])
                else:
                    contrib = contrib[:self.num_nodes]
            self.contribution_history[:, self.history_index] = contrib
        else:
            # Use energy * activity as proxy for contribution
            proxy_contribution = energy * activity
            self.contribution_history[:, self.history_index] = proxy_contribution

        # Advance history index
        self.history_index = (self.history_index + 1) % self.config.evaluation_window
        self.step_count += 1

    def assess_node_health(self) -> torch.Tensor:
        """Assess health status of all nodes."""

        # Calculate rolling averages
        mean_activity = self.activity_history.mean(dim=1)
        mean_energy = self.energy_history.mean(dim=1)
        mean_contribution = self.contribution_history.mean(dim=1)

        # Initialize health as healthy
        health = torch.full((self.num_nodes,), 0, dtype=torch.int8, device=self.device)  # 0 = healthy

        # Classify based on metrics
        # Dead nodes: very low activity, energy, and contribution
        dead_mask = (
            (mean_activity < self.config.activity_threshold * 0.1) &
            (mean_energy < self.config.energy_threshold * 0.1) &
            (mean_contribution < self.config.contribution_threshold * 0.1)
        )
        health[dead_mask] = 4  # NodeHealth.DEAD

        # Failing nodes: low on all metrics but not dead
        failing_mask = (
            (mean_activity < self.config.activity_threshold) &
            (mean_energy < self.config.energy_threshold) &
            (mean_contribution < self.config.contribution_threshold) &
            ~dead_mask
        )
        health[failing_mask] = 3  # NodeHealth.FAILING

        # Dormant nodes: low activity but reasonable energy
        dormant_mask = (
            (mean_activity < self.config.activity_threshold) &
            (mean_energy >= self.config.energy_threshold) &
            ~failing_mask & ~dead_mask
        )
        health[dormant_mask] = 2  # NodeHealth.DORMANT

        # Stressed nodes: moderate activity but low energy or contribution
        stressed_mask = (
            (mean_activity >= self.config.activity_threshold) &
            ((mean_energy < self.config.energy_threshold) |
             (mean_contribution < self.config.contribution_threshold)) &
            ~dormant_mask & ~failing_mask & ~dead_mask
        )
        health[stressed_mask] = 1  # NodeHealth.STRESSED

        # Remaining nodes are healthy (already initialized to 0)

        self.node_health = health
        return health

    def identify_pruning_candidates(self) -> torch.Tensor:
        """Identify nodes that are candidates for pruning."""

        if not self.should_run_pruning():
            return torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)

        # Assess current health
        health = self.assess_node_health()

        # Candidates are dead or failing nodes
        candidates = (health == 4) | (health == 3)  # DEAD or FAILING

        # Don't prune if it would go below minimum
        current_active = self.active_nodes.sum().item()
        max_to_prune = min(
            int(current_active * self.config.max_prune_rate),
            current_active - self.config.min_nodes
        )
        max_to_prune = max(0, max_to_prune)

        # If too many candidates, prioritize worst performers
        if candidates.sum() > max_to_prune:
            # Sort by combined score (lower is worse)
            activity_scores = self.activity_history.mean(dim=1)
            energy_scores = self.energy_history.mean(dim=1)
            contribution_scores = self.contribution_history.mean(dim=1)

            combined_scores = activity_scores + energy_scores + contribution_scores
            combined_scores[~candidates] = float('inf')  # Exclude non-candidates

            # Get indices of worst performers
            _, worst_indices = torch.topk(combined_scores, max_to_prune, largest=False)

            # Create new candidate mask
            new_candidates = torch.zeros_like(candidates)
            new_candidates[worst_indices] = True
            candidates = new_candidates

        return candidates

    def prune_nodes(self, node_indices: torch.Tensor) -> dict[str, Any]:
        """Prune specified nodes."""

        if not node_indices.any():
            return {"pruned_count": 0, "message": "No nodes to prune"}

        # Update active nodes mask
        self.active_nodes[node_indices] = False
        self.pruned_nodes[node_indices] = True

        # Zero out pruned node states
        self.node_state.activity[node_indices] = 0.0
        self.node_state.energy[node_indices] = 0.0

        # Update statistics
        pruned_count = node_indices.sum().item()
        self.pruning_stats["total_pruned"] += pruned_count
        self.pruning_stats["pruning_cycles"] += 1

        result = {
            "pruned_count": pruned_count,
            "pruned_indices": node_indices.nonzero().squeeze(-1).tolist(),
            "remaining_active": self.active_nodes.sum().item(),
            "message": f"Pruned {pruned_count} nodes"
        }

        return result

    def identify_healing_candidates(self, current_performance: float) -> tuple[torch.Tensor, bool]:
        """Identify if healing is needed and which nodes to restore."""

        if not self.config.healing_enabled or not self.should_run_healing():
            return torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device), False

        # Check if performance has degraded significantly
        performance_drop = self.performance_baseline - current_performance
        needs_healing = performance_drop > self.config.healing_threshold

        candidates = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)

        if needs_healing and self.pruned_nodes.any():
            # Identify best candidates among pruned nodes
            # Look at historical performance before pruning

            # For now, use simple heuristic: restore most recently pruned nodes first
            pruned_indices = self.pruned_nodes.nonzero().squeeze(-1)

            if len(pruned_indices) > 0:
                max_to_heal = min(
                    int(self.num_nodes * self.config.max_heal_rate),
                    len(pruned_indices)
                )

                # Select candidates (most recently pruned first - simple heuristic)
                heal_indices = pruned_indices[:max_to_heal]
                candidates[heal_indices] = True

        return candidates, needs_healing

    def heal_nodes(self, node_indices: torch.Tensor) -> dict[str, Any]:
        """Restore/heal pruned nodes."""

        if not node_indices.any():
            return {"healed_count": 0, "message": "No nodes to heal"}

        # Restore nodes
        self.active_nodes[node_indices] = True
        self.pruned_nodes[node_indices] = False

        # Initialize restored nodes with reasonable values
        self.node_state.energy[node_indices] = torch.rand(
            node_indices.sum(), device=self.device
        ) * 0.5 + 0.25  # Random energy between 0.25 and 0.75

        self.node_state.activity[node_indices] = torch.rand(
            node_indices.sum(), device=self.device
        ) * 0.3 + 0.1   # Random activity between 0.1 and 0.4

        # Update statistics
        healed_count = node_indices.sum().item()
        self.pruning_stats["total_healed"] += healed_count
        self.pruning_stats["healing_cycles"] += 1

        result = {
            "healed_count": healed_count,
            "healed_indices": node_indices.nonzero().squeeze(-1).tolist(),
            "total_active": self.active_nodes.sum().item(),
            "message": f"Healed {healed_count} nodes"
        }

        return result

    def should_run_pruning(self) -> bool:
        """Check if it's time to run pruning."""
        return (
            self.step_count > self.config.evaluation_window and
            self.step_count % self.config.pruning_frequency == 0
        )

    def should_run_healing(self) -> bool:
        """Check if it's time to evaluate healing."""
        return (
            self.step_count > self.config.evaluation_window and
            self.step_count % self.config.healing_frequency == 0
        )

    def update_performance_baseline(self, current_performance: float) -> None:
        """Update performance baseline for healing decisions."""
        self.performance_history.append(current_performance)

        # Keep only recent history
        max_history = 50
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]

        # Update baseline as moving average
        if len(self.performance_history) >= 10:
            self.performance_baseline = sum(self.performance_history[-10:]) / 10
        else:
            self.performance_baseline = current_performance

    def step(
        self,
        current_performance: float | None = None,
        contribution_scores: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """
        Perform one step of lifecycle management.
        
        Args:
            current_performance: Current model performance (for healing decisions)
            contribution_scores: Node contribution scores (optional)
            
        Returns:
            Dictionary with step results and any actions taken
        """
        results = {
            "step": self.step_count,
            "actions": [],
            "metrics": {},
            "node_health": {}
        }

        # Update node metrics
        self.update_node_metrics(contribution_scores)

        # Update performance baseline
        if current_performance is not None:
            self.update_performance_baseline(current_performance)

        # Check for pruning
        if self.should_run_pruning():
            pruning_candidates = self.identify_pruning_candidates()
            if pruning_candidates.any():
                prune_result = self.prune_nodes(pruning_candidates)
                results["actions"].append(("prune", prune_result))

        # Check for healing
        if current_performance is not None and self.should_run_healing():
            healing_candidates, needs_healing = self.identify_healing_candidates(current_performance)
            if needs_healing and healing_candidates.any():
                heal_result = self.heal_nodes(healing_candidates)
                results["actions"].append(("heal", heal_result))

        # Update health assessment
        health = self.assess_node_health()
        health_counts = {
            "healthy": (health == 0).sum().item(),
            "stressed": (health == 1).sum().item(),
            "dormant": (health == 2).sum().item(),
            "failing": (health == 3).sum().item(),
            "dead": (health == 4).sum().item(),
        }

        results["node_health"] = health_counts
        # Get flattened activity and energy with size matching
        activity_flat = self.node_state.activity.flatten()
        energy_flat = self.node_state.energy.flatten()

        # Ensure they match the number of nodes
        if activity_flat.shape[0] != self.num_nodes:
            if activity_flat.shape[0] < self.num_nodes:
                activity_flat = torch.cat([activity_flat, torch.zeros(self.num_nodes - activity_flat.shape[0], device=activity_flat.device)])
            else:
                activity_flat = activity_flat[:self.num_nodes]

        if energy_flat.shape[0] != self.num_nodes:
            if energy_flat.shape[0] < self.num_nodes:
                energy_flat = torch.cat([energy_flat, torch.zeros(self.num_nodes - energy_flat.shape[0], device=energy_flat.device)])
            else:
                energy_flat = energy_flat[:self.num_nodes]

        results["metrics"] = {
            "active_nodes": self.active_nodes.sum().item(),
            "pruned_nodes": self.pruned_nodes.sum().item(),
            "mean_activity": activity_flat[self.active_nodes].mean().item() if self.active_nodes.any() else 0,
            "mean_energy": energy_flat[self.active_nodes].mean().item() if self.active_nodes.any() else 0,
            "performance_baseline": self.performance_baseline,
        }

        return results

    def get_lifecycle_stats(self) -> dict[str, Any]:
        """Get comprehensive lifecycle management statistics."""
        return {
            "pruning_stats": self.pruning_stats.copy(),
            "current_state": {
                "total_nodes": self.num_nodes,
                "active_nodes": self.active_nodes.sum().item(),
                "pruned_nodes": self.pruned_nodes.sum().item(),
                "health_distribution": {
                    health.value: (self.node_health == i).sum().item()
                    for i, health in enumerate(NodeHealth)
                },
            },
            "performance": {
                "baseline": self.performance_baseline,
                "history_length": len(self.performance_history),
            },
            "config": {
                "activity_threshold": self.config.activity_threshold,
                "energy_threshold": self.config.energy_threshold,
                "contribution_threshold": self.config.contribution_threshold,
                "max_prune_rate": self.config.max_prune_rate,
                "min_nodes": self.config.min_nodes,
                "healing_enabled": self.config.healing_enabled,
            }
        }


def create_adaptive_pruning_manager(
    node_state: NodeState,
    phase_scheduler: PhaseScheduler | None = None,
    **config_kwargs
) -> NodeLifecycleManager:
    """
    Create an adaptive pruning manager with custom configuration.
    
    Args:
        node_state: Node state to manage
        phase_scheduler: Optional phase scheduler
        **config_kwargs: Configuration overrides
        
    Returns:
        Configured NodeLifecycleManager
    """
    config = PruningConfig(**config_kwargs)
    return NodeLifecycleManager(node_state, phase_scheduler, config)
