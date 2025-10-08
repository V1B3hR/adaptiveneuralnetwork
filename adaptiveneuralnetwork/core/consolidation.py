"""
Unified consolidation system for adaptive neural networks.

This module consolidates different types of consolidation mechanisms:
- Phase-based consolidation (during sleep phases) 
- Synaptic consolidation (EWC-based weight protection)
- Memory consolidation (episodic to semantic transfer)

The unified system ensures these mechanisms work together effectively.
"""

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from .nodes import NodeState
from .phases import PhaseScheduler


class ConsolidationType(Enum):
    """Types of consolidation mechanisms."""
    PHASE_BASED = "phase_based"      # Sleep-phase consolidation
    SYNAPTIC = "synaptic"           # EWC-based weight protection
    MEMORY = "memory"               # Episodic to semantic transfer


class ConsolidationMechanism(ABC):
    """Abstract base class for consolidation mechanisms."""

    def __init__(self, name: str, consolidation_type: ConsolidationType, **config):
        self.name = name
        self.consolidation_type = consolidation_type
        self.config = config
        self.is_active = False

    @abstractmethod
    def consolidate(self, **kwargs) -> dict[str, Any]:
        """
        Perform consolidation operation.
        
        Returns:
            Dictionary with consolidation results and metrics
        """
        pass

    @abstractmethod
    def get_consolidation_strength(self) -> float:
        """Get current consolidation strength."""
        pass

    def activate(self):
        """Activate this consolidation mechanism."""
        self.is_active = True

    def deactivate(self):
        """Deactivate this consolidation mechanism."""
        self.is_active = False


class PhaseBasedConsolidation(ConsolidationMechanism):
    """Phase-based consolidation during sleep phases."""

    def __init__(self, **config):
        default_config = {
            "memory_decay": 0.1,
            "stability_boost": 1.2,
            "consolidation_strength": 0.8,
        }
        default_config.update(config)
        super().__init__(
            "phase_consolidation",
            ConsolidationType.PHASE_BASED,
            **default_config
        )

    def consolidate(
        self,
        node_state: NodeState,
        phase_scheduler: PhaseScheduler,
        **kwargs
    ) -> dict[str, Any]:
        """Apply memory consolidation during sleep phases."""
        results = {"modifications": [], "metrics": {}}

        if not self.is_active:
            return results

        # Identify nodes in consolidation (sleep) phase
        sleep_nodes = phase_scheduler.node_phases == 1
        if not sleep_nodes.any():
            return results

        # Extract activity and energy tensors (handle batch dimension)
        activity = node_state.activity.squeeze(-1)  # [batch, nodes]
        energy = node_state.energy.squeeze(-1)     # [batch, nodes]

        # Apply consolidation to first batch (or average across batches)
        if activity.dim() > 1:
            activity = activity[0]  # Use first batch
            energy = energy[0]

        # Reduce activity but boost stability
        activity[sleep_nodes] *= (1 - self.config["memory_decay"])

        # Boost energy for important nodes (high activity before sleep)
        important_nodes = activity > 0.5
        consolidation_nodes = sleep_nodes & important_nodes

        if consolidation_nodes.any():
            energy[consolidation_nodes] *= self.config["stability_boost"]
            results["modifications"].append(
                f"Consolidated {consolidation_nodes.sum()} important nodes"
            )

        # Update the node state tensors
        node_state.activity[0, :, 0] = activity
        node_state.energy[0, :, 0] = energy

        results["metrics"] = {
            "sleep_nodes": sleep_nodes.sum().item(),
            "consolidated_nodes": consolidation_nodes.sum().item() if consolidation_nodes.any() else 0,
            "memory_decay_applied": self.config["memory_decay"],
            "stability_boost_applied": self.config["stability_boost"]
        }

        return results

    def get_consolidation_strength(self) -> float:
        """Get phase-based consolidation strength."""
        return self.config["consolidation_strength"]


class SynapticConsolidation(ConsolidationMechanism):
    """Synaptic consolidation using Elastic Weight Consolidation (EWC)."""

    def __init__(self, model: nn.Module, **config):
        default_config = {
            "consolidation_strength": 1.0,
            "fisher_samples": 1000,
            "importance_threshold": 1e-6,
        }
        default_config.update(config)
        super().__init__(
            "synaptic_consolidation",
            ConsolidationType.SYNAPTIC,
            **default_config
        )

        self.model = model
        self.fisher_information = {}
        self.optimal_params = {}

        # Initialize Fisher information storage
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)
                self.optimal_params[name] = param.clone().detach()

    def consolidate(self, data_loader=None, **kwargs) -> dict[str, Any]:
        """Estimate Fisher information and update optimal parameters."""
        results = {"modifications": [], "metrics": {}}

        if not self.is_active:
            return results

        if data_loader is not None:
            # Estimate Fisher information from data
            self._estimate_fisher_information(data_loader)
            results["modifications"].append(
                f"Updated Fisher information from {self.config['fisher_samples']} samples"
            )

        # Update optimal parameters
        self._update_optimal_params()
        results["modifications"].append("Updated optimal parameters")

        # Calculate consolidation metrics
        total_fisher = sum(f.sum().item() for f in self.fisher_information.values())
        important_params = sum(
            (f > self.config["importance_threshold"]).sum().item()
            for f in self.fisher_information.values()
        )

        results["metrics"] = {
            "total_fisher_information": total_fisher,
            "important_parameters": important_params,
            "consolidation_strength": self.config["consolidation_strength"]
        }

        return results

    def _estimate_fisher_information(self, data_loader):
        """Estimate Fisher information matrix for current task."""
        self.model.eval()

        # Reset Fisher information
        for name in self.fisher_information:
            self.fisher_information[name].zero_()

        sample_count = 0

        for batch_idx, (data, target) in enumerate(data_loader):
            if sample_count >= self.config["fisher_samples"]:
                break

            batch_size = data.size(0)
            output = self.model(data)
            pred_class = output.argmax(dim=1)

            # Compute gradients for sampled classes
            for i in range(batch_size):
                if sample_count >= self.config["fisher_samples"]:
                    break

                self.model.zero_grad()
                log_prob = torch.nn.functional.log_softmax(output[i:i+1], dim=1)
                loss = -log_prob[0, pred_class[i]]
                loss.backward(retain_graph=True)

                # Accumulate squared gradients (Fisher information)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_information[name] += param.grad.pow(2)

                sample_count += 1

        # Normalize Fisher information
        if sample_count > 0:
            for name in self.fisher_information:
                self.fisher_information[name] /= sample_count

    def _update_optimal_params(self):
        """Update optimal parameters after learning new task."""
        for name, param in self.model.named_parameters():
            if name in self.optimal_params:
                self.optimal_params[name] = param.clone().detach()

    def get_consolidation_loss(self) -> torch.Tensor:
        """Compute consolidation loss to prevent catastrophic forgetting."""
        loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                penalty = fisher * (param - optimal).pow(2)
                loss += penalty.sum()

        return self.config["consolidation_strength"] * loss

    def get_consolidation_strength(self) -> float:
        """Get synaptic consolidation strength."""
        return self.config["consolidation_strength"]


class MemoryConsolidation(ConsolidationMechanism):
    """Memory consolidation for episodic to semantic transfer."""

    def __init__(self, memory_dim: int, **config):
        default_config = {
            "consolidation_threshold": 0.7,
            "memory_decay": 0.05,
            "semantic_boost": 1.1,
        }
        default_config.update(config)
        super().__init__(
            "memory_consolidation",
            ConsolidationType.MEMORY,
            **default_config
        )

        self.memory_dim = memory_dim

        # Memory consolidation network
        self.consolidation_network = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Tanh()
        )

    def consolidate(
        self,
        episodic_memories: torch.Tensor,
        semantic_memories: torch.Tensor,
        importance_scores: torch.Tensor | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Consolidate episodic memories into semantic memory."""
        results = {"modifications": [], "metrics": {}}

        if not self.is_active:
            return results

        # Filter memories based on importance threshold
        if importance_scores is not None:
            consolidation_mask = importance_scores > self.config["consolidation_threshold"]
            if not consolidation_mask.any():
                return results

            episodic_to_consolidate = episodic_memories[consolidation_mask]
        else:
            episodic_to_consolidate = episodic_memories

        # Perform consolidation
        if episodic_to_consolidate.numel() > 0:
            # Combine episodic and semantic information
            batch_size = episodic_to_consolidate.size(0)
            semantic_context = semantic_memories.mean(dim=0, keepdim=True).expand(batch_size, -1)

            combined = torch.cat([episodic_to_consolidate, semantic_context], dim=1)
            consolidated = self.consolidation_network(combined)

            results["consolidated_memories"] = consolidated
            results["modifications"].append(
                f"Consolidated {batch_size} episodic memories to semantic"
            )

        results["metrics"] = {
            "memories_consolidated": episodic_to_consolidate.size(0) if episodic_to_consolidate.numel() > 0 else 0,
            "consolidation_threshold": self.config["consolidation_threshold"]
        }

        return results

    def get_consolidation_strength(self) -> float:
        """Get memory consolidation strength."""
        return self.config.get("consolidation_strength", 0.8)


class UnifiedConsolidationManager:
    """Unified manager for all consolidation mechanisms."""

    def __init__(self):
        self.mechanisms: dict[str, ConsolidationMechanism] = {}
        self.active_mechanisms: list[str] = []
        self.consolidation_history: list[dict[str, Any]] = []

    def register_mechanism(
        self,
        mechanism: ConsolidationMechanism,
        activate: bool = True
    ) -> None:
        """Register a consolidation mechanism."""
        if mechanism.name in self.mechanisms:
            warnings.warn(f"Mechanism '{mechanism.name}' already registered. Replacing.")

        self.mechanisms[mechanism.name] = mechanism

        if activate:
            self.activate_mechanism(mechanism.name)

    def activate_mechanism(self, mechanism_name: str) -> bool:
        """Activate a consolidation mechanism."""
        if mechanism_name not in self.mechanisms:
            warnings.warn(f"Mechanism '{mechanism_name}' not registered")
            return False

        mechanism = self.mechanisms[mechanism_name]
        mechanism.activate()

        if mechanism_name not in self.active_mechanisms:
            self.active_mechanisms.append(mechanism_name)

        return True

    def deactivate_mechanism(self, mechanism_name: str) -> bool:
        """Deactivate a consolidation mechanism."""
        if mechanism_name in self.active_mechanisms:
            self.mechanisms[mechanism_name].deactivate()
            self.active_mechanisms.remove(mechanism_name)
            return True
        return False

    def consolidate_all(self, **kwargs) -> dict[str, Any]:
        """Run all active consolidation mechanisms."""
        results = {
            "mechanisms": {},
            "summary": {
                "total_mechanisms": len(self.active_mechanisms),
                "total_modifications": 0
            }
        }

        for mechanism_name in self.active_mechanisms:
            mechanism = self.mechanisms[mechanism_name]

            try:
                # Filter kwargs for mechanism-specific parameters
                mechanism_kwargs = self._filter_kwargs_for_mechanism(
                    mechanism, kwargs
                )

                mechanism_results = mechanism.consolidate(**mechanism_kwargs)
                results["mechanisms"][mechanism_name] = mechanism_results

                # Update summary
                if "modifications" in mechanism_results:
                    results["summary"]["total_modifications"] += len(mechanism_results["modifications"])

            except Exception as e:
                warnings.warn(f"Mechanism '{mechanism_name}' failed: {e}")
                results["mechanisms"][mechanism_name] = {"error": str(e)}

        # Store in history
        self.consolidation_history.append(results)

        return results

    def _filter_kwargs_for_mechanism(
        self,
        mechanism: ConsolidationMechanism,
        kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter kwargs based on mechanism type."""
        filtered = {}

        if mechanism.consolidation_type == ConsolidationType.PHASE_BASED:
            # Phase-based consolidation needs node_state and phase_scheduler
            for key in ["node_state", "phase_scheduler"]:
                if key in kwargs:
                    filtered[key] = kwargs[key]

        elif mechanism.consolidation_type == ConsolidationType.SYNAPTIC:
            # Synaptic consolidation needs data_loader
            for key in ["data_loader"]:
                if key in kwargs:
                    filtered[key] = kwargs[key]

        elif mechanism.consolidation_type == ConsolidationType.MEMORY:
            # Memory consolidation needs memory tensors
            for key in ["episodic_memories", "semantic_memories", "importance_scores"]:
                if key in kwargs:
                    filtered[key] = kwargs[key]

        return filtered

    def get_total_consolidation_strength(self) -> float:
        """Get combined consolidation strength from all active mechanisms."""
        total_strength = 0.0

        for mechanism_name in self.active_mechanisms:
            mechanism = self.mechanisms[mechanism_name]
            total_strength += mechanism.get_consolidation_strength()

        return total_strength

    def get_consolidation_info(self) -> dict[str, Any]:
        """Get information about all consolidation mechanisms."""
        info = {
            "registered_mechanisms": list(self.mechanisms.keys()),
            "active_mechanisms": self.active_mechanisms.copy(),
            "total_consolidation_strength": self.get_total_consolidation_strength(),
            "mechanism_details": {}
        }

        for name, mechanism in self.mechanisms.items():
            info["mechanism_details"][name] = {
                "type": mechanism.consolidation_type.value,
                "is_active": mechanism.is_active,
                "strength": mechanism.get_consolidation_strength(),
                "config": mechanism.config.copy()
            }

        return info


# Convenience function for creating default consolidation manager
def create_default_consolidation_manager(
    model: nn.Module | None = None,
    memory_dim: int = 256
) -> UnifiedConsolidationManager:
    """Create consolidation manager with default mechanisms."""
    manager = UnifiedConsolidationManager()

    # Register phase-based consolidation
    phase_consolidation = PhaseBasedConsolidation()
    manager.register_mechanism(phase_consolidation)

    # Register synaptic consolidation if model provided
    if model is not None:
        synaptic_consolidation = SynapticConsolidation(model)
        manager.register_mechanism(synaptic_consolidation)

    # Register memory consolidation
    memory_consolidation = MemoryConsolidation(memory_dim)
    manager.register_mechanism(memory_consolidation)

    return manager
