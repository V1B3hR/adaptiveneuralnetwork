"""
Plugin system for custom phases in adaptive neural networks.

This module provides a flexible plugin architecture that allows users to
define custom phases and integrate them seamlessly with the core system.
"""

import importlib
import inspect
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from .consolidation import PhaseBasedConsolidation, UnifiedConsolidationManager
from .nodes import NodeState
from .phases import PhaseScheduler


class PhasePlugin(ABC):
    """Abstract base class for phase plugins."""

    def __init__(self, name: str, **config):
        self.name = name
        self.config = config
        self.is_active = False
        self.metadata = self._get_metadata()

    @abstractmethod
    def apply_phase_logic(
        self,
        node_state: NodeState,
        phase_scheduler: PhaseScheduler,
        step: int,
        **kwargs
    ) -> dict[str, Any]:
        """
        Apply custom phase logic to the network.
        
        Args:
            node_state: Current node state
            phase_scheduler: Phase scheduler instance
            step: Current training step
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with phase results and modifications
        """
        pass

    @abstractmethod
    def get_phase_transitions(self) -> dict[int, dict[int, float]]:
        """
        Define phase transition probabilities from this phase to others.
        
        Returns:
            Dictionary mapping from_phase -> {to_phase: probability}
        """
        pass

    def _get_metadata(self) -> dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "config": self.config,
        }

    def on_phase_enter(self, node_state: NodeState, **kwargs) -> None:
        """Called when nodes enter this phase."""
        self.is_active = True

    def on_phase_exit(self, node_state: NodeState, **kwargs) -> None:
        """Called when nodes exit this phase."""
        self.is_active = False

    def validate_config(self) -> list[str]:
        """Validate plugin configuration. Return list of warnings/errors."""
        warnings = []
        return warnings


class CreativePhase(PhasePlugin):
    """Example creative phase plugin that enhances exploration."""

    def __init__(self, **config):
        default_config = {
            "creativity_boost": 1.5,
            "exploration_noise": 0.1,
            "inspiration_threshold": 0.8,
        }
        default_config.update(config)
        super().__init__("creative", **default_config)

    def apply_phase_logic(
        self,
        node_state: NodeState,
        phase_scheduler: PhaseScheduler,
        step: int,
        **kwargs
    ) -> dict[str, Any]:
        """Apply creative phase enhancements."""
        results = {"modifications": [], "metrics": {}}

        # Boost activity for creative nodes
        creative_nodes = phase_scheduler.node_phases == 4  # Assuming phase 4 is creative
        if creative_nodes.any():
            boost = self.config["creativity_boost"]
            node_state.activity[creative_nodes] *= boost
            node_state.activity.clamp_(0, 1)  # Keep in valid range

            results["modifications"].append(f"Applied creativity boost {boost} to {creative_nodes.sum()} nodes")

        # Add exploration noise
        if self.config["exploration_noise"] > 0:
            noise = torch.randn_like(node_state.energy) * self.config["exploration_noise"]
            node_state.energy += noise
            results["modifications"].append(f"Added exploration noise (std={self.config['exploration_noise']})")

        # Track creative metrics
        results["metrics"] = {
            "creative_nodes": creative_nodes.sum().item(),
            "mean_creative_activity": node_state.activity[creative_nodes].mean().item() if creative_nodes.any() else 0,
            "energy_variance": node_state.energy.var().item(),
        }

        return results

    def get_phase_transitions(self) -> dict[int, dict[int, float]]:
        """Creative phase transition probabilities."""
        return {
            4: {  # From creative phase
                0: 0.4,  # To active
                1: 0.2,  # To sleep
                2: 0.3,  # To interactive
                4: 0.1,  # Stay creative
            }
        }


class ConsolidationPhase(PhasePlugin):
    """Plugin for memory consolidation during sleep-like phases."""

    def __init__(self, **config):
        default_config = {
            "consolidation_strength": 0.8,
            "memory_decay": 0.1,
            "stability_boost": 1.2,
        }
        default_config.update(config)
        super().__init__("consolidation", **default_config)

        # Use the unified consolidation system
        self.consolidation_manager = UnifiedConsolidationManager()
        self.phase_consolidation = PhaseBasedConsolidation(**default_config)
        self.consolidation_manager.register_mechanism(self.phase_consolidation)

    def apply_phase_logic(
        self,
        node_state: NodeState,
        phase_scheduler: PhaseScheduler,
        step: int,
        **kwargs
    ) -> dict[str, Any]:
        """Apply memory consolidation logic using unified system."""
        # Use the unified consolidation manager
        consolidation_results = self.consolidation_manager.consolidate_all(
            node_state=node_state,
            phase_scheduler=phase_scheduler,
            step=step,
            **kwargs
        )

        # Extract results for backward compatibility
        if "mechanisms" in consolidation_results and "phase_consolidation" in consolidation_results["mechanisms"]:
            phase_results = consolidation_results["mechanisms"]["phase_consolidation"]
            return {
                "modifications": phase_results.get("modifications", []),
                "metrics": phase_results.get("metrics", {}),
                "consolidation_summary": consolidation_results["summary"]
            }

        # Fallback to empty results
        return {"modifications": [], "metrics": {}, "consolidation_summary": {}}

    def get_phase_transitions(self) -> dict[int, dict[int, float]]:
        """Consolidation affects sleep phase transitions."""
        return {
            1: {  # From sleep phase
                0: 0.6,  # To active (refreshed)
                1: 0.2,  # Stay in sleep
                2: 0.2,  # To interactive
            }
        }


class PluginManager:
    """Manages phase plugins and their integration with the core system."""

    def __init__(self):
        self.plugins: dict[str, PhasePlugin] = {}
        self.active_plugins: list[str] = []
        self.plugin_phases: dict[str, int] = {}  # Map plugin names to phase IDs
        self.next_phase_id = 5  # Start after built-in phases (0-4)

    def register_plugin(
        self,
        plugin: PhasePlugin,
        phase_id: int | None = None,
        replace_existing: bool = False
    ) -> int:
        """
        Register a phase plugin.
        
        Args:
            plugin: Plugin instance to register
            phase_id: Specific phase ID to assign (optional)
            replace_existing: Whether to replace existing plugin with same name
            
        Returns:
            Assigned phase ID
        """
        if plugin.name in self.plugins and not replace_existing:
            raise ValueError(f"Plugin '{plugin.name}' already registered. Use replace_existing=True to override.")

        # Validate plugin configuration
        warnings_list = plugin.validate_config()
        if warnings_list:
            for warning in warnings_list:
                warnings.warn(f"Plugin '{plugin.name}': {warning}", stacklevel=2)

        # Assign phase ID
        if phase_id is None:
            phase_id = self.next_phase_id
            self.next_phase_id += 1
        elif phase_id in self.plugin_phases.values():
            if not replace_existing:
                raise ValueError(f"Phase ID {phase_id} already in use")

        # Register plugin
        self.plugins[plugin.name] = plugin
        self.plugin_phases[plugin.name] = phase_id

        print(f"Registered plugin '{plugin.name}' with phase ID {phase_id}")
        return phase_id

    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin."""
        if plugin_name not in self.plugins:
            return False

        # Deactivate if active
        if plugin_name in self.active_plugins:
            self.deactivate_plugin(plugin_name)

        # Remove from registry
        del self.plugins[plugin_name]
        del self.plugin_phases[plugin_name]

        print(f"Unregistered plugin '{plugin_name}'")
        return True

    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a registered plugin."""
        if plugin_name not in self.plugins:
            warnings.warn(f"Plugin '{plugin_name}' not registered", stacklevel=2)
            return False

        if plugin_name not in self.active_plugins:
            self.active_plugins.append(plugin_name)
            print(f"Activated plugin '{plugin_name}'")

        return True

    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate a plugin."""
        if plugin_name in self.active_plugins:
            self.active_plugins.remove(plugin_name)
            self.plugins[plugin_name].is_active = False
            print(f"Deactivated plugin '{plugin_name}'")
            return True
        return False

    def apply_plugin_phases(
        self,
        node_state: NodeState,
        phase_scheduler: PhaseScheduler,
        step: int,
        **kwargs
    ) -> dict[str, Any]:
        """Apply all active plugins to the current state."""
        all_results = {"plugins": {}, "summary": {"total_modifications": 0}}

        for plugin_name in self.active_plugins:
            plugin = self.plugins[plugin_name]
            phase_id = self.plugin_phases[plugin_name]

            # Check if any nodes are in this plugin's phase
            nodes_in_phase = (phase_scheduler.node_phases == phase_id).any()

            if nodes_in_phase or plugin.config.get("always_apply", False):
                try:
                    results = plugin.apply_phase_logic(
                        node_state, phase_scheduler, step, **kwargs
                    )
                    all_results["plugins"][plugin_name] = results

                    # Update summary
                    if "modifications" in results:
                        all_results["summary"]["total_modifications"] += len(results["modifications"])

                except Exception as e:
                    warnings.warn(f"Plugin '{plugin_name}' failed: {e}", stacklevel=2)
                    all_results["plugins"][plugin_name] = {"error": str(e)}

        return all_results

    def get_extended_transition_matrix(
        self,
        base_transitions: torch.Tensor,
        num_base_phases: int = 4
    ) -> torch.Tensor:
        """
        Extend transition matrix to include plugin phases.
        
        Args:
            base_transitions: Base transition matrix for built-in phases
            num_base_phases: Number of built-in phases
            
        Returns:
            Extended transition matrix including plugin phases
        """
        total_phases = num_base_phases + len(self.active_plugins)

        if total_phases == num_base_phases:
            return base_transitions

        # Create extended matrix
        extended_transitions = torch.zeros(total_phases, total_phases)
        extended_transitions[:num_base_phases, :num_base_phases] = base_transitions

        # Add plugin transition probabilities
        for plugin_name in self.active_plugins:
            plugin = self.plugins[plugin_name]
            phase_id = self.plugin_phases[plugin_name]

            if phase_id >= total_phases:
                continue

            plugin_transitions = plugin.get_phase_transitions()

            for from_phase, to_phases in plugin_transitions.items():
                if from_phase < total_phases:
                    for to_phase, prob in to_phases.items():
                        if to_phase < total_phases:
                            extended_transitions[from_phase, to_phase] = prob

        # Normalize rows to ensure they sum to 1
        row_sums = extended_transitions.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        extended_transitions = extended_transitions / row_sums

        return extended_transitions

    def load_plugin_from_file(self, filepath: str | Path) -> bool:
        """Load a plugin from a Python file."""
        filepath = Path(filepath)

        if not filepath.exists():
            warnings.warn(f"Plugin file not found: {filepath}", stacklevel=2)
            return False

        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("plugin_module", filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PhasePlugin) and
                    obj is not PhasePlugin):
                    plugin_classes.append(obj)

            if not plugin_classes:
                warnings.warn(f"No plugin classes found in {filepath}", stacklevel=2)
                return False

            # Register found plugins
            for plugin_class in plugin_classes:
                plugin_instance = plugin_class()
                self.register_plugin(plugin_instance)

            return True

        except Exception as e:
            warnings.warn(f"Failed to load plugin from {filepath}: {e}", stacklevel=2)
            return False

    def get_plugin_info(self) -> dict[str, Any]:
        """Get information about all registered plugins."""
        info = {
            "registered_plugins": list(self.plugins.keys()),
            "active_plugins": self.active_plugins.copy(),
            "plugin_phases": self.plugin_phases.copy(),
            "next_phase_id": self.next_phase_id,
            "plugin_details": {}
        }

        for name, plugin in self.plugins.items():
            info["plugin_details"][name] = {
                "metadata": plugin.metadata,
                "is_active": plugin.is_active,
                "phase_id": self.plugin_phases[name],
            }

        return info


class PluginAwarePhaseScheduler(PhaseScheduler):
    """Extended PhaseScheduler that supports plugins."""

    def __init__(self, plugin_manager: PluginManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plugin_manager = plugin_manager
        self._update_transition_matrix()

    def _update_transition_matrix(self):
        """Update transition matrix to include plugin phases."""
        if hasattr(self, 'transition_matrix'):
            base_transitions = self.transition_matrix
        else:
            base_transitions = self._build_transition_matrix()

        self.transition_matrix = self.plugin_manager.get_extended_transition_matrix(
            base_transitions, num_base_phases=4
        )

    def step(self, node_state: NodeState, current_step: int = None, **kwargs) -> dict[str, Any]:
        """Step with plugin application."""
        # Standard phase step - PhaseScheduler.step needs energy, activity, and anxiety levels
        step_results = super().step(
            node_state.energy,
            node_state.activity,
            getattr(node_state, 'anxiety', None)
        )

        # Apply plugins
        if current_step is not None:
            plugin_results = self.plugin_manager.apply_plugin_phases(
                node_state, self, current_step, **kwargs
            )
            # Merge plugin results
            if isinstance(step_results, dict):
                step_results.update(plugin_results)
            else:
                # If step_results is just a tensor, wrap it
                step_results = {
                    "phase_transitions": step_results,
                    "plugins": plugin_results
                }

        return step_results


# Convenience functions
def create_plugin_manager_with_defaults() -> PluginManager:
    """Create plugin manager with default plugins."""
    manager = PluginManager()

    # Register default plugins
    creative_plugin = CreativePhase()
    consolidation_plugin = ConsolidationPhase()

    manager.register_plugin(creative_plugin)
    manager.register_plugin(consolidation_plugin)

    return manager
