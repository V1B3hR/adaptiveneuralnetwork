"""
Tests for the unified consolidation system.

This module tests the consolidated consolidation mechanisms to ensure
they work together correctly and maintain backward compatibility.
"""

import pytest
import torch
import torch.nn as nn

from adaptiveneuralnetwork.core.consolidation import (
    ConsolidationType,
    MemoryConsolidation,
    PhaseBasedConsolidation,
    SynapticConsolidation,
    UnifiedConsolidationManager,
    create_default_consolidation_manager,
)
from adaptiveneuralnetwork.core.nodes import NodeConfig, NodeState
from adaptiveneuralnetwork.core.phases import PhaseScheduler


class SimpleModel(nn.Module):
    """Simple model for testing consolidation."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestUnifiedConsolidationManager:
    """Test the unified consolidation manager."""

    def test_manager_creation(self):
        """Test creating a consolidation manager."""
        manager = UnifiedConsolidationManager()
        assert len(manager.mechanisms) == 0
        assert len(manager.active_mechanisms) == 0

    def test_mechanism_registration(self):
        """Test registering consolidation mechanisms."""
        manager = UnifiedConsolidationManager()

        # Create and register phase-based consolidation
        phase_consolidation = PhaseBasedConsolidation()
        manager.register_mechanism(phase_consolidation)

        assert "phase_consolidation" in manager.mechanisms
        assert "phase_consolidation" in manager.active_mechanisms
        assert phase_consolidation.is_active

    def test_mechanism_activation_deactivation(self):
        """Test activating and deactivating mechanisms."""
        manager = UnifiedConsolidationManager()
        phase_consolidation = PhaseBasedConsolidation()
        manager.register_mechanism(phase_consolidation, activate=False)

        assert not phase_consolidation.is_active
        assert "phase_consolidation" not in manager.active_mechanisms

        # Activate
        manager.activate_mechanism("phase_consolidation")
        assert phase_consolidation.is_active
        assert "phase_consolidation" in manager.active_mechanisms

        # Deactivate
        manager.deactivate_mechanism("phase_consolidation")
        assert not phase_consolidation.is_active
        assert "phase_consolidation" not in manager.active_mechanisms


class TestPhaseBasedConsolidation:
    """Test phase-based consolidation."""

    def test_phase_consolidation_creation(self):
        """Test creating phase-based consolidation."""
        consolidation = PhaseBasedConsolidation()
        assert consolidation.consolidation_type == ConsolidationType.PHASE_BASED
        assert consolidation.config["memory_decay"] == 0.1
        assert consolidation.config["stability_boost"] == 1.2

    def test_phase_consolidation_logic(self):
        """Test phase consolidation logic."""
        consolidation = PhaseBasedConsolidation()
        consolidation.activate()

        # Create test node state and phase scheduler
        num_nodes = 8
        config = NodeConfig(num_nodes=num_nodes, device="cpu")
        node_state = NodeState(config)

        phase_scheduler = PhaseScheduler(num_nodes=num_nodes)
        # Set some nodes to sleep phase (phase 1)
        phase_scheduler.node_phases[2:4] = 1  # Two nodes in sleep phase

        # Make some nodes have high activity (important)
        node_state.activity[0, 2, 0] = 0.8  # Important sleep node
        node_state.activity[0, 3, 0] = 0.3  # Less important sleep node

        initial_energy = node_state.energy.clone()

        results = consolidation.consolidate(
            node_state=node_state,
            phase_scheduler=phase_scheduler
        )

        # Check results structure
        assert "modifications" in results
        assert "metrics" in results
        assert results["metrics"]["sleep_nodes"] == 2

        # Check that energy was boosted for important sleep nodes
        assert node_state.energy[0, 2, 0] > initial_energy[0, 2, 0]  # Important node boosted


class TestSynapticConsolidation:
    """Test synaptic consolidation."""

    def test_synaptic_consolidation_creation(self):
        """Test creating synaptic consolidation."""
        model = SimpleModel()
        consolidation = SynapticConsolidation(model)
        assert consolidation.consolidation_type == ConsolidationType.SYNAPTIC
        assert consolidation.model is model
        assert len(consolidation.fisher_information) > 0
        assert len(consolidation.optimal_params) > 0

    def test_consolidation_loss(self):
        """Test computing consolidation loss."""
        model = SimpleModel()
        consolidation = SynapticConsolidation(model)
        consolidation.activate()

        # Initialize some Fisher information
        for name, param in model.named_parameters():
            consolidation.fisher_information[name] = torch.ones_like(param) * 0.1

        loss = consolidation.get_consolidation_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative


class TestMemoryConsolidation:
    """Test memory consolidation."""

    def test_memory_consolidation_creation(self):
        """Test creating memory consolidation."""
        memory_dim = 64
        consolidation = MemoryConsolidation(memory_dim)
        assert consolidation.consolidation_type == ConsolidationType.MEMORY
        assert consolidation.memory_dim == memory_dim

    def test_memory_consolidation_logic(self):
        """Test memory consolidation logic."""
        memory_dim = 64
        batch_size = 4

        consolidation = MemoryConsolidation(memory_dim)
        consolidation.activate()

        episodic_memories = torch.randn(batch_size, memory_dim)
        semantic_memories = torch.randn(10, memory_dim)  # Bank of semantic memories
        importance_scores = torch.tensor([0.9, 0.8, 0.5, 0.6])  # Above threshold for first 2

        results = consolidation.consolidate(
            episodic_memories=episodic_memories,
            semantic_memories=semantic_memories,
            importance_scores=importance_scores
        )

        assert "consolidated_memories" in results
        assert "metrics" in results

        # Should consolidate memories with importance > 0.7
        expected_consolidated = (importance_scores > consolidation.config["consolidation_threshold"]).sum().item()
        assert results["metrics"]["memories_consolidated"] == expected_consolidated


class TestIntegratedConsolidation:
    """Test integrated consolidation system."""

    def test_default_manager_creation(self):
        """Test creating default consolidation manager."""
        model = SimpleModel()
        manager = create_default_consolidation_manager(model=model, memory_dim=64)

        assert len(manager.mechanisms) == 3  # Phase, synaptic, memory
        assert "phase_consolidation" in manager.mechanisms
        assert "synaptic_consolidation" in manager.mechanisms
        assert "memory_consolidation" in manager.mechanisms

    def test_integrated_consolidation(self):
        """Test running all consolidation mechanisms together."""
        model = SimpleModel()
        manager = create_default_consolidation_manager(model=model, memory_dim=64)

        # Create test data
        num_nodes = 8
        config = NodeConfig(num_nodes=num_nodes, device="cpu")
        node_state = NodeState(config)

        phase_scheduler = PhaseScheduler(num_nodes=num_nodes)
        phase_scheduler.node_phases[2:4] = 1  # Sleep phase

        episodic_memories = torch.randn(4, 64)
        semantic_memories = torch.randn(10, 64)

        # Run all consolidation
        results = manager.consolidate_all(
            node_state=node_state,
            phase_scheduler=phase_scheduler,
            episodic_memories=episodic_memories,
            semantic_memories=semantic_memories,
            importance_scores=torch.ones(4) * 0.8
        )

        # Check results structure
        assert "mechanisms" in results
        assert "summary" in results
        assert "phase_consolidation" in results["mechanisms"]
        assert "memory_consolidation" in results["mechanisms"]

        # Synaptic consolidation should run but may not have results without data_loader
        assert len(results["mechanisms"]) >= 2

    def test_consolidation_info(self):
        """Test getting consolidation information."""
        model = SimpleModel()
        manager = create_default_consolidation_manager(model=model)

        info = manager.get_consolidation_info()

        assert "registered_mechanisms" in info
        assert "active_mechanisms" in info
        assert "total_consolidation_strength" in info
        assert "mechanism_details" in info

        assert len(info["registered_mechanisms"]) == 3
        assert info["total_consolidation_strength"] > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_plugin_system_compatibility(self):
        """Test that plugin system still works with consolidated approach."""
        # Import here to avoid circular imports in testing
        from adaptiveneuralnetwork.core.plugin_system import ConsolidationPhase

        plugin = ConsolidationPhase()
        assert plugin.consolidation_manager is not None
        assert "phase_consolidation" in plugin.consolidation_manager.mechanisms

    def test_continual_learning_compatibility(self):
        """Test that continual learning still works with consolidated approach."""
        # Test direct consolidation imports instead of the full application to avoid dependencies
        try:
            from adaptiveneuralnetwork.applications.continual_learning import (
                SynapticConsolidation as AppSynapticConsolidation,
            )

            model = SimpleModel()
            consolidation = AppSynapticConsolidation(model)

            # Should maintain backward compatibility
            assert hasattr(consolidation, 'fisher_information')
            assert hasattr(consolidation, 'optimal_params')
            assert hasattr(consolidation, 'consolidation_loss')
            assert hasattr(consolidation, 'estimate_fisher_information')

            # Test consolidation loss computation
            loss = consolidation.consolidation_loss()
            assert isinstance(loss, torch.Tensor)

        except ImportError as e:
            # Skip if dependencies are missing
            pytest.skip(f"Skipping continual learning test due to missing dependency: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
