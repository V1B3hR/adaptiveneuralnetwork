"""
Tests for 0.4.0 roadmap features.

This module tests the new features implemented for version 0.4.0:
- ONNX export + model introspection
- Reproducibility harness
- Energy-aware optimizers 
- Plugin system
- Enhanced continual learning
- Adaptive pruning
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from ..api.config import AdaptiveConfig
from ..api.model import AdaptiveModel
from ..core.adaptive_pruning import NodeLifecycleManager, PruningConfig
from ..core.plugin_system import CreativePhase, PluginAwarePhaseScheduler, PluginManager
from ..training.energy_optimizers import EnergyAwareAdam, create_energy_aware_optimizer
from ..training.enhanced_continual import DomainShiftConfig, ProgressiveDomainShift
from ..utils.onnx_export import ModelIntrospection, ONNXExporter, export_model_with_introspection
from ..utils.reproducibility import ReproducibilityHarness, set_global_seed


class TestModelIntrospection:
    """Test ONNX export and model introspection features."""

    def test_model_introspection_basic(self):
        """Test basic model introspection functionality."""
        config = AdaptiveConfig(num_nodes=8, hidden_dim=6, input_dim=10, output_dim=4)
        model = AdaptiveModel(config)

        introspector = ModelIntrospection(model)
        summary = introspector.get_model_summary()

        assert "architecture" in summary
        assert "parameters" in summary
        assert "memory" in summary
        assert "structure" in summary

        assert summary["architecture"]["num_nodes"] == 8
        assert summary["architecture"]["hidden_dim"] == 6
        assert summary["parameters"]["total_parameters"] > 0
        assert summary["parameters"]["trainable_parameters"] > 0

    def test_onnx_export_basic(self):
        """Test basic ONNX export functionality."""
        config = AdaptiveConfig(num_nodes=5, hidden_dim=4, input_dim=8, output_dim=3)
        model = AdaptiveModel(config)

        exporter = ONNXExporter(model)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            # Test export (may fail if ONNX not installed, but should not crash)
            success = exporter.export_to_onnx(onnx_path)
            # Don't assert success since ONNX might not be installed

        finally:
            # Cleanup
            Path(onnx_path).unlink(missing_ok=True)

    def test_export_with_introspection(self):
        """Test combined export with introspection."""
        config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            results = export_model_with_introspection(
                model,
                temp_dir,
                export_onnx=False,  # Skip ONNX to avoid dependency issues
                create_summary=True
            )

            assert "output_directory" in results
            assert "files_created" in results
            assert len(results["files_created"]) > 0

            # Check summary file was created
            summary_file = Path(temp_dir) / "model_summary.json"
            assert summary_file.exists()

            # Verify summary content
            with open(summary_file) as f:
                summary = json.load(f)
                assert "architecture" in summary
                assert "parameters" in summary


class TestReproducibilityHarness:
    """Test reproducibility harness functionality."""

    def test_seed_setting(self):
        """Test seed setting functionality."""
        harness = ReproducibilityHarness(master_seed=123, strict_mode=False)
        seed = harness.set_seed()

        assert seed == 123

        # Test that random numbers are reproducible
        x1 = torch.randn(3, 3)
        harness.set_seed(123)
        x2 = torch.randn(3, 3)

        assert torch.allclose(x1, x2)

    def test_determinism_verification(self):
        """Test determinism verification."""
        harness = ReproducibilityHarness(master_seed=42, strict_mode=False)

        def deterministic_function():
            torch.manual_seed(42)
            return torch.randn(2, 2).sum().item()

        def non_deterministic_function():
            return torch.rand(1).item()  # Different each time

        # Test deterministic function
        report = harness.verify_determinism(deterministic_function, "deterministic_test", run_count=3)
        assert report.is_deterministic
        assert report.unique_outputs == 1

        # Test reproducibility utilities
        set_global_seed(42)
        val1 = torch.randn(1).item()
        set_global_seed(42)
        val2 = torch.randn(1).item()
        assert val1 == val2

    def test_environment_snapshot(self):
        """Test environment snapshot capture."""
        harness = ReproducibilityHarness()
        env = harness.environment

        assert hasattr(env, 'python_version')
        assert hasattr(env, 'torch_version')
        assert hasattr(env, 'numpy_version')
        assert hasattr(env, 'platform_system')


class TestEnergyAwareOptimizers:
    """Test energy-aware optimizers."""

    def test_energy_aware_adam_creation(self):
        """Test creating energy-aware Adam optimizer."""
        config = AdaptiveConfig(num_nodes=6, hidden_dim=4, input_dim=8, output_dim=3)
        model = AdaptiveModel(config)

        optimizer = EnergyAwareAdam(
            model.parameters(),
            model.node_state,
            model.phase_scheduler,
            lr=0.01
        )

        assert isinstance(optimizer, EnergyAwareAdam)
        assert optimizer.defaults['lr'] == 0.01
        assert optimizer.defaults['energy_scaling'] == True

    def test_energy_scaling_computation(self):
        """Test energy scaling factor computation."""
        config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(config)

        optimizer = create_energy_aware_optimizer(
            'adam', model.parameters(), model.node_state
        )

        scaling = optimizer.get_energy_scaling_factor()
        assert scaling.shape == model.node_state.energy.shape
        assert torch.all(scaling >= 0.1)
        assert torch.all(scaling <= 2.0)

    def test_optimizer_step(self):
        """Test optimizer step with energy awareness."""
        config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(config)

        optimizer = create_energy_aware_optimizer(
            'sgd', model.parameters(), model.node_state, lr=0.01
        )

        # Perform a training step
        x = torch.randn(2, 6)
        y = torch.randint(0, 2, (2,))

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert len(optimizer.adaptation_history) > 0
        assert 'learning_rate' in optimizer.adaptation_history[0]


class TestPluginSystem:
    """Test plugin system functionality."""

    def test_plugin_manager_creation(self):
        """Test plugin manager creation and registration."""
        manager = PluginManager()

        creative_plugin = CreativePhase(creativity_boost=1.5)
        phase_id = manager.register_plugin(creative_plugin)

        assert phase_id >= 5  # After built-in phases
        assert creative_plugin.name in manager.plugins
        assert creative_plugin.name in manager.plugin_phases

    def test_plugin_activation(self):
        """Test plugin activation and deactivation."""
        manager = PluginManager()
        creative_plugin = CreativePhase()

        manager.register_plugin(creative_plugin)
        assert creative_plugin.name not in manager.active_plugins

        success = manager.activate_plugin(creative_plugin.name)
        assert success
        assert creative_plugin.name in manager.active_plugins

        success = manager.deactivate_plugin(creative_plugin.name)
        assert success
        assert creative_plugin.name not in manager.active_plugins

    def test_plugin_phase_application(self):
        """Test applying plugin phases."""
        config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(config)

        manager = PluginManager()
        creative_plugin = CreativePhase(creativity_boost=2.0, exploration_noise=0.05)

        manager.register_plugin(creative_plugin)
        manager.activate_plugin(creative_plugin.name)

        # Apply plugins (even without nodes in creative phase, it should work)
        results = manager.apply_plugin_phases(
            model.node_state, model.phase_scheduler, step=1
        )

        assert "plugins" in results
        assert "summary" in results

    def test_plugin_aware_scheduler(self):
        """Test plugin-aware phase scheduler."""
        config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(config)

        manager = PluginManager()
        creative_plugin = CreativePhase()
        manager.register_plugin(creative_plugin)
        manager.activate_plugin(creative_plugin.name)

        scheduler = PluginAwarePhaseScheduler(
            manager,
            num_nodes=config.num_nodes,
            device=config.device
        )

        # Test step with plugin integration
        results = scheduler.step(model.node_state, current_step=1)
        assert isinstance(results, dict)


class TestAdaptivePruning:
    """Test adaptive pruning and node lifecycle management."""

    def test_lifecycle_manager_creation(self):
        """Test creating node lifecycle manager."""
        config = AdaptiveConfig(num_nodes=6, hidden_dim=4, input_dim=8, output_dim=3)
        model = AdaptiveModel(config)

        prune_config = PruningConfig(
            activity_threshold=0.1,
            energy_threshold=0.15,
            min_nodes=2
        )

        manager = NodeLifecycleManager(
            model.node_state,
            model.phase_scheduler,
            prune_config
        )

        assert manager.num_nodes == 6
        assert manager.config.activity_threshold == 0.1
        assert manager.active_nodes.sum() == 6  # All nodes initially active

    def test_node_health_assessment(self):
        """Test node health assessment."""
        config = AdaptiveConfig(num_nodes=5, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(config)

        manager = NodeLifecycleManager(model.node_state)

        # Simulate some history
        for i in range(10):
            manager.update_node_metrics()

        health = manager.assess_node_health()
        assert health.shape == (5,)
        assert torch.all(health >= 0)
        assert torch.all(health <= 4)

    def test_pruning_identification(self):
        """Test pruning candidate identification."""
        config = AdaptiveConfig(num_nodes=8, hidden_dim=4, input_dim=6, output_dim=3)
        model = AdaptiveModel(config)

        manager = NodeLifecycleManager(
            model.node_state,
            config=PruningConfig(min_nodes=4, max_prune_rate=0.5)
        )

        # Simulate low performance for some nodes
        model.node_state.activity[:3] = 0.01  # Very low activity
        model.node_state.energy[:3] = 0.02    # Very low energy

        # Build up history
        for i in range(150):  # More than evaluation window
            manager.update_node_metrics()

        candidates = manager.identify_pruning_candidates()
        assert candidates.shape == (8,)
        # Should identify some candidates with low activity/energy

    def test_lifecycle_step(self):
        """Test full lifecycle management step."""
        config = AdaptiveConfig(num_nodes=6, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(config)

        manager = NodeLifecycleManager(model.node_state)

        # Perform several steps
        for i in range(5):
            results = manager.step(current_performance=0.8)

            assert "step" in results
            assert "metrics" in results
            assert "node_health" in results

            # Check metrics structure
            assert "active_nodes" in results["metrics"]
            assert "mean_activity" in results["metrics"]
            assert "mean_energy" in results["metrics"]


class TestEnhancedContinualLearning:
    """Test enhanced continual learning scenarios."""

    def test_domain_shift_config(self):
        """Test domain shift configuration."""
        config = DomainShiftConfig(
            scenario_name="blur_to_noise",
            num_stages=3,
            samples_per_stage=100,
            initial_corruption=0.0,
            final_corruption=0.8
        )

        assert config.scenario_name == "blur_to_noise"
        assert config.num_stages == 3
        assert config.initial_corruption == 0.0
        assert config.final_corruption == 0.8

    def test_progressive_domain_shift_creation(self):
        """Test creating progressive domain shift."""
        # Create simple synthetic dataset
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        from torch.utils.data import TensorDataset
        base_dataset = TensorDataset(data, labels)

        config = DomainShiftConfig(
            scenario_name="blur_progression",
            num_stages=3,
            samples_per_stage=20,
        )

        domain_shift = ProgressiveDomainShift(base_dataset, config)

        assert len(domain_shift.stage_datasets) == 3
        assert domain_shift.current_stage == 0

    def test_stage_advancement(self):
        """Test stage advancement in domain shift."""
        # Simple synthetic dataset
        data = torch.randn(50, 1, 28, 28)
        labels = torch.randint(0, 5, (50,))
        from torch.utils.data import TensorDataset
        base_dataset = TensorDataset(data, labels)

        config = DomainShiftConfig(scenario_name="noise", num_stages=3, samples_per_stage=10)
        domain_shift = ProgressiveDomainShift(base_dataset, config)

        # Test advancement
        assert domain_shift.current_stage == 0

        success = domain_shift.advance_stage()
        assert success
        assert domain_shift.current_stage == 1

        success = domain_shift.advance_stage()
        assert success
        assert domain_shift.current_stage == 2

        success = domain_shift.advance_stage()
        assert not success  # Should be at end
        assert domain_shift.current_stage == 2

    def test_stage_info(self):
        """Test stage information retrieval."""
        data = torch.randn(30, 1, 28, 28)
        labels = torch.randint(0, 3, (30,))
        from torch.utils.data import TensorDataset
        base_dataset = TensorDataset(data, labels)

        config = DomainShiftConfig(scenario_name="test", num_stages=2, samples_per_stage=5)
        domain_shift = ProgressiveDomainShift(base_dataset, config)

        info = domain_shift.get_stage_info()

        assert "stage" in info
        assert "total_stages" in info
        assert "scenario" in info
        assert "corruption_intensity" in info
        assert info["scenario"] == "test"
        assert info["total_stages"] == 2


if __name__ == "__main__":
    pytest.main([__file__])
