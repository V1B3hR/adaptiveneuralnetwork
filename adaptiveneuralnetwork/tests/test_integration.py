"""
Integration tests for adaptive neural networks.

These tests verify that the complete system works together correctly.
"""

import pytest
import torch

from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel
from adaptiveneuralnetwork.training.datasets import create_synthetic_loaders
from adaptiveneuralnetwork.training.loops import quick_train


class TestBasicIntegration:
    """Test basic integration between components."""

    def test_model_creation(self):
        """Test that model can be created with default config."""
        config = AdaptiveConfig(num_nodes=10, hidden_dim=8, num_epochs=1, batch_size=4)

        model = AdaptiveModel(config)

        assert model.config == config
        assert model.node_state.config.num_nodes == 10
        assert model.node_state.config.hidden_dim == 8

    def test_forward_pass_smoke(self):
        """Test that forward pass works without errors."""
        config = AdaptiveConfig(num_nodes=5, hidden_dim=4, input_dim=10, output_dim=3)

        model = AdaptiveModel(config)

        # Create sample input
        batch_size = 2
        x = torch.randn(batch_size, config.input_dim)

        # Forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (batch_size, config.output_dim)

        # Output should be finite
        assert torch.all(torch.isfinite(output))

    def test_training_step_smoke(self):
        """Test that a single training step works."""
        config = AdaptiveConfig(
            num_nodes=8, hidden_dim=6, input_dim=20, output_dim=5, learning_rate=0.01
        )

        model = AdaptiveModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # Create sample data
        batch_size = 4
        x = torch.randn(batch_size, config.input_dim)
        y = torch.randint(0, config.output_dim, (batch_size,))

        # Training step
        model.train()
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(loss)

        # Model should have been updated
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))


class TestTrainingIntegration:
    """Test training loop integration."""

    def test_quick_training_synthetic(self):
        """Test quick training on synthetic data."""
        config = AdaptiveConfig(
            num_nodes=15,
            hidden_dim=10,
            input_dim=20,
            output_dim=3,
            num_epochs=2,
            batch_size=8,
            learning_rate=0.01,
            save_checkpoint=False,
            metrics_file=None,
        )

        # Create synthetic data
        train_loader, test_loader = create_synthetic_loaders(
            num_samples=100,
            batch_size=config.batch_size,
            input_dim=config.input_dim,
            num_classes=config.output_dim,
        )

        # Run training
        results = quick_train(
            config=config,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config.num_epochs,
        )

        # Check results structure
        assert "model" in results
        assert "trainer" in results
        assert "metrics_history" in results
        assert "final_metrics" in results

        # Check that training happened
        metrics_history = results["metrics_history"]
        assert len(metrics_history) == config.num_epochs

        # Check final metrics
        final_metrics = results["final_metrics"]
        assert "train_loss" in final_metrics
        assert "train_accuracy" in final_metrics
        assert "val_loss" in final_metrics
        assert "val_accuracy" in final_metrics
        assert "active_node_ratio" in final_metrics

        # Check metrics are reasonable
        assert 0 <= final_metrics["train_accuracy"] <= 100
        assert 0 <= final_metrics["val_accuracy"] <= 100
        assert 0 <= final_metrics["active_node_ratio"] <= 1
        assert final_metrics["train_loss"] >= 0
        assert final_metrics["val_loss"] >= 0

    def test_model_state_consistency(self):
        """Test that model state remains consistent during training."""
        config = AdaptiveConfig(num_nodes=6, hidden_dim=4, input_dim=8, output_dim=2, batch_size=4)

        model = AdaptiveModel(config)

        # Check initial state
        initial_energy = model.node_state.energy.clone()
        initial_activity = model.node_state.activity.clone()

        # Run several forward passes
        for _ in range(5):
            x = torch.randn(config.batch_size, config.input_dim)
            output = model(x)

            # State should remain valid
            assert torch.all(torch.isfinite(model.node_state.energy))
            assert torch.all(torch.isfinite(model.node_state.activity))
            assert torch.all(torch.isfinite(model.node_state.hidden_state))

            # Energy should be positive
            assert torch.all(model.node_state.energy >= 0)

            # Activity should be in [0, 1] range
            assert torch.all(model.node_state.activity >= 0)
            assert torch.all(model.node_state.activity <= 1)

        # State should have changed from initial
        assert not torch.equal(model.node_state.energy, initial_energy)

    def test_batch_size_handling(self):
        """Test handling of different batch sizes."""
        config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)

        model = AdaptiveModel(config)

        # Test different batch sizes
        for batch_size in [1, 3, 7, 10]:
            x = torch.randn(batch_size, config.input_dim)
            output = model(x)

            assert output.shape == (batch_size, config.output_dim)
            assert model.node_state.get_batch_size() >= batch_size

    def test_model_reset(self):
        """Test model state reset functionality."""
        config = AdaptiveConfig(num_nodes=5, hidden_dim=4, input_dim=8, output_dim=3)

        model = AdaptiveModel(config)

        # Run forward pass to change state
        x = torch.randn(2, config.input_dim)
        output1 = model(x)

        # Store state after forward pass
        energy_after = model.node_state.energy.clone()
        activity_after = model.node_state.activity.clone()

        # Reset model
        model.reset_state()

        # Run forward pass again
        output2 = model(x)

        # State should be reset (but outputs might be different due to learned parameters)
        assert model.node_state.get_batch_size() == 1  # Should reset to batch size 1
        assert model.phase_scheduler.current_step == 0  # Should reset scheduler

    def test_model_metrics(self):
        """Test model metrics collection."""
        config = AdaptiveConfig(num_nodes=8, hidden_dim=6, input_dim=10, output_dim=4)

        model = AdaptiveModel(config)

        # Run forward pass
        x = torch.randn(3, config.input_dim)
        output = model(x)

        # Get metrics
        metrics = model.get_metrics()

        # Check metric types and ranges
        assert isinstance(metrics["active_node_ratio"], float)
        assert isinstance(metrics["mean_energy"], float)
        assert isinstance(metrics["mean_activity"], float)

        assert 0 <= metrics["active_node_ratio"] <= 1
        assert metrics["mean_energy"] >= 0
        assert 0 <= metrics["mean_activity"] <= 1

        # Should have phase statistics
        assert "active_ratio" in metrics or "sleep_ratio" in metrics


class TestConfigIntegration:
    """Test configuration system integration."""

    def test_config_to_node_config(self):
        """Test conversion from AdaptiveConfig to NodeConfig."""
        config = AdaptiveConfig(
            num_nodes=12,
            hidden_dim=16,
            spatial_dim=3,
            energy_decay=0.05,
            learning_rate=0.002,
            device="cpu",
            dtype="float32",
        )

        node_config = config.to_node_config()

        assert node_config.num_nodes == 12
        assert node_config.hidden_dim == 16
        assert node_config.spatial_dim == 3
        assert node_config.energy_decay == 0.05
        assert node_config.learning_rate == 0.002
        assert node_config.device == "cpu"
        assert node_config.dtype == torch.float32

    def test_config_dict_conversion(self):
        """Test configuration dictionary conversion."""
        config = AdaptiveConfig(num_nodes=10, hidden_dim=8, learning_rate=0.01)

        # Convert to dict and back
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["num_nodes"] == 10
        assert config_dict["hidden_dim"] == 8
        assert config_dict["learning_rate"] == 0.01

    def test_config_update(self):
        """Test configuration update functionality."""
        config = AdaptiveConfig(num_nodes=10, hidden_dim=8, learning_rate=0.01)

        # Update configuration
        updated_config = config.update(num_nodes=20, learning_rate=0.001)

        # Original should be unchanged
        assert config.num_nodes == 10
        assert config.learning_rate == 0.01

        # Updated should have new values
        assert updated_config.num_nodes == 20
        assert updated_config.learning_rate == 0.001
        assert updated_config.hidden_dim == 8  # Unchanged
