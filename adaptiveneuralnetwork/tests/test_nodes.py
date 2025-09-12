"""
Tests for node state management.
"""

import pytest
import torch

from adaptiveneuralnetwork.core.nodes import NodeConfig, NodeState


class TestNodeConfig:
    """Test NodeConfig functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NodeConfig()

        assert config.num_nodes == 100
        assert config.hidden_dim == 64
        assert config.device == "cpu"
        assert config.dtype == torch.float32

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NodeConfig(num_nodes=50, hidden_dim=32, device="cuda", dtype=torch.float16)

        assert config.num_nodes == 50
        assert config.hidden_dim == 32
        assert config.device == "cuda"
        assert config.dtype == torch.float16


class TestNodeState:
    """Test NodeState functionality."""

    def test_initialization(self):
        """Test node state initialization."""
        config = NodeConfig(num_nodes=10, hidden_dim=8)
        state = NodeState(config)

        assert state.hidden_state.shape == (1, 10, 8)
        assert state.energy.shape == (1, 10, 1)
        assert state.activity.shape == (1, 10, 1)
        assert state.position.shape == (1, 10, 2)  # default spatial_dim=2

    def test_energy_clamping(self):
        """Test energy clamping functionality."""
        config = NodeConfig(num_nodes=5, hidden_dim=4)
        state = NodeState(config)

        # Set some extreme energy values
        state.energy = torch.tensor([[[-5.0], [25.0], [35.0], [100.0], [-10.0]]])

        # Clamp energy
        state.clamp_energy(min_val=0.0, max_val=50.0)

        # Check clamping worked
        assert torch.all(state.energy >= 0.0)
        assert torch.all(state.energy <= 50.0)
        assert state.energy[0, 0, 0] == 0.0  # Was -5.0
        assert state.energy[0, 1, 0] == 25.0  # Unchanged
        assert state.energy[0, 2, 0] == 35.0  # Was 35.0, within range so unchanged
        assert state.energy[0, 3, 0] == 50.0  # Was 100.0, now clamped

    def test_activity_clamping(self):
        """Test activity clamping functionality."""
        config = NodeConfig(num_nodes=3, hidden_dim=4)
        state = NodeState(config)

        # Set extreme activity values
        state.activity = torch.tensor([[[-2.0], [0.5], [3.0]]])

        # Clamp activity
        state.clamp_activity(min_val=0.0, max_val=1.0)

        # Check clamping worked
        assert torch.all(state.activity >= 0.0)
        assert torch.all(state.activity <= 1.0)
        assert state.activity[0, 0, 0] == 0.0  # Was -2.0
        assert state.activity[0, 1, 0] == 0.5  # Unchanged
        assert state.activity[0, 2, 0] == 1.0  # Was 3.0, now clamped

    def test_active_nodes_detection(self):
        """Test active node detection."""
        config = NodeConfig(num_nodes=4, hidden_dim=4, activity_threshold=0.5)
        state = NodeState(config)

        # Set energy and activity values
        state.energy = torch.tensor([[[2.0], [0.5], [3.0], [1.5]]])  # >1.0 threshold
        state.activity = torch.tensor([[[0.6], [0.4], [0.7], [0.3]]])  # >0.5 threshold

        active_mask = state.get_active_nodes()

        # Node 0: energy=2.0 >1.0, activity=0.6 >0.5 -> active
        # Node 1: energy=0.5 <1.0, activity=0.4 <0.5 -> inactive
        # Node 2: energy=3.0 >1.0, activity=0.7 >0.5 -> active
        # Node 3: energy=1.5 >1.0, activity=0.3 <0.5 -> inactive

        expected_mask = torch.tensor([[[True], [False], [True], [False]]])
        assert torch.equal(active_mask, expected_mask)

    def test_energy_update(self):
        """Test energy update functionality."""
        config = NodeConfig(num_nodes=3, hidden_dim=4, energy_decay=0.1)
        state = NodeState(config)

        # Set initial energy
        initial_energy = torch.tensor([[[10.0], [5.0], [8.0]]])
        state.energy = initial_energy.clone()

        # Apply energy update
        delta = torch.tensor([[[1.0], [-2.0], [3.0]]])
        state.update_energy(delta)

        # Check energy decay and delta application
        expected_energy = initial_energy * 0.9 + delta  # 0.9 = 1 - 0.1 decay

        # Should be clamped to valid range
        expected_energy = torch.clamp(expected_energy, 0.0, 50.0)

        assert torch.allclose(state.energy, expected_energy)

    def test_activity_update(self):
        """Test activity update functionality."""
        config = NodeConfig(num_nodes=2, hidden_dim=4)
        state = NodeState(config)

        # Set hidden state
        state.hidden_state = torch.randn(1, 2, 4)

        # Apply activity update
        stimulation = torch.tensor([[[0.5], [-0.3]]])
        state.update_activity(stimulation)

        # Activity should be between 0 and 1 (sigmoid output)
        assert torch.all(state.activity >= 0.0)
        assert torch.all(state.activity <= 1.0)

    def test_batch_expansion(self):
        """Test batch size expansion."""
        config = NodeConfig(num_nodes=3, hidden_dim=4)
        state = NodeState(config)

        # Initial batch size should be 1
        assert state.get_batch_size() == 1

        # Expand to batch size 4
        state.expand_batch(4)

        # Check all tensors expanded correctly
        assert state.hidden_state.shape == (4, 3, 4)
        assert state.energy.shape == (4, 3, 1)
        assert state.activity.shape == (4, 3, 1)
        assert state.position.shape == (4, 3, 2)
        assert state.phase_mask.shape == (4, 3, 1)

        # Expanding to smaller size should do nothing
        state.expand_batch(2)
        assert state.get_batch_size() == 4

    def test_device_movement(self):
        """Test moving state to different device."""
        config = NodeConfig(num_nodes=2, hidden_dim=3, device="cpu")
        state = NodeState(config)

        # Should start on CPU
        assert state.device.type == "cpu"
        assert state.hidden_state.device.type == "cpu"

        # Move to CPU (should be no-op but test the method)
        state.to(torch.device("cpu"))
        assert state.device.type == "cpu"
        assert state.hidden_state.device.type == "cpu"
