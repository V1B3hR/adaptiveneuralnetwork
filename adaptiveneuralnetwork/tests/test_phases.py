"""
Tests for phase scheduling system.
"""

import pytest
import torch
import numpy as np

from adaptiveneuralnetwork.core.phases import Phase, PhaseScheduler


class TestPhase:
    """Test Phase enumeration."""
    
    def test_phase_values(self):
        """Test phase enumeration values."""
        assert Phase.ACTIVE.value == 0
        assert Phase.SLEEP.value == 1
        assert Phase.INTERACTIVE.value == 2
        assert Phase.INSPIRED.value == 3


class TestPhaseScheduler:
    """Test PhaseScheduler functionality."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = PhaseScheduler(num_nodes=10, device="cpu")
        
        assert scheduler.num_nodes == 10
        assert scheduler.current_step == 0
        assert scheduler.node_phases.shape == (10,)
        assert torch.all(scheduler.node_phases == 0)  # Should start with ACTIVE phase
        
    def test_custom_circadian_period(self):
        """Test custom circadian period."""
        scheduler = PhaseScheduler(
            num_nodes=5,
            circadian_period=50
        )
        
        assert scheduler.circadian_period == 50
        
    def test_custom_phase_weights(self):
        """Test custom phase weights."""
        custom_weights = {
            Phase.ACTIVE: 0.8,
            Phase.INTERACTIVE: 0.15,
            Phase.SLEEP: 0.04,
            Phase.INSPIRED: 0.01
        }
        
        scheduler = PhaseScheduler(
            num_nodes=5,
            phase_weights=custom_weights
        )
        
        assert scheduler.phase_weights == custom_weights
        
    def test_step_basic(self):
        """Test basic step functionality."""
        scheduler = PhaseScheduler(num_nodes=3, device="cpu")
        
        # Create sample energy and activity
        energy = torch.tensor([[[5.0], [15.0], [1.0]]])  # Shape: [batch=1, nodes=3, energy_dim=1]
        activity = torch.tensor([[[0.3], [0.8], [0.2]]])  # Shape: [batch=1, nodes=3, activity_dim=1]
        
        # Step scheduler
        phases = scheduler.step(energy, activity)
        
        # Check output shape
        assert phases.shape == (1, 3)  # [batch_size, num_nodes]
        
        # Check phases are valid
        assert torch.all(phases >= 0)
        assert torch.all(phases < 4)  # 4 phases total
        
    def test_low_energy_sleep_transition(self):
        """Test that low energy leads to sleep phase.""" 
        scheduler = PhaseScheduler(num_nodes=5, device="cpu")
        
        # Very low energy should trigger sleep
        energy = torch.tensor([[[0.5], [1.0], [0.8], [1.5], [0.3]]])
        activity = torch.tensor([[[0.5], [0.5], [0.5], [0.5], [0.5]]])
        
        phases = scheduler.step(energy, activity)
        
        # Nodes with energy < 2.0 should be in sleep phase (Phase.SLEEP.value = 1)
        low_energy_mask = energy.squeeze(-1) < 2.0  # Remove energy dimension
        sleep_phases = phases[0, low_energy_mask[0]]  # Use batch 0
        
        # At least some low energy nodes should be in sleep
        # (Note: there's randomness, so we can't guarantee all will be sleep)
        assert torch.any(sleep_phases == Phase.SLEEP.value)
        
    def test_high_energy_inspired_potential(self):
        """Test that very high energy can lead to inspired phase."""
        scheduler = PhaseScheduler(num_nodes=3, device="cpu")
        
        # Very high energy, low activity (conditions for inspired phase)
        energy = torch.tensor([[[25.0], [30.0], [22.0]]])  
        activity = torch.tensor([[[0.1], [0.2], [0.15]]])
        
        phases = scheduler.step(energy, activity)
        
        # With high energy and low activity, some nodes might enter inspired phase
        # Due to randomness and circadian influence, we just check valid range
        assert torch.all(phases >= 0)
        assert torch.all(phases < 4)
        
    def test_get_phase_mask(self):
        """Test phase mask generation."""
        scheduler = PhaseScheduler(num_nodes=4, device="cpu")
        
        # Create sample phases
        phases = torch.tensor([[0, 1, 2, 0]])  # [batch=1, nodes=4]
        
        # Get mask for ACTIVE phase (value 0)
        active_mask = scheduler.get_phase_mask(phases, Phase.ACTIVE)
        expected_active = torch.tensor([[True, False, False, True]])
        assert torch.equal(active_mask, expected_active)
        
        # Get mask for SLEEP phase (value 1)
        sleep_mask = scheduler.get_phase_mask(phases, Phase.SLEEP)
        expected_sleep = torch.tensor([[False, True, False, False]])
        assert torch.equal(sleep_mask, expected_sleep)
        
    def test_get_active_mask(self):
        """Test active mask generation."""
        scheduler = PhaseScheduler(num_nodes=4, device="cpu")
        
        # Phases: ACTIVE, SLEEP, INTERACTIVE, INSPIRED
        phases = torch.tensor([[0, 1, 2, 3]])
        
        active_mask = scheduler.get_active_mask(phases)
        
        # ACTIVE, INTERACTIVE, and INSPIRED should be considered active
        # SLEEP should not be active
        expected_mask = torch.tensor([[True, False, True, True]])
        assert torch.equal(active_mask, expected_mask)
        
    def test_get_phase_stats(self):
        """Test phase statistics calculation."""
        scheduler = PhaseScheduler(num_nodes=4, device="cpu")
        
        # Create phases with known distribution
        # Batch 1: [ACTIVE, SLEEP, INTERACTIVE, INSPIRED] 
        # Batch 2: [ACTIVE, ACTIVE, SLEEP, INTERACTIVE]
        phases = torch.tensor([
            [0, 1, 2, 3],
            [0, 0, 1, 2]
        ])
        
        stats = scheduler.get_phase_stats(phases)
        
        # Total nodes: 2 batches * 4 nodes = 8 nodes
        # ACTIVE: 3 nodes -> 3/8 = 0.375
        # SLEEP: 2 nodes -> 2/8 = 0.25  
        # INTERACTIVE: 2 nodes -> 2/8 = 0.25
        # INSPIRED: 1 node -> 1/8 = 0.125
        
        assert abs(stats['active_ratio'] - 0.375) < 1e-6
        assert abs(stats['sleep_ratio'] - 0.25) < 1e-6
        assert abs(stats['interactive_ratio'] - 0.25) < 1e-6
        assert abs(stats['inspired_ratio'] - 0.125) < 1e-6
        
    def test_reset(self):
        """Test scheduler reset functionality."""
        scheduler = PhaseScheduler(num_nodes=3, device="cpu")
        
        # Advance scheduler state
        energy = torch.tensor([[[5.0], [10.0], [2.0]]])
        activity = torch.tensor([[[0.5], [0.8], [0.3]]])
        
        for _ in range(10):
            scheduler.step(energy, activity)
            
        # State should be advanced
        assert scheduler.current_step == 10
        
        # Reset scheduler
        scheduler.reset()
        
        # State should be reset
        assert scheduler.current_step == 0
        assert torch.all(scheduler.node_phases == 0)
        
    def test_circadian_influence(self):
        """Test that circadian rhythm influences phase transitions."""
        scheduler = PhaseScheduler(num_nodes=2, circadian_period=20, device="cpu")
        
        # Same energy/activity but at different circadian phases
        energy = torch.tensor([[[21.0], [21.0]]])  # High energy
        activity = torch.tensor([[[0.2], [0.2]]])   # Low activity
        
        phases_list = []
        
        # Collect phases over multiple steps to see circadian influence
        for _ in range(40):  # Two full circadian cycles
            phases = scheduler.step(energy, activity)
            phases_list.append(phases.clone())
            
        phases_tensor = torch.stack(phases_list)  # [steps, batch, nodes]
        
        # The phases should change over time due to circadian influence
        # (exact behavior depends on random elements, so we just check variability)
        unique_phases = torch.unique(phases_tensor)
        assert len(unique_phases) > 1  # Should have some phase variation
        
    def test_batch_processing(self):
        """Test processing multiple batches."""
        scheduler = PhaseScheduler(num_nodes=3, device="cpu")
        
        # Multiple batches
        batch_size = 2
        energy = torch.tensor([
            [[5.0], [10.0], [1.0]],  # Batch 1
            [[15.0], [0.5], [8.0]]   # Batch 2
        ])
        activity = torch.tensor([
            [[0.5], [0.8], [0.2]],   # Batch 1
            [[0.6], [0.3], [0.7]]    # Batch 2
        ])
        
        phases = scheduler.step(energy, activity)
        
        # Check shape
        assert phases.shape == (batch_size, 3)
        
        # Check valid phase values
        assert torch.all(phases >= 0)
        assert torch.all(phases < 4)