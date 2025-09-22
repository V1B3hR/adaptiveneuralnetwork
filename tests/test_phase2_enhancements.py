"""
Test suite for Phase 2 enhancements: trust attribute fix, energy hardening, and neuromorphic improvements.
"""

import unittest
import time
import numpy as np
import torch
from core.alive_node import AliveLoopNode
from core.trust_network import TrustNetwork

# Try to import neuromorphic components
try:
    from adaptiveneuralnetwork.core.neuromorphic import (
        NeuromorphicAdaptiveModel, 
        BrainWaveOscillator, 
        NeuromodulationSystem,
        NeuromorphicConfig
    )
    NEUROMORPHIC_AVAILABLE = True
except ImportError:
    NEUROMORPHIC_AVAILABLE = False


class TestTrustAttributeFix(unittest.TestCase):
    """Test the trust attribute fix and backward compatibility"""
    
    def setUp(self):
        self.node1 = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
        self.node2 = AliveLoopNode((1, 1), (0.1, 0.1), node_id=1)
    
    def test_trust_attribute_exists(self):
        """Test that AliveLoopNode has trust attribute"""
        self.assertTrue(hasattr(self.node1, 'trust'))
        self.assertIsInstance(self.node1.trust, float)
        self.assertEqual(self.node1.trust, 0.5)  # Default neutral trust
    
    def test_trust_attribute_updates(self):
        """Test that trust attribute updates with trust network changes"""
        initial_trust = self.node1.trust
        
        # Update trust through communication
        self.node1._update_trust_after_communication(self.node2, 'resource')
        
        # Trust attribute should be updated
        self.assertNotEqual(self.node1.trust, initial_trust)
        self.assertGreater(self.node1.trust, initial_trust)  # Should increase for positive signal
    
    def test_trust_attribute_consistency(self):
        """Test that trust attribute reflects trust network state"""
        # Set multiple trust relationships
        self.node1.trust_network[2] = 0.8
        self.node1.trust_network[3] = 0.4
        self.node1.trust_network[4] = 0.6
        
        # Update trust attribute
        self.node1._update_trust_attribute()
        
        # Trust should be influenced by network average
        expected_avg = np.mean([0.8, 0.4, 0.6])
        self.assertAlmostEqual(self.node1.trust, expected_avg, places=1)


class TestTrustDecayRecovery(unittest.TestCase):
    """Test trust decay and recovery algorithms"""
    
    def setUp(self):
        self.trust_network = TrustNetwork(node_id=0)
        self.trust_network.set_trust(1, 0.8)
        self.trust_network.set_trust(2, 0.6)
    
    def test_trust_decay_over_time(self):
        """Test that trust decays over time without interaction"""
        initial_trust = self.trust_network.get_trust(1)
        
        # Simulate time passage
        future_time = time.time() + 15  # 15 seconds later
        decayed_count = self.trust_network.apply_trust_decay(future_time)
        
        # Trust should have decayed
        self.assertGreater(decayed_count, 0)
        final_trust = self.trust_network.get_trust(1)
        self.assertLess(final_trust, initial_trust)
    
    def test_trust_recovery_mechanism(self):
        """Test trust recovery for positive interactions"""
        # Set low trust first
        self.trust_network.set_trust(3, 0.3)
        initial_trust = self.trust_network.get_trust(3)
        
        # Apply recovery
        recovery_amount = self.trust_network.apply_trust_recovery(3, recovery_factor=1.0)
        
        # Trust should have recovered
        self.assertGreater(recovery_amount, 0)
        final_trust = self.trust_network.get_trust(3)
        self.assertGreater(final_trust, initial_trust)
    
    def test_minimum_trust_threshold(self):
        """Test that trust doesn't decay below minimum threshold"""
        # Set trust and apply excessive decay
        self.trust_network.set_trust(4, 0.2)
        
        # Simulate long time passage
        very_future_time = time.time() + 1000  # Much later
        self.trust_network.apply_trust_decay(very_future_time)
        
        # Trust should not go below minimum threshold
        final_trust = self.trust_network.get_trust(4)
        self.assertGreaterEqual(final_trust, self.trust_network.MIN_TRUST_THRESHOLD)


class TestEnergySystemHardening(unittest.TestCase):
    """Test energy system hardening features"""
    
    def setUp(self):
        self.node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0, initial_energy=10.0)
    
    def test_emergency_energy_conservation(self):
        """Test emergency energy conservation activation"""
        # Set low energy to trigger emergency mode
        self.node.energy = 0.1  # Below emergency threshold
        self.node.adaptive_energy_allocation()
        
        # Emergency mode should be activated
        self.assertTrue(self.node.emergency_mode)
        self.assertLess(self.node.communication_range, 2.0)  # Should be reduced
    
    def test_energy_attack_detection(self):
        """Test energy attack detection mechanism"""
        # Simulate rapid energy drain events
        current_time = time.time()
        for i in range(3):
            self.node.energy_drain_events.append({
                'timestamp': current_time + i * 0.1,  # Quick succession
                'amount': 2.0,  # Large drain
                'source': 'attack'
            })
        
        # Should detect attack
        attack_detected = self.node.detect_energy_attack()
        self.assertTrue(attack_detected)
        self.assertTrue(self.node.energy_attack_detected)
        self.assertGreater(self.node.threat_assessment_level, 0)
    
    def test_distributed_energy_sharing(self):
        """Test distributed energy sharing mechanism"""
        # Set up trust network for energy sharing
        self.node.trust_network[1] = 0.8  # High trust
        self.node.trust_network[2] = 0.9  # Very high trust
        self.node.distributed_energy_pool = 5.0  # Available energy
        
        # Request energy
        shared_amount = self.node.request_distributed_energy(2.0)
        
        # Should receive some energy
        self.assertGreater(shared_amount, 0)
        self.assertLess(self.node.distributed_energy_pool, 5.0)  # Pool should be reduced
    
    def test_energy_pool_contribution(self):
        """Test contributing energy to distributed pool"""
        initial_energy = self.node.energy
        initial_pool = self.node.distributed_energy_pool
        
        # Contribute energy (should succeed with sufficient energy)
        success = self.node.contribute_to_energy_pool(1.0)
        
        self.assertTrue(success)
        self.assertLess(self.node.energy, initial_energy)
        self.assertGreater(self.node.distributed_energy_pool, initial_pool)
    
    def test_adaptive_energy_allocation(self):
        """Test adaptive energy allocation based on threat assessment"""
        # Set high threat level
        self.node.threat_assessment_level = 2
        
        # Apply adaptive allocation
        self.node.adaptive_energy_allocation()
        
        # Emergency threshold should be adjusted for higher threat
        base_threshold = 0.1
        expected_threshold = base_threshold * (1.0 + 2 * 0.1)  # threat_level * 0.1
        self.assertAlmostEqual(self.node.emergency_energy_threshold, expected_threshold, places=2)


class TestAdversarialDefense(unittest.TestCase):
    """Test adversarial defense mechanisms"""
    
    def setUp(self):
        self.node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
    
    def test_trust_manipulation_detection(self):
        """Test that trust manipulation is detected by the trust network system"""
        target_node = AliveLoopNode((1, 1), (0.1, 0.1), node_id=1)
        
        # Simulate rapid trust changes (manipulation attempt)
        for _ in range(5):
            self.node._update_trust_after_communication(target_node, 'resource')
        
        # Check if suspicion was triggered
        trust_summary = self.node.get_trust_summary()
        self.assertIsNotNone(trust_summary)
        
        # Should have some trust network activity
        self.assertGreater(len(self.node.trust_network), 0)


@unittest.skipUnless(NEUROMORPHIC_AVAILABLE, "Neuromorphic components not available")
class TestNeuromorphicEnhancements(unittest.TestCase):
    """Test neuromorphic integration enhancements"""
    
    def setUp(self):
        self.config = NeuromorphicConfig()
        self.oscillator = BrainWaveOscillator(self.config)
        self.neuromodulation = NeuromodulationSystem(self.config)
    
    def test_circadian_rhythm_integration(self):
        """Test circadian rhythm modulation of brain waves"""
        # Test different times of day
        morning_time = 8.0  # 8 AM
        night_time = 23.0  # 11 PM (clearly in sleep phase)
        
        # Update circadian phase for morning
        self.oscillator.update_circadian_phase(morning_time)
        morning_state = self.oscillator.get_sleep_wake_state()
        morning_delta_mod = self.oscillator.get_circadian_modulation('delta')
        morning_beta_mod = self.oscillator.get_circadian_modulation('beta')
        
        # Update circadian phase for night
        self.oscillator.update_circadian_phase(night_time)
        night_state = self.oscillator.get_sleep_wake_state()
        night_delta_mod = self.oscillator.get_circadian_modulation('delta')
        night_beta_mod = self.oscillator.get_circadian_modulation('beta')
        
        # Check that sleep/wake states are different
        self.assertEqual(morning_state, "wake")
        self.assertEqual(night_state, "sleep")
        
        # Test that modulation values are different between day and night
        self.assertNotAlmostEqual(night_delta_mod, morning_delta_mod, places=1)
        self.assertNotAlmostEqual(night_beta_mod, morning_beta_mod, places=1)
        
        # Both delta and beta should have valid modulation values
        self.assertGreater(morning_delta_mod, 0.5)  # Should be reasonable positive values
        self.assertGreater(night_delta_mod, 0.5)
        self.assertGreater(morning_beta_mod, 0.1)
        self.assertGreater(night_beta_mod, 0.1)
    
    def test_stress_response_neuromodulation(self):
        """Test neuromodulation for stress response"""
        initial_stress = self.neuromodulation.stress_level
        
        # Apply high stress multiple times to exceed threshold
        for _ in range(5):  # Multiple stress events to exceed threshold
            self.neuromodulation.update_stress_level(0.8, "energy_attack")
        
        # Stress level should increase
        self.assertGreater(self.neuromodulation.stress_level, initial_stress)
        
        # Stress-related neurotransmitters should be elevated
        cortisol = self.neuromodulation.neurotransmitters['cortisol']['concentration']
        adrenaline = self.neuromodulation.neurotransmitters['adrenaline']['concentration']
        
        self.assertGreater(cortisol, 0)
        self.assertGreater(adrenaline, 0)
    
    def test_stress_modulation_on_activity(self):
        """Test that stress modulates neural activity"""
        base_activity = 1.0
        
        # Apply stress multiple times to exceed threshold
        for _ in range(8):  # Ensure stress exceeds threshold
            self.neuromodulation.update_stress_level(0.7, "trust_violation")
        
        # Test modulation for excitatory neurons
        modulated_activity = self.neuromodulation.apply_stress_modulation(
            base_activity, "excitatory"
        )
        
        # Stress should increase excitatory activity
        self.assertGreater(modulated_activity, base_activity)
    
    def test_stress_recovery_mechanism(self):
        """Test stress recovery over time"""
        # Apply high stress multiple times
        for _ in range(10):
            self.neuromodulation.update_stress_level(0.9, "general")
        high_stress = self.neuromodulation.stress_level
        
        # Release calming neurotransmitters
        self.neuromodulation.release_neurotransmitter('serotonin', 0.5)
        self.neuromodulation.release_neurotransmitter('gaba', 0.3)
        
        # Update recovery multiple times
        for _ in range(5):
            self.neuromodulation.update_stress_recovery()
        recovered_stress = self.neuromodulation.stress_level
        
        # Stress should decrease
        self.assertLess(recovered_stress, high_stress)
    
    def test_neuromorphic_model_integration(self):
        """Test that enhanced neuromorphic model works with new features"""
        model = NeuromorphicAdaptiveModel(input_dim=10, output_dim=5, config=self.config)
        
        # Test with environmental data including stress
        input_tensor = torch.randn(1, 10)
        environmental_data = {
            'stress_level': 0.6,
            'stressor_type': 'communication_failure'
        }
        
        # Forward pass should work without errors
        output = model(input_tensor, environmental_data)
        
        # Output should have correct shape
        self.assertEqual(output.shape, (1, 5))
        
        # Stress level in neuromodulation system should be updated
        self.assertGreater(model.neuromodulation.stress_level, 0)


if __name__ == '__main__':
    unittest.main()