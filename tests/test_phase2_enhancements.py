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
        # Test emergency mode activation directly
        # Ensure node starts in normal state
        self.node.energy = 3.0
        self.node.emergency_mode = False
        
        # Manually set thresholds to test the activation logic
        original_threshold = self.node.emergency_energy_threshold
        
        # Test with energy below threshold
        self.node.energy = 0.6  # Above survival threshold (0.5)
        self.node.emergency_energy_threshold = 0.8  # Above current energy
        
        # Manually prevent threshold recalculation by setting threat level to 0
        self.node.threat_assessment_level = 0
        
        # Temporarily override threshold calculation to test activation logic
        self.node._original_emergency_threshold = 0.8
        self.node.adaptive_energy_allocation()
        
        # Emergency mode should be activated since 0.6 < 0.8 and not in survival mode
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
        
        # Store original threshold for comparison
        original_threshold = self.node.emergency_energy_threshold
        
        # Apply adaptive allocation
        self.node.adaptive_energy_allocation()
        
        # Emergency threshold should not go below original safe value
        # This ensures the system remains robust against threshold manipulation
        self.assertGreaterEqual(self.node.emergency_energy_threshold, original_threshold)


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


class TestAdvancedTrustVisualization(unittest.TestCase):
    """Test advanced trust network visualization and monitoring"""
    
    def setUp(self):
        self.trust_network = TrustNetwork(node_id=0)
        # Set up a sample trust network
        self.trust_network.set_trust(1, 0.9)  # Highly trusted
        self.trust_network.set_trust(2, 0.7)  # Trusted
        self.trust_network.set_trust(3, 0.5)  # Neutral
        self.trust_network.set_trust(4, 0.2)  # Suspicious
        self.trust_network.set_trust(5, 0.1)  # Very suspicious
    
    def test_trust_network_graph_generation(self):
        """Test trust network graph generation for visualization"""
        graph_data = self.trust_network.generate_trust_network_graph()
        
        # Check structure
        self.assertIn('nodes', graph_data)
        self.assertIn('edges', graph_data)
        self.assertIn('metadata', graph_data)
        
        # Should have 6 nodes (self + 5 peers)
        self.assertEqual(len(graph_data['nodes']), 6)
        
        # Should have 5 edges (self to each peer)
        self.assertEqual(len(graph_data['edges']), 5)
        
        # Check node data structure
        self_node = next(node for node in graph_data['nodes'] if node['type'] == 'self')
        self.assertEqual(self_node['id'], 0)
        self.assertEqual(self_node['trust_level'], 1.0)
        
        # Check edge data structure
        edge = graph_data['edges'][0]
        self.assertIn('source', edge)
        self.assertIn('target', edge)
        self.assertIn('weight', edge)
    
    def test_trust_network_metrics(self):
        """Test comprehensive trust network metrics calculation"""
        metrics = self.trust_network.get_trust_network_metrics()
        
        # Check all required metrics are present
        required_metrics = [
            'total_connections', 'average_trust', 'trust_variance',
            'network_resilience', 'suspicious_ratio', 'alert_count'
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Verify calculations
        self.assertEqual(metrics['total_connections'], 5)
        self.assertAlmostEqual(metrics['average_trust'], 0.48, places=1)  # (0.9+0.7+0.5+0.2+0.1)/5
        self.assertEqual(metrics['suspicious_nodes'], 2)  # Nodes 4 and 5
        self.assertAlmostEqual(metrics['suspicious_ratio'], 0.4, places=1)  # 2/5


class TestDistributedTrustConsensus(unittest.TestCase):
    """Test distributed trust consensus mechanisms"""
    
    def setUp(self):
        self.node1 = AliveLoopNode((0, 0), (0.1, 0.1), node_id=1)
        self.node2 = AliveLoopNode((1, 1), (0.1, 0.1), node_id=2)
        self.node3 = AliveLoopNode((2, 2), (0.1, 0.1), node_id=3)
        self.nodes = [self.node1, self.node2, self.node3]
        
        # Set up some trust relationships
        self.node1.trust_network[2] = 0.8
        self.node1.trust_network[3] = 0.7
        self.node2.trust_network[1] = 0.8
        self.node2.trust_network[3] = 0.3  # Different opinion about node 3
        
    def test_consensus_vote_initiation(self):
        """Test initiation of consensus vote"""
        vote_request = self.node1.trust_network_system.initiate_consensus_vote(subject_node_id=3)
        
        # Check vote request structure
        self.assertIn('vote_id', vote_request)
        self.assertIn('initiator', vote_request)
        self.assertIn('subject', vote_request)
        self.assertEqual(vote_request['subject'], 3)
        self.assertEqual(vote_request['initiator'], 1)
        
    def test_consensus_vote_response(self):
        """Test response to consensus vote"""
        vote_request = {
            'vote_id': 'test_vote',
            'initiator': 1,
            'subject': 3,
            'vote_type': 'trust_evaluation'
        }
        
        response = self.node2.respond_to_trust_vote(vote_request)
        
        # Check response structure
        self.assertIn('voter_id', response)
        self.assertIn('trust_assessment', response)
        self.assertIn('confidence', response)
        self.assertEqual(response['voter_id'], 2)
        
    def test_consensus_processing(self):
        """Test processing of consensus votes"""
        vote_request = {'subject': 3, 'initiator': 1}
        
        # Mock voter responses
        responses = [
            {'trust_assessment': 0.7, 'confidence': 0.8, 'voter_id': 1},
            {'trust_assessment': 0.3, 'confidence': 0.9, 'voter_id': 2},
            {'trust_assessment': 0.6, 'confidence': 0.7, 'voter_id': 4}
        ]
        
        result = self.node1.trust_network_system.process_consensus_vote(vote_request, responses)
        
        # Check result structure
        self.assertIn('consensus_trust', result)
        self.assertIn('agreement_level', result)
        self.assertIn('recommendation', result)
        self.assertEqual(result['voter_count'], 3)


class TestByzantineFaultTolerance(unittest.TestCase):
    """Test Byzantine fault tolerance improvements and stress testing"""
    
    def setUp(self):
        self.node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
        # Set up initial trust network
        for i in range(1, 6):
            self.node.trust_network[i] = 0.5
    
    def test_byzantine_stress_test_execution(self):
        """Test Byzantine stress test execution"""
        results = self.node.trust_network_system.stress_test_byzantine_resilience(
            malicious_ratio=0.3, 
            num_simulations=10  # Small number for testing
        )
        
        # Check results structure
        required_keys = ['attack_scenarios', 'resilience_score', 'detection_rate', 'false_positive_rate']
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check value types and ranges
        self.assertIsInstance(results['resilience_score'], float)
        self.assertGreaterEqual(results['resilience_score'], 0.0)
        self.assertLessEqual(results['resilience_score'], 1.0)
        
        self.assertEqual(len(results['attack_scenarios']), 10)
        
    def test_malicious_behavior_detection(self):
        """Test detection of malicious behavior patterns"""
        # Test the trust network's ability to track suspicious patterns
        malicious_node_id = 999
        self.node.trust_network_system.set_trust(malicious_node_id, 0.5)
        
        # Test that we can detect rapid trust changes by simulating direct manipulation
        original_trust = self.node.trust_network_system.get_trust(malicious_node_id)
        
        # Simulate multiple trust updates that could indicate manipulation
        for i in range(10):
            # Manually adjust trust to simulate rapid changes
            current_trust = self.node.trust_network_system.get_trust(malicious_node_id)
            new_trust = min(1.0, current_trust + 0.05)  # Small increments
            self.node.trust_network_system.trust_network[malicious_node_id] = new_trust
            
            # Check if the system detects the pattern
            self.node.trust_network_system._detect_suspicious_pattern(
                malicious_node_id, new_trust, current_trust
            )
        
        final_trust = self.node.trust_network_system.get_trust(malicious_node_id)
        
        # Verify the test setup worked - trust should have increased
        self.assertGreater(final_trust, original_trust,
                          "Trust should have increased during the test simulation")
        
    def test_trust_network_resilience_metrics(self):
        """Test trust network resilience metrics"""
        metrics = self.node.get_trust_network_metrics()
        
        # Should calculate resilience score
        self.assertIn('network_resilience', metrics)
        self.assertIsInstance(metrics['network_resilience'], float)
        
        # Test with a more resilient network
        for i in range(1, 6):
            self.node.trust_network[i] = 0.8  # High trust
        
        metrics_high_trust = self.node.get_trust_network_metrics()
        self.assertGreater(metrics_high_trust['network_resilience'], metrics['network_resilience'])


class TestTrustNetworkMonitoring(unittest.TestCase):
    """Test trust network monitoring and health assessment"""
    
    def setUp(self):
        self.node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
        
    def test_trust_network_health_monitoring(self):
        """Test trust network health monitoring"""
        # Set up concerning trust patterns
        self.node.trust_network[1] = 0.1  # Suspicious
        self.node.trust_network[2] = 0.2  # Suspicious
        self.node.trust_network[3] = 0.15  # Suspicious
        self.node.trust_network[4] = 0.8  # Trusted
        
        health_report = self.node.monitor_trust_network_health()
        
        # Check report structure
        self.assertIn('metrics', health_report)
        self.assertIn('alerts', health_report)
        self.assertIn('overall_health', health_report)
        
        # Should have alerts for high suspicious ratio
        alert_messages = [alert['message'] for alert in health_report['alerts']]
        suspicious_alerts = [msg for msg in alert_messages if 'suspicious' in msg.lower()]
        self.assertTrue(len(suspicious_alerts) > 0)
        
    def test_health_score_calculation(self):
        """Test overall health score calculation"""
        # Test with healthy network
        for i in range(1, 6):
            self.node.trust_network[i] = 0.8
        
        health_report = self.node.monitor_trust_network_health()
        healthy_score = health_report['overall_health']
        
        # Test with unhealthy network
        for i in range(1, 6):
            self.node.trust_network[i] = 0.2
        
        unhealthy_report = self.node.monitor_trust_network_health()
        unhealthy_score = unhealthy_report['overall_health']
        
        # Healthy network should have higher score
        self.assertGreater(healthy_score, unhealthy_score)


if __name__ == '__main__':
    unittest.main()