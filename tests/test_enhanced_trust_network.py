"""
Unit tests for enhanced trust network system

Tests trust updates, manipulation detection, and community verification
"""

import unittest

from core.alive_node import AliveLoopNode
from core.trust_network import TrustNetwork


class TestTrustNetwork(unittest.TestCase):
    """Test the TrustNetwork class functionality"""

    def setUp(self):
        self.trust_net = TrustNetwork(node_id=0)

        class MockTarget:
            def __init__(self, node_id):
                self.node_id = node_id

        self.target = MockTarget(1)

    def test_basic_trust_updates(self):
        """Test basic trust update functionality"""
        # Test positive signal
        initial_trust = self.trust_net.get_trust(self.target.node_id)
        self.assertEqual(initial_trust, 0.5)  # Default trust

        new_trust = self.trust_net.update_trust(self.target, 'resource')
        self.assertGreater(new_trust, initial_trust)
        self.assertLessEqual(new_trust, 1.0)

        # Test negative signal
        trust_before = new_trust
        new_trust = self.trust_net.update_trust(self.target, 'betrayal')
        self.assertLess(new_trust, trust_before)
        self.assertGreaterEqual(new_trust, 0.0)

    def test_trust_volatility_limits(self):
        """Test that trust changes are limited by volatility constraints"""
        # Try to make large changes
        for _ in range(10):
            initial = self.trust_net.get_trust(self.target.node_id)
            new_trust = self.trust_net.update_trust(self.target, 'resource')
            change = abs(new_trust - initial)
            self.assertLessEqual(change, self.trust_net.TRUST_VOLATILITY_LIMIT + 0.001)  # Small tolerance

    def test_manipulation_detection(self):
        """Test love bombing manipulation detection"""
        # Simulate love bombing pattern
        positive_signals = ['resource', 'joy_share', 'celebration_invite', 'comfort_request']

        for signal in positive_signals:
            self.trust_net.update_trust(self.target, signal)

        # Should detect manipulation pattern
        self.assertTrue(self.trust_net._detect_manipulation_pattern(self.target.node_id))

    def test_suspicion_threshold(self):
        """Test suspicion detection when trust drops below threshold"""
        # Set trust above threshold
        self.trust_net.set_trust(self.target.node_id, 0.4)

        # Apply negative signal to drop below threshold
        self.trust_net.update_trust(self.target, 'betrayal')

        # Should trigger suspicion
        self.assertIn(self.target.node_id, self.trust_net.suspicion_alerts)
        self.assertEqual(self.trust_net.suspicion_alerts[self.target.node_id]['status'],
                        'pending_verification')

    def test_community_feedback_processing(self):
        """Test community feedback processing"""
        # Set up a suspicious node
        self.trust_net.suspicion_alerts[self.target.node_id] = {
            'status': 'pending_verification',
            'trust_level': 0.2
        }

        # Simulate community feedback
        feedback = [
            {'trust_level': 0.1},
            {'trust_level': 0.2},
            {'trust_level': 0.15}
        ]

        initial_trust = self.trust_net.get_trust(self.target.node_id)
        self.trust_net.process_community_feedback(self.target.node_id, feedback)
        final_trust = self.trust_net.get_trust(self.target.node_id)

        # Trust should be adjusted based on community consensus
        self.assertNotEqual(initial_trust, final_trust)

    def test_trust_summary(self):
        """Test trust network summary functionality"""
        # Empty network
        summary = self.trust_net.get_trust_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['trusted_nodes'], 0)

        # Add some trust relationships
        self.trust_net.set_trust(1, 0.8)  # Trusted
        self.trust_net.set_trust(2, 0.2)  # Suspicious

        summary = self.trust_net.get_trust_summary()
        self.assertEqual(summary['trusted_nodes'], 1)
        self.assertEqual(summary['suspicious_nodes'], 1)


class TestAliveLoopNodeTrustIntegration(unittest.TestCase):
    """Test integration of enhanced trust system with AliveLoopNode"""

    def setUp(self):
        self.node1 = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
        self.node2 = AliveLoopNode((1, 1), (0.1, 0.1), node_id=1)

    def test_enhanced_trust_system_integration(self):
        """Test that AliveLoopNode uses enhanced trust system"""
        # Verify enhanced trust system is being used
        self.assertIsInstance(self.node1.trust_network_system, TrustNetwork)

        # Test trust update
        initial_trust = self.node1.trust_network.get(self.node2.node_id, 0.5)
        self.node1._update_trust_after_communication(self.node2, 'resource')
        new_trust = self.node1.trust_network.get(self.node2.node_id)

        self.assertGreater(new_trust, initial_trust)

    def test_backward_compatibility(self):
        """Test that old trust network interface still works"""
        # Test direct access to trust_network dict
        self.node1.trust_network[5] = 0.7
        self.assertEqual(self.node1.trust_network[5], 0.7)

        # Test that enhanced system can access it
        enhanced_trust = self.node1.trust_network_system.get_trust(5)
        self.assertEqual(enhanced_trust, 0.7)

    def test_trust_verification_request_handling(self):
        """Test community trust verification request processing"""
        verification_request = {
            'subject': self.node2.node_id,
            'requester': 5,
            'reason': 'test suspicion'
        }

        response = self.node1.process_trust_verification_request(verification_request)

        self.assertIsInstance(response, dict)
        self.assertIn('trust_level', response)
        self.assertIn('responder_id', response)
        self.assertEqual(response['responder_id'], self.node1.node_id)

    def test_trust_summary_access(self):
        """Test trust summary access through AliveLoopNode"""
        summary = self.node1.get_trust_summary()

        self.assertIsInstance(summary, dict)
        self.assertIn('average_trust', summary)
        self.assertIn('trusted_nodes', summary)
        self.assertIn('suspicious_nodes', summary)


if __name__ == '__main__':
    unittest.main()
