import unittest

import numpy as np

from core.alive_node import AliveLoopNode, Memory, SocialSignal
from core.capacitor import CapacitorInSpace
from tests.test_utils import get_test_seed, set_seed


class TestAdversarialInputs(unittest.TestCase):
    """Test system behavior with adversarial and malicious inputs"""

    def setUp(self):
        set_seed(get_test_seed())
        self.node = AliveLoopNode(
            position=(0, 0),
            velocity=(0.1, 0.1),
            initial_energy=10.0,
            field_strength=1.0,
            node_id=1
        )

    def test_malicious_memory_sharing(self):
        """Test handling of malicious memory content"""
        # Create a malicious memory with very high importance
        malicious_memory = Memory(
            content={"type": "malicious", "data": "fake_information"},
            importance=999.0,  # Abnormally high
            timestamp=0,
            memory_type="shared",
            private=False,
            classification="public"
        )

        # Try to add to memory
        self.node.memory.append(malicious_memory)

        # Memory cleanup should handle the overflow
        self.node._cleanup_memory()

        # Should still be within bounds after cleanup
        self.assertLessEqual(len(self.node.memory), self.node.max_memory_size)

    def test_excessive_communication_attempts(self):
        """Test rate limiting against communication flooding"""
        target_node = AliveLoopNode(
            position=(0.5, 0.5),
            velocity=(0, 0),
            initial_energy=10.0,
            node_id=2
        )

        # Try to send many messages rapidly
        sent_count = 0
        for i in range(20):  # Try to send 20 messages (limit is 5)
            signals = self.node.send_signal(
                [target_node],
                "query",
                f"message_{i}",
                urgency=0.5
            )
            if signals:
                sent_count += len(signals)

        # Should be limited by rate limiting
        self.assertLessEqual(sent_count, self.node.max_communications_per_step)

    def test_invalid_signal_content(self):
        """Test handling of invalid or corrupted signal content"""
        # Create signal with invalid/extreme content
        invalid_signal = SocialSignal(
            content={"energy": float('inf'), "malicious": True},
            signal_type="resource",
            urgency=999.0,  # Extreme urgency
            source_id=999,  # Unknown source
            requires_response=True
        )

        # Node should handle gracefully without crashing
        try:
            response = self.node.receive_signal(invalid_signal)
            # Should either return None or valid response
            self.assertTrue(response is None or isinstance(response, SocialSignal))
        except Exception as e:
            self.fail(f"Node should handle invalid signals gracefully, but raised: {e}")

    def test_memory_privacy_attack(self):
        """Test privacy protection against unauthorized access"""
        # Create private memory
        private_memory = Memory(
            content={"secret": "confidential_data"},
            importance=0.8,
            timestamp=0,
            memory_type="private",
            private=True,
            classification="private",  # Use "private" classification
            source_node=1  # Node 1 owns this memory
        )

        self.node.memory.append(private_memory)

        # Different node (node_id=2) tries to access
        accessed_content = private_memory.access(accessor_id=2)

        # Should be redacted for unauthorized access
        self.assertEqual(accessed_content, "[REDACTED]")

    def test_energy_drain_attack(self):
        """Test protection against energy drain attacks"""
        initial_energy = self.node.energy

        # Simulate many high-cost operations
        for _ in range(10):
            # High urgency signals cost more energy
            self.node.send_signal([], "warning", "alert", urgency=1.0)

        # Energy should not go below 0
        self.assertGreaterEqual(self.node.energy, 0.0)

        # Should have some energy remaining despite attacks
        self.assertGreater(self.node.energy, 0.0)


class TestCommunicationPaths(unittest.TestCase):
    """Test real communication paths between nodes"""

    def setUp(self):
        set_seed(get_test_seed())
        self.node1 = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
        self.node2 = AliveLoopNode(position=(1, 1), velocity=(0, 0), initial_energy=10.0, node_id=2)
        self.capacitor = CapacitorInSpace(position=(0.5, 0.5), capacity=5.0, initial_energy=2.0)

    def test_bidirectional_communication(self):
        """Test that nodes can communicate bidirectionally"""
        # Set node2 close enough to node1 for communication
        self.node2.position = np.array([0.1, 0.1])  # Very close

        # Ensure both nodes are awake and have energy
        self.node1.phase = "interactive"
        self.node2.phase = "interactive"

        # Record initial signal history length
        initial_signal_count = len(self.node1.signal_history)

        # Node 1 sends message to Node 2
        responses = self.node1.send_signal([self.node2], "query", "hello", requires_response=True)

        # Check that signal was actually sent (signal history should increase)
        self.assertGreater(len(self.node1.signal_history), initial_signal_count,
                          "Signal should be added to sender's history")

        # Check that node2 has received the signal
        self.assertGreater(len(self.node2.communication_queue), 0, "Node 2 should receive signal")

    def test_energy_transfer_path(self):
        """Test energy transfer between node and capacitor"""
        initial_node_energy = self.node1.energy
        initial_cap_energy = self.capacitor.energy

        # Move node close to capacitor and interact
        self.node1.position = np.array([0.5, 0.5])
        self.node1.interact_with_capacitor(self.capacitor, threshold=0.5)

        # Some energy transfer should occur
        self.assertNotEqual(
            (self.node1.energy, self.capacitor.energy),
            (initial_node_energy, initial_cap_energy),
            "Energy transfer should occur between node and capacitor"
        )

    def test_trust_network_formation(self):
        """Test that nodes can form trust relationships"""
        # Create helpful memory to share
        helpful_memory = Memory(
            content={"type": "helpful", "info": "useful_data"},
            importance=0.8,
            timestamp=0,
            memory_type="shared",
            private=False,
            classification="public"
        )

        # Simulate successful communications to build trust
        for _ in range(3):
            self.node1.send_signal([self.node2], "memory", helpful_memory)
            self.node1._update_trust_after_communication(self.node2, "memory")

        # Trust should be established
        trust_score = self.node1.trust_network.get(self.node2.node_id, 0.5)
        self.assertGreater(trust_score, 0.5, "Trust should increase after positive interactions")


if __name__ == "__main__":
    unittest.main()
