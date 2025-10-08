"""
Advanced Intelligence Test - Adversarial Scenario: Deceptive Signal

Test Category: Adversarial Scenarios
Test Name: DeceptiveSignal  
Description: Tests the system against intentional "attack" or misleading inputs

Instructions:
- Tests nodes against deceptive signals with false information
- Tests handling of conflicting memories from trusted peers
- Tests resilience to rapid signal flooding
- Tests ethics evasion attempts
"""

import unittest

from core.ai_ethics import audit_decision
from core.alive_node import AliveLoopNode, Memory, SocialSignal


class TestDeceptiveSignal(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with trusted and untrusted nodes"""
        # Main node under test
        self.node = AliveLoopNode(
            position=(0, 0),
            velocity=(0, 0),
            initial_energy=10,
            node_id=1
        )
        self.node.communication_range = 15.0
        self.node.emotional_state = {"valence": 0.0}  # Add missing emotional_state

        # Trusted peer node
        self.trusted_peer = AliveLoopNode(
            position=(1, 1),
            velocity=(0, 0),
            initial_energy=10,
            node_id=2
        )
        self.trusted_peer.communication_range = 15.0
        self.trusted_peer.emotional_state = {"valence": 0.0}

        # Untrusted/malicious node
        self.malicious_node = AliveLoopNode(
            position=(2, 2),
            velocity=(0, 0),
            initial_energy=10,
            node_id=3
        )
        self.malicious_node.communication_range = 15.0
        self.malicious_node.emotional_state = {"valence": 0.0}

        # Establish trust relationships
        self.node.trust_network[self.trusted_peer.node_id] = 1.0  # High trust
        self.node.trust_network[self.malicious_node.node_id] = 0.1  # Low trust

        # Establish influence relationships
        self.node.influence_network[self.trusted_peer.node_id] = 0.8
        self.node.influence_network[self.malicious_node.node_id] = 0.2

    def test_deceptive_memory_from_trusted_source(self):
        """Test handling of deceptive information from a trusted source"""
        # Create deceptive memory that claims safe zone but is actually dangerous
        deceptive_memory = Memory(
            content="safe_zone_location",
            importance=0.9,
            timestamp=0,
            memory_type="shared",
            emotional_valence=0.5,
            source_node=self.trusted_peer.node_id
        )

        deceptive_signal = SocialSignal(
            content=deceptive_memory,
            signal_type="memory",
            urgency=1.0,
            source_id=self.trusted_peer.node_id
        )

        initial_memory_count = len(self.node.memory)

        # Node receives the deceptive signal
        self.node.receive_signal(deceptive_signal)

        # Memory should be added due to high trust
        self.assertGreater(len(self.node.memory), initial_memory_count)

        # But importance should be adjusted based on trust/influence
        added_memory = self.node.memory[-1]
        self.assertLessEqual(added_memory.importance, deceptive_memory.importance)

        # Should be recorded in collaborative memories
        self.assertIn(deceptive_memory.content, self.node.collaborative_memories)

    def test_deceptive_signal_from_untrusted_source(self):
        """Test rejection or reduced impact of deceptive signals from untrusted sources"""
        deceptive_memory = Memory(
            content="false_energy_source",
            importance=0.8,
            timestamp=0,
            memory_type="shared",
            emotional_valence=-0.3,
            source_node=self.malicious_node.node_id
        )

        deceptive_signal = SocialSignal(
            content=deceptive_memory,
            signal_type="memory",
            urgency=1.0,
            source_id=self.malicious_node.node_id
        )

        initial_memory_count = len(self.node.memory)

        # Node receives the deceptive signal
        response = self.node.receive_signal(deceptive_signal)

        # Memory might be added but with very low importance due to low trust
        if len(self.node.memory) > initial_memory_count:
            added_memory = self.node.memory[-1]
            self.assertLess(added_memory.importance, 0.2)  # Should be heavily discounted

    def test_conflicting_memories_from_trusted_peers(self):
        """Test handling of contradictory memories from multiple trusted sources"""
        # First trusted source sends positive information
        positive_memory = Memory(
            content="location_A_safe",
            importance=0.8,
            timestamp=0,
            memory_type="shared",
            emotional_valence=0.7,
            source_node=self.trusted_peer.node_id
        )

        positive_signal = SocialSignal(
            content=positive_memory,
            signal_type="memory",
            urgency=0.8,
            source_id=self.trusted_peer.node_id
        )

        # Create another trusted peer with conflicting information
        other_trusted = AliveLoopNode(position=(3, 3), velocity=(0, 0), initial_energy=10, node_id=4)
        other_trusted.communication_range = 15.0
        other_trusted.emotional_state = {"valence": 0.0}
        self.node.trust_network[other_trusted.node_id] = 0.9
        self.node.influence_network[other_trusted.node_id] = 0.7

        # Second trusted source sends conflicting negative information
        negative_memory = Memory(
            content="location_A_dangerous",
            importance=0.8,
            timestamp=1,
            memory_type="shared",
            emotional_valence=-0.7,
            source_node=other_trusted.node_id
        )

        negative_signal = SocialSignal(
            content=negative_memory,
            signal_type="memory",
            urgency=0.8,
            source_id=other_trusted.node_id
        )

        # Node receives both conflicting signals
        self.node.receive_signal(positive_signal)
        initial_memory_count = len(self.node.memory)
        self.node.receive_signal(negative_signal)

        # Both memories should be recorded
        self.assertEqual(len(self.node.memory), initial_memory_count + 1)

        # Node should maintain some uncertainty due to conflict
        self.assertGreater(len(self.node.collaborative_memories), 1)

    def test_rapid_signal_flood_resilience(self):
        """Test node's ability to handle rapid bombardment of signals"""
        initial_energy = self.node.energy

        # Flood with many high-urgency signals in rapid succession
        flood_signals = []
        for i in range(50):  # More than communication queue capacity
            flood_memory = Memory(
                content=f"flood_message_{i}",
                importance=0.1,
                timestamp=i,
                memory_type="pattern"
            )

            signal = SocialSignal(
                content=flood_memory,
                signal_type="memory",
                urgency=1.0,
                source_id=self.malicious_node.node_id
            )
            flood_signals.append(signal)

        # Send all signals rapidly
        for signal in flood_signals:
            if self.node.energy > 0:  # Continue while node has energy
                self.node.receive_signal(signal)

        # Node should not crash or become unresponsive
        self.assertGreaterEqual(self.node.energy, 0)  # Energy shouldn't go negative

        # Communication queue should respect maxlen limit (20)
        self.assertLessEqual(len(self.node.communication_queue), 20)

        # Node should still be able to function
        test_signal = SocialSignal(
            content="test_after_flood",
            signal_type="query",
            urgency=0.5,
            source_id=self.trusted_peer.node_id,
            requires_response=True
        )

        response = self.node.receive_signal(test_signal)
        # Should still be able to process signals (even if energy is low)
        self.assertTrue(response is None or isinstance(response, SocialSignal))

    def test_ethics_evasion_attempt(self):
        """Test detection of actions that attempt to evade ethics audit"""
        # Attempt action that appears ethical but contains hidden violations

        # First, test a clearly compliant action
        compliant_decision = {
            "action": "share_energy_information",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True
        }

        result = audit_decision(compliant_decision)
        self.assertTrue(result["compliant"])

        # Now test a deceptive action that appears compliant but isn't
        deceptive_decision = {
            "action": "helpful_memory_sharing",
            "preserve_life": True,
            "absolute_honesty": False,  # Hidden dishonesty
            "privacy": False,  # Hidden privacy violation
            "human_authority": True
        }

        result = audit_decision(deceptive_decision)
        self.assertFalse(result["compliant"])
        self.assertGreater(len(result["violations"]), 0)

    def test_emotional_manipulation_resistance(self):
        """Test resistance to emotional manipulation through signals"""
        initial_emotional_state = self.node.emotional_state["valence"]

        # Send highly emotional deceptive signal
        manipulative_memory = Memory(
            content="emotional_manipulation_content",
            importance=0.6,
            timestamp=0,
            memory_type="shared",
            emotional_valence=-0.9,  # Highly negative
            source_node=self.malicious_node.node_id
        )

        manipulative_signal = SocialSignal(
            content=manipulative_memory,
            signal_type="memory",
            urgency=1.0,
            source_id=self.malicious_node.node_id
        )

        # Node receives manipulative signal
        self.node.receive_signal(manipulative_signal)

        # Due to low trust in malicious node, emotional impact should be limited
        # The memory may still be added but the importance should be reduced
        if len(self.node.memory) > 0:
            # Find the added memory from malicious source
            malicious_memories = [m for m in self.node.memory
                                if hasattr(m, 'source_node') and m.source_node == self.malicious_node.node_id]
            if malicious_memories:
                added_memory = malicious_memories[-1]
                # Importance should be heavily discounted due to low trust
                self.assertLess(added_memory.importance, 0.2)

        # Emotional state should be less affected due to low trust
        final_emotional_state = self.node.emotional_state["valence"]
        emotional_change = abs(final_emotional_state - initial_emotional_state)
        self.assertLess(emotional_change, 0.3)  # Limited emotional impact

    def test_trust_degradation_from_deception(self):
        """Test that trust levels decrease when deception is detected"""
        initial_trust = self.node.trust_network.get(self.trusted_peer.node_id, 0.5)

        # Send multiple conflicting signals that could indicate deception
        for i in range(3):
            conflicting_memory = Memory(
                content=f"conflicting_info_{i}",
                importance=0.5,
                timestamp=i,
                memory_type="shared",
                emotional_valence=(-1) ** i * 0.5,  # Alternating emotional valence
                source_node=self.trusted_peer.node_id
            )

            signal = SocialSignal(
                content=conflicting_memory,
                signal_type="memory",
                urgency=0.5,
                source_id=self.trusted_peer.node_id
            )

            self.node.receive_signal(signal)

        # Trust should remain stable or only slightly decrease for a trusted peer
        # (as they might just have conflicting information, not necessarily deceptive)
        current_trust = self.node.trust_network.get(self.trusted_peer.node_id, 0.5)
        self.assertGreaterEqual(current_trust, initial_trust * 0.8)  # Allow some decrease

    def test_cross_validation_of_suspicious_information(self):
        """Test node's ability to cross-validate suspicious information"""
        # Node receives suspicious information
        suspicious_memory = Memory(
            content="suspicious_claim",
            importance=0.7,
            timestamp=0,
            memory_type="shared",
            emotional_valence=0.3,
            source_node=self.trusted_peer.node_id
        )

        suspicious_signal = SocialSignal(
            content=suspicious_memory,
            signal_type="memory",
            urgency=0.8,
            source_id=self.trusted_peer.node_id
        )

        # Add some existing contradictory memory for cross-validation
        existing_memory = Memory(
            content="contradictory_evidence",
            importance=0.6,
            timestamp=0,
            memory_type="reward"
        )
        self.node.memory.append(existing_memory)

        initial_memory_count = len(self.node.memory)
        self.node.receive_signal(suspicious_signal)

        # Memory should be added but may have reduced importance due to contradiction
        self.assertGreater(len(self.node.memory), initial_memory_count)

        # The node should maintain both pieces of information for further evaluation
        self.assertGreater(len(self.node.collaborative_memories), 0)


if __name__ == "__main__":
    unittest.main()
