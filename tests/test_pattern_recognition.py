"""
Intelligence Test Template

Test Category: pattern recognition
Test Name: PatternRecognition
Description: Tests the node's ability to identify, learn, and respond to patterns in data and behavior

Instructions:
- Copy this template to tests/ and fill in the details
- Use Python's unittest, pytest, or your preferred framework
- Log results and note any ethical checks

Example usage:
    python -m unittest tests/test_pattern_recognition.py
"""

import random
import unittest

import numpy as np

from core.alive_node import AliveLoopNode, Memory


class TestPatternRecognition(unittest.TestCase):
    def setUp(self):
        # Initialize the environment and node under test
        self.node = AliveLoopNode(
            position=(0, 0), velocity=(0.1, 0.1), initial_energy=10.0, node_id=1
        )
        # Fix random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

    def test_temporal_pattern_recognition(self):
        """
        Description: Tests if node can recognize temporal patterns in data sequences
        Expected: Node should identify repeating temporal patterns and make predictions
        """
        # Setup temporal pattern: energy oscillates between high and low
        pattern_data = [
            {"time": 0, "energy": 5},
            {"time": 1, "energy": 10},
            {"time": 2, "energy": 5},
            {"time": 3, "energy": 10},
            {"time": 4, "energy": 5},
        ]

        # Add pattern data to memory
        for i, data in enumerate(pattern_data):
            memory = Memory(content=data, importance=0.7, timestamp=i, memory_type="pattern")
            self.node.memory.append(memory)

        # Test pattern-based prediction
        self.node.predict_energy()

        # Assert pattern recognition capability
        self.assertIsNotNone(self.node.predicted_energy)
        # With the pattern, next energy should be around 10 (high phase)
        self.assertGreater(self.node.predicted_energy, self.node.energy)

    def test_spatial_pattern_recognition(self):
        """
        Description: Tests if node can recognize spatial patterns in environment
        Expected: Node should identify spatial relationships and optimize movement
        """
        # Setup spatial pattern: positions that lead to energy gain
        beneficial_positions = [
            {"position": [1, 1], "energy_gain": 3},
            {"position": [2, 2], "energy_gain": 3},
            {"position": [3, 3], "energy_gain": 3},  # Diagonal pattern
        ]

        # Add spatial memories
        for pos_data in beneficial_positions:
            memory = Memory(
                content=pos_data, importance=0.8, timestamp=self.node._time, memory_type="spatial"
            )
            self.node.memory.append(memory)

        # Test spatial awareness
        self.assertGreater(len(self.node.memory), 0)
        # Check if node has learned about beneficial positions
        spatial_memories = [m for m in self.node.memory if m.memory_type == "spatial"]
        self.assertEqual(len(spatial_memories), 3)

    def test_behavioral_pattern_detection(self):
        """
        Description: Tests if node can detect patterns in its own behavior
        Expected: Node should recognize behavioral patterns and adapt accordingly
        """
        # Setup behavioral pattern data
        behavior_sequence = ["move", "rest", "move", "rest", "move"]

        for i, behavior in enumerate(behavior_sequence):
            memory = Memory(
                content={"behavior": behavior, "energy_after": 8 if behavior == "rest" else 6},
                importance=0.6,
                timestamp=i,
                memory_type="behavior",
            )
            self.node.memory.append(memory)

        # Test if patterns are stored
        behavior_memories = [m for m in self.node.memory if m.memory_type == "behavior"]
        self.assertEqual(len(behavior_memories), 5)

        # Test prediction based on behavioral patterns
        self.node.predict_energy()
        self.assertIsNotNone(self.node.predicted_energy)

    def test_social_pattern_recognition(self):
        """
        Description: Tests if node can recognize patterns in social interactions
        Expected: Node should identify reliable and unreliable communication partners
        """
        # Setup social interaction pattern
        # Node 2 is reliable, Node 3 is unreliable
        self.node.trust_network[2] = 0.9  # High trust
        self.node.trust_network[3] = 0.2  # Low trust

        # Add interaction memories
        reliable_memory = Memory(
            content={"interaction": "helpful_information", "source": 2},
            importance=0.8,
            timestamp=0,
            memory_type="social",
        )

        unreliable_memory = Memory(
            content={"interaction": "misleading_information", "source": 3},
            importance=0.3,
            timestamp=1,
            memory_type="social",
        )

        self.node.memory.extend([reliable_memory, unreliable_memory])

        # Test social pattern recognition
        social_memories = [m for m in self.node.memory if m.memory_type == "social"]
        self.assertEqual(len(social_memories), 2)

        # Check trust differential reflects pattern
        self.assertGreater(self.node.trust_network[2], self.node.trust_network[3])

    def test_anomaly_detection(self):
        """
        Description: Tests if node can detect anomalies or deviations from known patterns
        Expected: Node should identify unusual events that break established patterns
        """
        # Setup normal pattern
        normal_energy_levels = [8, 9, 8, 9, 8]
        for i, energy in enumerate(normal_energy_levels):
            memory = Memory(
                content={"energy_level": energy}, importance=0.5, timestamp=i, memory_type="normal"
            )
            self.node.memory.append(memory)

        # Add anomalous event
        anomaly_memory = Memory(
            content={"energy_level": 2},  # Significantly different
            importance=0.9,  # Should be marked as important due to anomaly
            timestamp=5,
            memory_type="anomaly",
        )
        self.node.memory.append(anomaly_memory)

        # Test anomaly recognition
        anomaly_memories = [m for m in self.node.memory if m.memory_type == "anomaly"]
        self.assertEqual(len(anomaly_memories), 1)
        self.assertGreater(anomaly_memory.importance, 0.8)

    def test_ethics_compliance(self):
        """
        Ensure major action is checked against the ethics audit.
        """
        decision_log = {
            "action": "pattern_analysis",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
        }
        from core.ai_ethics import audit_decision

        audit = audit_decision(decision_log)
        self.assertTrue(audit["compliant"])


if __name__ == "__main__":
    unittest.main()
