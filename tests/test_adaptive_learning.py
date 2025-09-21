"""
Intelligence Test Template

Test Category: learning ability
Test Name: AdaptiveLearning
Description: Tests the node's capacity for learning from experience and adapting behavior

Instructions:
- Copy this template to tests/ and fill in the details
- Use Python's unittest, pytest, or your preferred framework
- Log results and note any ethical checks

Example usage:
    python -m unittest tests/test_adaptive_learning.py
"""

import random
import unittest

import numpy as np

from core.alive_node import AliveLoopNode, Memory


class TestAdaptiveLearning(unittest.TestCase):
    def setUp(self):
        # Initialize the environment and node under test
        self.node = AliveLoopNode(
            position=(0, 0), velocity=(0.5, 0.5), initial_energy=15.0, node_id=1
        )
        # Fix random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

    def test_memory_consolidation(self):
        """
        Description: Tests if node can consolidate memories based on importance
        Expected: Important memories should be retained longer than trivial ones
        """
        # Setup memories with different importance levels
        important_memory = Memory(
            content="critical_energy_source",
            importance=0.9,
            timestamp=self.node._time,
            memory_type="reward",
        )
        trivial_memory = Memory(
            content="random_noise", importance=0.1, timestamp=self.node._time, memory_type="pattern"
        )

        self.node.memory.extend([important_memory, trivial_memory])

        # Simulate memory aging
        for memory in self.node.memory:
            memory.age()

        # Assert importance-based retention
        self.assertGreater(important_memory.importance, trivial_memory.importance)

    def test_social_learning(self):
        """
        Description: Tests if node can learn from shared memories of other nodes
        Expected: Node should integrate valuable shared knowledge
        """
        # Create a source node with valuable memory
        source_node = AliveLoopNode(
            position=(1, 1), velocity=(0, 0), initial_energy=10.0, node_id=2
        )

        # Establish trust relationship
        self.node.trust_network[2] = 0.8
        self.node.influence_network[2] = 0.7

        # Create valuable shared memory
        shared_memory = Memory(
            content="energy_efficient_path",
            importance=0.8,
            timestamp=0,
            memory_type="shared",
            source_node=2,
        )

        initial_memory_count = len(self.node.memory)

        # Simulate memory sharing (simplified)
        self.node.memory.append(shared_memory)
        self.node.collaborative_memories[shared_memory.content] = shared_memory

        # Assert learning occurred
        self.assertGreater(len(self.node.memory), initial_memory_count)
        self.assertIn(shared_memory.content, self.node.collaborative_memories)

    def test_behavioral_adaptation(self):
        """
        Description: Tests if node adapts behavior based on past experiences
        Expected: Node should modify behavior patterns based on success/failure
        """
        # Setup initial anxiety level
        initial_anxiety = self.node.anxiety = 8.0

        # Simulate stress reduction through sleep
        self.node.phase = "sleep"
        self.node.sleep_stage = "deep"
        self.node.clear_anxiety()

        # Assert adaptive behavior
        self.assertLess(self.node.anxiety, initial_anxiety)

    def test_pattern_recognition(self):
        """
        Description: Tests if node can recognize patterns in environmental data
        Expected: Node should identify recurring patterns and adjust predictions
        """
        # Setup pattern in energy changes
        energy_pattern = [{"energy": 2}, {"energy": 4}, {"energy": 2}, {"energy": 4}]

        for i, pattern_data in enumerate(energy_pattern):
            memory = Memory(
                content=pattern_data, importance=0.6, timestamp=i, memory_type="pattern"
            )
            self.node.memory.append(memory)

        # Execute prediction based on patterns
        self.node.predict_energy()

        # Assert pattern-based prediction
        self.assertIsNotNone(self.node.predicted_energy)
        self.assertGreater(self.node.predicted_energy, 0)

    def test_ethics_compliance(self):
        """
        Ensure major action is checked against the ethics audit.
        """
        decision_log = {
            "action": "memory_sharing",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
        }
        from core.ai_ethics import audit_decision

        audit = audit_decision(decision_log)
        self.assertTrue(audit["compliant"])


if __name__ == "__main__":
    unittest.main()
