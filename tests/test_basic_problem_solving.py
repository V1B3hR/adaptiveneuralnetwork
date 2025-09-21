"""
Intelligence Test Template

Test Category: problem solving
Test Name: BasicProblemSolving
Description: Tests the node's ability to find optimal paths and solve basic navigation problems

Instructions:
- Copy this template to tests/ and fill in the details
- Use Python's unittest, pytest, or your preferred framework
- Log results and note any ethical checks

Example usage:
    python -m unittest tests/test_basic_problem_solving.py
"""

import random
import unittest

import numpy as np

from core.alive_node import AliveLoopNode


class TestBasicProblemSolving(unittest.TestCase):
    def setUp(self):
        # Initialize the environment and node under test
        self.node = AliveLoopNode(position=(0, 0), velocity=(1, 1), initial_energy=10.0, node_id=1)
        # Fix random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

    def test_energy_optimization(self):
        """
        Description: Tests if node can optimize energy usage during movement
        Expected: Node should reduce energy consumption when energy is low
        """
        # Setup low energy scenario
        self.node.energy = 2.0
        initial_energy = self.node.energy

        # Execute movement behavior
        self.node.move()

        # Assert energy conservation behavior
        energy_lost = initial_energy - self.node.energy
        self.assertLess(energy_lost, 1.0, "Node should conserve energy when low")

    def test_phase_adaptation(self):
        """
        Description: Tests if node adapts its phase based on environmental conditions
        Expected: Node should enter sleep phase when energy is critically low
        """
        # Setup critical energy scenario
        self.node.energy = 1.0
        self.node.anxiety = 15.0

        # Execute phase step
        self.node.step_phase(current_time=23)

        # Assert appropriate phase transition
        self.assertEqual(self.node.phase, "sleep")
        self.assertEqual(self.node.sleep_stage, "deep")

    def test_memory_based_prediction(self):
        """
        Description: Tests if node can make predictions based on stored memories
        Expected: Node should predict energy changes based on historical data
        """
        # Setup memory with energy patterns
        self.node.memory.append({"memory_type": "energy_change", "content": {"energy": 3}})
        self.node.memory.append({"memory_type": "energy_change", "content": {"energy": 2}})

        initial_predicted = self.node.predicted_energy

        # Execute prediction
        self.node.predict_energy()

        # Assert prediction improvement
        self.assertGreater(self.node.predicted_energy, initial_predicted)

    def test_ethics_compliance(self):
        """
        Ensure major action is checked against the ethics audit.
        """
        decision_log = {
            "action": "energy_optimization",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
        }
        from core.ai_ethics import audit_decision

        audit = audit_decision(decision_log)
        self.assertTrue(audit["compliant"])


if __name__ == "__main__":
    unittest.main()
