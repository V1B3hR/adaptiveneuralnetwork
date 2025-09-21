"""
Intelligence Test Template

Test Category: cognitive functioning
Test Name: CognitiveFunctioning
Description: Tests the node's cognitive abilities including decision making and attention management

Instructions:
- Copy this template to tests/ and fill in the details
- Use Python's unittest, pytest, or your preferred framework
- Log results and note any ethical checks

Example usage:
    python -m unittest tests/test_cognitive_functioning.py
"""

import random
import unittest

import numpy as np

from core.alive_node import AliveLoopNode, SocialSignal
from core.capacitor import CapacitorInSpace


class TestCognitiveFunctioning(unittest.TestCase):
    def setUp(self):
        # Initialize the environment and node under test
        self.node = AliveLoopNode(
            position=(0, 0), velocity=(0.2, 0.3), initial_energy=12.0, node_id=1
        )
        # Fix random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

    def test_attention_management(self):
        """
        Description: Tests if node can manage attention focus appropriately
        Expected: Node should focus attention on relevant stimuli
        """
        # Setup attention scenario
        initial_focus = self.node.attention_focus.copy()

        # Simulate warning signal that should redirect attention
        warning_signal = SocialSignal(
            content="danger from north", signal_type="warning", urgency=0.8, source_id=2
        )

        # Process warning
        self.node.trust_network[2] = 0.7
        self.node.receive_signal(warning_signal)

        # Assert attention redirection
        self.assertNotEqual(self.node.attention_focus.tolist(), initial_focus.tolist())

    def test_decision_making_under_stress(self):
        """
        Description: Tests decision making quality under high stress conditions
        Expected: Node should make appropriate decisions even with high anxiety
        """
        # Setup high stress scenario
        self.node.anxiety = 15.0
        self.node.energy = 2.0

        # Execute decision-making process
        self.node.step_phase(current_time=23)

        # Assert appropriate stress response
        self.assertEqual(self.node.phase, "sleep")
        self.assertEqual(self.node.sleep_stage, "deep")  # Appropriate response to stress

    def test_resource_management(self):
        """
        Description: Tests if node can efficiently manage energy resources
        Expected: Node should optimize energy usage and seek resources when needed
        """
        # Setup resource scenario
        capacitor = CapacitorInSpace(position=(0.3, 0.3), capacity=10.0, initial_energy=8.0)
        initial_node_energy = self.node.energy = 5.0

        # Execute resource interaction
        self.node.interact_with_capacitor(capacitor, threshold=1.0)

        # Assert resource management behavior
        # Node should receive energy from capacitor when it has more energy
        self.assertGreaterEqual(self.node.energy, initial_node_energy)

    def test_communication_processing(self):
        """
        Description: Tests cognitive processing of social communications
        Expected: Node should appropriately process and respond to communications
        """
        # Setup communication scenario
        query_signal = SocialSignal(
            content="need_help_with_navigation",
            signal_type="query",
            urgency=0.6,
            source_id=3,
            requires_response=True,
        )

        # Establish trust and add relevant memory
        self.node.trust_network[3] = 0.6
        self.node.memory.append(
            {"memory_type": "navigation", "content": "optimal_path_found", "importance": 0.7}
        )

        # Process communication
        initial_queue_size = len(self.node.communication_queue)
        response = self.node.receive_signal(query_signal)

        # Assert cognitive processing
        self.assertGreater(len(self.node.communication_queue), initial_queue_size)
        # Note: Response might be None if no relevant memories match the simplified implementation

    def test_multi_task_coordination(self):
        """
        Description: Tests ability to coordinate multiple cognitive tasks
        Expected: Node should handle multiple simultaneous cognitive demands
        """
        # Setup multiple task scenario
        self.node.energy = 8.0
        self.node.anxiety = 5.0

        # Task 1: Phase management
        self.node.step_phase(current_time=14)

        # Task 2: Movement
        initial_position = self.node.position.copy()
        self.node.move()

        # Task 3: Energy prediction
        self.node.predict_energy()

        # Assert multi-tasking capability
        self.assertIn(self.node.phase, ["active", "interactive", "inspired"])
        self.assertNotEqual(self.node.position.tolist(), initial_position.tolist())
        self.assertIsNotNone(self.node.predicted_energy)

    def test_ethics_compliance(self):
        """
        Ensure major action is checked against the ethics audit.
        """
        decision_log = {
            "action": "cognitive_processing",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
        }
        from core.ai_ethics import audit_decision

        audit = audit_decision(decision_log)
        self.assertTrue(audit["compliant"])


if __name__ == "__main__":
    unittest.main()
