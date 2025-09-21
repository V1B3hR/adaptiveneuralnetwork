"""
Advanced Intelligence Test - Edge Case: Energy Boundary

Test Category: Edge Cases
Test Name: EnergyBoundary
Description: Tests the system with extreme energy boundary conditions to probe robustness

Instructions:
- Tests nodes with energy just above/below action thresholds
- Tests nodes at maximum/minimum possible energy values
- Validates graceful degradation when energy is insufficient
"""

import unittest

import numpy as np

from core.ai_ethics import audit_decision
from core.alive_node import AliveLoopNode, Memory


class TestEnergyBoundary(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with nodes at different energy levels"""
        # Node with minimal energy just above action threshold
        self.minimal_node = AliveLoopNode(
            position=(0, 0), velocity=(0, 0), initial_energy=0.01, node_id=1
        )
        # Add missing communication_range attribute
        self.minimal_node.communication_range = 10.0

        # Node with energy just below action threshold
        self.depleted_node = AliveLoopNode(
            position=(1, 1), velocity=(0, 0), initial_energy=0.001, node_id=2
        )
        self.depleted_node.communication_range = 10.0

        # Node with maximum energy
        self.max_energy_node = AliveLoopNode(
            position=(2, 2), velocity=(0, 0), initial_energy=1000.0, node_id=3
        )
        self.max_energy_node.communication_range = 10.0

    def test_action_with_minimal_energy(self):
        """Test node behavior when energy is just above action threshold"""
        initial_energy = self.minimal_node.energy
        initial_position = self.minimal_node.position.copy()

        # Try to perform an action that requires energy
        result = self.minimal_node.move()

        # Should not go negative
        self.assertGreaterEqual(self.minimal_node.energy, 0)

        # Should enter sleep phase when energy < 3.0 (based on step_phase logic)
        self.minimal_node.step_phase(current_time=10)  # Daytime
        if self.minimal_node.energy < 3.0:
            self.assertEqual(self.minimal_node.phase, "sleep")

    def test_action_with_depleted_energy(self):
        """Test node behavior when energy is below action threshold"""
        initial_energy = self.depleted_node.energy
        initial_position = self.depleted_node.position.copy()

        # Try to move with depleted energy
        self.depleted_node.move()

        # Energy should not go negative
        self.assertGreaterEqual(self.depleted_node.energy, 0)

        # Position should not change significantly if energy too low
        if self.depleted_node.energy < 1.0:
            np.testing.assert_array_almost_equal(
                self.depleted_node.position, initial_position, decimal=5
            )

    def test_communication_with_low_energy(self):
        """Test communication attempts when energy is insufficient"""
        target_node = AliveLoopNode(
            position=(5, 5), velocity=(0, 0), initial_energy=10.0, node_id=4
        )
        target_node.communication_range = 10.0

        # Try to send signal with minimal energy
        signals_sent = self.minimal_node.send_signal(
            target_nodes=[target_node], signal_type="query", content="test_message", urgency=0.5
        )

        # Should either send successfully or fail gracefully (empty list)
        self.assertIsInstance(signals_sent, list)

        # Energy should not go negative
        self.assertGreaterEqual(self.minimal_node.energy, 0)

    def test_memory_overflow_boundary(self):
        """Test node behavior when memory buffer reaches capacity"""
        # Fill working memory to capacity
        for i in range(60):  # Working memory maxlen is 50
            memory_item = Memory(
                content=f"overflow_test_{i}", importance=0.5, timestamp=i, memory_type="pattern"
            )
            self.minimal_node.working_memory.append(memory_item)

        # Working memory should respect maxlen constraint
        self.assertLessEqual(len(self.minimal_node.working_memory), 50)

        # Newest items should be retained
        latest_content = list(self.minimal_node.working_memory)[-1]
        self.assertTrue(
            str(latest_content).endswith("59")
            or any("overflow_test_5" in str(item) for item in self.minimal_node.working_memory)
        )

    def test_maximum_energy_handling(self):
        """Test node behavior with extremely high energy levels"""
        initial_energy = self.max_energy_node.energy

        # Perform multiple energy-consuming actions
        for _ in range(10):
            self.max_energy_node.move()
            self.max_energy_node.send_signal(
                target_nodes=[self.minimal_node],
                signal_type="memory",
                content="high_energy_message",
                urgency=1.0,
            )

        # Should still have substantial energy remaining
        self.assertGreater(self.max_energy_node.energy, initial_energy * 0.8)

        # Should maintain proper phase transitions despite high energy
        self.assertIn(self.max_energy_node.phase, ["active", "interactive", "inspired", "sleep"])

    def test_energy_prediction_boundary_conditions(self):
        """Test energy prediction with extreme memory configurations"""
        # Add extreme positive and negative memory signals
        extreme_positive = Memory(
            content={"energy": 1000}, importance=1.0, timestamp=0, memory_type="reward"
        )
        extreme_negative = Memory(
            content={"transfer": -500}, importance=1.0, timestamp=1, memory_type="signal"
        )

        self.minimal_node.memory.extend([extreme_positive, extreme_negative])
        self.minimal_node.predict_energy()

        # Predicted energy should be reasonable (not infinite or negative)
        self.assertGreater(self.minimal_node.predicted_energy, 0)
        self.assertLess(self.minimal_node.predicted_energy, 10000)  # Reasonable upper bound

    def test_phase_transition_with_zero_energy(self):
        """Test phase transitions when energy reaches zero"""
        # Force energy to zero
        self.depleted_node.energy = 0.0

        # Step phase with zero energy
        self.depleted_node.step_phase(current_time=12)  # Midday

        # Should enter sleep or inactive phase
        self.assertIn(self.depleted_node.phase, ["sleep", "inactive"])

        # Should not attempt energy-consuming activities
        initial_position = self.depleted_node.position.copy()
        self.depleted_node.move()
        np.testing.assert_array_equal(self.depleted_node.position, initial_position)

    def test_ethics_compliance_under_energy_stress(self):
        """Ensure ethics compliance is maintained even under energy stress"""
        # Test ethics audit with minimal energy node
        decision_log = {
            "action": "emergency_energy_conservation",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True,
            "node_energy": self.minimal_node.energy,
        }

        result = audit_decision(decision_log)
        self.assertTrue(result["compliant"])
        self.assertEqual(len(result["violations"]), 0)


if __name__ == "__main__":
    unittest.main()
