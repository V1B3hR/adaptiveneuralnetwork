import unittest
import numpy as np
from core.alive_node import AliveLoopNode


class TestAliveLoopNode(unittest.TestCase):
    def setUp(self):
        """Initialize an AliveLoopNode instance for testing"""
        self.node = AliveLoopNode(
            position=(0, 0),
            velocity=(1, 1),
            initial_energy=10.0,
            field_strength=1.0,
            node_id=1
        )

    def test_step_phase(self):
        """Validate phase transitions under different conditions"""
        self.node.energy = 1
        self.node.anxiety = 16
        self.node.step_phase(current_time=23)
        self.assertEqual(self.node.phase, "sleep")
        self.assertEqual(self.node.sleep_stage, "deep")  # Stress-induced deep sleep

        self.node.energy = 25
        self.node.anxiety = 3
        self.node.step_phase(current_time=10)
        self.assertEqual(self.node.phase, "inspired")  # High energy, low anxiety

    def test_move(self):
        """Validate position and energy updates during movement"""
        self.node.step_phase(current_time=10)  # Ensure phase is active
        initial_position = self.node.position.copy()
        self.node.move()
        self.assertNotEqual(self.node.position.tolist(), initial_position.tolist())
        self.assertLess(self.node.energy, 10.0)  # Energy decreases after moving

    def test_interact_with_capacitor(self):
        """Validate energy transfers with capacitors"""
        class MockCapacitor:
            def __init__(self, position, energy, capacity):
                self.position = np.array(position, dtype=float)
                self.energy = energy
                self.capacity = capacity

        capacitor = MockCapacitor(position=(1, 1), energy=5, capacity=20)
        self.node.interact_with_capacitor(capacitor, threshold=2)
        self.assertGreater(capacitor.energy, 5)  # Energy transferred to capacitor
        self.assertLess(self.node.energy, 10.0)  # Node energy decreases

    def test_energy_prediction(self):
        """Validate advanced energy prediction logic"""
        self.node.memory.append({"memory_type": "reward", "content": {"transfer": 5}})
        self.node.memory.append({"memory_type": "signal", "content": {"energy": 2}})
        self.node.predict_energy()
        self.assertGreater(self.node.predicted_energy, 10.0)  # Predicted energy increases

    def test_anxiety_reduction(self):
        """Validate anxiety reduction during sleep"""
        self.node.phase = "sleep"
        self.node.sleep_stage = "deep"
        self.node.anxiety = 10
        self.node.clear_anxiety()
        self.assertLess(self.node.anxiety, 10)  # Anxiety reduces after deep sleep


if __name__ == "__main__":
    unittest.main()
