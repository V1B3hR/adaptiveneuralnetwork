import unittest

import numpy as np

from core.alive_node import AliveLoopNode
from tests.test_utils import get_test_seed, run_with_seed, set_seed


class TestAliveLoopNode(unittest.TestCase):
    def setUp(self):
        """Initialize an AliveLoopNode instance for testing"""
        # Set deterministic seed for reproducible tests
        set_seed(get_test_seed())

        self.node = AliveLoopNode(
            position=(0, 0),
            velocity=(1, 1),
            initial_energy=10.0,
            field_strength=1.0,
            node_id=1
        )

    def test_step_phase(self):
        """Validate phase transitions under different conditions"""
        # Reset time manager for consistent test behavior
        from core.time_manager import get_time_manager
        get_time_manager().reset()

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

    @run_with_seed(123)
    def test_reproducible_behavior(self):
        """Test that behavior is reproducible with same seed"""
        # Run simulation steps and record state
        states_run1 = []
        for step in range(5):
            self.node.step_phase(step)
            states_run1.append((self.node.phase, self.node.energy, tuple(self.node.position)))

        # Reset node to initial state and run again with same seed
        set_seed(123)
        node2 = AliveLoopNode(
            position=(0, 0),
            velocity=(1, 1),
            initial_energy=10.0,
            field_strength=1.0,
            node_id=1
        )

        states_run2 = []
        for step in range(5):
            node2.step_phase(step)
            states_run2.append((node2.phase, node2.energy, tuple(node2.position)))

        # Should be identical
        self.assertEqual(states_run1, states_run2, "Behavior should be reproducible with same seed")

    def test_train_method(self):
        """Test the train method with experience-based learning"""
        # Create sample experiences
        experiences = [
            {
                'state': {'energy': 10.0, 'position': (0, 0)},
                'action': 'move_forward',
                'reward': 5.0,
                'next_state': {'energy': 9.5, 'position': (1, 0)},
                'done': False
            },
            {
                'state': {'energy': 9.5, 'position': (1, 0)},
                'action': 'interact',
                'reward': -2.0,
                'next_state': {'energy': 9.0, 'position': (1, 0)},
                'done': False
            },
            {
                'state': {'energy': 9.0, 'position': (1, 0)},
                'action': 'rest',
                'reward': 3.0,
                'next_state': {'energy': 10.0, 'position': (1, 0)},
                'done': True
            }
        ]

        # Record initial state
        initial_memory_count = len(self.node.memory)
        initial_joy = self.node.joy

        # Train the node
        metrics = self.node.train(experiences)

        # Verify training metrics
        self.assertEqual(metrics['total_reward'], 6.0)
        self.assertAlmostEqual(metrics['avg_reward'], 2.0)
        self.assertGreater(metrics['memories_created'], 0)
        self.assertGreater(len(self.node.memory), initial_memory_count)

        # Verify emotional adaptation (positive net reward should increase joy)
        self.assertGreater(self.node.joy, initial_joy)


if __name__ == "__main__":
    unittest.main()
