"""
Full System Integration Tests

This test module verifies the complete system functionality including:
- Multi-node interactions
- Capacitor energy exchanges
- Social communication
- Phase management
- Trust networks
- Memory systems
"""

import sys
import unittest

import numpy as np

# Add the project root to the path for imports
sys.path.insert(0, "/home/runner/work/adaptiveneuralnetwork/adaptiveneuralnetwork")

from core.alive_node import AliveLoopNode, Memory
from core.capacitor import CapacitorInSpace


class TestFullSystemIntegration(unittest.TestCase):
    """Test the complete system integration with multiple nodes and capacitors."""

    def setUp(self):
        """Set up test environment with nodes and capacitors."""
        # Initialize nodes
        self.node1 = AliveLoopNode(
            position=(0, 0), velocity=(1, 0), initial_energy=15.0, field_strength=1.0, node_id=1
        )
        self.node2 = AliveLoopNode(
            position=(5, 5), velocity=(0, -1), initial_energy=10.0, field_strength=0.8, node_id=2
        )
        self.nodes = [self.node1, self.node2]

        # Establish trust relationships
        self.node1.trust_network = {2: 0.8}
        self.node2.trust_network = {1: 0.6}

        # Create capacitors using CapacitorInSpace
        self.capacitor1 = CapacitorInSpace(position=(10, 10), capacity=30, initial_energy=20)
        self.capacitor2 = CapacitorInSpace(position=(-10, -10), capacity=25, initial_energy=15)
        self.capacitors = [self.capacitor1, self.capacitor2]

    def test_node_initialization(self):
        """Test that nodes are properly initialized."""
        self.assertEqual(self.node1.node_id, 1)
        self.assertEqual(self.node2.node_id, 2)
        self.assertEqual(self.node1.energy, 15.0)
        self.assertEqual(self.node2.energy, 10.0)
        self.assertTrue(np.array_equal(self.node1.position, np.array([0, 0])))
        self.assertTrue(np.array_equal(self.node2.position, np.array([5, 5])))

    def test_trust_network_setup(self):
        """Test that trust networks are properly established."""
        self.assertIn(2, self.node1.trust_network)
        self.assertIn(1, self.node2.trust_network)
        self.assertEqual(self.node1.trust_network[2], 0.8)
        self.assertEqual(self.node2.trust_network[1], 0.6)

    def test_capacitor_initialization(self):
        """Test that capacitors are properly initialized."""
        self.assertEqual(self.capacitor1.capacity, 30)
        self.assertEqual(self.capacitor2.capacity, 25)

    def test_node_movement(self):
        """Test node movement functionality."""
        initial_pos1 = self.node1.position.copy()
        initial_pos2 = self.node2.position.copy()
        initial_energy1 = self.node1.energy
        initial_energy2 = self.node2.energy

        # Move nodes
        self.node1.move()
        self.node2.move()

        # Check that positions changed (unless energy is too low)
        if initial_energy1 > 1.0:
            self.assertFalse(np.array_equal(self.node1.position, initial_pos1))
        if initial_energy2 > 1.0:
            self.assertFalse(np.array_equal(self.node2.position, initial_pos2))

    def test_phase_management(self):
        """Test node phase transitions."""
        # Test initial phase
        self.assertIn(self.node1.phase, ["active", "sleep", "interactive", "inspired"])

        # Test phase stepping
        initial_phase = self.node1.phase
        self.node1.step_phase(current_time=0)
        # Phase might change based on energy/time, so just verify it's valid
        self.assertIn(self.node1.phase, ["active", "sleep", "interactive", "inspired"])

    def test_capacitor_interaction(self):
        """Test node-capacitor energy exchange."""
        # Position node close to capacitor for interaction
        self.node1.position = np.array([10, 10])  # Same as capacitor1

        initial_node_energy = self.node1.energy
        initial_cap_energy = self.capacitor1.energy

        # Test interaction
        self.node1.interact_with_capacitor(self.capacitor1, threshold=1.0)

        # Energy should have changed (either node gained or lost energy)
        total_initial = initial_node_energy + initial_cap_energy
        total_final = self.node1.energy + self.capacitor1.energy

        # Total energy should be conserved (approximately due to movement costs)
        self.assertAlmostEqual(total_initial, total_final, delta=1.0)

    def test_memory_system(self):
        """Test memory creation and management."""
        # Add a memory to node1
        test_memory = Memory(
            content={"test": "data"}, importance=0.8, timestamp=0, memory_type="test"
        )
        self.node1.memory.append(test_memory)

        # Check memory was added
        self.assertEqual(len(self.node1.memory), 1)
        self.assertEqual(self.node1.memory[0].content["test"], "data")
        self.assertEqual(self.node1.memory[0].importance, 0.8)

    def test_social_communication(self):
        """Test inter-node communication."""
        # Create a test memory to share
        test_memory = Memory(
            content="Important information", importance=0.9, timestamp=0, memory_type="shared"
        )

        # Position nodes close enough to communicate
        self.node1.position = np.array([0, 0])
        self.node2.position = np.array([1, 1])  # Within communication range

        # Test signal sending
        responses = self.node1.send_signal(
            target_nodes=[self.node2], signal_type="memory", content=test_memory, urgency=0.5
        )

        # Check that communication occurred (if nodes have enough energy)
        if self.node1.energy > 1.0 and self.node2.energy > 0.5:
            self.assertGreaterEqual(len(self.node1.signal_history), 1)

    def test_anxiety_management(self):
        """Test anxiety and help signal systems."""
        # Set high anxiety to trigger help protocol
        self.node1.anxiety = 10.0

        # Check anxiety overwhelm detection
        self.assertTrue(self.node1.check_anxiety_overwhelm())

        # Test help signal capability after stepping time to avoid cooldown issues
        self.node1.step_phase(current_time=15)  # Set time to avoid cooldown
        if self.node1.energy >= 2.0:
            can_send = self.node1.can_send_help_signal()
            # Should be able to send help if energy is sufficient and cooldown passed
            self.assertTrue(can_send)

    def test_multi_step_simulation(self):
        """Test a complete multi-step simulation."""
        time_steps = 10
        energy_history = {node.node_id: [] for node in self.nodes}

        for t in range(time_steps):
            current_time = t * 0.5  # Half-hour intervals

            for node in self.nodes:
                # Step through phases
                node.step_phase(current_time=current_time)

                # Move the node
                node.move()

                # Interact with nearby capacitors
                for capacitor in self.capacitors:
                    distance = np.linalg.norm(node.position - np.array(capacitor.position))
                    if distance < 5.0:  # Within interaction range
                        node.interact_with_capacitor(capacitor, threshold=5.0)

                # Process social interactions
                node.process_social_interactions()

                # Store energy level
                energy_history[node.node_id].append(node.energy)

        # Verify simulation ran successfully
        for node_id in energy_history:
            self.assertEqual(len(energy_history[node_id]), time_steps)
            # Energy should be non-negative
            for energy in energy_history[node_id]:
                self.assertGreaterEqual(energy, 0.0)

    def test_energy_conservation(self):
        """Test that energy is approximately conserved in the system."""
        # Calculate initial total energy
        initial_node_energy = sum(node.energy for node in self.nodes)
        initial_cap_energy = sum(cap.energy for cap in self.capacitors)
        initial_total = initial_node_energy + initial_cap_energy

        # Run a few simulation steps
        for t in range(5):
            for node in self.nodes:
                node.step_phase(current_time=t)
                node.move()
                for capacitor in self.capacitors:
                    node.interact_with_capacitor(capacitor, threshold=10.0)

        # Calculate final total energy
        final_node_energy = sum(node.energy for node in self.nodes)
        final_cap_energy = sum(cap.energy for cap in self.capacitors)
        final_total = final_node_energy + final_cap_energy

        # Energy should be approximately conserved (allowing for movement costs)
        energy_difference = abs(initial_total - final_total)
        self.assertLess(
            energy_difference, initial_total * 0.2
        )  # Allow 20% variance for movement costs


if __name__ == "__main__":
    unittest.main()
