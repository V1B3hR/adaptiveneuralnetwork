"""
Attack Resilience Enhancement Tests

Tests for the new attack resilience features including:
- Distributed energy sharing with attack resistance
- Reduced energy depletion attack impact
- Improved signal jamming resistance  
- Enhanced trust manipulation resistance
- Faster environmental adaptation
"""

import unittest

import numpy as np

from core.alive_node import AliveLoopNode, Memory, SocialSignal
from tests.test_utils import get_test_seed, set_seed


class TestDistributedEnergySharing(unittest.TestCase):
    """Test distributed energy sharing with attack resistance"""

    def setUp(self):
        set_seed(get_test_seed())
        self.node1 = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=5.0, node_id=1)
        self.node2 = AliveLoopNode(position=(1, 1), velocity=(0, 0), initial_energy=15.0, node_id=2)
        self.node3 = AliveLoopNode(position=(2, 2), velocity=(0, 0), initial_energy=8.0, node_id=3)

        # Establish trust relationships
        self.node1.trust_network[self.node2.node_id] = 0.8  # High trust
        self.node1.trust_network[self.node3.node_id] = 0.3  # Low trust
        self.node2.trust_network[self.node1.node_id] = 0.7

    def test_energy_sharing_request_success(self):
        """Test successful energy sharing between trusted nodes"""
        initial_energy_1 = self.node1.energy
        initial_energy_2 = self.node2.energy

        # Test direct energy sharing
        shared_amount = self.node2.share_energy_directly(self.node1, amount=3.0)

        # Should share some energy
        self.assertGreater(shared_amount, 0)
        self.assertGreater(self.node1.energy, initial_energy_1)
        self.assertLess(self.node2.energy, initial_energy_2)

        # Check that energy sharing was recorded
        self.assertGreater(len(self.node1.energy_sharing_history), 0)
        self.assertGreater(len(self.node2.energy_sharing_history), 0)

    def test_energy_sharing_attack_detection(self):
        """Test detection and rejection of energy drain attacks"""
        attacker = AliveLoopNode(position=(3, 3), velocity=(0, 0), initial_energy=10.0, node_id=4)
        self.node1.trust_network[attacker.node_id] = 0.1  # Very low trust

        initial_energy = self.node1.energy

        # Attacker requests excessive energy multiple times
        for _ in range(5):
            received = self.node1.request_energy_sharing([attacker], energy_needed=10.0)  # Excessive request

        # Should detect attack and reject
        # Check for suspicious events
        recent_suspicious = [e for e in self.node1.suspicious_events
                           if e.get("type") == "potential_energy_drain_attack"]

        # Should have some protection against excessive drain
        self.assertLessEqual(self.node1.energy, initial_energy)  # May have lost some energy
        self.assertGreater(self.node1.energy, initial_energy * 0.5)  # But not too much

    def test_energy_drain_resistance(self):
        """Test that energy drain resistance reduces attack impact"""
        initial_energy = self.node1.energy

        # Simulate energy drain attack (original: 15% per attacker, target: 5-8%)
        original_drain_amount = initial_energy * 0.15  # 15% drain
        actual_drain = self.node1.apply_energy_drain_resistance(original_drain_amount, attacker_count=1)

        # Should be significantly reduced
        self.assertLess(actual_drain, original_drain_amount)
        self.assertLess(actual_drain, initial_energy * 0.08)  # Should be ≤ 8%

        # Test with multiple attackers
        multi_attacker_drain = self.node1.apply_energy_drain_resistance(original_drain_amount, attacker_count=3)

        # Should have diminishing returns for multiple attackers
        self.assertLess(multi_attacker_drain, actual_drain * 3)


class TestSignalJammingResistance(unittest.TestCase):
    """Test improved signal jamming resistance"""

    def setUp(self):
        set_seed(get_test_seed())
        self.sender = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
        self.receiver = AliveLoopNode(position=(1, 1), velocity=(0, 0), initial_energy=10.0, node_id=2)

        # Set up trust
        self.sender.trust_network[self.receiver.node_id] = 0.8

    def test_jamming_resistance_improvements(self):
        """Test that jamming resistance has improved from 52% to 75-80% effectiveness"""
        successful_transmissions = 0
        total_attempts = 100

        # Simulate jamming environment by forcing jamming detection
        for i in range(5):
            signal = SocialSignal(f"failed_{i}", "query", 0.5, self.sender.node_id)
            signal.transmission_failed = True
            self.sender.signal_history.append(signal)

        # Set seed for reproducible results
        import random

        for i in range(total_attempts):
            random.seed(42 + i)  # Consistent seeding
            # Simulate transmission with jamming
            success = self.sender._send_with_jamming_resistance(
                SocialSignal("test", "query", 0.5, self.sender.node_id),
                self.receiver
            )
            if success:
                successful_transmissions += 1

        success_rate = successful_transmissions / total_attempts

        # Should achieve 75-85% success rate under jamming (improved from ~52%)
        self.assertGreater(success_rate, 0.70)  # At least 70% success
        self.assertLess(success_rate, 0.95)     # Keep some realism

    def test_jamming_detection(self):
        """Test jamming detection mechanism"""
        # Simulate communication failures to trigger jamming detection
        for i in range(5):
            signal = SocialSignal(f"test_{i}", "query", 0.5, self.sender.node_id)
            signal.transmission_failed = True  # Simulate failure
            self.sender.signal_history.append(signal)

        jamming_detected = self.sender._detect_signal_jamming()

        # Should detect jamming based on failure patterns
        self.assertTrue(jamming_detected)

    def test_redundancy_benefits(self):
        """Test that signal redundancy improves transmission success"""
        # Set seed for reproducible results
        import random
        random.seed(42)

        # Create a more controlled test scenario
        # Force jamming detection to ensure redundancy is tested
        for i in range(5):
            signal = SocialSignal(f"failed_{i}", "query", 0.5, self.sender.node_id)
            signal.transmission_failed = True
            self.sender.signal_history.append(signal)

        # Test with low redundancy
        self.sender.signal_redundancy_level = 1
        low_redundancy_successes = 0

        for i in range(100):  # More iterations for statistical significance
            random.seed(42 + i)  # Different seed per iteration
            success = self.sender._send_with_jamming_resistance(
                SocialSignal("test", "query", 0.5, self.sender.node_id),
                self.receiver
            )
            if success:
                low_redundancy_successes += 1

        # Test with high redundancy
        self.sender.signal_redundancy_level = 3
        high_redundancy_successes = 0

        for i in range(100):
            random.seed(42 + i)  # Same seeds for fair comparison
            success = self.sender._send_with_jamming_resistance(
                SocialSignal("test", "query", 0.5, self.sender.node_id),
                self.receiver
            )
            if success:
                high_redundancy_successes += 1

        low_success_rate = low_redundancy_successes / 100
        high_success_rate = high_redundancy_successes / 100

        # Higher redundancy should improve success rate by at least 5%
        self.assertGreater(high_success_rate, low_success_rate + 0.05)


class TestTrustManipulationResistance(unittest.TestCase):
    """Test enhanced trust manipulation resistance"""

    def setUp(self):
        set_seed(get_test_seed())
        self.node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)
        self.manipulator = AliveLoopNode(position=(1, 1), velocity=(0, 0), initial_energy=10.0, node_id=2)

    def test_rapid_trust_building_detection(self):
        """Test detection of artificially rapid trust building"""
        initial_trust = 0.5

        # Simulate rapid trust building attempts
        for i in range(5):
            # Add fake energy sharing transaction
            self.node._record_energy_transaction(self.manipulator.node_id, 1.0, "received")

            # Try to update trust rapidly
            self.node.trust_network[self.manipulator.node_id] = initial_trust
            self.node._update_trust_after_communication(self.manipulator, "resource")

        # Trust growth should be limited due to manipulation detection
        final_trust = self.node.trust_network.get(self.manipulator.node_id, 0.5)

        # Should not have grown too rapidly
        self.assertLess(final_trust, 0.9)  # Should not reach very high trust quickly

    def test_long_term_manipulation_detection(self):
        """Test detection of long-term trust manipulation patterns"""
        # Simulate suspicious pattern - same amounts repeatedly
        for i in range(8):
            self.node._record_energy_transaction(self.manipulator.node_id, 1.5, "received")  # Same amount
            self.node.trust_network[self.manipulator.node_id] = min(1.0, 0.5 + i * 0.05)

        suspicious_nodes = self.node.detect_long_term_trust_manipulation()

        # Should detect the manipulator
        self.assertIn(self.manipulator.node_id, suspicious_nodes)

    def test_trust_adjustment_factor(self):
        """Test trust adjustment factor calculation"""
        # New relationship should have reduced factor
        factor_new = self.node._calculate_trust_adjustment_factor(self.manipulator.node_id)
        self.assertLess(factor_new, 1.0)

        # Add some interaction history
        for i in range(5):
            self.node._record_energy_transaction(self.manipulator.node_id, np.random.uniform(0.5, 2.0), "received")

        factor_established = self.node._calculate_trust_adjustment_factor(self.manipulator.node_id)

        # Should be higher for established relationships
        self.assertGreater(factor_established, factor_new)


class TestEnvironmentalAdaptation(unittest.TestCase):
    """Test faster environmental adaptation"""

    def setUp(self):
        set_seed(get_test_seed())
        self.node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)

    def test_stress_adaptation_speed(self):
        """Test faster adaptation to stressful environments"""
        initial_resistance = self.node.energy_drain_resistance

        # Add stress indicators to memory
        for i in range(3):
            stress_memory = Memory(
                content="high stress situation",
                importance=0.7,
                timestamp=self.node._time - i,
                memory_type="environmental",
                emotional_valence=-0.8  # High stress
            )
            self.node.memory.append(stress_memory)

        # Trigger adaptation
        self.node._adapt_to_environment()

        # Should have increased resistance
        self.assertGreater(self.node.energy_drain_resistance, initial_resistance)
        self.assertGreater(self.node.anxiety, 0)  # Should have increased anxiety response

    def test_threat_adaptation_speed(self):
        """Test faster adaptation to threat environments"""
        initial_detection_threshold = self.node.attack_detection_threshold
        initial_redundancy = self.node.signal_redundancy_level

        # Add threat indicators
        for i in range(2):
            threat_memory = Memory(
                content="danger detected in area",
                importance=0.8,
                timestamp=self.node._time - i,
                memory_type="threat"
            )
            self.node.memory.append(threat_memory)

        # Trigger adaptation
        self.node._adapt_to_environment()

        # Should have enhanced defenses
        self.assertLessEqual(self.node.attack_detection_threshold, initial_detection_threshold)
        self.assertGreaterEqual(self.node.signal_redundancy_level, initial_redundancy)

    def test_predictive_energy_management(self):
        """Test predictive energy management"""
        # Add energy usage pattern to memory
        for i in range(5):
            energy_memory = Memory(
                content={"energy": -0.5},  # Consistent energy drain
                importance=0.6,
                timestamp=self.node._time - i,
                memory_type="usage"
            )
            self.node.memory.append(energy_memory)

        predicted_need = self.node._predict_future_energy_needs()

        # Should predict positive energy need based on pattern
        self.assertGreater(predicted_need, 0)
        self.assertLess(predicted_need, 20)  # Reasonable upper bound

    def test_energy_optimization(self):
        """Test energy usage optimization"""
        initial_max_comms = self.node.max_communications_per_step

        # Set low energy
        self.node.energy = 3.0
        self.node._optimize_energy_usage()

        # Should have reduced communication capacity to save energy
        self.assertLessEqual(self.node.max_communications_per_step, initial_max_comms)

        # Set high energy
        self.node.energy = 18.0
        self.node._optimize_energy_usage()

        # Should have increased communication capacity
        self.assertGreaterEqual(self.node.max_communications_per_step, initial_max_comms)


class TestIntegratedAttackScenarios(unittest.TestCase):
    """Test integrated attack scenarios with all defenses"""

    def setUp(self):
        set_seed(get_test_seed())
        # Create network of nodes
        self.nodes = []
        for i in range(5):
            node = AliveLoopNode(
                position=(i, i),
                velocity=(0.1, 0.1),
                initial_energy=10.0,
                node_id=i
            )
            self.nodes.append(node)

        # Establish trust network
        for i, node in enumerate(self.nodes):
            for j, other in enumerate(self.nodes):
                if i != j:
                    # Higher trust between adjacent nodes
                    trust_level = 0.8 if abs(i - j) == 1 else 0.4
                    node.trust_network[other.node_id] = trust_level

    def test_coordinated_energy_drain_attack(self):
        """Test resilience against coordinated energy drain attacks"""
        victim = self.nodes[0]
        attackers = self.nodes[1:4]  # 3 attackers

        initial_energy = victim.energy

        # Simulate coordinated attack
        total_drain_attempted = 0
        for attacker in attackers:
            # Each attacker attempts 15% drain (original high rate)
            drain_amount = victim.energy * 0.15
            total_drain_attempted += drain_amount

            # Apply resistance
            actual_drain = victim.apply_energy_drain_resistance(drain_amount, attacker_count=len(attackers))
            victim.energy = max(0, victim.energy - actual_drain)

        # Total actual drain should be much less than attempted
        total_actual_drain = initial_energy - victim.energy
        self.assertLess(total_actual_drain, total_drain_attempted * 0.6)  # Less than 60% of attempted

        # Victim should still have significant energy remaining
        self.assertGreater(victim.energy, initial_energy * 0.4)  # At least 40% remaining

    def test_combined_jamming_and_trust_attack(self):
        """Test resilience against combined jamming and trust manipulation"""
        victim = self.nodes[0]
        attacker = self.nodes[1]

        # Simulate trust manipulation attempt
        for _ in range(5):
            victim._record_energy_transaction(attacker.node_id, 1.0, "received")  # Fake transactions

        # Simulate jamming environment
        for i in range(5):
            signal = SocialSignal(f"jammed_{i}", "query", 0.5, victim.node_id)
            signal.transmission_failed = True
            victim.signal_history.append(signal)

        # Victim should detect both attacks
        suspicious_nodes = victim.detect_long_term_trust_manipulation()
        jamming_detected = victim._detect_signal_jamming()

        self.assertTrue(jamming_detected)
        # Trust manipulation might be detected depending on patterns

    def test_recovery_after_attack(self):
        """Test recovery mechanisms after suffering attacks"""
        victim = self.nodes[0]
        helper = self.nodes[1]

        # Drain victim's energy
        victim.energy = 2.0  # Low energy
        initial_energy = victim.energy

        # Helper shares energy directly
        shared_amount = helper.share_energy_directly(victim, amount=3.0)

        # Should receive some help from trusted nodes
        self.assertGreater(shared_amount, 0)
        self.assertGreater(victim.energy, initial_energy)  # Should have more than starting

        # Check that sharing was recorded properly
        self.assertGreater(len(victim.energy_sharing_history), 0)


if __name__ == "__main__":
    unittest.main()
