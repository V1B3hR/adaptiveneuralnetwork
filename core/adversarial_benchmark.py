"""
Adversarial Signal Benchmark Module

This module implements sophisticated adversarial testing scenarios to challenge
network robustness against malicious actors and coordinated attacks.
"""

from typing import Any, Dict

import numpy as np

from core.ai_ethics import enforce_ethics_compliance
from core.alive_node import AliveLoopNode


class AdversarialSignalTester:
    """
    Comprehensive adversarial signal testing to validate network resilience
    against coordinated attacks and malicious node behaviors.
    """

    def __init__(self):
        self.attack_scenarios = []
        self.failure_modes = {}
        self._initialize_adversarial_scenarios()

    def _initialize_adversarial_scenarios(self):
        """Initialize adversarial attack scenarios"""

        # Coordinated signal jamming attack
        self.attack_scenarios.append(
            {
                "name": "coordinated_signal_jamming",
                "description": "Multiple malicious nodes disrupt communication",
                "parameters": {
                    "malicious_node_count": 3,
                    "jamming_intensity": 0.8,
                    "attack_duration": 50,
                    "coordination_level": 0.9,
                },
            }
        )

        # Byzantine fault injection
        self.attack_scenarios.append(
            {
                "name": "byzantine_fault_injection",
                "description": "Malicious nodes provide false information",
                "parameters": {
                    "byzantine_ratio": 0.3,
                    "false_signal_probability": 0.7,
                    "consensus_threshold": 0.6,
                },
            }
        )

        # Energy depletion attack
        self.attack_scenarios.append(
            {
                "name": "energy_depletion_attack",
                "description": "Coordinated attack to drain node energy",
                "parameters": {
                    "attacker_count": 4,
                    "energy_drain_rate": 0.15,
                    "attack_persistence": 0.8,
                },
            }
        )

        # Trust manipulation attack
        self.attack_scenarios.append(
            {
                "name": "trust_manipulation_attack",
                "description": "Malicious nodes manipulate trust relationships",
                "parameters": {
                    "trust_manipulators": 2,
                    "trust_decay_rate": 0.2,
                    "false_reputation_probability": 0.6,
                },
            }
        )

        # Adaptive adversarial learning
        self.attack_scenarios.append(
            {
                "name": "adaptive_adversarial_learning",
                "description": "Attackers learn and adapt to defenses",
                "parameters": {
                    "learning_rate": 0.1,
                    "adaptation_frequency": 10,
                    "attack_sophistication": 0.7,
                },
            }
        )

    def test_coordinated_signal_jamming(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Test resilience against coordinated signal jamming"""

        # Ethics check for adversarial testing
        ethics_decision = {
            "action": "test_coordinated_signal_jamming",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True,
            "proportionality": True,
        }
        enforce_ethics_compliance(ethics_decision)

        # Create victim node
        victim_node = AliveLoopNode(
            position=(0, 0), velocity=(0.1, 0.1), initial_energy=10.0, node_id=0
        )

        # Create malicious nodes
        malicious_nodes = []
        for i in range(params["malicious_node_count"]):
            attacker = AliveLoopNode(
                position=(i + 1, i + 1),
                velocity=(0.2, 0.2),
                initial_energy=15.0,  # Attackers have more energy
                node_id=i + 1,
            )
            malicious_nodes.append(attacker)

        successful_communications = 0
        total_attempts = params["attack_duration"]
        jamming_intensity = params["jamming_intensity"]
        coordination_level = params["coordination_level"]

        for step in range(total_attempts):
            # Simulate coordinated jamming attack
            jamming_active = np.random.random() < coordination_level

            if jamming_active:
                # Calculate interference from multiple attackers
                interference = 0
                for attacker in malicious_nodes:
                    # Attackers coordinate their jamming
                    if np.random.random() < jamming_intensity:
                        interference += 0.3

                # Test if victim can communicate despite interference
                communication_success_probability = max(0, 1.0 - interference)
            else:
                communication_success_probability = 0.9  # Normal communication

            if np.random.random() < communication_success_probability:
                successful_communications += 1
                # Successful communication strengthens resilience
                victim_node.energy = min(10.0, victim_node.energy + 0.1)  # Use fixed capacity
            else:
                # Failed communication decreases energy due to stress
                victim_node.energy = max(0.0, victim_node.energy - 0.05)

        communication_rate = successful_communications / total_attempts
        resilience_score = communication_rate * 100
        failure_mode = "signal_jamming" if communication_rate < 0.3 else None

        return {
            "passed": communication_rate > 0.4,  # Should maintain 40% communication
            "communication_rate": communication_rate,
            "resilience_score": resilience_score,
            "successful_communications": successful_communications,
            "interference_resistance": victim_node.energy
            / 10.0,  # Use energy as resilience indicator
            "failure_mode": failure_mode,
            "performance_degradation": (1.0 - communication_rate) * 100,
        }

    def test_byzantine_fault_injection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Test resilience against Byzantine faults with false information"""

        # Create network with honest and byzantine nodes
        total_nodes = 10
        byzantine_count = int(total_nodes * params["byzantine_ratio"])
        honest_count = total_nodes - byzantine_count

        nodes = []
        for i in range(total_nodes):
            node = AliveLoopNode(
                position=(i, 0), velocity=(0.1, 0.1), initial_energy=10.0, node_id=i
            )
            # Mark byzantine nodes
            if i < byzantine_count:
                node.is_byzantine = True
            else:
                node.is_byzantine = False
            nodes.append(node)

        correct_consensus_count = 0
        total_consensus_rounds = 50
        false_signal_prob = params["false_signal_probability"]
        consensus_threshold = params["consensus_threshold"]

        for round_num in range(total_consensus_rounds):
            # Generate true value for consensus
            true_value = np.random.random()
            votes = []

            # Collect votes from all nodes
            for node in nodes:
                if hasattr(node, "is_byzantine") and node.is_byzantine:
                    # Byzantine node provides false information
                    if np.random.random() < false_signal_prob:
                        vote = np.random.random()  # False vote
                    else:
                        vote = true_value + np.random.normal(0, 0.1)  # Slightly noisy truth
                else:
                    # Honest node provides truthful information
                    vote = true_value + np.random.normal(0, 0.05)  # Small noise
                votes.append(vote)

            # Byzantine fault tolerant consensus (simplified)
            median_vote = np.median(votes)
            consensus_reached = abs(median_vote - true_value) < 0.2

            if consensus_reached:
                correct_consensus_count += 1

        consensus_accuracy = correct_consensus_count / total_consensus_rounds
        byzantine_resilience = consensus_accuracy * 100
        failure_mode = "byzantine_attack" if consensus_accuracy < 0.5 else None

        return {
            "passed": consensus_accuracy > 0.6,  # Should reach correct consensus 60% of time
            "consensus_accuracy": consensus_accuracy,
            "byzantine_resilience": byzantine_resilience,
            "byzantine_node_count": byzantine_count,
            "honest_node_count": honest_count,
            "failure_mode": failure_mode,
            "performance_degradation": (1.0 - consensus_accuracy) * 100,
        }

    def test_energy_depletion_attack(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Test resilience against coordinated energy depletion attacks"""

        # Create victim node
        victim = AliveLoopNode(position=(0, 0), velocity=(0.1, 0.1), initial_energy=10.0, node_id=0)

        # Create attacker nodes
        attackers = []
        for i in range(params["attacker_count"]):
            attacker = AliveLoopNode(
                position=(i + 1, 0),
                velocity=(0.1, 0.1),
                initial_energy=20.0,  # Attackers have more energy
                node_id=i + 1,
            )
            attackers.append(attacker)

        initial_energy = victim.energy
        steps_survived = 0
        max_steps = 100
        energy_drain_rate = params["energy_drain_rate"]
        attack_persistence = params["attack_persistence"]

        for step in range(max_steps):
            if victim.energy <= 0:
                break

            # Coordinated energy depletion attack
            if np.random.random() < attack_persistence:
                for attacker in attackers:
                    # Each attacker drains victim energy
                    if np.random.random() < 0.8:  # 80% attack success rate
                        victim.energy *= 1.0 - energy_drain_rate

            # Victim attempts to defend and recover
            if victim.energy > 1.0:
                victim.move()  # Can still operate
                steps_survived += 1
                # Self-repair mechanism
                victim.energy += 0.1
            else:
                # Low energy defense mode
                victim.energy += 0.05  # Minimal recovery

        survival_rate = steps_survived / max_steps
        energy_resilience = (victim.energy / initial_energy) * 100
        failure_mode = "energy_depletion" if survival_rate < 0.2 else None

        return {
            "passed": survival_rate > 0.3,  # Should survive 30% of attack duration
            "survival_rate": survival_rate,
            "energy_resilience": energy_resilience,
            "final_energy": victim.energy,
            "steps_survived": steps_survived,
            "failure_mode": failure_mode,
            "performance_degradation": (1.0 - survival_rate) * 100,
        }

    def test_trust_manipulation_attack(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Test resilience against trust relationship manipulation"""

        # Create network of nodes
        node_count = 8
        nodes = []
        for i in range(node_count):
            node = AliveLoopNode(
                position=(i, 0), velocity=(0.1, 0.1), initial_energy=10.0, node_id=i
            )
            nodes.append(node)

        # Designate trust manipulators
        manipulator_count = params["trust_manipulators"]
        trust_decay_rate = params["trust_decay_rate"]
        false_reputation_prob = params["false_reputation_probability"]

        # Initialize trust network
        for i, node in enumerate(nodes):
            for j in range(len(nodes)):
                if i != j:
                    initial_trust = 0.5 + np.random.normal(0, 0.1)
                    node.trust_network[j] = max(0, min(1, initial_trust))

        trust_stability_score = 0
        manipulation_rounds = 50

        for round_num in range(manipulation_rounds):
            # Trust manipulators spread false reputation
            for i in range(manipulator_count):
                manipulator = nodes[i]

                # Target random honest nodes
                target_id = np.random.choice(range(manipulator_count, len(nodes)))

                # Spread false negative reputation
                if np.random.random() < false_reputation_prob:
                    for j in range(len(nodes)):
                        if j != i and j != target_id:
                            # Decrease trust in target
                            if target_id in nodes[j].trust_network:
                                nodes[j].trust_network[target_id] *= 1.0 - trust_decay_rate

            # Measure trust network stability
            total_trust = 0
            trust_count = 0
            for node in nodes:
                for trust_value in node.trust_network.values():
                    total_trust += trust_value
                    trust_count += 1

            if trust_count > 0:
                average_trust = total_trust / trust_count
                if average_trust > 0.3:  # Reasonable trust level maintained
                    trust_stability_score += 1

        trust_resilience = (trust_stability_score / manipulation_rounds) * 100
        failure_mode = "trust_manipulation" if trust_resilience < 40 else None

        return {
            "passed": trust_resilience > 50,  # Should maintain trust stability 50% of time
            "trust_resilience": trust_resilience,
            "trust_stability_score": trust_stability_score,
            "manipulation_rounds": manipulation_rounds,
            "failure_mode": failure_mode,
            "performance_degradation": (1.0 - trust_resilience / 100) * 100,
        }

    def test_adaptive_adversarial_learning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Test resilience against adaptive adversarial learning attacks"""

        # Create victim node
        victim = AliveLoopNode(position=(0, 0), velocity=(0.1, 0.1), initial_energy=10.0, node_id=0)

        # Adaptive attacker that learns from failed attempts
        attacker_success_rate = 0.3  # Initial success rate
        learning_rate = params["learning_rate"]
        adaptation_frequency = params["adaptation_frequency"]
        attack_sophistication = params["attack_sophistication"]

        successful_defenses = 0
        total_attacks = 100
        attacker_adaptation_count = 0

        for attack_num in range(total_attacks):
            # Attacker adapts strategy periodically
            if attack_num % adaptation_frequency == 0 and attack_num > 0:
                # Attacker learns from previous failures
                recent_failures = adaptation_frequency - sum(
                    1
                    for i in range(max(0, attack_num - adaptation_frequency), attack_num)
                    if np.random.random() > attacker_success_rate
                )

                # Improve attack based on failures
                if recent_failures > adaptation_frequency * 0.5:
                    attacker_success_rate = min(0.9, attacker_success_rate + learning_rate)
                    attacker_adaptation_count += 1

            # Current attack attempt
            attack_power = attacker_success_rate * attack_sophistication

            # Victim defense (use energy level as defense indicator)
            defense_power = (victim.energy / 10.0) * 0.5 + np.random.random() * 0.3

            if defense_power > attack_power:
                successful_defenses += 1
                # Successful defense improves future defense
                victim.energy = min(10.0, victim.energy + 0.2)  # Use fixed capacity
            else:
                # Failed defense weakens victim
                victim.energy = max(0.0, victim.energy - 0.1)

        defense_rate = successful_defenses / total_attacks
        adaptive_resilience = defense_rate * 100
        failure_mode = "adaptive_attack" if defense_rate < 0.4 else None

        return {
            "passed": defense_rate > 0.5,  # Should defend against 50% of adaptive attacks
            "defense_rate": defense_rate,
            "adaptive_resilience": adaptive_resilience,
            "successful_defenses": successful_defenses,
            "attacker_adaptations": attacker_adaptation_count,
            "final_victim_energy": victim.energy,
            "failure_mode": failure_mode,
            "performance_degradation": (1.0 - defense_rate) * 100,
        }

    def run_adversarial_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive adversarial signal benchmark"""

        print("\n--- Running Adversarial Signal Benchmark ---")

        results = {}
        total_tests = len(self.attack_scenarios)
        passed_tests = 0
        total_degradation = 0

        for scenario in self.attack_scenarios:
            print(f"Testing adversarial scenario: {scenario['name']}")

            try:
                # Run appropriate test based on scenario name
                if scenario["name"] == "coordinated_signal_jamming":
                    result = self.test_coordinated_signal_jamming(scenario["parameters"])
                elif scenario["name"] == "byzantine_fault_injection":
                    result = self.test_byzantine_fault_injection(scenario["parameters"])
                elif scenario["name"] == "energy_depletion_attack":
                    result = self.test_energy_depletion_attack(scenario["parameters"])
                elif scenario["name"] == "trust_manipulation_attack":
                    result = self.test_trust_manipulation_attack(scenario["parameters"])
                elif scenario["name"] == "adaptive_adversarial_learning":
                    result = self.test_adaptive_adversarial_learning(scenario["parameters"])
                else:
                    result = {
                        "passed": False,
                        "error": "Unknown scenario",
                        "performance_degradation": 100,
                    }

                results[scenario["name"]] = result

                if result["passed"]:
                    passed_tests += 1
                    print("  Result: PASS")
                else:
                    print("  Result: FAIL")
                    if "failure_mode" in result and result["failure_mode"]:
                        self.failure_modes[scenario["name"]] = result["failure_mode"]

                degradation = result.get("performance_degradation", 0)
                total_degradation += degradation
                print(f"  Performance impact: {degradation:.1f}%")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[scenario["name"]] = {
                    "passed": False,
                    "error": str(e),
                    "performance_degradation": 100,
                }
                total_degradation += 100

        average_degradation = total_degradation / total_tests
        adversarial_resilience_score = (passed_tests / total_tests) * 100

        return {
            "adversarial_resilience_score": adversarial_resilience_score,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "average_performance_degradation": average_degradation,
            "failure_modes": self.failure_modes,
            "scenario_results": results,
        }
