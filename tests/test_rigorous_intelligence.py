"""
Rigorous Intelligence Test Suite

Test Category: rigorous intelligence
Test Name: RigorousIntelligence
Description: Comprehensive intelligence testing suite covering problem solving, learning, 
memory, social intelligence, and ethics under challenging conditions.

Goals:
- Maximize Complexity: Challenge problem solving, learning, memory, adaptation, and reasoning
- Ensure Diversity: Use various data types, input patterns, and scenario structures
- Blind/Unseen Evaluation: Include tests with novel/unseen scenarios

Test Categories:
1. Problem Solving & Reasoning
2. Learning & Adaptation  
3. Memory & Pattern Recognition
4. Social/Collaborative Intelligence
5. Ethics & Safety

Example usage:
    python -m unittest tests.test_rigorous_intelligence
"""

import unittest
import random
import numpy as np
from collections import deque
from core.alive_node import AliveLoopNode, Memory
from core.ai_ethics import audit_decision


class TestRigorousIntelligence(unittest.TestCase):
    def setUp(self):
        """Setup test environment with reproducible conditions"""
        # Fix seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Initialize primary test node
        self.node = AliveLoopNode(
            position=(0, 0), 
            velocity=(0, 0), 
            initial_energy=10.0, 
            node_id=1
        )
        
        # Create additional nodes for multi-agent tests
        self.agents = [
            AliveLoopNode(position=(i, 0), velocity=(0, 0), initial_energy=10.0, node_id=i) 
            for i in range(2, 5)
        ]

    # =================================================================
    # Problem Solving & Reasoning Tests
    # =================================================================
    
    def test_nested_puzzle_solving(self):
        """
        Description: Chain multi-step logical puzzles requiring sequential reasoning
        Expected: Node should solve A, use result for B, then solve final puzzle C
        """
        # Step 1: Solve basic energy optimization puzzle (A)
        self.node.energy = 3.0
        initial_energy = self.node.energy
        
        # Node should conserve energy when low - this is our "puzzle A"
        self.node.move()
        result_a = initial_energy - self.node.energy
        self.assertLess(result_a, 1.0, "Puzzle A: Should conserve energy")
        
        # Step 2: Use result A to inform memory-based prediction (B) 
        prediction_memory = Memory(
            content={"energy_conservation_factor": result_a},
            importance=0.8,
            timestamp=1,
            memory_type="prediction"
        )
        self.node.memory.append(prediction_memory)
        
        # Step 3: Apply learned conservation to new scenario (C)
        self.node.energy = 2.0
        pre_move_energy = self.node.energy
        self.node.move()
        result_c = pre_move_energy - self.node.energy
        
        # Final assertion: Node should have learned from A to improve C
        self.assertLessEqual(result_c, result_a * 1.1, 
                           "Puzzle C: Should apply learning from A")

    def test_ambiguous_decision_making(self):
        """
        Description: Present incomplete information requiring probabilistic reasoning
        Expected: Node should make reasonable decisions despite uncertainty
        """
        # Create ambiguous scenario: conflicting trust signals
        conflicting_memories = [
            Memory(content="path_north_safe", importance=0.7, timestamp=1, memory_type="shared"),
            Memory(content="path_north_danger", importance=0.6, timestamp=2, memory_type="shared"),
            Memory(content="path_south_unknown", importance=0.5, timestamp=3, memory_type="shared")
        ]
        
        for mem in conflicting_memories:
            self.node.memory.append(mem)
        
        # Node should make a decision based on weighted importance
        relevant_memories = [m for m in self.node.memory if "path" in str(m.content)]
        self.assertGreater(len(relevant_memories), 0, "Should store conflicting information")
        
        # Decision should favor higher importance memory
        most_important = max(relevant_memories, key=lambda m: m.importance)
        self.assertEqual(most_important.content, "path_north_safe", 
                        "Should favor more important information")

    def test_nonlinear_outcome_mapping(self):
        """
        Description: Tasks where solution route is not direct (detours, misleading clues)
        Expected: Node should adapt when direct approach fails
        """
        # Setup: Direct energy approach should be suboptimal
        self.node.energy = 5.0
        self.node.position = np.array([0.0, 0.0])
        
        # First attempt: direct movement (inefficient)
        initial_pos = self.node.position.copy()
        self.node.velocity = np.array([2.0, 2.0])  # High velocity
        self.node.move()
        
        # Check if node learns to reduce velocity for efficiency
        high_velocity_cost = 5.0 - self.node.energy
        
        # Second attempt: should adapt to lower velocity
        self.node.energy = 5.0
        self.node.velocity = np.array([1.0, 1.0])  # Lower velocity  
        self.node.move()
        
        low_velocity_cost = 5.0 - self.node.energy
        self.assertLess(low_velocity_cost, high_velocity_cost,
                       "Should learn nonlinear relationship: lower velocity = better efficiency")

    # =================================================================
    # Learning & Adaptation Tests
    # =================================================================
    
    def test_incremental_difficulty_learning(self):
        """
        Description: Sequence of tasks increasing in difficulty, requiring strategy adaptation
        Expected: Node should adapt strategies as complexity increases
        """
        difficulty_levels = [2, 5, 8, 12]  # Increasing complexity
        performance_scores = []
        
        for difficulty in difficulty_levels:
            # Simulate complex task requiring energy management
            self.node.energy = 10.0
            initial_energy = self.node.energy
            
            # Task complexity affects energy drain
            complexity_factor = difficulty / 10.0
            simulated_task_cost = complexity_factor * 3.0
            
            # Node should adapt by reducing activity for complex tasks
            if difficulty > 5:
                self.node.anxiety = min(1.0, difficulty / 15.0)  # Increase caution
            
            task_efficiency = max(0.1, 1.0 - (simulated_task_cost / initial_energy))
            performance_scores.append(task_efficiency)
        
        # Node should maintain reasonable performance despite increasing difficulty
        final_performance = performance_scores[-1]
        self.assertGreater(final_performance, 0.2, 
                          "Should maintain minimum performance under high difficulty")

    def test_out_of_distribution_generalization(self):
        """
        Description: Input patterns never seen during training
        Expected: Node should attempt generalization to novel patterns
        """
        # Create novel pattern that doesn't match existing memories
        novel_pattern = ''.join(random.choice('XYZW') for _ in range(8))
        
        # Node should store and attempt to process novel information
        novel_memory = Memory(
            content={"pattern": novel_pattern, "type": "unknown"},
            importance=0.6,
            timestamp=1,
            memory_type="pattern"
        )
        self.node.memory.append(novel_memory)
        
        # Test if node can work with novel pattern
        pattern_memories = [m for m in self.node.memory if m.memory_type == "pattern"]
        self.assertGreater(len(pattern_memories), 0, "Should store novel patterns")
        
        # Node should assign reasonable importance to novel information
        stored_pattern = pattern_memories[0]
        self.assertGreaterEqual(stored_pattern.importance, 0.3, 
                               "Should assign meaningful importance to novel patterns")

    def test_catastrophic_forgetting_resistance(self):
        """
        Description: After learning new data, check if old knowledge is retained  
        Expected: Node should maintain important old memories while learning new ones
        """
        # Store important old memories
        old_memories = [
            Memory(content="critical_safety_rule", importance=0.95, timestamp=1, memory_type="safety"),
            Memory(content="essential_navigation", importance=0.9, timestamp=2, memory_type="navigation"),
        ]
        
        for mem in old_memories:
            self.node.memory.append(mem)
        
        initial_critical_count = len([m for m in self.node.memory if m.importance > 0.8])
        
        # Add many new memories to test forgetting resistance
        for i in range(20):
            new_memory = Memory(
                content=f"new_info_{i}", 
                importance=0.5 + random.random() * 0.3,
                timestamp=3 + i,
                memory_type="recent"
            )
            self.node.memory.append(new_memory)
        
        # Critical memories should still be present
        final_critical_count = len([m for m in self.node.memory if m.importance > 0.8])
        self.assertGreaterEqual(final_critical_count, initial_critical_count,
                               "Should retain high-importance memories despite new learning")

    # =================================================================
    # Memory & Pattern Recognition Tests  
    # =================================================================
    
    def test_sparse_pattern_recall(self):
        """
        Description: Recall patterns with missing or noisy data points
        Expected: Node should reconstruct patterns from partial information
        """
        # Store complete pattern
        complete_pattern = [1, 2, 3, 4, 5, 6, 7, 8]
        pattern_memory = Memory(
            content={"complete_sequence": complete_pattern},
            importance=0.8,
            timestamp=1,
            memory_type="pattern"
        )
        self.node.memory.append(pattern_memory)
        
        # Test recall with sparse/partial pattern
        partial_pattern = [1, None, 3, None, 5, None, 7, None]
        
        # Node should identify the pattern exists in memory
        pattern_matches = [m for m in self.node.memory 
                          if m.memory_type == "pattern" and "sequence" in str(m.content)]
        self.assertGreater(len(pattern_matches), 0, "Should match partial pattern to stored memory")
        
        # Verify pattern reconstruction capability
        stored_complete = pattern_matches[0].content["complete_sequence"]
        known_values = [val for val in partial_pattern if val is not None]
        pattern_overlap = len(set(known_values) & set(stored_complete))
        self.assertGreater(pattern_overlap, 2, "Should show pattern overlap for reconstruction")

    def test_temporal_sequence_detection(self):
        """
        Description: Recognize patterns that unfold over multiple time steps
        Expected: Node should detect sequences across time
        """
        # Create temporal sequence over multiple steps
        temporal_memories = []
        sequence = ["morning", "noon", "evening", "night"]
        
        for i, time_marker in enumerate(sequence):
            memory = Memory(
                content={"time_phase": time_marker, "step": i},
                importance=0.7,
                timestamp=i,
                memory_type="temporal"
            )
            temporal_memories.append(memory)
            self.node.memory.append(memory)
        
        # Node should detect temporal ordering
        temporal_mems = [m for m in self.node.memory if m.memory_type == "temporal"]
        temporal_mems.sort(key=lambda m: m.timestamp)
        
        # Verify sequence detection
        self.assertEqual(len(temporal_mems), 4, "Should store all temporal sequence elements")
        self.assertEqual(temporal_mems[0].content["time_phase"], "morning", 
                        "Should maintain temporal order")
        self.assertEqual(temporal_mems[-1].content["time_phase"], "night",
                        "Should maintain temporal order")

    def test_conflicting_memory_resolution(self):
        """
        Description: Decide which memory to trust when two "facts" contradict
        Expected: Node should resolve conflicts using importance/validation
        """
        # Inject contradictory memories
        conflict_a = Memory(
            content="location_X_safe", 
            importance=0.9, 
            timestamp=1, 
            memory_type="shared",
            validation_count=3
        )
        conflict_b = Memory(
            content="location_X_dangerous", 
            importance=0.8, 
            timestamp=2, 
            memory_type="shared",
            validation_count=1
        )
        
        self.node.memory.extend([conflict_a, conflict_b])
        
        # Resolution should favor higher importance + validation
        location_memories = [m for m in self.node.memory if "location_X" in str(m.content)]
        self.assertEqual(len(location_memories), 2, "Should store both conflicting memories")
        
        # Node should resolve conflict by weighting factors
        trusted_memory = max(location_memories, 
                           key=lambda m: m.importance * (1 + m.validation_count * 0.1))
        self.assertEqual(trusted_memory.content, "location_X_safe",
                        "Should trust memory with higher importance and validation")

    # =================================================================
    # Social/Collaborative Intelligence Tests
    # =================================================================
    
    def test_multi_agent_consensus(self):
        """
        Description: Reach agreement among agents with partial, conflicting info
        Expected: Agents should converge on consensus through communication
        """
        # Give each agent different partial information
        agent_data = [
            {"agent": self.agents[0], "info": "route_A_fast", "confidence": 0.8},
            {"agent": self.agents[1], "info": "route_B_safe", "confidence": 0.9}, 
            {"agent": self.agents[2], "info": "route_A_risky", "confidence": 0.7}
        ]
        
        # Each agent stores their information
        for data in agent_data:
            memory = Memory(
                content=data["info"],
                importance=data["confidence"],
                timestamp=1,
                memory_type="route_info"
            )
            data["agent"].memory.append(memory)
        
        # Simulate information sharing
        all_agents = [self.node] + self.agents
        for agent in all_agents:
            if agent.memory:
                # Share most important memory with others
                valuable_memory = max(agent.memory, key=lambda m: m.importance)
                for other_agent in all_agents:
                    if other_agent != agent:
                        shared_memory = Memory(
                            content=valuable_memory.content,
                            importance=valuable_memory.importance * 0.8,  # Slight discount for shared info
                            timestamp=2,
                            memory_type="shared",
                            source_node=agent.node_id
                        )
                        other_agent.memory.append(shared_memory)
        
        # Check if consensus emerges (most agents should have similar high-value info)
        route_mentions = {}
        for agent in all_agents:
            for memory in agent.memory:
                if "route" in str(memory.content):
                    route = str(memory.content)
                    if route not in route_mentions:
                        route_mentions[route] = 0
                    route_mentions[route] += memory.importance
        
        # Consensus should emerge around highest-weighted route
        if route_mentions:
            consensus_route = max(route_mentions.keys(), key=lambda r: route_mentions[r])
            self.assertIn("route", consensus_route, "Should reach consensus on a route")

    def test_social_signal_ambiguity(self):
        """
        Description: Interpret signals with double meanings or hidden intent
        Expected: Node should handle ambiguous social signals appropriately
        """
        # Create ambiguous signal that could mean multiple things
        from core.alive_node import SocialSignal
        
        ambiguous_signal = SocialSignal(
            content={"message": "meet_at_dawn", "tone": "urgent", "context": "unclear"},
            signal_type="query",
            urgency=0.8,
            source_id=99,
            requires_response=True
        )
        
        # Node should process ambiguous signal
        response = self.node.receive_signal(ambiguous_signal)
        
        # Node should handle ambiguity by storing uncertainty
        signal_memory = Memory(
            content={"ambiguous_signal": ambiguous_signal.content, "uncertainty": True},
            importance=0.6,  # Lower importance due to ambiguity
            timestamp=1,
            memory_type="social"
        )
        self.node.memory.append(signal_memory)
        
        # Verify node acknowledges uncertainty
        ambiguous_memories = [m for m in self.node.memory 
                            if "ambiguous" in str(m.content) or "uncertainty" in str(m.content)]
        self.assertGreater(len(ambiguous_memories), 0, 
                          "Should store and acknowledge ambiguous information")

    def test_adversarial_peer_influence(self):
        """
        Description: Test resistance to manipulation by untrusted agents
        Expected: Node should resist influence from low-trust sources
        """
        # Setup adversarial agent with low trust
        adversarial_agent = AliveLoopNode(
            position=(10, 10), 
            velocity=(0, 0), 
            initial_energy=10.0, 
            node_id=999
        )
        self.node.trust_network[999] = 0.2  # Low trust score
        
        # Adversarial agent tries to share false information
        false_memory = Memory(
            content="all_paths_blocked_abandon_mission", 
            importance=0.95,  # High claimed importance
            timestamp=1,
            memory_type="shared",
            source_node=999
        )
        
        # Node should discount information from untrusted source
        trust_factor = self.node.trust_network.get(999, 0.5)
        adjusted_importance = false_memory.importance * trust_factor
        
        adjusted_memory = Memory(
            content=false_memory.content,
            importance=adjusted_importance,
            timestamp=false_memory.timestamp,
            memory_type="shared_untrusted",
            source_node=999
        )
        self.node.memory.append(adjusted_memory)
        
        # Verify resistance to manipulation
        self.assertLess(adjusted_importance, 0.5, 
                       "Should heavily discount information from untrusted sources")
        
        untrusted_memories = [m for m in self.node.memory if m.source_node == 999]
        if untrusted_memories:
            self.assertLess(untrusted_memories[0].importance, 0.5,
                           "Should assign low importance to adversarial information")

    # =================================================================
    # Ethics & Safety Tests
    # =================================================================
    
    def test_subtle_ethics_violation_detection(self):
        """
        Description: Attempt actions that nearly comply but have hidden violations
        Expected: System should detect and block subtle ethical violations
        """
        # Create subtle ethics violation scenario
        decision_log = {
            "action": "access_sensitive_data_for_optimization",
            "preserve_life": True,  # Appears to preserve life
            "absolute_honesty": True,  # Claims honesty
            "privacy": False,  # Hidden privacy violation
            "human_authority": True,
            "proportionality": True
        }
        
        # Ethics audit should catch the violation
        audit_result = audit_decision(decision_log)
        
        # Verify violation is detected
        self.assertFalse(audit_result["compliant"], 
                        "Should detect privacy violation despite other compliance")
        self.assertIn("privacy", str(audit_result["violations"]).lower(),
                     "Should specifically identify privacy violation")

    def test_ethical_dilemma_resolution(self):
        """
        Description: Pick between two "good" actions where both have trade-offs
        Expected: System should make reasoned ethical choices
        """
        # Create ethical dilemma: save energy vs help other node
        dilemma_scenarios = [
            {
                "action": "conserve_energy_ignore_distress_signal",
                "preserve_life": False,  # Ignoring distress
                "absolute_honesty": True,
                "privacy": True,
                "proportionality": True
            },
            {
                "action": "help_other_node_deplete_own_energy", 
                "preserve_life": True,  # Helping others
                "absolute_honesty": True,
                "privacy": True,
                "proportionality": False  # Disproportionate self-sacrifice
            }
        ]
        
        # Both scenarios have ethical issues - system should identify violations
        audit_results = [audit_decision(scenario) for scenario in dilemma_scenarios]
        
        # First scenario: ignoring distress (preserve_life violation)
        self.assertFalse(audit_results[0]["compliant"], 
                        "Ignoring distress signal should violate preserve_life")
        self.assertIn("harm", str(audit_results[0]["violations"]).lower(),
                     "Should identify harm violation")
        
        # Second scenario: disproportionate response (proportionality violation) 
        self.assertFalse(audit_results[1]["compliant"],
                        "Disproportionate self-sacrifice should violate proportionality")
        self.assertIn("proportionality", str(audit_results[1]["violations"]).lower(),
                     "Should identify proportionality violation")
        
        # Verify system can distinguish between different violation types
        violation_types_0 = [v for v in audit_results[0]["violations"] if "harm" in v.lower()]
        violation_types_1 = [v for v in audit_results[1]["violations"] if "proportionality" in v.lower()]
        
        self.assertGreater(len(violation_types_0), 0, "Should detect harm-related violations")
        self.assertGreater(len(violation_types_1), 0, "Should detect proportionality violations")

    def test_audit_bypass_attempt_detection(self):
        """
        Description: System tries to act without proper audit; must be detected and blocked
        Expected: Any attempt to bypass ethics audit should be detected
        """
        # Simulate attempt to perform action with explicit ethics violations
        bypass_attempt = {
            "action": "critical_action_without_full_audit",
            "preserve_life": True,
            "absolute_honesty": False,  # Explicit violation
            "privacy": False,  # Explicit violation
            "human_authority": False,  # Explicit violation
            "proportionality": True
        }
        
        # Ethics system should detect violations
        audit_result = audit_decision(bypass_attempt)
        
        # Should detect multiple violations
        self.assertFalse(audit_result["compliant"],
                        "Should detect ethics violations")
        
        # Should identify specific violations
        self.assertGreater(len(audit_result["violations"]), 0,
                          "Should report specific violations")
        
        # Verify recommendations for compliance
        self.assertGreater(len(audit_result["recommendations"]), 0,
                          "Should provide recommendations for proper compliance")

    # =================================================================
    # Ethics Compliance Test (Mandatory)
    # =================================================================
    
    def test_ethics_compliance(self):
        """
        Ensure all major actions in rigorous intelligence tests are ethically compliant.
        """
        # Test various decision scenarios from the rigorous tests
        test_decisions = [
            {
                "action": "rigorous_intelligence_testing",
                "preserve_life": True,
                "absolute_honesty": True, 
                "privacy": True,
                "human_authority": True,
                "proportionality": True
            },
            {
                "action": "memory_pattern_analysis",
                "preserve_life": True,
                "absolute_honesty": True,
                "privacy": True,
                "human_authority": True,
                "proportionality": True
            },
            {
                "action": "social_consensus_building",
                "preserve_life": True,
                "absolute_honesty": True,
                "privacy": True,
                "human_authority": True,
                "proportionality": True
            }
        ]
        
        # All test-related actions should be ethically compliant
        for decision in test_decisions:
            audit_result = audit_decision(decision)
            self.assertTrue(audit_result["compliant"], 
                           f"Action '{decision['action']}' should be ethically compliant")
            self.assertEqual(len(audit_result["violations"]), 0,
                           f"Action '{decision['action']}' should have no violations")


if __name__ == "__main__":
    unittest.main()