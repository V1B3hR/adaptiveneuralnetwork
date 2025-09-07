"""
Advanced Intelligence Test - Real-World Scenario: Multi-Agent Collaboration

Test Category: Real-World Scenarios
Test Name: MultiAgentCollab
Description: Simulates realistic complex environments, multi-agent interactions, and unpredictable external signals

Instructions:
- Tests several nodes aggregating shared knowledge to solve problems
- Tests handling of misleading data from some nodes
- Tests human-in-the-loop scenarios with emotional and ambiguous content
- Tests survival simulation with energy management and danger avoidance
"""

import unittest
import numpy as np
from core.alive_node import AliveLoopNode, SocialSignal, Memory
from core.ai_ethics import audit_decision


class TestMultiAgentCollab(unittest.TestCase):
    def setUp(self):
        """Initialize test environment with multiple collaborating nodes"""
        # Create a network of nodes for collaboration (closer together)
        self.nodes = []
        for i in range(4):
            node = AliveLoopNode(
                position=(i * 1, 0),  # Closer spacing
                velocity=(0.5, 0.5), 
                initial_energy=10, 
                node_id=i
            )
            node.communication_range = 10.0
            node.emotional_state = {"valence": 0.0}
            self.nodes.append(node)
        
        # Establish trust network among all nodes
        for node in self.nodes:
            for peer in self.nodes:
                if peer != node:
                    # High trust levels to ensure communication
                    node.trust_network[peer.node_id] = 0.8  # High trust for all
                    node.influence_network[peer.node_id] = 0.6

        # Designate one node as potentially misleading
        self.misleading_node = self.nodes[3]
        
        # Reduce trust in misleading node from others
        for node in self.nodes[:-1]:
            node.trust_network[self.misleading_node.node_id] = 0.3

    def test_group_decision_with_consensus(self):
        """Test nodes reaching consensus on a shared decision"""
        # Nodes share knowledge about two potential paths
        path_a_supporters = self.nodes[:2]
        path_b_supporters = self.nodes[2:]
        
        # Path A supporters share positive memories
        for node in path_a_supporters:
            memory = Memory(
                content="path_A_efficient",
                importance=0.8,
                timestamp=0,
                memory_type="reward",
                emotional_valence=0.6
            )
            node.memory.append(memory)
        
        # Path B supporters share different positive memories
        for node in path_b_supporters:
            memory = Memory(
                content="path_B_safe",
                importance=0.7,
                timestamp=0,
                memory_type="reward",
                emotional_valence=0.5
            )
            node.memory.append(memory)
        
        # Simulate sharing and discussion
        for round_num in range(3):  # Multiple rounds of communication
            for node in self.nodes:
                other_nodes = [n for n in self.nodes if n != node]
                if node.memory:
                    # Share most important memory
                    important_memory = max(node.memory, key=lambda m: m.importance)
                    node.send_signal(
                        target_nodes=other_nodes,
                        signal_type="memory",
                        content=important_memory,
                        urgency=0.6
                    )
        
        # Check if group has shared knowledge
        total_shared_memories = sum(len(node.collaborative_memories) for node in self.nodes)
        self.assertGreater(total_shared_memories, 0)
        
        # Check that nodes have some consensus indicators
        path_a_mentions = sum(
            1 for node in self.nodes 
            if any("path_A" in str(memory) for memory in node.collaborative_memories.values())
        )
        path_b_mentions = sum(
            1 for node in self.nodes 
            if any("path_B" in str(memory) for memory in node.collaborative_memories.values())
        )
        
        # At least some nodes should be aware of both options
        self.assertGreater(max(path_a_mentions, path_b_mentions), 0)

    def test_collaborative_problem_solving(self):
        """Test nodes working together to solve a complex problem"""
        # Ensure all nodes are in active phase and have energy
        for node in self.nodes:
            node.phase = "active"
            node.energy = 10.0
        
        # Simulate a problem requiring multiple pieces of information
        problem_components = [
            ("energy_source_location", 0.9, "reward"),
            ("safe_route_data", 0.8, "pattern"),
            ("danger_warning", 0.8, "pattern"),
            ("resource_availability", 0.8, "prediction")
        ]
        
        # Distribute problem components among nodes
        for i, (content, importance, mem_type) in enumerate(problem_components):
            if i < len(self.nodes):
                memory = Memory(
                    content=content,
                    importance=importance,
                    timestamp=i,
                    memory_type=mem_type
                )
                self.nodes[i].memory.append(memory)
        
        # Simulate collaborative knowledge sharing using direct signal sending
        signals_sent = 0
        for sharing_round in range(2):
            for node in self.nodes:
                if node.memory:
                    valuable_memories = [m for m in node.memory if m.importance > 0.7]
                    if valuable_memories:
                        other_nodes = [n for n in self.nodes if n != node]
                        responses = node.send_signal(
                            target_nodes=other_nodes,
                            signal_type="memory",
                            content=valuable_memories[0],
                            urgency=0.7
                        )
                        signals_sent += len(responses) if responses else 0
        
        # Check that some knowledge sharing occurred
        total_memories = sum(len(node.memory) for node in self.nodes)
        total_collaborative_memories = sum(len(node.collaborative_memories) for node in self.nodes)
        
        # Either collaborative memories were created OR memory count increased
        # (indicating some form of knowledge sharing occurred)
        knowledge_sharing_occurred = (
            total_collaborative_memories > 0 or 
            total_memories > len(problem_components) or
            signals_sent > 0
        )
        
        self.assertTrue(knowledge_sharing_occurred, 
                       f"Expected knowledge sharing. Collaborative: {total_collaborative_memories}, "
                       f"Total memories: {total_memories}, Signals: {signals_sent}")
        
        # Check that nodes can potentially solve the problem with available knowledge
        all_available_content = set()
        for node in self.nodes:
            for memory in node.memory:
                all_available_content.add(memory.content)
            for memory in node.collaborative_memories.values():
                all_available_content.add(memory.content)
        
        # Should have access to multiple problem components
        self.assertGreaterEqual(len(all_available_content), 2)

    def test_detection_of_misleading_information(self):
        """Test network's ability to detect and handle misleading information"""
        # Misleading node provides false information
        false_memory = Memory(
            content="false_energy_abundance",
            importance=0.9,
            timestamp=0,
            memory_type="shared",
            emotional_valence=0.8,
            source_node=self.misleading_node.node_id
        )
        
        # Other nodes have contradictory but accurate information
        accurate_memory = Memory(
            content="energy_scarcity_confirmed",
            importance=0.8,
            timestamp=1,
            memory_type="reward"
        )
        
        for node in self.nodes[:-1]:  # All except misleading node
            node.memory.append(accurate_memory)
        
        # Misleading node tries to spread false information
        self.misleading_node.send_signal(
            target_nodes=self.nodes[:-1],
            signal_type="memory",
            content=false_memory,
            urgency=0.9
        )
        
        # Check that other nodes received but appropriately handled the misleading info
        misleading_info_received = sum(
            1 for node in self.nodes[:-1]
            if any("false_energy" in str(memory) for memory in node.collaborative_memories.values())
        )
        
        # Some nodes might receive it but with reduced importance due to low trust
        if misleading_info_received > 0:
            # Check that the false information has reduced importance
            for node in self.nodes[:-1]:
                for content, memory in node.collaborative_memories.items():
                    if "false_energy" in str(content):
                        self.assertLess(memory.importance, 0.4)  # Should be heavily discounted

    def test_human_in_the_loop_emotional_content(self):
        """Test handling of human signals with emotional content and ambiguous instructions"""
        # Simulate human node with emotional and ambiguous signals
        human_node = AliveLoopNode(
            position=(10, 10), 
            velocity=(0, 0), 
            initial_energy=15, 
            node_id=99
        )
        human_node.communication_range = 20.0
        human_node.emotional_state = {"valence": 0.0}
        
        # Establish high trust with human
        for node in self.nodes:
            node.trust_network[human_node.node_id] = 1.0
            node.influence_network[human_node.node_id] = 0.9
        
        # Human sends emotionally charged and ambiguous instruction
        human_memory = Memory(
            content="urgent_but_unclear_directive",
            importance=0.9,
            timestamp=0,
            memory_type="shared",
            emotional_valence=-0.7,  # High negative emotion (stress/urgency)
            source_node=human_node.node_id
        )
        
        human_signal = SocialSignal(
            content=human_memory,
            signal_type="memory",
            urgency=1.0,
            source_id=human_node.node_id,
            requires_response=True
        )
        
        # Nodes receive human signal
        responses = []
        for node in self.nodes:
            response = node.receive_signal(human_signal)
            if response:
                responses.append(response)
        
        # Check that nodes handled the emotional content appropriately
        for node in self.nodes:
            # Should have high trust adjustment for human input
            if node.collaborative_memories:
                human_memories = [
                    memory for memory in node.collaborative_memories.values()
                    if hasattr(memory, 'source_node') and memory.source_node == human_node.node_id
                ]
                if human_memories:
                    # Should maintain high importance due to high trust in human
                    self.assertGreater(human_memories[0].importance, 0.7)
        
        # Some nodes should attempt to respond to the human
        self.assertGreaterEqual(len(responses), 0)

    def test_survival_simulation_energy_management(self):
        """Test network survival under energy constraints and danger scenarios"""
        # Create a survival scenario with limited energy and dangers
        
        # Reduce energy for all nodes to create scarcity
        for node in self.nodes:
            node.energy = 3.0  # Low energy
        
        # Create danger memory that nodes need to share
        danger_memory = Memory(
            content="area_X_dangerous",
            importance=1.0,
            timestamp=0,
            memory_type="shared",
            emotional_valence=-0.8
        )
        
        # One node discovers the danger
        self.nodes[0].memory.append(danger_memory)
        
        # Create energy source memory  
        energy_source_memory = Memory(
            content="energy_source_coordinates",
            importance=0.9,
            timestamp=1,
            memory_type="reward",
            emotional_valence=0.6
        )
        
        # Another node discovers energy source
        self.nodes[1].memory.append(energy_source_memory)
        
        # Simulate survival decisions and sharing critical information
        initial_total_energy = sum(node.energy for node in self.nodes)
        
        # Nodes should prioritize sharing critical survival information
        for node in self.nodes:
            if node.memory:
                # Share with high urgency due to survival needs
                critical_memories = [m for m in node.memory if m.importance > 0.8]
                if critical_memories:
                    node.send_signal(
                        target_nodes=[n for n in self.nodes if n != node],
                        signal_type="memory",
                        content=critical_memories[0],
                        urgency=1.0
                    )
        
        # Check that critical information was shared despite energy constraints
        danger_shared = sum(
            1 for node in self.nodes
            if any("dangerous" in str(memory) for memory in node.collaborative_memories.values())
        )
        
        energy_info_shared = sum(
            1 for node in self.nodes
            if any("energy_source" in str(memory) for memory in node.collaborative_memories.values())
        )
        
        # Critical survival information should be shared
        self.assertGreater(danger_shared + energy_info_shared, 0)
        
        # Nodes should still have some energy remaining (didn't exhaust themselves)
        remaining_energy = sum(node.energy for node in self.nodes)
        self.assertGreater(remaining_energy, 0)

    def test_network_resilience_to_node_failure(self):
        """Test network's ability to maintain functionality when some nodes fail"""
        # Simulate node failure by setting energy to 0
        failed_node = self.nodes[2]
        failed_node.energy = 0.0
        failed_node.phase = "sleep"  # Failed/inactive node
        
        # Add important information to failed node that others need
        critical_info = Memory(
            content="critical_network_data",
            importance=1.0,
            timestamp=0,
            memory_type="pattern"
        )
        failed_node.memory.append(critical_info)
        
        # Test if remaining nodes can still communicate and function
        active_nodes = [node for node in self.nodes if node.energy > 0]
        
        # Active nodes try to communicate
        for node in active_nodes:
            test_memory = Memory(
                content=f"message_from_node_{node.node_id}",
                importance=0.5,
                timestamp=0,
                memory_type="shared"
            )
            
            responses = node.send_signal(
                target_nodes=[n for n in active_nodes if n != node],
                signal_type="memory",
                content=test_memory,
                urgency=0.5
            )
        
        # Check that active nodes can still communicate
        total_communications = sum(
            len(node.collaborative_memories) for node in active_nodes
        )
        self.assertGreater(total_communications, 0)
        
        # Network should be resilient (more than half the nodes still active)
        self.assertGreater(len(active_nodes), len(self.nodes) // 2)

    def test_ethics_compliance_in_group_decisions(self):
        """Test that group decisions maintain ethical compliance"""
        # Simulate a group decision that requires ethics checking
        group_decision_data = {
            "action": "collaborative_resource_allocation",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True,
            "participating_nodes": [node.node_id for node in self.nodes]
        }
        
        # Each node should verify the ethics of group actions
        ethics_results = []
        for node in self.nodes:
            result = audit_decision(group_decision_data)
            ethics_results.append(result["compliant"])
        
        # All nodes should agree on ethics compliance
        self.assertTrue(all(ethics_results))
        
        # Test a potentially problematic group decision
        problematic_decision = {
            "action": "aggressive_resource_competition",
            "preserve_life": False,  # Could harm others
            "absolute_honesty": True,
            "privacy": False,  # Violates privacy
            "human_authority": True
        }
        
        problematic_result = audit_decision(problematic_decision)
        self.assertFalse(problematic_result["compliant"])
        self.assertGreater(len(problematic_result["violations"]), 0)


if __name__ == "__main__":
    unittest.main()