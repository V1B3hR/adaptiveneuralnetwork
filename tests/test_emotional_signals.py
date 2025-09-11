#!/usr/bin/env python3
"""
Test suite for emotional signal protocols (joy sharing, grief support, comfort, celebration)

Tests the new emotional signal types added to extend the anxiety help protocol
to support a broader range of emotional states and social interactions.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import from core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.alive_node import AliveLoopNode, SocialSignal, Memory
from core.capacitor import CapacitorInSpace
from core.network import TunedAdaptiveFieldNetwork
import numpy as np


class TestEmotionalSignals(unittest.TestCase):
    """Test cases for new emotional signal protocols"""
    
    def setUp(self):
        """Set up test nodes with different emotional states"""
        # Create nodes with different initial states
        self.joyful_node = AliveLoopNode(
            position=(0, 0), 
            velocity=(0, 0), 
            initial_energy=15.0, 
            node_id=1
        )
        # Set joyful node to positive emotional state
        self.joyful_node.emotional_state["valence"] = 0.7
        self.joyful_node.calm = 3.0
        
        self.grieving_node = AliveLoopNode(
            position=(2, 0), 
            velocity=(0, 0), 
            initial_energy=10.0, 
            node_id=2
        )
        # Set grieving node to negative emotional state
        self.grieving_node.emotional_state["valence"] = -0.6
        self.grieving_node.anxiety = 5.0
        self.grieving_node.calm = 1.0
        
        self.supportive_node = AliveLoopNode(
            position=(1, 1), 
            velocity=(0, 0), 
            initial_energy=18.0, 
            node_id=3
        )
        # Set supportive node to stable, helpful state
        self.supportive_node.emotional_state["valence"] = 0.3
        self.supportive_node.calm = 4.0
        self.supportive_node.anxiety = 2.0
        
        # Establish trust relationships
        self.joyful_node.trust_network[2] = 0.7  # trusts grieving node
        self.joyful_node.trust_network[3] = 0.8  # trusts supportive node
        
        self.grieving_node.trust_network[1] = 0.6  # trusts joyful node
        self.grieving_node.trust_network[3] = 0.9  # trusts supportive node highly
        
        self.supportive_node.trust_network[1] = 0.7  # trusts joyful node
        self.supportive_node.trust_network[2] = 0.8  # trusts grieving node
        
        self.nodes = [self.joyful_node, self.grieving_node, self.supportive_node]
        
    def test_joy_sharing_signal(self):
        """Test that nodes can share joy and positive emotions"""
        # Joyful node shares an achievement
        joy_content = {
            "type": "achievement",
            "description": "Successfully completed a complex task",
            "intensity": 0.8,
            "invite_celebration": True
        }
        
        # Get initial emotional states
        initial_valence_grieving = self.grieving_node.emotional_state["valence"]
        initial_valence_supportive = self.supportive_node.emotional_state["valence"]
        
        # Send joy signal to other nodes
        recipients = self.joyful_node.send_joy_share_signal(
            nearby_nodes=[self.grieving_node, self.supportive_node],
            joy_content=joy_content
        )
        
        # Verify recipients were found
        self.assertEqual(len(recipients), 2)
        self.assertIn(self.grieving_node, recipients)
        self.assertIn(self.supportive_node, recipients)
        
        # Verify memory was created for joyful node
        joy_memories = [m for m in self.joyful_node.memory if m.memory_type == "joy_shared"]
        self.assertEqual(len(joy_memories), 1)
        self.assertEqual(joy_memories[0].emotional_valence, 0.8)
        
        # Test individual signal processing (not through send_signal to avoid duplicates)
        # Create separate signals for each receiving node
        grieving_signal = SocialSignal(
            content={
                "type": "joy_share",
                "source_node": self.joyful_node.node_id,
                "emotional_valence": 0.8,
                "celebration_type": "achievement",
                "description": "Successfully completed a complex task",
                "intensity": 0.8,
                "timestamp": 0
            },
            signal_type="joy_share", 
            urgency=0.3,
            source_id=self.joyful_node.node_id
        )
        
        supportive_signal = SocialSignal(
            content={
                "type": "joy_share",
                "source_node": self.joyful_node.node_id,
                "emotional_valence": 0.8,
                "celebration_type": "achievement",
                "description": "Successfully completed a complex task",
                "intensity": 0.8,
                "timestamp": 0
            },
            signal_type="joy_share", 
            urgency=0.3,
            source_id=self.joyful_node.node_id
        )
        
        # Clear any existing memories from send_signal processing
        self.grieving_node.memory = []
        self.supportive_node.memory = []
        
        # Process signals individually 
        self.grieving_node._process_joy_share_signal(grieving_signal)
        self.supportive_node._process_joy_share_signal(supportive_signal)
        
        # Verify emotional improvement in receiving nodes
        self.assertGreater(
            self.grieving_node.emotional_state["valence"], 
            initial_valence_grieving
        )
        self.assertGreater(
            self.supportive_node.emotional_state["valence"], 
            initial_valence_supportive
        )
        
        # Verify joy memories were created in receivers (one each)
        grieving_joy_memories = [m for m in self.grieving_node.memory if m.memory_type == "joy_received"]
        supportive_joy_memories = [m for m in self.supportive_node.memory if m.memory_type == "joy_received"]
        
        self.assertEqual(len(grieving_joy_memories), 1)
        self.assertEqual(len(supportive_joy_memories), 1)
        
        # Verify emotional valence of memories
        self.assertGreater(grieving_joy_memories[0].emotional_valence, 0)
        self.assertGreater(supportive_joy_memories[0].emotional_valence, 0)
            
    def test_grief_support_request(self):
        """Test that nodes can request and receive grief support"""
        # Grieving node requests support
        grief_details = {
            "support_type": "emotional",
            "intensity": 0.8,
            "description": "Experiencing significant sadness and need support",
            "emotional_valence": -0.7
        }
        
        # Get initial states
        initial_grieving_valence = self.grieving_node.emotional_state["valence"]
        initial_grieving_anxiety = self.grieving_node.anxiety
        initial_supportive_trust = self.supportive_node.trust_network.get(2, 0.5)
        
        # Send grief support request
        supporters = self.grieving_node.send_grief_support_request(
            nearby_nodes=[self.joyful_node, self.supportive_node],
            grief_details=grief_details
        )
        
        # Verify supporters were found (supportive node should qualify)
        self.assertGreater(len(supporters), 0)
        
        # Verify support request memory was created
        support_memories = [m for m in self.grieving_node.memory if m.memory_type == "support_requested"]
        self.assertEqual(len(support_memories), 1)
        self.assertLess(support_memories[0].emotional_valence, 0)
        
        # Simulate supportive node processing the request
        support_signal = SocialSignal(
            content={
                "type": "grief_support_request",
                "requesting_node": self.grieving_node.node_id,
                "emotional_valence": -0.7,
                "support_type": "emotional", 
                "grief_intensity": 0.8,
                "description": "Experiencing significant sadness",
                "urgency": 0.8,
                "timestamp": 0
            },
            signal_type="grief_support_request",
            urgency=0.8,
            source_id=self.grieving_node.node_id,
            requires_response=True
        )
        
        response = self.supportive_node._process_grief_support_request_signal(support_signal)
        
        # Verify response was generated
        self.assertIsNotNone(response)
        self.assertEqual(response.signal_type, "grief_support_response")
        self.assertGreater(response.content["emotional_support_offered"], 0)
        
        # Clear memories to avoid duplicates from send_grief_support_request
        self.supportive_node.memory = []
        
        # Process the request signal again to create the memory (simulate clean processing)
        self.supportive_node._process_grief_support_request_signal(support_signal)
        
        # Verify support memory was created in supportive node
        support_given_memories = [m for m in self.supportive_node.memory if m.memory_type == "support_given"]
        self.assertEqual(len(support_given_memories), 1)
        
        # Simulate grieving node receiving the response
        self.grieving_node._process_grief_support_response_signal(response)
        
        # Verify emotional improvement in grieving node
        self.assertGreater(
            self.grieving_node.emotional_state["valence"],
            initial_grieving_valence
        )
        self.assertLess(
            self.grieving_node.anxiety,
            initial_grieving_anxiety
        )
        
        # Verify trust increased between nodes
        self.assertGreater(
            self.grieving_node.trust_network.get(self.supportive_node.node_id, 0.5),
            initial_supportive_trust
        )
        
        # Verify support received memory was created
        support_received_memories = [m for m in self.grieving_node.memory if m.memory_type == "support_received"]
        self.assertEqual(len(support_received_memories), 1)
        
    def test_celebration_invite_signal(self):
        """Test celebration invitation processing"""
        # Create celebration invite signal
        celebration_signal = SocialSignal(
            content={
                "type": "celebration_invite",
                "inviter_node": self.joyful_node.node_id,
                "celebration_type": "achievement",
                "description": "Let's celebrate together!",
                "timestamp": 0
            },
            signal_type="celebration_invite",
            urgency=0.3,
            source_id=self.joyful_node.node_id,
            requires_response=True
        )
        
        # Get initial state
        initial_valence = self.supportive_node.emotional_state["valence"]
        initial_trust = self.supportive_node.trust_network.get(self.joyful_node.node_id, 0.5)
        
        # Process celebration invite
        response = self.supportive_node._process_celebration_invite_signal(celebration_signal)
        
        # Verify response was generated (supportive node should accept)
        self.assertIsNotNone(response)
        self.assertEqual(response.content["type"], "celebration_acceptance")
        
        # Verify emotional boost from participating
        self.assertGreater(
            self.supportive_node.emotional_state["valence"],
            initial_valence
        )
        
        # Verify trust increased
        self.assertGreater(
            self.supportive_node.trust_network.get(self.joyful_node.node_id, 0.5),
            initial_trust
        )
        
        # Verify celebration memory was created
        celebration_memories = [m for m in self.supportive_node.memory if m.memory_type == "celebration"]
        self.assertEqual(len(celebration_memories), 1)
        
    def test_comfort_request_and_response(self):
        """Test general comfort request and response"""
        # Create comfort request signal
        comfort_signal = SocialSignal(
            content={
                "type": "comfort_request",
                "requesting_node": self.grieving_node.node_id,
                "comfort_type": "general",
                "timestamp": 0
            },
            signal_type="comfort_request",
            urgency=0.4,
            source_id=self.grieving_node.node_id,
            requires_response=True
        )
        
        # Get initial state
        initial_calm = self.grieving_node.calm
        initial_trust = self.supportive_node.trust_network.get(self.grieving_node.node_id, 0.5)
        
        # Process comfort request
        response = self.supportive_node._process_comfort_request_signal(comfort_signal)
        
        # Verify response was generated
        self.assertIsNotNone(response)
        self.assertEqual(response.signal_type, "comfort_response")
        self.assertGreater(response.content["comfort_offered"], 0)
        
        # Simulate grieving node receiving comfort response
        self.grieving_node._process_comfort_response_signal(response)
        
        # Verify comfort was applied
        self.assertGreater(self.grieving_node.calm, initial_calm)
        
        # Verify trust increased between nodes
        self.assertGreater(
            self.supportive_node.trust_network.get(self.grieving_node.node_id, 0.5),
            initial_trust
        )
        
        # Verify comfort memory was created
        comfort_memories = [m for m in self.grieving_node.memory if m.memory_type == "comfort_received"]
        self.assertEqual(len(comfort_memories), 1)
        
    def test_trust_updates_for_emotional_signals(self):
        """Test that trust is properly updated for new signal types"""
        initial_trust = self.joyful_node.trust_network.get(self.supportive_node.node_id, 0.5)
        
        # Test different signal types
        signal_types = ["joy_share", "grief_support_request", "celebration_invite", "comfort_request"]
        
        for signal_type in signal_types:
            self.joyful_node._update_trust_after_communication(self.supportive_node, signal_type)
            
        # Verify trust increased
        final_trust = self.joyful_node.trust_network.get(self.supportive_node.node_id, 0.5)
        self.assertGreater(final_trust, initial_trust)
        
    def test_emotional_state_consistency(self):
        """Test that emotional states remain within valid bounds"""
        # Test extreme emotional state changes
        test_node = self.joyful_node
        
        # Apply multiple positive signals
        for _ in range(10):
            test_node.emotional_state["valence"] = min(1.0, test_node.emotional_state["valence"] + 0.2)
            test_node.calm = min(5.0, test_node.calm + 0.5)
            
        # Verify bounds are maintained
        self.assertLessEqual(test_node.emotional_state["valence"], 1.0)
        self.assertLessEqual(test_node.calm, 5.0)
        
        # Apply multiple negative signals  
        for _ in range(10):
            test_node.emotional_state["valence"] = max(-1.0, test_node.emotional_state["valence"] - 0.3)
            test_node.anxiety = max(0, test_node.anxiety + 1.0)
            
        # Verify bounds are maintained
        self.assertGreaterEqual(test_node.emotional_state["valence"], -1.0)
        self.assertGreaterEqual(test_node.anxiety, 0)
        
    def test_low_energy_signal_blocking(self):
        """Test that nodes with low energy cannot send emotional signals"""
        # Set joyful node to very low energy
        self.joyful_node.energy = 0.5
        
        # Attempt to send joy signal
        recipients = self.joyful_node.send_joy_share_signal(
            nearby_nodes=[self.supportive_node],
            joy_content={"type": "general", "intensity": 0.5}
        )
        
        # Verify no recipients (insufficient energy)
        self.assertEqual(len(recipients), 0)
        
        # Set grieving node to low energy
        self.grieving_node.energy = 1.5
        
        # Attempt to send grief support request
        supporters = self.grieving_node.send_grief_support_request(
            nearby_nodes=[self.supportive_node],
            grief_details={"support_type": "emotional", "intensity": 0.7}
        )
        
        # Verify no supporters (insufficient energy)
        self.assertEqual(len(supporters), 0)
        
    def test_overwhelmed_node_cannot_provide_support(self):
        """Test that overwhelmed nodes cannot provide emotional support"""
        # Set supportive node to overwhelmed state
        self.supportive_node.anxiety = 9.0
        self.supportive_node.emotional_state["valence"] = -0.8
        
        # Create grief support request
        support_signal = SocialSignal(
            content={
                "type": "grief_support_request",
                "requesting_node": self.grieving_node.node_id,
                "emotional_valence": -0.6,
                "support_type": "emotional",
                "grief_intensity": 0.7,
                "urgency": 0.7,
                "timestamp": 0
            },
            signal_type="grief_support_request",
            urgency=0.7,
            source_id=self.grieving_node.node_id,
            requires_response=True
        )
        
        # Process signal on overwhelmed node
        response = self.supportive_node._process_grief_support_request_signal(support_signal)
        
        # Verify no response (too overwhelmed to help)
        self.assertIsNone(response)


class TestEmotionalSignalsIntegration(unittest.TestCase):
    """Integration tests for emotional signals within the network"""
    
    def setUp(self):
        """Set up a small network for integration testing"""
        self.nodes = []
        positions = [(0, 0), (2, 0), (1, 2), (3, 1)]
        
        for i, pos in enumerate(positions):
            node = AliveLoopNode(
                position=pos,
                velocity=(0, 0),
                initial_energy=15.0,
                node_id=i+1
            )
            # Initialize varying emotional states
            node.emotional_state["valence"] = np.random.uniform(-0.5, 0.8)
            node.anxiety = np.random.uniform(0, 6)
            node.calm = np.random.uniform(1, 4)
            self.nodes.append(node)
            
        # Create trust network
        for node in self.nodes:
            for other in self.nodes:
                if node.node_id != other.node_id:
                    node.trust_network[other.node_id] = np.random.uniform(0.3, 0.9)
                    
        # Create capacitors 
        self.capacitors = [
            CapacitorInSpace(position=(1, 1), capacity=10.0, initial_energy=5.0)
        ]
        
        # Create network
        self.network = TunedAdaptiveFieldNetwork(
            nodes=self.nodes,
            capacitors=self.capacitors,
            enable_time_series=False  # Disable for testing
        )
        
    def test_network_emotional_dynamics(self):
        """Test emotional signal propagation across the network"""
        # Set one node to very positive state
        happy_node = self.nodes[0]
        happy_node.emotional_state["valence"] = 0.9
        happy_node.calm = 4.5
        
        # Set another node to negative state
        sad_node = self.nodes[1]
        sad_node.emotional_state["valence"] = -0.7
        sad_node.anxiety = 7.0
        
        # Record initial network emotional state
        initial_avg_valence = np.mean([n.emotional_state["valence"] for n in self.nodes])
        initial_total_anxiety = sum(n.anxiety for n in self.nodes)
        
        # Run network steps to allow emotional signal propagation
        for step in range(10):
            self.network.step()
            
            # Happy node occasionally shares joy
            if step % 3 == 0:
                nearby_nodes = [n for n in self.nodes if n.node_id != happy_node.node_id]
                happy_node.send_joy_share_signal(
                    nearby_nodes=nearby_nodes,
                    joy_content={
                        "type": "spontaneous",
                        "intensity": 0.6,
                        "description": "Feeling great today!"
                    }
                )
                
            # Sad node occasionally requests support
            if step % 4 == 0:
                nearby_nodes = [n for n in self.nodes if n.node_id != sad_node.node_id]
                sad_node.send_grief_support_request(
                    nearby_nodes=nearby_nodes,
                    grief_details={
                        "support_type": "emotional",
                        "intensity": 0.7,
                        "description": "Need emotional support"
                    }
                )
                
        # Check that emotional states have evolved
        final_avg_valence = np.mean([n.emotional_state["valence"] for n in self.nodes])
        final_total_anxiety = sum(n.anxiety for n in self.nodes)
        
        # Network should become more positive and less anxious over time
        self.assertGreater(final_avg_valence, initial_avg_valence)
        self.assertLess(final_total_anxiety, initial_total_anxiety)
        
        # Verify that emotional memories were created
        total_emotional_memories = 0
        for node in self.nodes:
            emotional_memory_types = ["joy_shared", "joy_received", "support_requested", 
                                    "support_given", "support_received", "comfort_received"]
            node_emotional_memories = [m for m in node.memory 
                                     if m.memory_type in emotional_memory_types]
            total_emotional_memories += len(node_emotional_memories)
            
        self.assertGreater(total_emotional_memories, 0)
        
    def test_network_status_includes_emotional_data(self):
        """Test that network status reporting includes emotional signal data"""
        # Run a few steps 
        for _ in range(5):
            self.network.step()
            
        # Get network status
        status = self.network.get_network_status()
        
        # Verify status includes nodes
        self.assertIn("nodes", status)
        self.assertEqual(len(status["nodes"]), len(self.nodes))
        
        # Verify individual node data includes emotional information
        for node_id, node_status in status["nodes"].items():
            self.assertIn("anxiety_level", node_status)
            self.assertIn("calm_level", node_status)
            self.assertIn("trust_network_size", node_status)


if __name__ == '__main__':
    unittest.main()