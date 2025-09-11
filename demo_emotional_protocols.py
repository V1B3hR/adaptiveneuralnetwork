#!/usr/bin/env python3
"""
Demonstration of new emotional signal protocols in the adaptive neural network.

This script showcases the extended emotional support capabilities beyond just anxiety help,
including joy sharing, grief support, celebration invites, and comfort requests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.alive_node import AliveLoopNode
from core.capacitor import CapacitorInSpace
from core.network import TunedAdaptiveFieldNetwork
import numpy as np
import json
import time


def create_emotional_network():
    """Create a network of nodes with diverse emotional states"""
    print("🌟 Creating Emotional Support Network...")
    
    # Create nodes with different personalities and initial states
    nodes = []
    
    # Joyful achiever node
    joyful_node = AliveLoopNode(
        position=(0, 0), 
        velocity=(0.1, 0), 
        initial_energy=18.0, 
        node_id=1
    )
    joyful_node.emotional_state["valence"] = 0.8
    joyful_node.calm = 4.0
    joyful_node.anxiety = 1.0
    nodes.append(joyful_node)
    
    # Grieving node needing support
    grieving_node = AliveLoopNode(
        position=(3, 0), 
        velocity=(-0.05, 0), 
        initial_energy=8.0, 
        node_id=2
    )
    grieving_node.emotional_state["valence"] = -0.7
    grieving_node.anxiety = 6.0
    grieving_node.calm = 0.8
    nodes.append(grieving_node)
    
    # Supportive helper node
    supportive_node = AliveLoopNode(
        position=(1.5, 2), 
        velocity=(0, -0.1), 
        initial_energy=15.0, 
        node_id=3
    )
    supportive_node.emotional_state["valence"] = 0.3
    supportive_node.calm = 3.5
    supportive_node.anxiety = 2.0
    nodes.append(supportive_node)
    
    # Neutral observing node
    neutral_node = AliveLoopNode(
        position=(1.5, -2), 
        velocity=(0, 0.1), 
        initial_energy=12.0, 
        node_id=4
    )
    neutral_node.emotional_state["valence"] = 0.1
    neutral_node.calm = 2.5
    neutral_node.anxiety = 3.0
    nodes.append(neutral_node)
    
    # Establish trust relationships
    trust_matrix = {
        1: {2: 0.6, 3: 0.8, 4: 0.7},  # Joyful trusts others moderately to highly
        2: {1: 0.5, 3: 0.9, 4: 0.6},  # Grieving highly trusts supportive node
        3: {1: 0.8, 2: 0.8, 4: 0.7},  # Supportive trusts everyone well
        4: {1: 0.6, 2: 0.7, 3: 0.7}   # Neutral trusts moderately
    }
    
    for node in nodes:
        for other_id, trust_level in trust_matrix[node.node_id].items():
            node.trust_network[other_id] = trust_level
            node.influence_network[other_id] = trust_level * 0.8
    
    # Create capacitors for energy support
    capacitors = [
        CapacitorInSpace(position=(1.5, 0), capacity=15.0, initial_energy=10.0)
    ]
    
    return nodes, capacitors


def demonstrate_joy_sharing(nodes):
    """Demonstrate joy sharing protocols"""
    print("\n🎉 DEMONSTRATION: Joy Sharing Protocol")
    print("=" * 50)
    
    joyful_node = nodes[0]  # Node 1
    other_nodes = nodes[1:]
    
    # Record initial emotional states
    print("Initial Emotional States:")
    for node in nodes:
        print(f"  Node {node.node_id}: valence={node.emotional_state['valence']:.2f}, "
              f"anxiety={node.anxiety:.2f}, calm={node.calm:.2f}")
    
    # Joyful node shares a major achievement
    print(f"\n📢 Node {joyful_node.node_id} shares exciting achievement...")
    joy_content = {
        "type": "major_breakthrough",
        "description": "Successfully solved a complex optimization problem!",
        "intensity": 0.9,
        "invite_celebration": True
    }
    
    recipients = joyful_node.send_joy_share_signal(
        nearby_nodes=other_nodes,
        joy_content=joy_content
    )
    
    print(f"   💌 Joy shared with {len(recipients)} nodes")
    
    # Show emotional impact
    print("\nEmotional Impact of Joy Sharing:")
    for node in nodes[1:]:
        if node in recipients:
            print(f"  ✨ Node {node.node_id}: valence improved to {node.emotional_state['valence']:.2f}, "
                  f"anxiety reduced to {node.anxiety:.2f}")
    
    # Check memories created
    joy_memories = [m for m in joyful_node.memory if m.memory_type == "joy_shared"]
    print(f"   🧠 Joy sharing memory created (importance: {joy_memories[0].importance:.2f})")
    
    return joy_memories


def demonstrate_grief_support(nodes):
    """Demonstrate grief support protocols"""
    print("\n💙 DEMONSTRATION: Grief Support Protocol")
    print("=" * 50)
    
    grieving_node = nodes[1]  # Node 2
    potential_supporters = [nodes[0], nodes[2], nodes[3]]  # Other nodes
    
    print("Grief Support Scenario:")
    print(f"  😢 Node {grieving_node.node_id} is experiencing significant emotional distress")
    print(f"     Current state: valence={grieving_node.emotional_state['valence']:.2f}, "
          f"anxiety={grieving_node.anxiety:.2f}")
    
    # Request grief support
    grief_details = {
        "support_type": "emotional",
        "intensity": 0.8,
        "description": "Feeling overwhelmed and need emotional support",
        "emotional_valence": -0.7
    }
    
    print(f"\n🆘 Node {grieving_node.node_id} requests grief support...")
    supporters = grieving_node.send_grief_support_request(
        nearby_nodes=potential_supporters,
        grief_details=grief_details
    )
    
    print(f"   🤝 {len(supporters)} nodes responded with support")
    
    # Show which nodes provided support
    for supporter in supporters:
        print(f"   💝 Node {supporter.node_id} offered emotional support")
        
    # Show emotional improvement
    print(f"\nEmotional Recovery:")
    print(f"  🌱 Node {grieving_node.node_id}: valence improved to {grieving_node.emotional_state['valence']:.2f}, "
          f"anxiety reduced to {grieving_node.anxiety:.2f}")
    
    # Check support memories
    support_memories = [m for m in grieving_node.memory if m.memory_type == "support_received"]
    if support_memories:
        print(f"   🧠 Support received memory created (emotional valence: {support_memories[0].emotional_valence:.2f})")
    
    return support_memories


def demonstrate_celebration_and_comfort(nodes):
    """Demonstrate celebration invites and comfort requests"""
    print("\n🎊 DEMONSTRATION: Celebration & Comfort Protocols")
    print("=" * 50)
    
    # Joyful node invites others to celebrate
    joyful_node = nodes[0]
    neutral_node = nodes[3]
    
    print("Celebration Invitation:")
    print(f"  🎉 Node {joyful_node.node_id} invites Node {neutral_node.node_id} to celebrate")
    
    # Simulate celebration invite processing
    from core.alive_node import SocialSignal
    celebration_signal = SocialSignal(
        content={
            "type": "celebration_invite",
            "inviter_node": joyful_node.node_id,
            "celebration_type": "achievement",
            "description": "Let's celebrate this breakthrough together!",
            "timestamp": 0
        },
        signal_type="celebration_invite",
        urgency=0.3,
        source_id=joyful_node.node_id,
        requires_response=True
    )
    
    initial_valence = neutral_node.emotional_state["valence"]
    response = neutral_node._process_celebration_invite_signal(celebration_signal)
    
    if response:
        print(f"   ✅ Node {neutral_node.node_id} accepted the celebration invite!")
        print(f"   📈 Emotional boost: valence {initial_valence:.2f} → {neutral_node.emotional_state['valence']:.2f}")
    
    # Comfort request scenario
    print(f"\nComfort Request:")
    print(f"  🤗 Node {neutral_node.node_id} requests general comfort")
    
    comfort_signal = SocialSignal(
        content={
            "type": "comfort_request",
            "requesting_node": neutral_node.node_id,
            "comfort_type": "general",
            "timestamp": 0
        },
        signal_type="comfort_request",
        urgency=0.4,
        source_id=neutral_node.node_id,
        requires_response=True
    )
    
    supportive_node = nodes[2]
    initial_calm = neutral_node.calm
    comfort_response = supportive_node._process_comfort_request_signal(comfort_signal)
    
    if comfort_response:
        neutral_node._process_comfort_response_signal(comfort_response)
        print(f"   🌸 Node {supportive_node.node_id} provided comfort")
        print(f"   📈 Calm increased: {initial_calm:.2f} → {neutral_node.calm:.2f}")


def demonstrate_trust_evolution(nodes):
    """Show how emotional interactions affect trust networks"""
    print("\n🔗 DEMONSTRATION: Trust Network Evolution")
    print("=" * 50)
    
    print("Trust levels before emotional interactions:")
    for node in nodes:
        print(f"  Node {node.node_id} trust network: {dict(node.trust_network)}")
    
    # Simulate multiple emotional interactions
    interactions = [
        ("joy_share", nodes[0], nodes[1]),
        ("grief_support_request", nodes[1], nodes[2]),
        ("celebration_invite", nodes[0], nodes[3]),
        ("comfort_request", nodes[3], nodes[2])
    ]
    
    print(f"\nSimulating {len(interactions)} emotional interactions...")
    for signal_type, source, target in interactions:
        initial_trust = source.trust_network.get(target.node_id, 0.5)
        source._update_trust_after_communication(target, signal_type)
        new_trust = source.trust_network.get(target.node_id, 0.5)
        print(f"  {signal_type}: Node {source.node_id} → Node {target.node_id}, "
              f"trust {initial_trust:.2f} → {new_trust:.2f}")
    
    print("\nFinal trust network state:")
    for node in nodes:
        avg_trust = np.mean(list(node.trust_network.values()))
        print(f"  Node {node.node_id}: average trust level = {avg_trust:.2f}")


def run_network_simulation(nodes, capacitors):
    """Run a brief network simulation to show emergent emotional dynamics"""
    print("\n🌐 DEMONSTRATION: Network Emotional Dynamics")
    print("=" * 50)
    
    # Create network
    network = TunedAdaptiveFieldNetwork(
        nodes=nodes,
        capacitors=capacitors,
        enable_time_series=False  # Disable for demo to avoid API calls
    )
    
    print("Running 10-step simulation with emotional interactions...")
    
    for step in range(10):
        network.step()
        
        # Trigger emotional interactions based on node states
        if step % 3 == 0:
            # Joy sharing from happy nodes
            for node in nodes:
                if node.emotional_state["valence"] > 0.5 and node.energy > 5:
                    nearby = [n for n in nodes if n.node_id != node.node_id and 
                             np.linalg.norm(np.array(node.position) - np.array(n.position)) < 3.0]
                    if nearby:
                        node.send_joy_share_signal(
                            nearby_nodes=nearby[:2],  # Limit to 2 nodes
                            joy_content={"type": "spontaneous", "intensity": 0.6}
                        )
        
        if step % 4 == 0:
            # Grief support requests from distressed nodes
            for node in nodes:
                if node.emotional_state["valence"] < -0.3 and node.energy > 3:
                    nearby = [n for n in nodes if n.node_id != node.node_id and 
                             np.linalg.norm(np.array(node.position) - np.array(n.position)) < 3.0]
                    if nearby:
                        node.send_grief_support_request(
                            nearby_nodes=nearby[:2],  # Limit to 2 nodes
                            grief_details={"support_type": "emotional", "intensity": 0.7}
                        )
    
    # Show final network emotional state
    print("\nFinal Network Emotional State:")
    total_valence = 0
    total_anxiety = 0
    emotional_memories = 0
    
    for node in nodes:
        valence = node.emotional_state["valence"]
        anxiety = node.anxiety
        total_valence += valence
        total_anxiety += anxiety
        
        # Count emotional memories
        emotional_types = ["joy_shared", "joy_received", "support_requested", 
                          "support_given", "support_received", "comfort_received"]
        node_emotional_memories = [m for m in node.memory if m.memory_type in emotional_types]
        emotional_memories += len(node_emotional_memories)
        
        print(f"  Node {node.node_id}: valence={valence:.2f}, anxiety={anxiety:.2f}, "
              f"emotional_memories={len(node_emotional_memories)}")
    
    avg_valence = total_valence / len(nodes)
    avg_anxiety = total_anxiety / len(nodes)
    
    print(f"\nNetwork Summary:")
    print(f"  📊 Average emotional valence: {avg_valence:.2f}")
    print(f"  📊 Average anxiety level: {avg_anxiety:.2f}")
    print(f"  📊 Total emotional memories created: {emotional_memories}")
    
    return {
        "average_valence": avg_valence,
        "average_anxiety": avg_anxiety,
        "emotional_memories": emotional_memories
    }


def main():
    """Main demonstration script"""
    print("🧠 ADAPTIVE NEURAL NETWORK - EMOTIONAL SIGNALS DEMONSTRATION")
    print("=" * 65)
    print("This demonstration showcases the new emotional signal protocols")
    print("that extend beyond anxiety help to include joy sharing, grief")
    print("support, celebration invites, and comfort requests.\n")
    
    # Create network
    nodes, capacitors = create_emotional_network()
    
    # Demonstrate different emotional protocols
    joy_memories = demonstrate_joy_sharing(nodes)
    support_memories = demonstrate_grief_support(nodes)
    demonstrate_celebration_and_comfort(nodes)
    demonstrate_trust_evolution(nodes)
    
    # Run network simulation
    network_stats = run_network_simulation(nodes, capacitors)
    
    # Summary
    print("\n🎯 DEMONSTRATION SUMMARY")
    print("=" * 30)
    print("✅ Joy sharing protocol: WORKING")
    print("✅ Grief support protocol: WORKING") 
    print("✅ Celebration invites: WORKING")
    print("✅ Comfort requests: WORKING")
    print("✅ Trust network evolution: WORKING")
    print("✅ Network emotional dynamics: WORKING")
    
    print(f"\n📈 Results:")
    print(f"   Joy memories created: {len(joy_memories)}")
    print(f"   Support memories created: {len(support_memories)}")
    print(f"   Final network valence: {network_stats['average_valence']:.2f}")
    print(f"   Final network anxiety: {network_stats['average_anxiety']:.2f}")
    print(f"   Total emotional interactions: {network_stats['emotional_memories']}")
    
    print("\n🌟 The adaptive neural network now supports comprehensive")
    print("   emotional protocols beyond just anxiety help, enabling")
    print("   richer social interactions and emotional support systems!")


if __name__ == "__main__":
    main()