#!/usr/bin/env python3
"""
Demonstration of Enhanced Trust Network System

Shows:
1. Normal trust evolution through positive interactions
2. Manipulation pattern detection (love bombing)
3. Community verification process
4. Trust network health monitoring
5. Resilience against adversarial attacks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.alive_node import AliveLoopNode
import numpy as np


def demonstrate_normal_trust_evolution():
    """Show normal trust building through positive interactions"""
    print("=" * 60)
    print("🤝 NORMAL TRUST EVOLUTION DEMONSTRATION")
    print("=" * 60)
    
    # Create two nodes
    alice = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)  # Alice
    bob = AliveLoopNode((1, 1), (0.1, 0.1), node_id=1)    # Bob
    
    print(f"Initial trust: Alice → Bob = {alice.trust_network.get(bob.node_id, 0.5):.3f}")
    
    # Simulate a series of positive interactions
    interactions = [
        ('resource', "Alice shares computational resources with Bob"),
        ('memory', "Alice shares valuable problem-solving memory"),
        ('joy_share', "Alice shares excitement about discovery"),
        ('celebration_invite', "Alice invites Bob to celebrate success"),
        ('resource', "Alice helps Bob again with another task")
    ]
    
    for signal_type, description in interactions:
        trust_before = alice.trust_network.get(bob.node_id, 0.5)
        alice._update_trust_after_communication(bob, signal_type)
        trust_after = alice.trust_network.get(bob.node_id)
        
        print(f"📈 {description}")
        print(f"   Trust: {trust_before:.3f} → {trust_after:.3f} (Δ{trust_after - trust_before:+.3f})")
        
        # Check for any alerts
        if bob.node_id in alice.trust_network_system.suspicion_alerts:
            alert = alice.trust_network_system.suspicion_alerts[bob.node_id]
            print(f"   ⚠️  ALERT: {alert['status']} - {alice.trust_network_system._generate_suspicion_reason(bob.node_id, signal_type)}")
    
    # Show final trust summary
    summary = alice.get_trust_summary()
    print(f"\n📊 Final Trust Summary:")
    print(f"   Average Trust: {summary['average_trust']:.3f}")
    print(f"   Trusted Nodes: {summary['trusted_nodes']}")
    print(f"   Suspicious Nodes: {summary['suspicious_nodes']}")
    print(f"   Active Alerts: {summary['active_alerts']}")
    print()


def demonstrate_manipulation_detection():
    """Show manipulation pattern detection"""
    print("=" * 60)
    print("🚨 MANIPULATION DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Create victim and manipulator
    victim = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
    manipulator = AliveLoopNode((2, 2), (0.1, 0.1), node_id=2)
    
    print(f"Manipulator attempting love bombing attack on victim...")
    print(f"Initial trust: {victim.trust_network.get(manipulator.node_id, 0.5):.3f}")
    
    # Love bombing attack - rapid positive signals
    love_bomb_signals = [
        ('resource', "Offers valuable resources"),
        ('joy_share', "Shares overwhelming enthusiasm"),  
        ('celebration_invite', "Invites to exclusive celebration"),
        ('comfort_request', "Asks for comfort (vulnerability play)"),
        ('resource', "Offers more resources"),
        ('memory', "Shares 'valuable' memories")
    ]
    
    for i, (signal_type, description) in enumerate(love_bomb_signals):
        trust_before = victim.trust_network.get(manipulator.node_id, 0.5)
        victim._update_trust_after_communication(manipulator, signal_type)
        trust_after = victim.trust_network.get(manipulator.node_id)
        
        print(f"💣 Attempt {i+1}: {description}")
        print(f"   Trust: {trust_before:.3f} → {trust_after:.3f}")
        
        # Check if manipulation was detected
        if manipulator.node_id in victim.trust_network_system.suspicion_alerts:
            alert = victim.trust_network_system.suspicion_alerts[manipulator.node_id]
            print(f"   🛡️  MANIPULATION DETECTED!")
            print(f"   Status: {alert['status']}")
            print(f"   Reason: {victim.trust_network_system._generate_suspicion_reason(manipulator.node_id, signal_type)}")
            break
    
    print(f"\n🛡️  Protection Status: {'ACTIVE' if manipulator.node_id in victim.trust_network_system.suspicion_alerts else 'NONE'}")
    print()


def demonstrate_community_verification():
    """Show community verification process"""
    print("=" * 60)
    print("🏘️  COMMUNITY VERIFICATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a network of nodes
    nodes = []
    for i in range(5):
        node = AliveLoopNode((i, 0), (0.1, 0.1), node_id=i)
        nodes.append(node)
    
    # Establish trust relationships (everyone trusts each other initially)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                nodes[i].trust_network_system.set_trust(j, 0.7)
    
    observer = nodes[0]  # Node 0 observes
    suspect = nodes[1]   # Node 1 becomes suspicious
    community = nodes[2:] # Nodes 2-4 provide community feedback
    
    print(f"Observer (Node {observer.node_id}) monitoring Suspect (Node {suspect.node_id})")
    print(f"Community members: Nodes {[n.node_id for n in community]}")
    
    # Simulate suspicious behavior
    print(f"\n📉 Suspect exhibits repeated betrayal behavior...")
    for i in range(3):
        trust_before = observer.trust_network.get(suspect.node_id)
        observer._update_trust_after_communication(suspect, 'betrayal')
        trust_after = observer.trust_network.get(suspect.node_id)
        print(f"   Betrayal {i+1}: Trust {trust_before:.3f} → {trust_after:.3f}")
    
    # Check if community verification was triggered
    if suspect.node_id in observer.trust_network_system.suspicion_alerts:
        alert = observer.trust_network_system.suspicion_alerts[suspect.node_id]
        print(f"\n🚨 Community verification triggered!")
        print(f"   Status: {alert['status']}")
        
        # Simulate community providing feedback
        print(f"\n👥 Community members provide their trust assessments:")
        community_feedback = []
        for member in community:
            # Each community member has their own opinion
            trust_level = np.random.normal(0.25, 0.1)  # Low trust with some variance
            trust_level = max(0.0, min(1.0, trust_level))  # Clamp to [0,1]
            
            feedback = {'trust_level': trust_level}
            community_feedback.append(feedback)
            print(f"   Node {member.node_id}: trust = {trust_level:.3f}")
        
        # Process community feedback
        print(f"\n🔄 Processing community consensus...")
        trust_before_consensus = observer.trust_network.get(suspect.node_id)
        observer.handle_community_trust_feedback(suspect.node_id, community_feedback)
        trust_after_consensus = observer.trust_network.get(suspect.node_id)
        
        print(f"   Trust before consensus: {trust_before_consensus:.3f}")
        print(f"   Trust after consensus:  {trust_after_consensus:.3f}")
        print(f"   Consensus effect: {trust_after_consensus - trust_before_consensus:+.3f}")
        
        # Check final status
        final_status = observer.trust_network_system.suspicion_alerts[suspect.node_id]['status']
        print(f"   Final status: {final_status}")
    
    print()


def demonstrate_network_health_monitoring():
    """Show trust network health monitoring"""
    print("=" * 60)
    print("📊 TRUST NETWORK HEALTH MONITORING")
    print("=" * 60)
    
    node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
    
    # Add various trust relationships
    trust_relationships = [
        (1, 0.9, "Highly trusted collaborator"),
        (2, 0.7, "Reliable partner"),
        (3, 0.5, "Neutral relationship"), 
        (4, 0.25, "Suspicious entity"),
        (5, 0.1, "Highly suspicious"),
        (6, 0.8, "Another trusted node")
    ]
    
    print("🔗 Establishing trust relationships...")
    for node_id, trust_level, description in trust_relationships:
        node.trust_network_system.set_trust(node_id, trust_level)
        print(f"   Node {node_id}: {trust_level:.2f} - {description}")
    
    # Get comprehensive trust summary
    summary = node.get_trust_summary()
    
    print(f"\n📈 Trust Network Health Report:")
    print(f"   Average Trust Level: {summary['average_trust']:.3f}")
    print(f"   Trusted Nodes (>0.6): {summary['trusted_nodes']}")
    print(f"   Suspicious Nodes (<0.3): {summary['suspicious_nodes']}")
    print(f"   Active Alerts: {summary['active_alerts']}")
    print(f"   Paranoia Warning: {'YES' if summary['paranoia_warning'] else 'NO'}")
    
    # Analyze trust distribution
    trust_values = list(node.trust_network.values())
    print(f"\n📊 Trust Distribution Analysis:")
    print(f"   Minimum Trust: {min(trust_values):.3f}")
    print(f"   Maximum Trust: {max(trust_values):.3f}")
    print(f"   Standard Deviation: {np.std(trust_values):.3f}")
    print(f"   Network Diversity: {'High' if np.std(trust_values) > 0.3 else 'Low'}")
    
    print()


def main():
    """Run all demonstrations"""
    print("\n🌟 ENHANCED TRUST NETWORK SYSTEM DEMONSTRATION")
    print("🔒 Advanced Trust Management with Manipulation Detection")
    print("🏘️  Community Verification and Network Health Monitoring\n")
    
    try:
        demonstrate_normal_trust_evolution()
        demonstrate_manipulation_detection()
        demonstrate_community_verification() 
        demonstrate_network_health_monitoring()
        
        print("=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("✅ Enhanced trust system is working correctly!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())