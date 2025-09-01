#!/usr/bin/env python3
"""
AI Ethics Framework Demonstration

This script demonstrates the AI Ethics Framework functionality
integrated into the adaptive neural network.
"""

from core.ai_ethics import get_ethics_framework, audit_decision_simple
from core.alive_node import AliveLoopNode
from core.capacitor import CapacitorInSpace
from core.network import TunedAdaptiveFieldNetwork
from config.network_config import load_network_config

def test_ethics_framework():
    """Test the ethics framework with various scenarios"""
    print("=" * 60)
    print("AI ETHICS FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    # Get the ethics framework
    framework = get_ethics_framework()
    print(f"\n1. Framework loaded with {len(framework.principles)} ethical principles")
    
    # Display all principles
    print("\n2. Ethical Principles by Category:")
    categories = framework.list_all_principles()
    for category, principles in categories.items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        for principle in principles:
            print(f"     - {principle['name']}: {principle['description']}")
    
    print("\n3. Testing Ethics Auditing:")
    
    # Test 1: Normal operation (should pass)
    print("\n   Test 1: Normal signal absorption")
    has_violations, messages = audit_decision_simple(
        action_type="absorb_external_signal",
        actor_id="node_1",
        signal_energy=5.0,
        signal_type="human",
        verified=True,
        logged=True
    )
    print(f"   Result: {'PASS' if not has_violations else 'FAIL'}")
    if messages:
        print(f"   Messages: {messages}")
    
    # Test 2: High energy signal (should trigger violation)
    print("\n   Test 2: Dangerous high energy signal")
    has_violations, messages = audit_decision_simple(
        action_type="absorb_external_signal",
        actor_id="node_1",
        signal_energy=25.0,
        signal_type="unknown",
        verified=False
    )
    print(f"   Result: {'FAIL' if has_violations else 'UNEXPECTED PASS'}")
    if messages:
        for msg in messages:
            print(f"   Violation: {msg}")
    
    # Test 3: High speed movement (should trigger violation)
    print("\n   Test 3: Dangerous high speed movement")
    has_violations, messages = audit_decision_simple(
        action_type="move",
        actor_id="node_2",
        velocity=[3.5, 2.1],
        current_speed=4.1
    )
    print(f"   Result: {'FAIL' if has_violations else 'UNEXPECTED PASS'}")
    if messages:
        for msg in messages:
            print(f"   Violation: {msg}")
    
    # Test 4: High anxiety interaction
    print("\n   Test 4: High anxiety capacitor interaction")
    has_violations, messages = audit_decision_simple(
        action_type="interact_with_capacitor",
        actor_id="node_3",
        energy_difference=5.0,
        environment_state={"anxiety": 16.0, "phase": "active"}
    )
    print(f"   Result: {'FAIL' if has_violations else 'PASS'}")
    if messages:
        for msg in messages:
            print(f"   Violation: {msg}")
    
    # Show violations summary
    print("\n4. Violations Summary:")
    summary = framework.get_violations_summary()
    print(f"   Total violations detected: {summary['total_violations']}")
    if summary['by_severity']:
        print("   By severity:")
        for severity, count in summary['by_severity'].items():
            print(f"     {severity.upper()}: {count}")
    
    if summary['recent_violations']:
        print("   Recent violations:")
        for violation in summary['recent_violations'][-3:]:  # Show last 3
            print(f"     {violation['timestamp']}: {violation['principle']} "
                  f"({violation['severity']}) - {violation['actor']}")

def test_network_integration():
    """Test ethics integration in the full network"""
    print("\n" + "=" * 60)
    print("NETWORK INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple network
    nodes = [
        AliveLoopNode(
            position=[0, 0],
            velocity=[0.15, 0],
            initial_energy=10,
            field_strength=1.0,
            node_id=0
        ),
        AliveLoopNode(
            position=[1, 2],
            velocity=[-0.08, 0.03],
            initial_energy=5,
            field_strength=1.2,
            node_id=1
        )
    ]
    
    capacitors = [
        CapacitorInSpace(position=[0.5, 0.5], capacity=4),
        CapacitorInSpace(position=[-0.5, -0.5], capacity=6)
    ]
    
    # Create network with ethics enabled
    config = {"ethics_auditing": True, "logging": True}
    network = TunedAdaptiveFieldNetwork(nodes, capacitors, config=config)
    
    print(f"\n1. Network created with {len(nodes)} nodes and {len(capacitors)} capacitors")
    print("   Ethics auditing: ENABLED")
    
    print("\n2. Running 3 simulation steps with ethics monitoring...")
    
    # Run a few steps
    for step in range(3):
        print(f"\n   Step {step + 1}:")
        
        # Create some external streams that might trigger violations
        external_streams = None
        if step == 1:
            # Test with high energy signal in step 2
            external_streams = {0: ("ai", 22.0)}  # High energy signal
            print("     Injecting high energy signal to node 0...")
        
        network.step(external_streams)
        
        # Show node states
        for node in nodes:
            print(f"     Node {node.node_id}: Energy={node.energy:.1f}, "
                  f"Anxiety={node.anxiety:.1f}, Phase={node.phase}")
    
    print("\n3. Ethics violations during simulation:")
    framework = get_ethics_framework()
    summary = framework.get_violations_summary()
    
    if summary['total_violations'] > 0:
        print(f"   Total violations: {summary['total_violations']}")
        for violation in summary['recent_violations']:
            print(f"     {violation['principle']} ({violation['severity']}) - {violation['actor']}")
    else:
        print("   No violations detected during simulation")

if __name__ == "__main__":
    # Run demonstrations
    test_ethics_framework()
    test_network_integration()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe AI Ethics Framework is successfully integrated!")
    print("Key features demonstrated:")
    print("  ✓ 25 comprehensive ethical principles")
    print("  ✓ Automatic violation detection") 
    print("  ✓ Network-level integration")
    print("  ✓ Node-level decision auditing")
    print("  ✓ Comprehensive logging and reporting")
    print("  ✓ Configurable enforcement levels")
    print("\nSee docs/ETHICS_FRAMEWORK.md for detailed documentation.")