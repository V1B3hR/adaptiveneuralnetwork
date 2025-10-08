#!/usr/bin/env python3
"""
Demo script showing extended emotional states with history tracking and prediction.

This demonstrates the new emotional states (joy, grief, sadness) alongside existing
anxiety and calm, with historical tracking in deques and predictive behavior for
future node states.
"""

import os
import sys

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys

# Add the parent directory to the path so we can import from core
sys.path.insert(0, os.path.dirname(__file__))


from core.alive_node import AliveLoopNode, SocialSignal


def create_emotional_demo_nodes():
    """Create nodes with different emotional profiles for demonstration"""
    print("🎭 Creating nodes with diverse emotional profiles...")

    # Joyful node experiencing increasing joy
    joyful_node = AliveLoopNode(
        position=(0, 0),
        velocity=(0.1, 0),
        initial_energy=18.0,
        node_id=1
    )
    joyful_node.emotional_state["valence"] = 0.8
    joyful_node.calm = 4.0
    joyful_node.joy = 3.5
    print(f"   😊 Joyful Node (ID: {joyful_node.node_id}): joy={joyful_node.joy}, calm={joyful_node.calm}")

    # Grieving node with high grief and sadness
    grieving_node = AliveLoopNode(
        position=(3, 0),
        velocity=(-0.05, 0),
        initial_energy=8.0,
        node_id=2
    )
    grieving_node.emotional_state["valence"] = -0.7
    grieving_node.anxiety = 6.0
    grieving_node.grief = 4.2
    grieving_node.sadness = 3.0
    print(f"   😢 Grieving Node (ID: {grieving_node.node_id}): grief={grieving_node.grief}, sadness={grieving_node.sadness}, anxiety={grieving_node.anxiety}")

    # Supportive node with balanced emotional state
    supportive_node = AliveLoopNode(
        position=(1.5, 2),
        velocity=(0, -0.1),
        initial_energy=15.0,
        node_id=3
    )
    supportive_node.emotional_state["valence"] = 0.3
    supportive_node.calm = 4.5
    supportive_node.anxiety = 2.0
    supportive_node.joy = 2.0
    print(f"   🤝 Supportive Node (ID: {supportive_node.node_id}): calm={supportive_node.calm}, joy={supportive_node.joy}")

    # Establish trust relationships
    trust_matrix = {
        1: {2: 0.6, 3: 0.8},  # Joyful trusts others moderately to highly
        2: {1: 0.5, 3: 0.9},  # Grieving highly trusts supportive node
        3: {1: 0.8, 2: 0.8}   # Supportive trusts everyone well
    }

    nodes = [joyful_node, grieving_node, supportive_node]
    for node in nodes:
        for other_id, trust_level in trust_matrix[node.node_id].items():
            node.trust_network[other_id] = trust_level

    return nodes


def demonstrate_emotional_history_tracking(nodes):
    """Show how emotional states are tracked in history over time"""
    print("\n📊 Demonstrating emotional history tracking...")

    node = nodes[0]  # Use joyful node

    # Simulate some emotional changes over time
    print(f"   Initial state: joy={node.joy:.2f}, grief={node.grief:.2f}, sadness={node.sadness:.2f}")

    for step in range(5):
        # Simulate varying emotional changes
        node.update_joy(0.2 + (step * 0.1))
        node.update_grief(-0.1 if step > 2 else 0.05)  # Grief decreases later
        node.update_sadness(-0.05)
        node.update_emotional_states()  # Record current states

        print(f"   Step {step+1}: joy={node.joy:.2f}, grief={node.grief:.2f}, sadness={node.sadness:.2f}")

    print(f"   History lengths: joy={len(node.joy_history)}, grief={len(node.grief_history)}, sadness={len(node.sadness_history)}")

    return node


def demonstrate_emotional_prediction(node):
    """Show predictive behavior for future emotional states"""
    print("\n🔮 Demonstrating emotional state prediction...")

    # Predict future states
    for state in ['joy', 'grief', 'sadness', 'calm', 'anxiety']:
        current = getattr(node, state)
        predicted_3_steps = node.predict_emotional_state(state, 3)
        predicted_5_steps = node.predict_emotional_state(state, 5)

        print(f"   {state.capitalize()}: current={current:.2f}, "
              f"predicted(+3)={predicted_3_steps:.2f}, predicted(+5)={predicted_5_steps:.2f}")

    # Show trend analysis
    trends = node.get_emotional_trends()
    print(f"   Emotional trends: {trends}")


def demonstrate_proactive_interventions(nodes):
    """Show proactive intervention assessment based on emotional trends"""
    print("\n🚨 Demonstrating proactive intervention assessment...")

    for node in nodes:
        assessment = node.assess_intervention_need()

        print(f"\n   Node {node.node_id} assessment:")
        print(f"     Intervention needed: {assessment['intervention_needed']}")
        if assessment['intervention_needed']:
            print(f"     Type: {assessment['intervention_type']}")
            print(f"     Urgency: {assessment['urgency']:.2f}")
            print(f"     Reasons: {assessment['reasons']}")

        print("     Current emotional state:")
        emotional_summary = assessment['emotional_summary']
        for emotion, value in emotional_summary.items():
            print(f"       {emotion}: {value:.2f}")


def demonstrate_enhanced_emotional_interactions(nodes):
    """Show enhanced emotional signal processing with new factors"""
    print("\n💫 Demonstrating enhanced emotional interactions...")

    joyful_node, grieving_node, supportive_node = nodes

    # 1. Joyful node shares joy with enhanced effects
    joy_content = {
        "type": "joy_share",
        "source_node": joyful_node.node_id,
        "intensity": 0.8,
        "description": "Wonderful achievement to celebrate!"
    }

    joy_signal = SocialSignal(
        content=joy_content,
        signal_type="joy_share",
        urgency=0.4,
        source_id=joyful_node.node_id
    )

    print(f"   Grieving node before joy sharing: joy={grieving_node.joy:.2f}, "
          f"sadness={grieving_node.sadness:.2f}, grief={grieving_node.grief:.2f}")

    grieving_node._process_joy_share_signal(joy_signal)

    print(f"   Grieving node after joy sharing: joy={grieving_node.joy:.2f}, "
          f"sadness={grieving_node.sadness:.2f}, grief={grieving_node.grief:.2f}")

    # 2. Grief support with enhanced decision making
    grief_request = {
        "type": "grief_support_request",
        "requesting_node": grieving_node.node_id,
        "grief_intensity": 0.9
    }

    grief_signal = SocialSignal(
        content=grief_request,
        signal_type="grief_support_request",
        urgency=0.7,
        source_id=grieving_node.node_id
    )

    print(f"\n   Supportive node energy before helping: {supportive_node.energy:.2f}")
    response = supportive_node._process_grief_support_request_signal(grief_signal)

    if response:
        print("   Supportive node provided grief support successfully")
        print(f"   Energy after helping: {supportive_node.energy:.2f}")
    else:
        print("   Supportive node could not provide grief support")


def run_simulation_with_emotional_tracking(nodes, steps=10):
    """Run a simulation showing emotional tracking over time"""
    print(f"\n⚡ Running {steps}-step simulation with emotional tracking...")

    for step in range(steps):
        print(f"\n   === Step {step + 1} ===")

        for node in nodes:
            # Update emotional states during simulation
            node.update_emotional_states()

            # Simulate some emotional dynamics
            if node.node_id == 1:  # Joyful node
                node.update_joy(0.1)
            elif node.node_id == 2:  # Grieving node
                node.update_grief(-0.05)  # Slowly healing
                node.update_sadness(-0.03)
            elif node.node_id == 3:  # Supportive node
                node.update_calm(0.02)

            # Check for interventions every few steps
            if step % 3 == 0:
                assessment = node.assess_intervention_need()
                if assessment['intervention_needed']:
                    print(f"     Node {node.node_id} needs intervention: {assessment['intervention_type']}")

        # Show network emotional state
        if step % 5 == 4:  # Every 5 steps
            print("   Network emotional summary:")
            for node in nodes:
                print(f"     Node {node.node_id}: joy={node.joy:.1f}, grief={node.grief:.1f}, "
                      f"sadness={node.sadness:.1f}, anxiety={node.anxiety:.1f}")


def main():
    """Main demonstration script"""
    print("🌟 Extended Emotional States Demonstration")
    print("=" * 50)

    # Create demo nodes
    nodes = create_emotional_demo_nodes()

    # Demonstrate history tracking
    tracked_node = demonstrate_emotional_history_tracking(nodes)

    # Demonstrate prediction
    demonstrate_emotional_prediction(tracked_node)

    # Demonstrate proactive interventions
    demonstrate_proactive_interventions(nodes)

    # Demonstrate enhanced interactions
    demonstrate_enhanced_emotional_interactions(nodes)

    # Run simulation
    run_simulation_with_emotional_tracking(nodes)

    print("\n" + "=" * 50)
    print("✅ Extended emotional states demonstration complete!")
    print("\nKey features demonstrated:")
    print("   🎭 Extended emotional states: joy, grief, sadness")
    print("   📊 Historical tracking in deques for all emotional states")
    print("   🔮 Predictive behavior for future emotional states")
    print("   🚨 Proactive intervention assessment using emotional trends")
    print("   💫 Enhanced decision-making in emotional interactions")
    print("   ⚡ Integration into simulation step process")


if __name__ == '__main__':
    main()
