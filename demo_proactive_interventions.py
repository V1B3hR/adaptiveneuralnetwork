#!/usr/bin/env python3
"""
Proactive Intervention Demonstration

This script demonstrates the proactive intervention system
triggering based on concerning trends in rolling history.
"""

import json

import matplotlib.pyplot as plt

from core.alive_node import AliveLoopNode


def demonstrate_proactive_interventions():
    """Demonstrate proactive intervention system in action"""
    print("=" * 60)
    print("PROACTIVE INTERVENTION DEMONSTRATION")
    print("=" * 60)

    # Create a node for demonstration
    node = AliveLoopNode(
        position=(0, 0),
        velocity=(0.1, 0.1),
        initial_energy=10.0,
        field_strength=1.0,
        node_id=1
    )

    print(f"Node {node.node_id} initialized with:")
    print(f"  Initial Energy: {node.energy:.2f}")
    print(f"  Initial Anxiety: {node.anxiety:.2f}")
    print(f"  Initial Calm: {node.calm:.2f}")
    print()

    # Simulation to trigger specific interventions
    scenarios = [
        {
            "name": "Escalating Anxiety Crisis",
            "duration": 10,
            "effects": lambda t, n: [
                setattr(n, 'anxiety', min(15.0, n.anxiety + 0.8)),  # Rapid anxiety increase
                setattr(n, 'calm', max(0.1, n.calm - 0.2))          # Calm decrease
            ]
        },
        {
            "name": "Energy Depletion Emergency",
            "duration": 8,
            "effects": lambda t, n: [
                setattr(n, 'energy', max(0.5, n.energy - 0.6)),     # Rapid energy loss
                setattr(n, 'anxiety', min(15.0, n.anxiety + 0.3))   # Added stress from fatigue
            ]
        },
        {
            "name": "Combined Crisis State",
            "duration": 6,
            "effects": lambda t, n: [
                setattr(n, 'anxiety', min(15.0, n.anxiety + 1.0)),  # Very high anxiety
                setattr(n, 'calm', max(0.1, n.calm - 0.3)),         # Minimal calm
                setattr(n, 'energy', max(0.5, n.energy - 0.4))      # Low energy
            ]
        }
    ]

    time_step = 0
    intervention_log = []
    history_data = {
        'time': [],
        'anxiety': [],
        'calm': [],
        'energy': [],
        'phase': [],
        'interventions_applied': []
    }

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ({scenario['duration']} steps) ---")

        for step in range(scenario['duration']):
            # Apply scenario effects
            scenario['effects'](time_step, node)

            # Store pre-step values for comparison
            pre_anxiety = node.anxiety
            pre_calm = node.calm
            pre_energy = node.energy

            # Normal node processing (which includes proactive intervention)
            node.step_phase(current_time=time_step)

            # Check what interventions were applied
            interventions_applied = []
            if len(node.anxiety_history) >= 5:
                # Get fresh intervention analysis
                intervention_result = node.apply_proactive_intervention()
                if intervention_result['applied_interventions']:
                    interventions_applied = intervention_result['applied_interventions']
                    intervention_log.append({
                        'time': time_step,
                        'scenario': scenario['name'],
                        'interventions': intervention_result['applied_interventions'],
                        'analysis': intervention_result['intervention_analysis'],
                        'before_state': {'anxiety': pre_anxiety, 'calm': pre_calm, 'energy': pre_energy},
                        'after_state': intervention_result['node_state_after']
                    })

                    print(f"  Step {time_step}: Applied {len(interventions_applied)} interventions")
                    for intervention in interventions_applied:
                        print(f"    - {intervention}")

            # Record data
            history_data['time'].append(time_step)
            history_data['anxiety'].append(node.anxiety)
            history_data['calm'].append(node.calm)
            history_data['energy'].append(node.energy)
            history_data['phase'].append(node.phase)
            history_data['interventions_applied'].append(interventions_applied)

            time_step += 1

        # Print scenario summary
        print(f"  End State - Energy: {node.energy:.2f}, Anxiety: {node.anxiety:.2f}, Calm: {node.calm:.2f}")

        # Analyze trend after scenario
        if len(node.anxiety_history) >= 5:
            intervention_analysis = node.detect_intervention_needs()
            print(f"  Urgency Level: {intervention_analysis['urgency_level']}")
            print(f"  Combined Risk: {intervention_analysis['combined_risk']}")

    print("\n--- INTERVENTION SUMMARY ---")
    print(f"Total interventions applied: {len(intervention_log)}")

    # Group interventions by type
    intervention_types = {}
    for log_entry in intervention_log:
        for intervention in log_entry['interventions']:
            intervention_type = intervention.split(':')[0]
            if intervention_type not in intervention_types:
                intervention_types[intervention_type] = 0
            intervention_types[intervention_type] += 1

    for int_type, count in intervention_types.items():
        print(f"  {int_type}: {count} times")

    # Show final comprehensive analysis
    final_analysis = node.detect_intervention_needs()
    print("\nFINAL STATE ANALYSIS:")
    print(f"  Anxiety Trend: {final_analysis['anxiety_trend']['trend']} (slope: {final_analysis['anxiety_trend']['slope']:.3f})")
    print(f"  Calm Trend: {final_analysis['calm_trend']['trend']} (slope: {final_analysis['calm_trend']['slope']:.3f})")
    print(f"  Energy Trend: {final_analysis['energy_trend']['trend']} (slope: {final_analysis['energy_trend']['slope']:.3f})")
    print(f"  Overall Urgency: {final_analysis['urgency_level']}")

    return history_data, intervention_log, node


def plot_intervention_demo(history_data, intervention_log):
    """Create visualization showing interventions in action"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    time_points = history_data['time']

    # Plot anxiety with intervention markers
    axes[0].plot(time_points, history_data['anxiety'], 'r-', linewidth=2, label='Anxiety')
    axes[0].axhline(y=8.0, color='r', linestyle='--', alpha=0.5, label='High Anxiety Threshold')
    axes[0].set_ylabel('Anxiety Level')
    axes[0].set_title('Anxiety with Proactive Interventions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot calm with intervention markers
    axes[1].plot(time_points, history_data['calm'], 'g-', linewidth=2, label='Calm')
    axes[1].axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label='Low Calm Threshold')
    axes[1].set_ylabel('Calm Level')
    axes[1].set_title('Calm with Proactive Interventions')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot energy with intervention markers
    axes[2].plot(time_points, history_data['energy'], 'b-', linewidth=2, label='Energy')
    axes[2].axhline(y=3.0, color='b', linestyle='--', alpha=0.5, label='Low Energy Threshold')
    axes[2].set_ylabel('Energy Level')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Energy with Proactive Interventions')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Mark intervention points
    intervention_times = [log['time'] for log in intervention_log]
    for int_time in intervention_times:
        for ax in axes:
            ax.axvline(x=int_time, color='orange', linestyle=':', alpha=0.8, linewidth=2)

    # Add text annotations for major interventions
    for i, log in enumerate(intervention_log):
        if len(log['interventions']) > 1:  # Only annotate comprehensive interventions
            axes[0].annotate(f"Multi-Intervention\n({len(log['interventions'])} actions)",
                           xy=(log['time'], history_data['anxiety'][log['time']]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.suptitle('Proactive Intervention System in Action', y=1.02, fontsize=16)

    # Save the plot
    plt.savefig('demo_proactive_interventions.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'demo_proactive_interventions.png'")

    return fig


def export_intervention_data(history_data, intervention_log, node, filename='proactive_interventions_demo.json'):
    """Export intervention demonstration data"""
    export_data = {
        'simulation_summary': {
            'total_time_steps': len(history_data['time']),
            'total_interventions': len(intervention_log),
            'intervention_types': {}
        },
        'history_data': history_data,
        'intervention_log': intervention_log,
        'final_state': {
            'anxiety': node.anxiety,
            'calm': node.calm,
            'energy': node.energy,
            'phase': node.phase
        },
        'rolling_histories': {
            'anxiety_history': list(node.anxiety_history),
            'calm_history': list(node.calm_history),
            'energy_history': list(node.energy_history)
        },
        'final_analysis': node.detect_intervention_needs()
    }

    # Count intervention types
    for log_entry in intervention_log:
        for intervention in log_entry['interventions']:
            intervention_type = intervention.split(':')[0]
            if intervention_type not in export_data['simulation_summary']['intervention_types']:
                export_data['simulation_summary']['intervention_types'][intervention_type] = 0
            export_data['simulation_summary']['intervention_types'][intervention_type] += 1

    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"Intervention demonstration data exported to '{filename}'")
    return export_data


if __name__ == "__main__":
    # Run the proactive intervention demonstration
    history_data, intervention_log, node = demonstrate_proactive_interventions()

    # Create visualization
    plot_intervention_demo(history_data, intervention_log)

    # Export data
    export_intervention_data(history_data, intervention_log, node)

    print("\n" + "=" * 60)
    print("PROACTIVE INTERVENTION DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Real-time trend analysis during crisis scenarios")
    print("✓ Automatic intervention triggering based on concerning patterns")
    print("✓ Multiple intervention types (anxiety management, calm restoration, energy conservation)")
    print("✓ Combined risk detection and comprehensive support")
    print("✓ Intervention effectiveness tracking and reporting")
    print("✓ Rolling history integration with proactive system")
