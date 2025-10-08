#!/usr/bin/env python3
"""
Rolling History Demonstration

This script demonstrates the new 20-entry rolling history functionality
for calm, energy, and anxiety tracking with proactive intervention.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.alive_node import AliveLoopNode
from core.time_series_tracker import TimeSeriesTracker
import json


def demonstrate_rolling_history():
    """Demonstrate rolling history functionality with trend analysis"""
    print("=" * 60)
    print("ROLLING HISTORY DEMONSTRATION")
    print("=" * 60)
    
    # Create a node for demonstration
    node = AliveLoopNode(
        position=(0, 0),
        velocity=(0.1, 0.1),
        initial_energy=10.0,
        field_strength=1.0,
        node_id=1
    )
    
    # Create time series tracker
    tracker = TimeSeriesTracker(persist_to_disk=False)
    
    print(f"Node {node.node_id} initialized with:")
    print(f"  Initial Energy: {node.energy:.2f}")
    print(f"  Initial Anxiety: {node.anxiety:.2f}")
    print(f"  Initial Calm: {node.calm:.2f}")
    print()
    
    # Simulate different scenarios to demonstrate rolling history
    scenarios = [
        ("Normal Operation", 10, lambda t: None),
        ("Stress Buildup", 15, lambda t: setattr(node, 'anxiety', node.anxiety + 0.3)),
        ("Energy Depletion", 12, lambda t: setattr(node, 'energy', max(0.5, node.energy - 0.4))),
        ("Recovery Period", 8, lambda t: setattr(node, 'calm', min(5.0, node.calm + 0.2)))
    ]
    
    time_step = 0
    all_history = {
        'time': [],
        'anxiety': [],
        'calm': [],
        'energy': [],
        'interventions': []
    }
    
    for scenario_name, duration, scenario_effect in scenarios:
        print(f"\n--- {scenario_name} ({duration} steps) ---")
        
        for step in range(duration):
            # Apply scenario effect
            scenario_effect(time_step)
            
            # Normal node processing
            node.step_phase(current_time=time_step)
            
            # Record state for tracking
            tracker.record_node_state(node.node_id, {
                'energy': node.energy,
                'anxiety': node.anxiety,
                'calm': node.calm,
                'phase': node.phase
            }, timestamp=time_step)
            
            # Record for our demonstration
            all_history['time'].append(time_step)
            all_history['anxiety'].append(node.anxiety)
            all_history['calm'].append(node.calm)
            all_history['energy'].append(node.energy)
            
            # Check for interventions
            if len(node.anxiety_history) >= 5:
                intervention_analysis = node.detect_intervention_needs()
                if intervention_analysis['interventions_needed']:
                    all_history['interventions'].append({
                        'time': time_step,
                        'interventions': intervention_analysis['interventions_needed'],
                        'urgency': intervention_analysis['urgency_level']
                    })
                    print(f"  Step {time_step}: Interventions needed: {intervention_analysis['interventions_needed']} "
                          f"(Urgency: {intervention_analysis['urgency_level']})")
            
            time_step += 1
            
        # Print current state
        print(f"  End State - Energy: {node.energy:.2f}, Anxiety: {node.anxiety:.2f}, Calm: {node.calm:.2f}")
        print(f"  History Lengths - Anxiety: {len(node.anxiety_history)}, Calm: {len(node.calm_history)}, Energy: {len(node.energy_history)}")
    
    print(f"\n--- Final Rolling History Analysis ---")
    
    # Demonstrate trend analysis for each metric
    metrics = ['anxiety', 'calm', 'energy']
    histories = [node.anxiety_history, node.calm_history, node.energy_history]
    
    for metric, history in zip(metrics, histories, strict=False):
        if len(history) > 0:
            trend_analysis = node.analyze_trend(history)
            print(f"\n{metric.upper()} TREND ANALYSIS:")
            print(f"  Trend: {trend_analysis['trend']}")
            print(f"  Slope: {trend_analysis['slope']:.3f}")
            print(f"  Recent Average: {trend_analysis['recent_avg']:.2f}")
            print(f"  Volatility: {trend_analysis['volatility']:.3f}")
            print(f"  Values Count: {trend_analysis['values_count']}")
    
    # Overall intervention analysis
    final_intervention_analysis = node.detect_intervention_needs()
    print(f"\nFINAL INTERVENTION ANALYSIS:")
    print(f"  Interventions Needed: {final_intervention_analysis['interventions_needed']}")
    print(f"  Urgency Level: {final_intervention_analysis['urgency_level']}")
    print(f"  Combined Risk: {final_intervention_analysis['combined_risk']}")
    
    # Get comprehensive status
    status = node.get_anxiety_status()
    print(f"\nCOMPREHENSIVE NODE STATUS:")
    print(f"  Current - Anxiety: {status['anxiety_level']:.2f}, Calm: {status['calm_level']:.2f}, Energy: {status['energy_level']:.2f}")
    print(f"  History Lengths: {status['history_lengths']}")
    print(f"  Is Overwhelmed: {status['is_overwhelmed']}")
    
    return all_history, node, tracker


def plot_rolling_history(history_data, node):
    """Create visualization of rolling history data"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each metric
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    time_points = history_data['time']
    
    # Plot anxiety
    axes[0].plot(time_points, history_data['anxiety'], 'r-', linewidth=2, label='Anxiety')
    axes[0].axhline(y=8.0, color='r', linestyle='--', alpha=0.5, label='High Anxiety Threshold')
    axes[0].set_ylabel('Anxiety Level')
    axes[0].set_title('Anxiety Over Time (Rolling History)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot calm
    axes[1].plot(time_points, history_data['calm'], 'g-', linewidth=2, label='Calm')
    axes[1].axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label='Low Calm Threshold')
    axes[1].set_ylabel('Calm Level')
    axes[1].set_title('Calm Over Time (Rolling History)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot energy
    axes[2].plot(time_points, history_data['energy'], 'b-', linewidth=2, label='Energy')
    axes[2].axhline(y=3.0, color='b', linestyle='--', alpha=0.5, label='Low Energy Threshold')
    axes[2].set_ylabel('Energy Level')
    axes[2].set_xlabel('Time Step')
    axes[2].set_title('Energy Over Time (Rolling History)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Mark intervention points
    for intervention in history_data['interventions']:
        for ax in axes:
            ax.axvline(x=intervention['time'], color='orange', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle('Rolling History with Proactive Intervention Analysis', y=1.02, fontsize=16)
    
    # Save the plot
    plt.savefig('demo_rolling_history.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'demo_rolling_history.png'")
    
    return fig


def export_history_data(history_data, node, filename='rolling_history_demo.json'):
    """Export rolling history data to JSON for analysis"""
    export_data = {
        'simulation_data': history_data,
        'final_node_state': {
            'energy': node.energy,
            'anxiety': node.anxiety,
            'calm': node.calm,
            'phase': node.phase
        },
        'rolling_histories': {
            'anxiety_history': list(node.anxiety_history),
            'calm_history': list(node.calm_history),
            'energy_history': list(node.energy_history)
        },
        'final_trend_analysis': {
            'anxiety': node.analyze_trend(node.anxiety_history),
            'calm': node.analyze_trend(node.calm_history),
            'energy': node.analyze_trend(node.energy_history)
        },
        'intervention_analysis': node.detect_intervention_needs()
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"Rolling history data exported to '{filename}'")
    return export_data


if __name__ == "__main__":
    # Run the demonstration
    history_data, node, tracker = demonstrate_rolling_history()
    
    # Create visualization
    plot_rolling_history(history_data, node)
    
    # Export data
    export_history_data(history_data, node)
    
    print("\n" + "=" * 60)
    print("ROLLING HISTORY DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ 20-entry rolling history for anxiety, calm, and energy")
    print("✓ Automatic trend analysis with slope and volatility calculations")
    print("✓ Proactive intervention detection based on concerning trends")
    print("✓ Integration with existing TimeSeriesTracker system")
    print("✓ Comprehensive status reporting with history information")
    print("✓ Visualization and data export capabilities")