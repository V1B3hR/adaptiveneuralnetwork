#!/usr/bin/env python3
"""
Demo script showcasing the newly implemented continual learning features for version 0.2.0.

This script demonstrates:
1. Split MNIST continual learning benchmark
2. Sleep-phase ablation studies  
3. Anxiety and restorative behavior analysis
4. Enhanced phase controllers with sparsity metrics
"""

import sys
import os
# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel
from adaptiveneuralnetwork.training.continual import (
    split_mnist_benchmark,
    ablation_study_sleep_phases, 
    anxiety_restorative_analysis
)

def demo_split_mnist():
    """Demonstrate Split MNIST continual learning."""
    print("=" * 50)
    print("Split MNIST Continual Learning Demo")
    print("=" * 50)
    
    config = AdaptiveConfig(
        num_nodes=4,  # Reduced to avoid tensor size issues
        hidden_dim=8,  # Reduced
        input_dim=784,
        output_dim=10,
        num_epochs=1,  # Reduced for demo
        batch_size=4,  # Reduced
        device="cpu"
    )
    
    model = AdaptiveModel(config)
    
    # Run Split MNIST benchmark with synthetic data
    print("Running Split MNIST benchmark with 2 tasks...")
    results = split_mnist_benchmark(model, config, num_tasks=2, use_synthetic=True)
    
    print(f"\nResults Summary:")
    print(f"Final Average Accuracy: {results['final_average_accuracy']:.4f}")
    print(f"Total Forgetting: {results['total_forgetting']:.4f}")
    print(f"Number of Tasks: {results['num_tasks']}")
    
    for task_name, task_result in results['task_results'].items():
        print(f"\n{task_name.upper()}:")
        print(f"  Classes: {task_result['classes']}")
        print(f"  Accuracy: {task_result['accuracy']:.4f}")
        print(f"  Average Forgetting: {task_result['average_forgetting']:.4f}")

def demo_ablation_studies():
    """Demonstrate sleep-phase ablation studies."""
    print("\n" + "=" * 50)
    print("Sleep-Phase Ablation Studies Demo")
    print("=" * 50)
    
    config = AdaptiveConfig(
        num_nodes=8,
        hidden_dim=12,
        input_dim=32,
        output_dim=4,
        num_epochs=1,
        batch_size=4,
        device="cpu"
    )
    
    print("Running ablation study with different phase configurations...")
    results = ablation_study_sleep_phases(config)
    
    print(f"\nBaseline Accuracy: {results['baseline_accuracy']:.4f}")
    print(f"Best Configuration: {results['summary']['best_config']}")
    print(f"Most Efficient Configuration: {results['summary']['most_efficient']}")
    
    print("\nConfiguration Results:")
    for config_name, result in results['configurations'].items():
        print(f"\n{config_name.upper()}:")
        print(f"  Disabled Phases: {result['disabled_phases']}")
        print(f"  Final Accuracy: {result['final_accuracy']:.4f}")
        print(f"  Energy Efficiency: {result['energy_efficiency']:.4f}")
        if 'accuracy_drop' in result:
            print(f"  Accuracy Drop: {result['accuracy_drop']:.4f}")

def demo_anxiety_analysis():
    """Demonstrate anxiety and restorative behavior analysis."""
    print("\n" + "=" * 50)
    print("Anxiety & Restorative Analysis Demo")
    print("=" * 50)
    
    config = AdaptiveConfig(
        num_nodes=6,
        hidden_dim=10,
        input_dim=32,
        output_dim=3,
        device="cpu"
    )
    
    model = AdaptiveModel(config)
    
    stress_conditions = {
        'high_loss_threshold': 2.0,
        'stress_duration': 5,
        'recovery_duration': 8
    }
    
    print("Running anxiety analysis with different stress scenarios...")
    results = anxiety_restorative_analysis(model, stress_conditions)
    
    print(f"\nOverall Analysis:")
    print(f"Most Stressful Scenario: {results['overall_analysis']['most_stressful_scenario']}")
    print(f"Best Recovery Scenario: {results['overall_analysis']['best_recovery_scenario']}")
    print(f"Average Resilience: {results['overall_analysis']['average_resilience']:.4f}")
    print(f"Stress Sensitivity: {results['overall_analysis']['stress_sensitivity']:.4f}")
    
    # Show details for one scenario
    scenario = 'high_loss'
    if scenario in results and 'summary' in results[scenario]:
        summary = results[scenario]['summary']
        print(f"\n{scenario.upper()} Scenario Summary:")
        print(f"  Max Stress Anxiety: {summary['max_stress_anxiety']:.4f}")
        print(f"  Final Recovery Anxiety: {summary['final_recovery_anxiety']:.4f}")
        print(f"  Anxiety Resilience: {summary['anxiety_resilience']:.4f}")
        print(f"  Recovery Effectiveness: {summary['recovery_effectiveness']:.4f}")

def demo_enhanced_metrics():
    """Demonstrate enhanced phase scheduler with sparsity metrics."""
    print("\n" + "=" * 50)
    print("Enhanced Phase Scheduler & Sparsity Metrics Demo")
    print("=" * 50)
    
    config = AdaptiveConfig(
        num_nodes=8,
        hidden_dim=12,
        device="cpu"
    )
    
    model = AdaptiveModel(config)
    
    # Create sample input
    sample_input = torch.randn(4, config.input_dim)
    
    # Forward pass to populate node states
    _ = model(sample_input)
    
    # Get enhanced metrics
    metrics = model.get_metrics()
    
    print("Standard Metrics:")
    print(f"  Active Node Ratio: {metrics['active_node_ratio']:.4f}")
    print(f"  Mean Energy: {metrics['mean_energy']:.4f}")
    print(f"  Mean Activity: {metrics['mean_activity']:.4f}")
    
    # Show phase distribution
    print("\nPhase Distribution:")
    for phase in ['active', 'sleep', 'interactive', 'inspired']:
        key = f"{phase}_ratio"
        if key in metrics:
            print(f"  {phase.title()} Ratio: {metrics[key]:.4f}")
    
    # Show anxiety metrics if available
    if 'mean_anxiety' in metrics:
        print("\nAnxiety Metrics:")
        print(f"  Mean Anxiety: {metrics['mean_anxiety']:.4f}")
        print(f"  Max Anxiety: {metrics['max_anxiety']:.4f}")
        print(f"  Anxious Nodes Ratio: {metrics['anxious_nodes_ratio']:.4f}")
    
    # Show sparsity metrics if available
    if 'energy_sparsity' in metrics:
        print("\nSparsity Metrics:")
        print(f"  Energy Sparsity: {metrics['energy_sparsity']:.4f}")
        print(f"  Activity Sparsity: {metrics['activity_sparsity']:.4f}")
        print(f"  Combined Sparsity: {metrics['combined_sparsity']:.4f}")
        print(f"  Active Phase Ratio: {metrics['active_phase_ratio']:.4f}")

def main():
    """Run all demonstrations."""
    print("Adaptive Neural Network - Continual Learning Features Demo")
    print("Version 0.2.0 Implementation")
    
    try:
        demo_split_mnist()
        demo_enhanced_metrics()
        demo_ablation_studies()  
        demo_anxiety_analysis()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("Features implemented:")
        print("✓ Split MNIST continual learning benchmark")
        print("✓ Advanced phase controllers with anxiety/restorative mechanics")
        print("✓ Energy/activity sparsity metrics")
        print("✓ Sleep-phase ablation studies")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()