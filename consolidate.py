#!/usr/bin/env python
"""
Simple consolidation runner script.

This script demonstrates the unified consolidation system by running
a basic consolidation operation across all three mechanisms:
- Phase-based consolidation (sleep-phase memory strengthening)
- Synaptic consolidation (EWC-based weight protection)
- Memory consolidation (episodic-to-semantic transfer)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adaptiveneuralnetwork.core.consolidation import create_default_consolidation_manager
from adaptiveneuralnetwork.core.nodes import NodeConfig, NodeState
from adaptiveneuralnetwork.core.phases import PhaseScheduler


class SimpleConsolidationModel(nn.Module):
    """Simple neural network for consolidation demonstration."""

    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def consolidate():
    """
    Execute consolidation across all mechanisms.
    
    This function demonstrates the unified consolidation system by:
    1. Creating a simple neural network model
    2. Setting up node states and phase scheduler
    3. Running consolidation across all three mechanisms
    4. Displaying results
    """
    print("=" * 70)
    print("CONSOLIDATION SYSTEM - EXECUTION")
    print("=" * 70)

    # Setup
    print("\n[1/4] Setting up consolidation system...")
    model = SimpleConsolidationModel()
    num_nodes = 16
    memory_dim = 128

    # Create consolidation manager with all mechanisms
    manager = create_default_consolidation_manager(
        model=model,
        memory_dim=memory_dim
    )

    print(f"  ✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  ✓ Initialized {len(manager.mechanisms)} consolidation mechanisms")

    # Setup node state and phase scheduler for phase-based consolidation
    print("\n[2/4] Preparing consolidation data...")
    config = NodeConfig(num_nodes=num_nodes, device="cpu")
    node_state = NodeState(config)
    phase_scheduler = PhaseScheduler(num_nodes=num_nodes)

    # Set some nodes to sleep phase
    phase_scheduler.node_phases[4:8] = 1  # Nodes 4-7 in sleep phase
    node_state.activity[0, 4:6, 0] = 0.8  # High activity (important)
    node_state.activity[0, 6:8, 0] = 0.3  # Low activity

    # Create synthetic data for synaptic consolidation
    batch_size = 16
    synthetic_data = torch.randn(batch_size, 128)
    synthetic_labels = torch.randint(0, 10, (batch_size,))
    dataset = TensorDataset(synthetic_data, synthetic_labels)
    data_loader = DataLoader(dataset, batch_size=8)

    # Create memory data for memory consolidation
    episodic_memories = torch.randn(8, memory_dim)
    semantic_memories = torch.randn(20, memory_dim)
    importance_scores = torch.tensor([0.9, 0.8, 0.75, 0.6, 0.85, 0.55, 0.9, 0.7])

    print(f"  ✓ Configured {num_nodes} nodes ({phase_scheduler.node_phases.eq(1).sum()} in sleep phase)")
    print(f"  ✓ Prepared {len(episodic_memories)} episodic memories")
    print(f"  ✓ Created data loader with {batch_size} samples")

    # Run consolidation
    print("\n[3/4] Running consolidation...")
    results = manager.consolidate_all(
        node_state=node_state,
        phase_scheduler=phase_scheduler,
        data_loader=data_loader,
        episodic_memories=episodic_memories,
        semantic_memories=semantic_memories,
        importance_scores=importance_scores
    )

    # Display results
    print("\n[4/4] Consolidation Results:")
    print("-" * 70)

    for mechanism_name, mechanism_results in results["mechanisms"].items():
        print(f"\n{mechanism_name}:")

        if "error" in mechanism_results:
            print(f"  ⚠ Error: {mechanism_results['error']}")
            continue

        if "modifications" in mechanism_results:
            for mod in mechanism_results["modifications"]:
                print(f"  ✓ {mod}")

        if "metrics" in mechanism_results:
            metrics = mechanism_results["metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"    • {key}: {value:.4f}")
                else:
                    print(f"    • {key}: {value}")

    # Summary
    print("\n" + "=" * 70)
    print("CONSOLIDATION SUMMARY")
    print("=" * 70)
    print(f"Total mechanisms executed: {results['summary']['total_mechanisms']}")
    print(f"Total modifications: {results['summary']['total_modifications']}")
    print(f"Combined consolidation strength: {manager.get_total_consolidation_strength():.2f}")

    # Get detailed info
    info = manager.get_consolidation_info()
    print("\nActive Mechanisms:")
    for name in info["active_mechanisms"]:
        details = info["mechanism_details"][name]
        print(f"  • {name}: {details['type']} (strength={details['strength']:.2f})")

    print("\n✅ Consolidation completed successfully!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    consolidate()
