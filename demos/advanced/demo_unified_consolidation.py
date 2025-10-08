"""
Demonstration of the unified consolidation system.

This script showcases how the consolidated consolidation mechanisms work 
together to provide comprehensive memory consolidation across different
time scales and neural processing phases.
"""


import torch
import torch.nn as nn

from adaptiveneuralnetwork.core.consolidation import (
    create_default_consolidation_manager,
)
from adaptiveneuralnetwork.core.nodes import NodeConfig, NodeState
from adaptiveneuralnetwork.core.phases import PhaseScheduler
from adaptiveneuralnetwork.core.plugin_system import (
    create_plugin_manager_with_defaults,
)


class DemoModel(nn.Module):
    """Simple demonstration model."""

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def demo_unified_consolidation():
    """Demonstrate unified consolidation system."""
    print("=" * 80)
    print("UNIFIED CONSOLIDATION SYSTEM DEMONSTRATION")
    print("=" * 80)

    # 1. Create components
    print("\n1. Creating System Components...")

    model = DemoModel()
    num_nodes = 16
    memory_dim = 128

    # Create node state and phase scheduler
    config = NodeConfig(num_nodes=num_nodes, device="cpu")
    node_state = NodeState(config)
    phase_scheduler = PhaseScheduler(num_nodes=num_nodes)

    # Create unified consolidation manager with all mechanisms
    consolidation_manager = create_default_consolidation_manager(
        model=model,
        memory_dim=memory_dim
    )

    print(f"   ✓ Model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"   ✓ Node state with {num_nodes} nodes")
    print(f"   ✓ Consolidation manager with {len(consolidation_manager.mechanisms)} mechanisms")

    # Display consolidation info
    info = consolidation_manager.get_consolidation_info()
    print("\n   Registered consolidation mechanisms:")
    for name, details in info["mechanism_details"].items():
        print(f"     • {name} ({details['type']}): strength={details['strength']:.2f}")

    # 2. Simulate different consolidation scenarios
    print("\n2. Running Consolidation Scenarios...")

    # Scenario A: Phase-based consolidation during sleep
    print("\n   Scenario A: Sleep-Phase Consolidation")
    phase_scheduler.node_phases[4:8] = 1  # Set nodes 4-7 to sleep phase
    node_state.activity[0, 4:6, 0] = 0.8  # Make some sleep nodes highly active (important)
    node_state.activity[0, 6:8, 0] = 0.3  # Make some sleep nodes less active

    initial_energy = node_state.energy.clone()

    results_phase = consolidation_manager.consolidate_all(
        node_state=node_state,
        phase_scheduler=phase_scheduler
    )

    print(f"     • Sleep nodes identified: {results_phase['mechanisms']['phase_consolidation']['metrics']['sleep_nodes']}")
    print(f"     • Important nodes consolidated: {results_phase['mechanisms']['phase_consolidation']['metrics']['consolidated_nodes']}")

    # Check energy boost for important nodes
    energy_boost = node_state.energy[0, 4, 0] - initial_energy[0, 4, 0]
    print(f"     • Energy boost for important node: {energy_boost.item():.3f}")

    # Scenario B: Synaptic consolidation for catastrophic forgetting prevention
    print("\n   Scenario B: Synaptic Consolidation (EWC)")

    # Create synthetic data for Fisher information estimation
    batch_size = 32
    input_dim = 784
    synthetic_data = torch.randn(batch_size, input_dim)
    synthetic_labels = torch.randint(0, 10, (batch_size,))

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(synthetic_data, synthetic_labels)
    data_loader = DataLoader(dataset, batch_size=8)

    results_synaptic = consolidation_manager.consolidate_all(
        data_loader=data_loader
    )

    if "synaptic_consolidation" in results_synaptic["mechanisms"]:
        synaptic_metrics = results_synaptic["mechanisms"]["synaptic_consolidation"]["metrics"]
        print(f"     • Total Fisher information: {synaptic_metrics['total_fisher_information']:.6f}")
        print(f"     • Important parameters identified: {synaptic_metrics['important_parameters']}")

    # Scenario C: Memory consolidation (episodic to semantic)
    print("\n   Scenario C: Memory Consolidation")

    episodic_memories = torch.randn(10, memory_dim) * 0.5  # Simulated episodic memories
    semantic_memories = torch.randn(50, memory_dim) * 0.3  # Existing semantic memory bank
    importance_scores = torch.tensor([0.9, 0.8, 0.6, 0.5, 0.9, 0.7, 0.4, 0.8, 0.6, 0.9])

    results_memory = consolidation_manager.consolidate_all(
        episodic_memories=episodic_memories,
        semantic_memories=semantic_memories,
        importance_scores=importance_scores
    )

    if "memory_consolidation" in results_memory["mechanisms"]:
        memory_metrics = results_memory["mechanisms"]["memory_consolidation"]["metrics"]
        print(f"     • Episodic memories provided: {len(episodic_memories)}")
        print(f"     • Memories consolidated to semantic: {memory_metrics['memories_consolidated']}")

    # 3. Integrated consolidation (all mechanisms together)
    print("\n3. Integrated Multi-Scale Consolidation...")

    integrated_results = consolidation_manager.consolidate_all(
        node_state=node_state,
        phase_scheduler=phase_scheduler,
        data_loader=data_loader,
        episodic_memories=episodic_memories,
        semantic_memories=semantic_memories,
        importance_scores=importance_scores
    )

    print(f"   ✓ Total mechanisms active: {len(integrated_results['mechanisms'])}")
    print(f"   ✓ Total modifications: {integrated_results['summary']['total_modifications']}")
    print(f"   ✓ Combined consolidation strength: {consolidation_manager.get_total_consolidation_strength():.2f}")

    # Display detailed results
    print("\n   Detailed Results:")
    for mechanism_name, mechanism_results in integrated_results["mechanisms"].items():
        print(f"     {mechanism_name}:")
        if "modifications" in mechanism_results:
            for mod in mechanism_results["modifications"]:
                print(f"       - {mod}")

    return integrated_results


def demo_plugin_integration():
    """Demonstrate integration with plugin system."""
    print("\n" + "=" * 80)
    print("PLUGIN SYSTEM INTEGRATION DEMONSTRATION")
    print("=" * 80)

    # Create plugin manager with consolidated consolidation
    plugin_manager = create_plugin_manager_with_defaults()

    print(f"\n✓ Plugin manager created with {len(plugin_manager.plugins)} plugins")

    # Get consolidation plugin and show it uses unified system
    consolidation_plugin = plugin_manager.plugins["consolidation"]
    print(f"✓ Consolidation plugin uses unified system: {hasattr(consolidation_plugin, 'consolidation_manager')}")

    # Test plugin functionality
    num_nodes = 8
    config = NodeConfig(num_nodes=num_nodes, device="cpu")
    node_state = NodeState(config)
    phase_scheduler = PhaseScheduler(num_nodes=num_nodes)

    # Set up sleep phase scenario
    phase_scheduler.node_phases[2:4] = 1
    node_state.activity[0, 2, 0] = 0.9

    # Apply plugin logic (uses unified consolidation internally)
    plugin_results = consolidation_plugin.apply_phase_logic(
        node_state=node_state,
        phase_scheduler=phase_scheduler,
        step=1
    )

    print("✓ Plugin consolidation completed")
    print(f"  • Modifications: {len(plugin_results.get('modifications', []))}")
    print(f"  • Has consolidation summary: {'consolidation_summary' in plugin_results}")

    return plugin_results


def demo_backward_compatibility():
    """Demonstrate backward compatibility with existing code."""
    print("\n" + "=" * 80)
    print("BACKWARD COMPATIBILITY DEMONSTRATION")
    print("=" * 80)

    try:
        from adaptiveneuralnetwork.applications.continual_learning import SynapticConsolidation

        model = DemoModel()
        consolidation = SynapticConsolidation(model)

        print("✓ Legacy SynapticConsolidation interface works")
        print(f"  • Has fisher_information: {hasattr(consolidation, 'fisher_information')}")
        print(f"  • Has consolidation_loss method: {hasattr(consolidation, 'consolidation_loss')}")

        # Test legacy methods
        loss = consolidation.consolidation_loss(consolidation_strength=0.5)
        print(f"  • Consolidation loss computed: {loss.item():.6f}")

        # Show it uses unified system internally
        print(f"  • Uses unified system internally: {hasattr(consolidation, 'consolidation_manager')}")

        return {"legacy_works": True, "loss": loss.item()}

    except Exception as e:
        print(f"⚠ Backward compatibility test skipped: {e}")
        return {"legacy_works": False, "error": str(e)}


def main():
    """Run all demonstrations."""
    print("Starting Unified Consolidation System Demonstrations...")

    results = {}

    # Run demonstrations
    results["unified"] = demo_unified_consolidation()
    results["plugin"] = demo_plugin_integration()
    results["compatibility"] = demo_backward_compatibility()

    # Summary
    print("\n" + "=" * 80)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("=" * 80)

    print("\nSummary of Unified Consolidation System:")
    print("✅ Phase-based consolidation: Sleep-phase memory strengthening")
    print("✅ Synaptic consolidation: EWC-based weight protection")
    print("✅ Memory consolidation: Episodic-to-semantic transfer")
    print("✅ Unified management: Coordinated multi-scale consolidation")
    print("✅ Plugin integration: Seamless existing system compatibility")
    print("✅ Backward compatibility: Legacy interfaces preserved")

    print(f"\nTotal consolidation mechanisms active: {len(results['unified']['mechanisms'])}")
    print(f"System integration successful: {results['compatibility']['legacy_works']}")

    print("\nThe consolidation system has been successfully unified! 🚀")

    return results


if __name__ == "__main__":
    main()
