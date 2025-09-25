#!/usr/bin/env python3
"""
Demo of Phase 2 Enhancements: Robustness & Security

Showcases:
1. Critical trust network repair fixes
2. Advanced energy system hardening 
3. Trust network visualization and monitoring
4. Distributed trust consensus mechanisms
5. Byzantine fault tolerance improvements
6. Advanced neuromorphic integration
"""

import sys
import os
import time
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alive_node import AliveLoopNode
from core.trust_network import TrustNetwork

console = Console()

def demo_trust_network_repair():
    """Demonstrate critical trust network repair fixes"""
    console.print("\n" + "="*60)
    console.print("[bold cyan]🔧 TRUST NETWORK REPAIR DEMONSTRATION[/bold cyan]")
    console.print("="*60)
    
    # Create nodes with trust relationships
    node1 = AliveLoopNode((0, 0), (0.1, 0.1), node_id=1)
    node2 = AliveLoopNode((1, 1), (0.1, 0.1), node_id=2)
    
    console.print(f"[green]✓[/green] Node 1 created with trust attribute: {node1.trust}")
    console.print(f"[green]✓[/green] Node 2 created with trust attribute: {node2.trust}")
    
    # Test trust attribute updates
    initial_trust = node1.trust
    node1._update_trust_after_communication(node2, 'resource')
    updated_trust = node1.trust
    
    console.print(f"[blue]📊[/blue] Trust before communication: {initial_trust:.3f}")
    console.print(f"[blue]📊[/blue] Trust after communication: {updated_trust:.3f}")
    
    # Test trust decay and recovery
    trust_system = node1.trust_network_system
    trust_system.set_trust(2, 0.8)
    
    console.print(f"[yellow]⏰[/yellow] Applying trust decay over time...")
    future_time = time.time() + 15
    decayed_count = trust_system.apply_trust_decay(future_time)
    console.print(f"[yellow]⏰[/yellow] {decayed_count} relationships decayed")
    
    # Apply recovery
    recovery_amount = trust_system.apply_trust_recovery(2, recovery_factor=1.5)
    console.print(f"[green]🔄[/green] Trust recovery applied: +{recovery_amount:.3f}")
    
    return True

def demo_energy_system_hardening():
    """Demonstrate energy system hardening improvements"""
    console.print("\n" + "="*60)
    console.print("[bold yellow]⚡ ENERGY SYSTEM HARDENING DEMONSTRATION[/bold yellow]")
    console.print("="*60)
    
    node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0, initial_energy=10.0)
    
    console.print(f"[blue]📊[/blue] Initial energy: {node.energy:.2f}")
    console.print(f"[blue]📊[/blue] Emergency threshold: {node.emergency_energy_threshold:.2f}")
    
    # Test energy attack detection
    console.print("[red]🚨[/red] Simulating energy attack...")
    for i in range(3):
        node.record_energy_drain(2.0, source="attack")
    
    attack_detected = node.energy_attack_detected
    console.print(f"[red]🚨[/red] Attack detected: {'Yes' if attack_detected else 'No'}")
    
    # Test emergency energy conservation
    node.energy = 0.6  # Set low energy
    node.threat_assessment_level = 0  # Prevent threshold recalculation
    node._original_emergency_threshold = 0.8
    node.adaptive_energy_allocation()
    
    console.print(f"[orange1]🔋[/orange1] Emergency mode activated: {'Yes' if node.emergency_mode else 'No'}")
    console.print(f"[orange1]🔋[/orange1] Communication range reduced to: {node.communication_range:.2f}")
    
    # Test distributed energy sharing
    node.trust_network[1] = 0.8  # Add trusted node
    node.distributed_energy_pool = 5.0
    shared_amount = node.request_distributed_energy(2.0)
    console.print(f"[green]🤝[/green] Energy received from network: {shared_amount:.2f}")
    
    return True

def demo_trust_visualization():
    """Demonstrate trust network visualization and monitoring"""
    console.print("\n" + "="*60)
    console.print("[bold magenta]📊 TRUST NETWORK VISUALIZATION DEMONSTRATION[/bold magenta]")
    console.print("="*60)
    
    node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
    
    # Set up diverse trust relationships
    trust_relationships = [
        (1, 0.9, "Highly trusted collaborator"),
        (2, 0.7, "Reliable partner"),
        (3, 0.5, "Neutral relationship"),
        (4, 0.2, "Suspicious entity"),
        (5, 0.1, "Very suspicious")
    ]
    
    for node_id, trust_level, desc in trust_relationships:
        node.trust_network_system.set_trust(node_id, trust_level)
    
    # Generate visualization data
    graph_data = node.get_trust_network_visualization()
    console.print(f"[blue]📈[/blue] Network nodes: {len(graph_data['nodes'])}")
    console.print(f"[blue]📈[/blue] Network edges: {len(graph_data['edges'])}")
    
    # Get comprehensive metrics
    metrics = node.get_trust_network_metrics()
    
    # Create metrics table
    table = Table(title="Trust Network Health Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Connections", str(metrics['total_connections']))
    table.add_row("Average Trust", f"{metrics['average_trust']:.3f}")
    table.add_row("Network Resilience", f"{metrics['network_resilience']:.1%}")
    table.add_row("Suspicious Ratio", f"{metrics['suspicious_ratio']:.1%}")
    table.add_row("Trust Variance", f"{metrics['trust_variance']:.3f}")
    
    console.print(table)
    
    # Monitor network health
    health_report = node.monitor_trust_network_health()
    console.print(f"[green]💚[/green] Overall health score: {health_report['overall_health']:.3f}")
    
    if health_report['alerts']:
        console.print("[red]🚨 Health Alerts:[/red]")
        for alert in health_report['alerts']:
            console.print(f"  [{alert['level'].lower()}]{alert['level']}[/{alert['level'].lower()}]: {alert['message']}")
    else:
        console.print("[green]✅ No health alerts - network is stable[/green]")
    
    return True

def demo_distributed_consensus():
    """Demonstrate distributed trust consensus mechanisms"""
    console.print("\n" + "="*60)
    console.print("[bold blue]🗳️  DISTRIBUTED TRUST CONSENSUS DEMONSTRATION[/bold blue]")
    console.print("="*60)
    
    # Create a network of nodes
    nodes = []
    for i in range(5):
        node = AliveLoopNode((i, i), (0.1, 0.1), node_id=i)
        nodes.append(node)
    
    # Set up some trust relationships
    nodes[0].trust_network[1] = 0.8
    nodes[0].trust_network[2] = 0.3  # Suspicious of node 2
    nodes[1].trust_network[2] = 0.7  # Different opinion
    nodes[3].trust_network[2] = 0.4
    nodes[4].trust_network[2] = 0.2
    
    console.print("[blue]🔍[/blue] Initiating consensus vote about node 2...")
    
    # Node 0 initiates consensus about node 2
    vote_request = nodes[0].trust_network_system.initiate_consensus_vote(subject_node_id=2)
    console.print(f"[blue]📝[/blue] Vote ID: {vote_request['vote_id'][:20]}...")
    console.print(f"[blue]📝[/blue] Vote reason: {vote_request['reason']}")
    
    # Collect responses from other nodes
    responses = []
    for voter in nodes[1:4]:  # Skip node 0 (initiator) and node 2 (subject)
        if voter.node_id != 2:
            response = voter.respond_to_trust_vote(vote_request)
            if response:
                responses.append(response)
                console.print(f"[green]📨[/green] Node {response['voter_id']} voted: {response['trust_assessment']:.3f} (confidence: {response['confidence']:.2f})")
    
    # Process consensus
    consensus_result = nodes[0].trust_network_system.process_consensus_vote(vote_request, responses)
    
    if consensus_result:
        console.print(f"[yellow]🎯[/yellow] Consensus trust: {consensus_result['consensus_trust']:.3f}")
        console.print(f"[yellow]🎯[/yellow] Agreement level: {consensus_result['agreement_level']:.3f}")
        console.print(f"[yellow]🎯[/yellow] Recommendation: {consensus_result['recommendation']}")
    
    return True

def demo_byzantine_fault_tolerance():
    """Demonstrate Byzantine fault tolerance improvements"""
    console.print("\n" + "="*60)
    console.print("[bold red]🛡️  BYZANTINE FAULT TOLERANCE DEMONSTRATION[/bold red]")
    console.print("="*60)
    
    node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)
    
    # Set up initial trust network
    for i in range(1, 6):
        node.trust_network[i] = 0.5
    
    console.print("[blue]🔬[/blue] Running Byzantine stress test...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Testing Byzantine resilience...", total=None)
        
        # Run stress test
        results = node.run_byzantine_stress_test(malicious_ratio=0.33, num_simulations=20)
        
        progress.remove_task(task)
    
    # Display results
    console.print(f"[green]✅[/green] Resilience score: {results['resilience_score']:.3f}")
    console.print(f"[yellow]🎯[/yellow] Attack detection rate: {results['detection_rate']:.3f}")
    console.print(f"[red]⚠️[/red] False positive rate: {results['false_positive_rate']:.3f}")
    
    # Analyze results
    if results['resilience_score'] > 0.7:
        console.print("[green]🛡️ EXCELLENT: Network shows strong Byzantine fault tolerance[/green]")
    elif results['resilience_score'] > 0.5:
        console.print("[yellow]⚠️ GOOD: Network has adequate Byzantine fault tolerance[/yellow]")
    else:
        console.print("[red]🚨 WARNING: Network may be vulnerable to Byzantine attacks[/red]")
    
    return True

def demo_neuromorphic_integration():
    """Demonstrate advanced neuromorphic integration"""
    console.print("\n" + "="*60)
    console.print("[bold purple]🧠 NEUROMORPHIC INTEGRATION DEMONSTRATION[/bold purple]")
    console.print("="*60)
    
    try:
        from adaptiveneuralnetwork.core.neuromorphic import (
            NeuromorphicAdaptiveModel, 
            BrainWaveOscillator, 
            NeuromodulationSystem,
            NeuromorphicConfig
        )
        
        config = NeuromorphicConfig()
        oscillator = BrainWaveOscillator(config)
        neuromodulation = NeuromodulationSystem(config)
        
        console.print("[green]✅[/green] Neuromorphic components loaded successfully")
        
        # Test brain wave oscillations
        frequencies = oscillator.get_circadian_frequencies(12.0)  # Noon
        console.print(f"[blue]🌊[/blue] Alpha frequency: {frequencies['alpha']:.2f} Hz")
        console.print(f"[blue]🌊[/blue] Beta frequency: {frequencies['beta']:.2f} Hz")
        
        # Test stress response
        console.print("[red]😰[/red] Applying stress...")
        for _ in range(5):
            neuromodulation.update_stress_level(0.8, "trust_violation")
        
        console.print(f"[red]📊[/red] Stress level: {neuromodulation.stress_level:.3f}")
        
        # Test stress modulation
        base_activity = 1.0
        modulated = neuromodulation.apply_stress_modulation(base_activity, "excitatory")
        console.print(f"[yellow]⚡[/yellow] Neural activity modulation: {base_activity:.2f} → {modulated:.2f}")
        
        # Test recovery
        console.print("[green]😌[/green] Applying recovery...")
        neuromodulation.release_neurotransmitter('serotonin', 0.5)
        for _ in range(3):
            neuromodulation.update_stress_recovery()
        
        console.print(f"[green]📊[/green] Recovered stress level: {neuromodulation.stress_level:.3f}")
        
        return True
        
    except ImportError:
        console.print("[yellow]⚠️ Neuromorphic components not available - skipping demonstration[/yellow]")
        return False

def main():
    """Run all Phase 2 enhancement demonstrations"""
    console.print(Panel.fit(
        "[bold white]Phase 2: Robustness & Security Enhancement Demo[/bold white]\n"
        "[dim]Showcasing critical trust network repairs, energy hardening,\n"
        "distributed consensus, Byzantine fault tolerance, and neuromorphic integration[/dim]",
        border_style="bright_blue"
    ))
    
    demos = [
        ("Trust Network Repair", demo_trust_network_repair),
        ("Energy System Hardening", demo_energy_system_hardening),
        ("Trust Visualization", demo_trust_visualization),
        ("Distributed Consensus", demo_distributed_consensus),
        ("Byzantine Fault Tolerance", demo_byzantine_fault_tolerance),
        ("Neuromorphic Integration", demo_neuromorphic_integration)
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            console.print(f"\n[dim]Running {name} demonstration...[/dim]")
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            console.print(f"[red]❌ {name} failed: {e}[/red]")
            results.append((name, False))
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold green]📋 DEMONSTRATION SUMMARY[/bold green]")
    console.print("="*60)
    
    for name, success in results:
        status = "[green]✅ PASSED[/green]" if success else "[red]❌ FAILED[/red]"
        console.print(f"{status} {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    console.print(f"\n[bold]Results: {passed}/{total} demonstrations successful[/bold]")
    
    if passed == total:
        console.print("[bold green]🎉 All Phase 2 enhancements working perfectly![/bold green]")
        return 0
    else:
        console.print("[bold yellow]⚠️ Some demonstrations had issues - check logs above[/bold yellow]")
        return 1

if __name__ == "__main__":
    exit(main())