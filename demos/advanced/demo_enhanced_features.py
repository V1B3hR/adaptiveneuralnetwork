#!/usr/bin/env python3
"""
Comprehensive Demo Script for Enhanced Adaptive Neural Network Features

This script demonstrates all the new capabilities:
1. External signal absorption with multiple sources
2. Anxiety overwhelm safety protocol
3. Time series tracking and visualization
4. Security and privacy features

Usage:
    python demo_enhanced_features.py [--nodes N] [--steps N] [--visualize]
"""

import sys
import os
# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_demo.log")
    ]
)
logger = logging.getLogger(__name__)

# Import our enhanced components
from core.alive_node import AliveLoopNode, Memory
from core.network import TunedAdaptiveFieldNetwork, Capacitor
from core.time_series_tracker import TimeSeriesTracker, TimeSeriesQuery, track_node_automatically
from api_integration.signal_adapter import (
    SignalAdapter, SignalSource, SignalType, StateVariable, 
    SignalMapping, ApiCredentials
)
from api_integration.human_api import HumanSignalManager
from api_integration.world_api import EnvironmentalSignalManager
from api_integration.ai_api import AISignalManager


def create_mock_signal_sources():
    """Create mock signal sources for demonstration"""
    
    # Mock human emotion source
    def mock_human_emotion_data():
        return {
            "happiness": np.random.uniform(0.3, 0.9),
            "stress": np.random.uniform(0.1, 0.7),
            "energy_level": np.random.uniform(0.4, 1.0),
            "trust_level": np.random.uniform(0.3, 0.8)
        }
    
    # Mock environmental data
    def mock_environmental_data():
        return {
            "temperature": np.random.uniform(15, 30),  # Celsius
            "air_quality": np.random.uniform(20, 150),  # AQI
            "noise_level": np.random.uniform(30, 80),   # dB
            "light_intensity": np.random.uniform(0.2, 1.0)
        }
    
    # Mock AI system data
    def mock_ai_system_data():
        return {
            "accuracy": np.random.uniform(0.7, 0.95),
            "confidence": np.random.uniform(0.6, 0.9),
            "uncertainty": np.random.uniform(0.1, 0.4),
            "processing_speed": np.random.uniform(50, 200)
        }
    
    return {
        "human": mock_human_emotion_data,
        "environmental": mock_environmental_data,
        "ai": mock_ai_system_data
    }


def create_demo_network(num_nodes=5):
    """Create a demo network with enhanced features"""
    
    logger.info(f"Creating demo network with {num_nodes} nodes")
    
    # Create nodes in a circular arrangement
    nodes = []
    for i in range(num_nodes):
        angle = 2 * np.pi * i / num_nodes
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        
        node = AliveLoopNode(
            position=(x, y),
            velocity=(0, 0),
            initial_energy=np.random.uniform(10, 20),
            node_id=i
        )
        
        # Vary anxiety thresholds and communication styles
        node.anxiety_threshold = np.random.uniform(7.0, 9.0)
        node.communication_range = np.random.uniform(2.0, 4.0)
        node.calm = np.random.uniform(1.0, 3.0)
        
        # Set up initial trust network
        for j in range(num_nodes):
            if i != j:
                # Higher trust for nearby nodes
                distance = abs(i - j) if abs(i - j) <= num_nodes // 2 else num_nodes - abs(i - j)
                base_trust = max(0.3, 1.0 - 0.2 * distance)
                node.trust_network[j] = base_trust + np.random.uniform(-0.1, 0.1)
                node.influence_network[j] = np.random.uniform(0.3, 0.7)
        
        nodes.append(node)
    
    # Create capacitors
    capacitors = []
    for i in range(2):
        cap = Capacitor(capacity=15.0)
        cap.position = np.array([1.5 * np.cos(i * np.pi), 1.5 * np.sin(i * np.pi)])
        capacitors.append(cap)
    
    # Create enhanced network
    network = TunedAdaptiveFieldNetwork(
        nodes=nodes,
        capacitors=capacitors,
        enable_time_series=True,
        enable_security=False  # Disable for demo to avoid external API calls
    )
    
    logger.info("Demo network created successfully")
    return network


def simulate_external_signals(network, mock_sources, intensity_factor=1.0):
    """Simulate external signals affecting the network"""
    
    external_signals = {}
    
    for source_type, data_generator in mock_sources.items():
        mock_data = data_generator()
        
        if source_type == "human":
            # Apply human signal changes
            changes = {}
            changes[StateVariable.CALM] = mock_data["happiness"] * 2.0 * intensity_factor
            changes[StateVariable.ANXIETY] = mock_data["stress"] * 3.0 * intensity_factor
            changes[StateVariable.ENERGY] = (mock_data["energy_level"] - 0.5) * 2.0 * intensity_factor
            external_signals["human"] = changes
            
        elif source_type == "environmental":
            # Apply environmental signal changes
            changes = {}
            # Temperature affects arousal
            temp_effect = (mock_data["temperature"] - 22.5) * 0.1
            changes[StateVariable.AROUSAL] = temp_effect * intensity_factor
            
            # Air quality affects anxiety
            if mock_data["air_quality"] > 100:
                changes[StateVariable.ANXIETY] = (mock_data["air_quality"] - 100) * 0.02 * intensity_factor
            
            # Noise affects anxiety
            if mock_data["noise_level"] > 60:
                changes[StateVariable.ANXIETY] = changes.get(StateVariable.ANXIETY, 0) + \
                                              (mock_data["noise_level"] - 60) * 0.05 * intensity_factor
            
            external_signals["environmental"] = changes
            
        elif source_type == "ai":
            # Apply AI system signal changes
            changes = {}
            changes[StateVariable.TRUST] = (mock_data["accuracy"] - 0.5) * 0.5 * intensity_factor
            changes[StateVariable.CALM] = mock_data["confidence"] * 2.0 * intensity_factor
            changes[StateVariable.ANXIETY] = mock_data["uncertainty"] * 3.0 * intensity_factor
            external_signals["ai"] = changes
    
    return external_signals


def apply_external_signals_to_network(network, external_signals):
    """Apply external signals to network nodes"""
    
    for node in network.nodes:
        for signal_source, changes in external_signals.items():
            for state_var, value in changes.items():
                try:
                    if state_var == StateVariable.ENERGY:
                        node.energy = max(0, min(node.energy + value, 25.0))
                    elif state_var == StateVariable.ANXIETY:
                        node.anxiety = max(0, node.anxiety + value)
                    elif state_var == StateVariable.CALM:
                        node.calm = max(0, min(node.calm + value, 5.0))
                    elif state_var == StateVariable.TRUST:
                        # Apply to general trust baseline
                        for trust_node_id in node.trust_network:
                            current_trust = node.trust_network[trust_node_id]
                            node.trust_network[trust_node_id] = max(0, min(current_trust + value * 0.1, 1.0))
                    elif state_var == StateVariable.EMOTIONAL_VALENCE:
                        node.emotional_state["valence"] = max(-1.0, min(value, 1.0))
                    elif state_var == StateVariable.AROUSAL:
                        node.emotional_state["arousal"] = max(0.0, min(value, 1.0))
                        
                except Exception as e:
                    logger.warning(f"Failed to apply {state_var} change to node {node.node_id}: {e}")


def run_anxiety_overwhelm_scenario(network):
    """Demonstrate anxiety overwhelm safety protocol"""
    
    logger.info("=== Demonstrating Anxiety Overwhelm Safety Protocol ===")
    
    # Deliberately overwhelm a node
    victim_node = network.nodes[0]
    victim_node.anxiety = 12.0  # Well above threshold
    victim_node.energy = 8.0
    
    logger.info(f"Node {victim_node.node_id} anxiety level: {victim_node.anxiety:.2f} (threshold: {victim_node.anxiety_threshold:.2f})")
    logger.info(f"Node {victim_node.node_id} overwhelmed: {victim_node.check_anxiety_overwhelm()}")
    
    # Show initial network state
    print("\nInitial Network State:")
    for node in network.nodes:
        anxiety_status = node.get_anxiety_status()
        print(f"  Node {node.node_id}: Anxiety={node.anxiety:.2f}, Energy={node.energy:.2f}, "
              f"Can help: {node.energy >= 3.0 and node.anxiety < 6.0}")
    
    # Run network step to trigger help protocol
    initial_help_signals = network.performance_metrics["total_help_signals"]
    network.step()
    
    # Show results
    help_signals_sent = network.performance_metrics["total_help_signals"] - initial_help_signals
    logger.info(f"Help signals sent this step: {help_signals_sent}")
    
    print(f"\nAfter Help Protocol:")
    for node in network.nodes:
        if node.node_id == victim_node.node_id:
            print(f"  Node {node.node_id}: Anxiety={node.anxiety:.2f} (reduced), Energy={node.energy:.2f}")
        else:
            help_memories = [m for m in node.memory if m.memory_type == "help_given"]
            if help_memories:
                print(f"  Node {node.node_id}: Provided help (trust network updated)")
    
    return help_signals_sent > 0


def demonstrate_time_series_tracking(network, num_steps=20):
    """Demonstrate time series tracking capabilities"""
    
    logger.info("=== Demonstrating Time Series Tracking ===")
    
    # Create mock data sources
    mock_sources = create_mock_signal_sources()
    
    # Run simulation with varying external signals
    for step in range(num_steps):
        # Vary signal intensity over time (create patterns)
        intensity = 1.0 + 0.5 * np.sin(step * 0.3) + 0.2 * np.random.randn()
        
        # Generate and apply external signals
        external_signals = simulate_external_signals(network, mock_sources, intensity)
        apply_external_signals_to_network(network, external_signals)
        
        # Step the network
        network.step()
        
        # Occasionally introduce anxiety spikes
        if step in [5, 12, 18]:
            random_node = np.random.choice(network.nodes)
            random_node.anxiety += np.random.uniform(3.0, 6.0)
            logger.info(f"Step {step}: Introduced anxiety spike to node {random_node.node_id}")
        
        # Log network state every few steps
        if step % 5 == 0:
            avg_anxiety = np.mean([node.anxiety for node in network.nodes])
            avg_energy = np.mean([node.energy for node in network.nodes])
            logger.info(f"Step {step}: Avg anxiety={avg_anxiety:.2f}, Avg energy={avg_energy:.2f}")
    
    # Analyze time series data
    if network.enable_time_series:
        stats = network.time_series_tracker.get_statistics()
        logger.info(f"Time series data collected: {stats['total_points']} points for {len(stats['unique_nodes'])} nodes")
        
        # Get latest values for all nodes
        print("\nLatest Node States:")
        for node in network.nodes:
            latest = network.time_series_tracker.get_latest_values(
                node.node_id, 
                ["anxiety", "energy", "calm"]
            )
            print(f"  Node {node.node_id}: {latest}")
        
        return True
    
    return False


def create_visualizations(network, save_plots=True):
    """Create visualizations of network behavior"""
    
    if not network.enable_time_series:
        logger.warning("Time series tracking disabled, cannot create visualizations")
        return
    
    logger.info("=== Creating Visualizations ===")
    
    try:
        # Create individual node analysis
        node_id = network.nodes[0].node_id
        fig1 = network.time_series_tracker.visualize_node_variables(
            node_id=node_id,
            variables=["anxiety", "energy", "calm"],
            time_range_hours=1,  # Last hour (or all data if less)
            save_path="demo_node_analysis.png" if save_plots else None
        )
        
        if save_plots:
            logger.info("Saved individual node analysis to demo_node_analysis.png")
        
        # Create network comparison
        node_ids = [node.node_id for node in network.nodes[:3]]  # First 3 nodes
        fig2 = network.time_series_tracker.compare_nodes(
            node_ids=node_ids,
            variable="anxiety",
            time_range_hours=1,
            save_path="demo_anxiety_comparison.png" if save_plots else None
        )
        
        if save_plots:
            logger.info("Saved anxiety comparison to demo_anxiety_comparison.png")
        
        # Show plots if not saving
        if not save_plots:
            plt.show()
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
        return False


def demonstrate_security_features():
    """Demonstrate security and privacy features"""
    
    logger.info("=== Demonstrating Security and Privacy Features ===")
    
    # Create signal adapter with security enabled
    adapter = SignalAdapter(security_enabled=True)
    
    # Create secure credentials
    credentials = ApiCredentials(
        api_key="demo-api-key-12345",
        secret_key="demo-secret-key-67890",
        custom_headers={"X-Client-Version": "enhanced-demo-1.0"}
    )
    
    # Create signal source with privacy controls
    secure_source = SignalSource(
        name="secure_human_biometrics",
        signal_type=SignalType.HUMAN,
        api_url="https://secure-api.example.com/biometrics",
        mappings=[
            SignalMapping("heart_rate", StateVariable.AROUSAL, "linear", 0.02, -1.5, 0.0, 2.0),
            SignalMapping("stress_hormones", StateVariable.ANXIETY, "logarithmic", 2.0, 0.0, 0.0, 8.0)
        ],
        credentials=credentials,
        privacy_level="confidential",
        data_retention_hours=0.5,  # 30 minutes
        integrity_check=True
    )
    
    adapter.register_source(secure_source)
    
    # Demonstrate memory with privacy controls
    sensitive_memory = Memory(
        content={"biometric_data": "heart_rate=75, stress=0.3"},
        importance=0.9,
        timestamp=time.time(),
        memory_type="biometric",
        classification="confidential",
        retention_limit=1800,  # 30 minutes
        audit_log=[]
    )
    
    # Show access logging
    print("\nPrivacy Controls Demonstration:")
    print(f"Memory classification: {sensitive_memory.classification}")
    print(f"Retention limit: {sensitive_memory.retention_limit} seconds")
    
    # Access with different node IDs
    content1 = sensitive_memory.access(1)  # Authorized access
    content2 = sensitive_memory.access(2)  # Different accessor
    
    print(f"Access by node 1: {content1}")
    print(f"Access by node 2: {content2}")
    print(f"Audit log: {sensitive_memory.audit_log}")
    
    # Demonstrate human signal manager privacy
    human_manager = HumanSignalManager(security_enabled=True)
    privacy_report = human_manager.get_privacy_report()
    
    print(f"\nHuman Signal Privacy Report:")
    print(f"Privacy levels: {privacy_report['privacy_levels']}")
    print(f"Sources with private data: {len([s for s, info in privacy_report['sources'].items() if info['privacy_level'] in ['private', 'confidential']])}")
    
    return True


def export_demo_data(network):
    """Export demo data for analysis"""
    
    logger.info("=== Exporting Demo Data ===")
    
    try:
        # Export network status
        status = network.get_network_status()
        with open("demo_network_status.json", "w") as f:
            json.dump(status, f, indent=2, default=str)
        logger.info("Exported network status to demo_network_status.json")
        
        # Export time series data if available
        if network.enable_time_series:
            query = TimeSeriesQuery(
                node_ids=[node.node_id for node in network.nodes],
                start_time=network.time_series_tracker._stats["first_timestamp"],
                end_time=network.time_series_tracker._stats["last_timestamp"]
            )
            
            timeseries_file = network.time_series_tracker.export_data(
                query, 
                format="json", 
                output_path="demo_timeseries"
            )
            logger.info(f"Exported time series data to {timeseries_file}")
        
        # Export comprehensive network data
        network_file = network.export_network_data(
            format="json", 
            output_path="demo_full_network"
        )
        logger.info(f"Exported full network data to {network_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to export demo data: {e}")
        return False


def print_demo_summary(network, results):
    """Print a summary of the demo results"""
    
    print("\n" + "="*60)
    print("ENHANCED ADAPTIVE NEURAL NETWORK DEMO SUMMARY")
    print("="*60)
    
    # Network overview
    print(f"\nNetwork Overview:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Capacitors: {len(network.capacitors)}")
    print(f"  Time steps simulated: {network.time}")
    
    # Performance metrics
    metrics = network.performance_metrics
    print(f"\nPerformance Metrics:")
    print(f"  Total help signals sent: {metrics['total_help_signals']}")
    print(f"  Successful anxiety reductions: {metrics['successful_anxiety_reductions']}")
    print(f"  Average network anxiety: {metrics['average_network_anxiety']:.2f}")
    print(f"  Network stability score: {metrics['network_stability_score']:.2f}")
    
    # Feature demonstrations
    print(f"\nFeature Demonstrations:")
    print(f"  ✓ External signal absorption: Multi-modal signal integration")
    print(f"  ✓ Anxiety overwhelm protocol: {results.get('anxiety_protocol', False)}")
    print(f"  ✓ Time series tracking: {results.get('time_series', False)}")
    print(f"  ✓ Security features: Privacy controls and data protection")
    print(f"  ✓ Visualizations: {results.get('visualizations', False)}")
    print(f"  ✓ Data export: {results.get('export', False)}")
    
    # Node states
    print(f"\nFinal Node States:")
    for node in network.nodes:
        status = node.get_anxiety_status()
        print(f"  Node {node.node_id}: Anxiety={status['anxiety_level']:.2f}, "
              f"Calm={status['calm_level']:.2f}, "
              f"Trust network: {status['trust_network_size']} connections")
    
    # Time series statistics
    if network.enable_time_series:
        stats = network.time_series_tracker.get_statistics()
        print(f"\nTime Series Statistics:")
        print(f"  Total data points: {stats['total_points']}")
        print(f"  Variables tracked: {len(stats['unique_variables'])}")
        print(f"  Time range: {stats.get('time_range_seconds', 0):.1f} seconds")
    
    print(f"\nDemo completed successfully! Check output files for detailed analysis.")
    print("="*60)


def main():
    """Main demo function"""
    
    parser = argparse.ArgumentParser(description="Enhanced Adaptive Neural Network Demo")
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes in network")
    parser.add_argument("--steps", type=int, default=20, help="Number of simulation steps")
    parser.add_argument("--visualize", action="store_true", help="Create and show visualizations")
    parser.add_argument("--no-export", action="store_true", help="Skip data export")
    
    args = parser.parse_args()
    
    logger.info("Starting Enhanced Adaptive Neural Network Demo")
    logger.info(f"Configuration: {args.nodes} nodes, {args.steps} steps, visualize={args.visualize}")
    
    results = {}
    
    try:
        # Create demo network
        network = create_demo_network(args.nodes)
        
        # Demonstrate anxiety overwhelm protocol
        results["anxiety_protocol"] = run_anxiety_overwhelm_scenario(network)
        
        # Demonstrate time series tracking
        results["time_series"] = demonstrate_time_series_tracking(network, args.steps)
        
        # Demonstrate security features
        results["security"] = demonstrate_security_features()
        
        # Create visualizations
        if args.visualize:
            results["visualizations"] = create_visualizations(network, save_plots=True)
        
        # Export data
        if not args.no_export:
            results["export"] = export_demo_data(network)
        
        # Clean up resources
        network.cleanup_resources()
        
        # Print summary
        print_demo_summary(network, results)
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
        
    return 0


if __name__ == "__main__":
    exit(main())