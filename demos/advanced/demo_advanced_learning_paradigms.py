#!/usr/bin/env python3
"""
Demo script for Advanced Learning Paradigms (Phase 3.2).

Showcases the new continual learning 2.0 features and advanced multi-agent systems:
- Progressive neural networks for task sequences
- Advanced image processing capabilities
- Memory-augmented architectures
- Lifelong learning benchmarks
- Swarm intelligence behaviors
- Negotiation and consensus protocols
- Competitive and cooperative multi-agent environments
"""

import time

import numpy as np
import torch

from adaptiveneuralnetwork.applications.continual_learning import (
    AdvancedImageProcessor,
    ContinualLearningConfig,
    ContinualLearningSystem,
    LifelongLearningBenchmark,
    MemoryAugmentedArchitecture,
    ProgressiveNeuralNetwork,
)
from core.social_learning import (
    CompetitiveCooperativeEnvironment,
    NegotiationProtocol,
    SwarmIntelligenceBehavior,
)


def demo_progressive_neural_networks():
    """Demonstrate progressive neural networks for continual learning."""
    print("=" * 60)
    print("🔄 PROGRESSIVE NEURAL NETWORKS DEMO")
    print("=" * 60)

    # Create progressive network
    network = ProgressiveNeuralNetwork(
        input_size=784,  # MNIST-like input
        hidden_sizes=[256, 128],
        output_size=10  # 10 classes
    )

    print("Created progressive network: 784 -> [256, 128] -> 10")

    # Add first task (e.g., digits 0-4)
    network.add_task(0)
    print(f"Added Task 0: Network has {network.num_tasks} task(s)")

    # Test forward pass on first task
    x = torch.randn(32, 784)
    output_task0 = network(x, task_id=0)
    print(f"Task 0 output shape: {output_task0.shape}")

    # Add second task (e.g., digits 5-9)
    network.add_task(1)
    print(f"Added Task 1: Network has {network.num_tasks} task(s)")

    # Test both tasks - first task parameters are frozen
    output_task0_after = network(x, task_id=0)
    output_task1 = network(x, task_id=1)

    print(f"Task 0 output (after Task 1): {output_task0_after.shape}")
    print(f"Task 1 output: {output_task1.shape}")

    # Verify parameter freezing
    frozen_params = sum(1 for p in network.task_columns[0].parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in network.task_columns[1].parameters() if p.requires_grad)

    print(f"Task 0 has {frozen_params} frozen parameters")
    print(f"Task 1 has {trainable_params} trainable parameters")
    print("✅ Progressive networks prevent catastrophic forgetting by freezing old tasks\n")


def demo_advanced_image_processing():
    """Demonstrate advanced image processing capabilities."""
    print("=" * 60)
    print("🖼️  ADVANCED IMAGE PROCESSING DEMO")
    print("=" * 60)

    config = ContinualLearningConfig(
        enable_spatial_attention=True,
        enable_temporal_pooling=True,
        visual_feature_dim=512
    )

    processor = AdvancedImageProcessor(
        input_channels=3,
        feature_dim=512,
        config=config
    )

    print("Created advanced image processor: 3 channels -> 512 features")
    print("✓ Spatial attention enabled")
    print("✓ Temporal pooling enabled")

    # Process single images
    batch_images = torch.randn(8, 3, 224, 224)  # Standard ImageNet size
    features = processor(batch_images)
    print(f"Single image batch: {batch_images.shape} -> {features.shape}")

    # Process temporal sequences (video)
    video_sequence = torch.randn(4, 10, 3, 224, 224)  # Batch, Time, Channels, H, W
    temporal_features = processor(video_sequence, temporal_sequence=True)
    print(f"Video sequence: {video_sequence.shape} -> {temporal_features.shape}")

    # Add task-specific adaptation
    processor.add_task_adaptation(0)  # Task 0: natural images
    processor.add_task_adaptation(1)  # Task 1: medical images

    task0_features = processor(batch_images, task_id=0)
    task1_features = processor(batch_images, task_id=1)

    print(f"Task 0 (natural) features: {task0_features.shape}")
    print(f"Task 1 (medical) features: {task1_features.shape}")
    print("✅ Advanced image processing with multi-scale attention and temporal modeling\n")


def demo_memory_augmented_architecture():
    """Demonstrate memory-augmented architectures."""
    print("=" * 60)
    print("🧠 MEMORY-AUGMENTED ARCHITECTURE DEMO")
    print("=" * 60)

    config = ContinualLearningConfig(
        memory_consolidation_strength=0.5,
        memory_retrieval_temperature=1.0
    )

    memory_arch = MemoryAugmentedArchitecture(
        feature_dim=256,
        memory_dim=128,
        config=config
    )

    print("Created memory-augmented architecture: 256 -> 128")
    print("✓ Working memory (LSTM)")
    print("✓ Semantic memory (Transformer)")
    print("✓ Attention-based retrieval")

    # Process features
    features = torch.randn(16, 256)
    enhanced_features, memory_state = memory_arch(features)

    print(f"Input features: {features.shape}")
    print(f"Enhanced features: {enhanced_features.shape}")
    print(f"Memory state: {memory_state.shape}")

    # Update semantic memory with important experiences
    importance = torch.rand(16) * 0.8 + 0.2  # Random importance scores
    memory_arch.update_semantic_memory(memory_state, importance)

    print(f"Updated semantic memory with {len(importance)} experiences")

    # Retrieve relevant memories for new query
    query = torch.randn(4, 128)
    relevant_memories = memory_arch.retrieve_relevant_memories(query)
    print(f"Retrieved memories for query {query.shape}: {relevant_memories.shape}")
    print("✅ Memory-augmented processing enhances knowledge retention and transfer\n")


def demo_lifelong_learning_benchmark():
    """Demonstrate lifelong learning benchmark system."""
    print("=" * 60)
    print("📊 LIFELONG LEARNING BENCHMARK DEMO")
    print("=" * 60)

    config = ContinualLearningConfig(num_tasks=3)
    benchmark = LifelongLearningBenchmark(config)

    print("Created lifelong learning benchmark system")
    print(f"Tracking metrics: {', '.join(benchmark.metric_names)}")

    # Simulate learning multiple tasks
    for task_id in range(3):
        print(f"\nSimulating Task {task_id}...")

        # Start timing
        benchmark.start_task_timing(task_id)

        # Simulate learning time
        time.sleep(0.1)

        # End timing
        benchmark.end_task_timing(task_id)

        # Record memory usage (mock model)
        mock_model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.Linear(50, 10)
        )
        benchmark.record_memory_usage(task_id, mock_model)

        # Record adaptation events
        benchmark.record_adaptation_event(task_id, 'adaptation', np.random.uniform(0.1, 0.5))
        benchmark.record_adaptation_event(task_id, 'drift_detection', np.random.uniform(0.0, 0.3))

        duration = benchmark.task_timings[task_id]['duration']
        print(f"  Task {task_id} completed in {duration:.3f}s")

    # Generate comprehensive benchmark report
    print("\n📈 Generating benchmark report...")

    # Mock task loaders and model for evaluation
    mock_task_loaders = dict.fromkeys(range(3))  # Placeholder
    baseline_performances = {0: 0.75, 1: 0.70, 2: 0.73}

    # Create mock continual learning system
    config = ContinualLearningConfig(
        input_size=64, output_size=10, hidden_layers=[32, 16],
        enable_lifelong_benchmarking=True,
        enable_metaplasticity=False, synaptic_consolidation=False
    )
    mock_system = ContinualLearningSystem(config)

    # Simulate metrics
    mock_metrics = {
        'average_accuracy': 0.72,
        'backward_transfer': 0.05,
        'average_forgetting': 0.02,
        'learning_efficiency': 2.4,
        'memory_stability': 0.85,
        'adaptation_speed': 1.2,
        'knowledge_retention': 0.78
    }

    benchmark.metrics_history.append({
        'timestamp': time.time(),
        'num_tasks': 3,
        'metrics': mock_metrics
    })

    report = benchmark.generate_benchmark_report()

    print("Benchmark Report:")
    print(f"  Total tasks learned: {report['performance_summary']['total_tasks_learned']}")
    print(f"  Learning efficiency: {report['latest_metrics']['learning_efficiency']:.2f}")
    print(f"  Memory stability: {report['latest_metrics']['memory_stability']:.2f}")
    print(f"  Knowledge retention: {report['latest_metrics']['knowledge_retention']:.2f}")
    print("✅ Comprehensive benchmarking tracks lifelong learning performance\n")


def demo_swarm_intelligence():
    """Demonstrate swarm intelligence behaviors."""
    print("=" * 60)
    print("🐝 SWARM INTELLIGENCE DEMO")
    print("=" * 60)

    # Create swarm agents
    swarm_size = 5
    agents = []

    for i in range(swarm_size):
        agent = SwarmIntelligenceBehavior(agent_id=i, swarm_size=swarm_size)
        agents.append(agent)

    print(f"Created swarm with {swarm_size} agents")
    print("✓ Particle Swarm Optimization")
    print("✓ Flocking behaviors")
    print("✓ Ant Colony Optimization")

    # Demonstrate PSO convergence
    print("\n🎯 Particle Swarm Optimization:")
    global_best_position = np.random.randn(10)

    for iteration in range(3):
        print(f"  Iteration {iteration + 1}:")
        for agent in agents:
            new_position = agent.update_pso_position(global_best_position)
            distance_to_best = np.linalg.norm(new_position - global_best_position)
            print(f"    Agent {agent.agent_id}: distance to global best = {distance_to_best:.3f}")

    # Demonstrate flocking behavior
    print("\n🐦 Flocking Behavior:")
    for agent in agents[:2]:  # Show first 2 agents
        neighbor_positions = [a.position for a in agents if a.agent_id != agent.agent_id]
        neighbor_velocities = [a.velocity for a in agents if a.agent_id != agent.agent_id]

        flocking_force = agent.flocking_behavior(neighbor_positions, neighbor_velocities)
        force_magnitude = np.linalg.norm(flocking_force)
        print(f"  Agent {agent.agent_id}: flocking force magnitude = {force_magnitude:.3f}")

    # Demonstrate ant colony decision making
    print("\n🐜 Ant Colony Decision Making:")
    options = [
        {'key': 'path_A', 'quality': 0.8},
        {'key': 'path_B', 'quality': 0.6},
        {'key': 'path_C', 'quality': 0.9}
    ]

    pheromone_trails = {
        'path_A': 0.5,
        'path_B': 0.8,
        'path_C': 0.3
    }

    decisions = []
    for agent in agents:
        decision = agent.ant_colony_decision(options, pheromone_trails)
        decisions.append(decision)
        chosen_path = options[decision]['key']
        print(f"  Agent {agent.agent_id}: chose {chosen_path}")

    # Show collective decision pattern
    decision_counts = np.bincount(decisions)
    print(f"  Collective decision: {decision_counts} votes for paths A, B, C")
    print("✅ Swarm intelligence enables emergent collective behavior\n")


def demo_negotiation_protocol():
    """Demonstrate negotiation and consensus protocols."""
    print("=" * 60)
    print("🤝 NEGOTIATION & CONSENSUS DEMO")
    print("=" * 60)

    # Create negotiating agents
    agents = []
    for i in range(3):
        agent = NegotiationProtocol(agent_id=i)
        agents.append(agent)

    print(f"Created {len(agents)} negotiating agents")

    # Demonstrate auction mechanism
    print("\n💰 Auction Mechanism:")

    # Agent 0 initiates auction
    item = {'name': 'territory_X', 'value': 100, 'duration': 30}
    auction = agents[0].initiate_auction(item, auction_type='first_price')

    print(f"  Agent {auction['auctioneer']} initiated auction for {item['name']}")
    print(f"  Auction type: {auction['type']}")

    # Other agents submit bids
    private_values = [0, 85, 92, 78]  # Agent 0 doesn't bid on own auction

    for i, agent in enumerate(agents):
        if i != auction['auctioneer']:
            bid = agent.submit_bid(auction, private_values[i])
            if bid:
                auction['bids'].append(bid)
                print(f"  Agent {i} bid ${bid['amount']:.2f} (private value: ${private_values[i]})")

    # Resolve auction
    result = agents[0].resolve_auction(auction)
    print(f"  Winner: Agent {result['winner']}")
    print(f"  Price: ${result['price']:.2f}")
    print(f"  Efficiency: {result['efficiency']:.2f}")

    # Demonstrate multi-issue negotiation
    print("\n📋 Multi-Issue Negotiation:")

    issues = [
        {'name': 'price', 'range': [100, 200], 'importance': 1.0},
        {'name': 'delivery_time', 'range': [1, 10], 'importance': 0.8},
        {'name': 'quality', 'range': [0.5, 1.0], 'importance': 0.9}
    ]

    opponent_preferences = {
        'price': 0.9,        # Opponent cares a lot about price
        'delivery_time': 0.3, # Opponent doesn't care much about delivery
        'quality': 0.8       # Opponent cares about quality
    }

    negotiation_result = agents[1].multi_issue_negotiation(issues, opponent_preferences)

    print(f"  Agent {agents[1].agent_id} negotiation offer:")
    for issue, value in negotiation_result['offer'].items():
        print(f"    {issue}: {value:.2f}")
    print(f"  Expected value: {negotiation_result['expected_value']:.2f}")
    print(f"  Strategy: {negotiation_result['strategy']}")
    print(f"  Concessions: {negotiation_result['concessions']}")
    print("✅ Advanced negotiation enables win-win outcomes\n")


def demo_competitive_cooperative_environment():
    """Demonstrate competitive and cooperative multi-agent environments."""
    print("=" * 60)
    print("⚔️🤝 COMPETITIVE-COOPERATIVE ENVIRONMENT DEMO")
    print("=" * 60)

    # Create mixed environment
    env = CompetitiveCooperativeEnvironment(num_agents=4, environment_type='mixed')

    print(f"Created mixed environment for {env.num_agents} agents")
    print(f"Environment type: {env.environment_type}")

    # Add agents with different resource profiles
    agent_profiles = [
        {'food': 20, 'territory': 5, 'information': 10},   # Food specialist
        {'food': 8, 'territory': 18, 'information': 12},   # Territory specialist
        {'food': 12, 'territory': 8, 'information': 25},   # Information specialist
        {'food': 15, 'territory': 15, 'information': 15}   # Generalist
    ]

    for i, profile in enumerate(agent_profiles):
        env.add_agent(i, profile)
        print(f"  Agent {i}: {profile}")

    print("\nInitial environment state:")
    state = env.get_environment_state()
    print(f"  Round: {state['round']}")
    print(f"  Global resources: {state['resources']}")
    print(f"  Competition level: {state['competition_level']}")

    # Run competitive round
    print("\n⚔️  Competitive Round:")
    comp_result = env.run_competitive_round()
    print(f"  Round {comp_result['round']} - {comp_result['type']}")
    print(f"  Interactions: {len(comp_result['interactions'])}")

    for interaction in comp_result['interactions']:
        winner = interaction['winner']
        loser = interaction['loser']
        resource = interaction['resource']
        transfer = interaction['transfer_amount']
        print(f"    Agent {winner} won {resource} from Agent {loser} (+{transfer:.1f})")

    # Run cooperative round
    print("\n🤝 Cooperative Round:")
    coop_result = env.run_cooperative_round()
    print(f"  Round {coop_result['round']} - {coop_result['type']}")
    print(f"  Coalitions formed: {len(coop_result['coalitions_formed'])}")

    for coalition_result in coop_result['coalitions_formed']:
        members = coalition_result['coalition']
        success = coalition_result['success']
        reward = coalition_result['collective_reward']
        print(f"    Coalition {members}: {'succeeded' if success else 'failed'} (reward: {reward:.1f})")

    # Show final scores
    print("\n🏆 Final Scores:")
    for agent_id, score in env.agent_scores.items():
        print(f"  Agent {agent_id}: {score:.1f} points")

    # Final environment state
    final_state = env.get_environment_state()
    print("\nFinal environment state:")
    print(f"  Total rounds: {final_state['round']}")
    print(f"  Average score: {final_state['average_score']:.1f}")
    print(f"  Cooperation events: {final_state['cooperation_events']}")
    print("✅ Dynamic environment balances competition and cooperation\n")


def main():
    """Run all advanced learning paradigms demos."""
    print("🚀 ADVANCED LEARNING PARADIGMS DEMO")
    print("Phase 3.2: Continual Learning 2.0 & Advanced Multi-Agent Systems")
    print("=" * 80)

    try:
        # Continual Learning 2.0 demos
        demo_progressive_neural_networks()
        demo_advanced_image_processing()
        demo_memory_augmented_architecture()
        demo_lifelong_learning_benchmark()

        # Advanced Multi-Agent Systems demos
        demo_swarm_intelligence()
        demo_negotiation_protocol()
        demo_competitive_cooperative_environment()

        print("=" * 80)
        print("✅ ALL ADVANCED LEARNING PARADIGMS DEMONSTRATED SUCCESSFULLY!")
        print("🎉 Phase 3.2 implementation complete!")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
