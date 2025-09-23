"""
Test suite for Advanced Learning Paradigms (Phase 3.2).

Tests the new continual learning 2.0 features and advanced multi-agent systems.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List

from adaptiveneuralnetwork.applications.continual_learning import (
    ContinualLearningConfig, ContinualLearningSystem, ProgressiveNeuralNetwork,
    AdvancedImageProcessor, MemoryAugmentedArchitecture, LifelongLearningBenchmark
)
from core.social_learning import (
    SwarmIntelligenceBehavior, NegotiationProtocol, CompetitiveCooperativeEnvironment
)


class TestProgressiveNeuralNetworks:
    """Test progressive neural networks for continual learning."""
    
    def test_progressive_network_creation(self):
        """Test creating a progressive neural network."""
        network = ProgressiveNeuralNetwork(
            input_size=784, 
            hidden_sizes=[256, 128], 
            output_size=10
        )
        
        assert network.num_tasks == 0
        assert len(network.task_columns) == 0
        assert len(network.lateral_connections) == 0
        assert len(network.output_heads) == 0
    
    def test_adding_tasks(self):
        """Test adding tasks to progressive network."""
        network = ProgressiveNeuralNetwork(
            input_size=784, 
            hidden_sizes=[256, 128], 
            output_size=10
        )
        
        # Add first task
        network.add_task(0)
        assert network.num_tasks == 1
        assert len(network.task_columns) == 1
        assert len(network.output_heads) == 1
        
        # Add second task
        network.add_task(1)
        assert network.num_tasks == 2
        assert len(network.task_columns) == 2
        assert len(network.output_heads) == 2
        assert len(network.lateral_connections) == 2  # Including empty list for first task
    
    def test_progressive_forward_pass(self):
        """Test forward pass through progressive network."""
        network = ProgressiveNeuralNetwork(
            input_size=10, 
            hidden_sizes=[8, 6], 
            output_size=3
        )
        
        # Add tasks
        network.add_task(0)
        network.add_task(1)
        
        # Test forward pass
        x = torch.randn(2, 10)
        
        # Task 0 forward pass
        output0 = network(x, task_id=0)
        assert output0.shape == (2, 3)
        
        # Task 1 forward pass
        output1 = network(x, task_id=1)
        assert output1.shape == (2, 3)
    
    def test_parameter_freezing(self):
        """Test that previous task parameters are frozen."""
        network = ProgressiveNeuralNetwork(
            input_size=10, 
            hidden_sizes=[8], 
            output_size=3
        )
        
        # Add first task
        network.add_task(0)
        
        # Check that parameters are trainable
        for param in network.task_columns[0].parameters():
            assert param.requires_grad
        
        # Add second task
        network.add_task(1)
        
        # Check that first task parameters are frozen
        for param in network.task_columns[0].parameters():
            assert not param.requires_grad
        
        # Check that second task parameters are trainable
        for param in network.task_columns[1].parameters():
            assert param.requires_grad


class TestAdvancedImageProcessor:
    """Test advanced image processing capabilities."""
    
    def test_image_processor_creation(self):
        """Test creating advanced image processor."""
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
        
        assert processor.feature_dim == 512
        assert hasattr(processor, 'spatial_attention')
        assert hasattr(processor, 'temporal_pooling')
    
    def test_image_processing(self):
        """Test image processing forward pass."""
        config = ContinualLearningConfig(
            enable_spatial_attention=True,
            visual_feature_dim=256
        )
        
        processor = AdvancedImageProcessor(
            input_channels=3, 
            feature_dim=256, 
            config=config
        )
        
        # Test single image
        x = torch.randn(2, 3, 64, 64)
        features = processor(x)
        assert features.shape == (2, 256)
        
        # Test temporal sequence
        x_temporal = torch.randn(2, 5, 3, 64, 64)  # Batch, Time, Channels, Height, Width
        features_temporal = processor(x_temporal, temporal_sequence=True)
        assert features_temporal.shape == (2, 256) 
    
    def test_task_adaptation(self):
        """Test task-specific adaptation."""
        config = ContinualLearningConfig(visual_feature_dim=128)
        
        processor = AdvancedImageProcessor(
            input_channels=3, 
            feature_dim=128, 
            config=config
        )
        
        # Add task-specific adaptation
        processor.add_task_adaptation(0)
        processor.add_task_adaptation(1)
        
        assert len(processor.task_bn_layers) == 2
        
        # Test with task-specific processing
        x = torch.randn(2, 3, 32, 32)
        features0 = processor(x, task_id=0)
        features1 = processor(x, task_id=1)
        
        assert features0.shape == (2, 128)
        assert features1.shape == (2, 128)


class TestMemoryAugmentedArchitecture:
    """Test memory-augmented architectures."""
    
    def test_memory_architecture_creation(self):
        """Test creating memory-augmented architecture."""
        config = ContinualLearningConfig()
        
        memory_arch = MemoryAugmentedArchitecture(
            feature_dim=256, 
            memory_dim=128, 
            config=config
        )
        
        assert memory_arch.feature_dim == 256
        assert memory_arch.memory_dim == 128
        assert hasattr(memory_arch, 'working_memory')
        assert hasattr(memory_arch, 'semantic_memory')
    
    def test_memory_processing(self):
        """Test memory-augmented processing."""
        config = ContinualLearningConfig()
        
        memory_arch = MemoryAugmentedArchitecture(
            feature_dim=128, 
            memory_dim=64, 
            config=config
        )
        
        # Test forward pass
        features = torch.randn(4, 128)
        enhanced_features, memory_state = memory_arch(features)
        
        assert enhanced_features.shape == (4, 64)
        assert memory_state.shape == (4, 64)
    
    def test_memory_update(self):
        """Test semantic memory updates."""
        config = ContinualLearningConfig()
        
        memory_arch = MemoryAugmentedArchitecture(
            feature_dim=64, 
            memory_dim=32, 
            config=config
        )
        
        # Update semantic memory
        features = torch.randn(3, 32)
        importance = torch.tensor([0.8, 0.6, 0.9])
        
        initial_pointer = memory_arch.memory_write_pointer.item()
        memory_arch.update_semantic_memory(features, importance)
        
        assert memory_arch.memory_write_pointer.item() == initial_pointer + 3


class TestLifelongLearningBenchmark:
    """Test lifelong learning benchmark system."""
    
    def test_benchmark_creation(self):
        """Test creating benchmark system."""
        config = ContinualLearningConfig()
        benchmark = LifelongLearningBenchmark(config)
        
        assert len(benchmark.metrics_history) == 0
        assert len(benchmark.task_timings) == 0
        assert 'average_accuracy' in benchmark.metric_names
        assert 'knowledge_retention' in benchmark.metric_names
    
    def test_timing_tracking(self):
        """Test task timing tracking."""
        config = ContinualLearningConfig()
        benchmark = LifelongLearningBenchmark(config)
        
        # Start timing
        benchmark.start_task_timing(0)
        assert 0 in benchmark.task_timings
        assert 'start' in benchmark.task_timings[0]
        
        # End timing
        import time
        time.sleep(0.01)  # Small delay
        benchmark.end_task_timing(0)
        
        assert 'end' in benchmark.task_timings[0]
        assert 'duration' in benchmark.task_timings[0]
        assert benchmark.task_timings[0]['duration'] > 0
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        config = ContinualLearningConfig()
        benchmark = LifelongLearningBenchmark(config)
        
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 2)
        )
        
        benchmark.record_memory_usage(0, model)
        
        assert 0 in benchmark.memory_usage
        assert 'total_parameters' in benchmark.memory_usage[0]
        assert 'trainable_parameters' in benchmark.memory_usage[0]
        assert benchmark.memory_usage[0]['total_parameters'] > 0
    
    def test_adaptation_events(self):
        """Test adaptation event recording."""
        config = ContinualLearningConfig()
        benchmark = LifelongLearningBenchmark(config)
        
        # Record adaptation events
        benchmark.record_adaptation_event(0, 'adaptation', 0.5)
        benchmark.record_adaptation_event(0, 'drift_detection', 0.3)
        
        assert 0 in benchmark.adaptation_events
        assert len(benchmark.adaptation_events[0]) == 2


class TestSwarmIntelligence:
    """Test swarm intelligence behaviors."""
    
    def test_swarm_behavior_creation(self):
        """Test creating swarm intelligence behavior."""
        behavior = SwarmIntelligenceBehavior(agent_id=0, swarm_size=10)
        
        assert behavior.agent_id == 0
        assert behavior.swarm_size == 10
        assert len(behavior.position) == 10
        assert len(behavior.velocity) == 10
    
    def test_pso_update(self):
        """Test particle swarm optimization update."""
        behavior = SwarmIntelligenceBehavior(agent_id=0, swarm_size=5)
        
        initial_position = behavior.position.copy()
        global_best = np.random.randn(10)
        
        new_position = behavior.update_pso_position(global_best)
        
        assert len(new_position) == 10
        assert not np.array_equal(initial_position, new_position)
    
    def test_flocking_behavior(self):
        """Test flocking behavior implementation."""
        behavior = SwarmIntelligenceBehavior(agent_id=0, swarm_size=5)
        
        # Create neighbor data
        neighbor_positions = [np.random.randn(10) for _ in range(3)]
        neighbor_velocities = [np.random.randn(10) for _ in range(3)]
        
        flocking_force = behavior.flocking_behavior(neighbor_positions, neighbor_velocities)
        
        assert len(flocking_force) == 10
        assert isinstance(flocking_force, np.ndarray)
    
    def test_ant_colony_decision(self):
        """Test ant colony optimization decision making."""
        behavior = SwarmIntelligenceBehavior(agent_id=0, swarm_size=5)
        
        options = [
            {'key': 'option1', 'quality': 0.8},
            {'key': 'option2', 'quality': 0.6},
            {'key': 'option3', 'quality': 0.9}
        ]
        
        pheromone_trails = {
            'option1': 0.5,
            'option2': 0.8,
            'option3': 0.3
        }
        
        decision = behavior.ant_colony_decision(options, pheromone_trails)
        
        assert 0 <= decision < len(options)
    
    def test_pheromone_deposit(self):
        """Test pheromone deposition."""
        behavior = SwarmIntelligenceBehavior(agent_id=0, swarm_size=5)
        
        path = ['step1', 'step2', 'step3']
        success = 0.8
        
        pheromone_deposit = behavior.deposit_pheromone(path, success)
        
        assert len(pheromone_deposit) == 3
        for step in path:
            assert step in pheromone_deposit
            assert pheromone_deposit[step] > 0


class TestNegotiationProtocol:
    """Test negotiation and consensus protocols."""
    
    def test_negotiation_creation(self):
        """Test creating negotiation protocol."""
        protocol = NegotiationProtocol(agent_id=1)
        
        assert protocol.agent_id == 1
        assert len(protocol.negotiation_history) == 0
        assert protocol.current_strategy in protocol.negotiation_strategies
    
    def test_auction_initiation(self):
        """Test auction initiation."""
        protocol = NegotiationProtocol(agent_id=1)
        
        item = {'name': 'resource', 'value': 100, 'duration': 30}
        auction = protocol.initiate_auction(item, 'first_price')
        
        assert auction['auctioneer'] == 1
        assert auction['item'] == item
        assert auction['type'] == 'first_price'
        assert auction['status'] == 'open'
    
    def test_bid_submission(self):
        """Test bid submission."""
        protocol = NegotiationProtocol(agent_id=1)
        
        # Create auction
        item = {'name': 'resource', 'value': 100}
        auction = protocol.initiate_auction(item)
        
        # Submit bid
        bid = protocol.submit_bid(auction, private_value=80)
        
        assert bid is not None
        assert bid['bidder'] == 1
        assert bid['amount'] > 0
        assert bid['private_value'] == 80
    
    def test_auction_resolution(self):
        """Test auction resolution."""
        protocol = NegotiationProtocol(agent_id=1)
        
        # Create auction with bids
        auction = {
            'auction_id': 'test_auction',
            'type': 'first_price',
            'bids': [
                {'bidder': 1, 'amount': 80, 'private_value': 90},
                {'bidder': 2, 'amount': 85, 'private_value': 100},
                {'bidder': 3, 'amount': 75, 'private_value': 80}
            ]
        }
        
        result = protocol.resolve_auction(auction)
        
        assert result['winner'] == 2  # Highest bidder
        assert result['price'] == 85   # First-price auction
        assert result['status'] == 'completed'
        assert 'efficiency' in result
    
    def test_multi_issue_negotiation(self):
        """Test multi-issue negotiation."""
        protocol = NegotiationProtocol(agent_id=1)
        
        issues = [
            {'name': 'price', 'range': [100, 200], 'importance': 1.0},
            {'name': 'delivery', 'range': [1, 10], 'importance': 0.8},
            {'name': 'quality', 'range': [0.5, 1.0], 'importance': 0.9}
        ]
        
        opponent_preferences = {
            'price': 0.9,
            'delivery': 0.3,
            'quality': 0.8
        }
        
        result = protocol.multi_issue_negotiation(issues, opponent_preferences)
        
        assert 'offer' in result
        assert 'expected_value' in result
        assert 'strategy' in result
        assert len(result['offer']) == 3


class TestCompetitiveCooperativeEnvironment:
    """Test competitive and cooperative multi-agent environments."""
    
    def test_environment_creation(self):
        """Test creating competitive-cooperative environment."""
        env = CompetitiveCooperativeEnvironment(num_agents=5, environment_type='mixed')
        
        assert env.num_agents == 5
        assert env.environment_type == 'mixed'
        assert len(env.agents) == 0
        assert len(env.resources) > 0
    
    def test_agent_addition(self):
        """Test adding agents to environment."""
        env = CompetitiveCooperativeEnvironment(num_agents=3)
        
        # Add agents
        env.add_agent(0)
        env.add_agent(1, {'food': 15, 'territory': 8, 'information': 12})
        
        assert len(env.agents) == 2
        assert env.agents[0]['id'] == 0
        assert env.agents[1]['resources']['food'] == 15
    
    def test_competitive_round(self):
        """Test competitive interaction round."""
        env = CompetitiveCooperativeEnvironment(num_agents=4)
        
        # Add agents
        for i in range(4):
            env.add_agent(i)
        
        # Run competitive round
        result = env.run_competitive_round()
        
        assert result['type'] == 'competitive'
        assert 'interactions' in result
        assert len(result['interactions']) >= 0  # May be 0 if odd number of agents
    
    def test_cooperative_round(self):
        """Test cooperative interaction round."""
        env = CompetitiveCooperativeEnvironment(num_agents=4)
        
        # Add agents with complementary resources
        env.add_agent(0, {'food': 20, 'territory': 2, 'information': 5})
        env.add_agent(1, {'food': 2, 'territory': 18, 'information': 8})
        env.add_agent(2, {'food': 8, 'territory': 6, 'information': 20})
        env.add_agent(3, {'food': 10, 'territory': 10, 'information': 10})
        
        # Run cooperative round
        result = env.run_cooperative_round()
        
        assert result['type'] == 'cooperative'
        assert 'coalitions_formed' in result
    
    def test_mixed_round(self):
        """Test mixed competitive-cooperative round."""
        env = CompetitiveCooperativeEnvironment(num_agents=3, environment_type='mixed')
        
        # Add agents
        for i in range(3):
            env.add_agent(i)
        
        # Run mixed round
        result = env.run_mixed_round()
        
        assert result['type'] in ['competitive', 'cooperative']
        assert 'round' in result
    
    def test_environment_state(self):
        """Test environment state tracking."""
        env = CompetitiveCooperativeEnvironment(num_agents=2)
        
        env.add_agent(0)
        env.add_agent(1)
        
        state = env.get_environment_state()
        
        assert state['agents'] == 2
        assert 'resources' in state
        assert 'round' in state
        assert 'average_score' in state


@pytest.mark.integration
class TestIntegratedAdvancedLearning:
    """Integration tests for advanced learning paradigms."""
    
    def test_continual_learning_with_advanced_features(self):
        """Test continual learning system with advanced features enabled."""
        config = ContinualLearningConfig(
            input_size=64,
            output_size=5,
            hidden_layers=[32, 16],
            enable_progressive_networks=True,
            enable_memory_augmentation=True,
            enable_lifelong_benchmarking=True,
            enable_metaplasticity=False,  # Disable to avoid neuromorphic complexity
            enable_homeostatic_scaling=False,
            synaptic_consolidation=False,
            num_tasks=3
        )
        
        system = ContinualLearningSystem(config)
        
        # Verify advanced components are initialized
        assert system.progressive_network is not None
        assert system.memory_augmented_arch is not None
        assert system.benchmark_system is not None
        
        # Test individual components work
        x = torch.randn(4, 64)
        
        # Test progressive network
        system.progressive_network.add_task(0)
        prog_output = system.progressive_network(x, task_id=0)
        assert prog_output.shape == (4, 5)
        
        # Test memory augmented architecture
        features = torch.randn(4, 16)  # Last hidden layer size
        enhanced_features, memory_state = system.memory_augmented_arch(features)
        assert enhanced_features.shape == (4, 16)
        assert memory_state.shape == (4, 16)
    
    def test_multi_agent_swarm_with_negotiation(self):
        """Test integrated multi-agent system with swarm intelligence and negotiation."""
        # Create environment
        env = CompetitiveCooperativeEnvironment(num_agents=3, environment_type='mixed')
        
        # Create agents with swarm and negotiation capabilities
        agents = []
        for i in range(3):
            env.add_agent(i)
            
            swarm_behavior = SwarmIntelligenceBehavior(agent_id=i, swarm_size=3)
            negotiation = NegotiationProtocol(agent_id=i)
            
            agents.append({
                'id': i,
                'swarm': swarm_behavior,
                'negotiation': negotiation
            })
        
        # Test swarm coordination
        positions = [agent['swarm'].position for agent in agents]
        global_best = np.mean(positions, axis=0)
        
        # Update positions using PSO
        for agent in agents:
            new_pos = agent['swarm'].update_pso_position(global_best)
            assert len(new_pos) == 10
        
        # Test negotiation scenario
        item = {'name': 'territory', 'value': 50}
        auction = agents[0]['negotiation'].initiate_auction(item)
        
        # Other agents bid
        for agent in agents[1:]:
            bid = agent['negotiation'].submit_bid(auction, private_value=np.random.uniform(30, 70))
            if bid:
                auction['bids'].append(bid)
        
        # Resolve auction
        if auction['bids']:
            result = agents[0]['negotiation'].resolve_auction(auction)
            assert result['status'] == 'completed'
        
        # Run environment rounds
        comp_result = env.run_competitive_round()
        coop_result = env.run_cooperative_round()
        
        assert comp_result['type'] == 'competitive'
        assert coop_result['type'] == 'cooperative'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])