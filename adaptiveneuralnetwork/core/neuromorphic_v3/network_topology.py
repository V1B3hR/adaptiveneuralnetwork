"""
Advanced network topology and connectivity for 3rd generation neuromorphic computing.

This module implements hierarchical network structures, dynamic connectivity,
population-based processing, and realistic spike propagation delays.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .advanced_neurons import AdaptiveThresholdNeuron, MultiCompartmentNeuron, NeuronV3Config

logger = logging.getLogger(__name__)


@dataclass
class TopologyConfig:
    """Configuration for network topology parameters."""
    # Hierarchical structure
    num_layers: int = 3
    layer_sizes: list[int] = None  # Will default to [100, 50, 25] if None

    # Connectivity parameters
    connection_probability: float = 0.1
    small_world_rewiring: float = 0.1  # Small-world network parameter
    scale_free_exponent: float = 2.5  # Scale-free network parameter

    # Dynamic connectivity
    synapse_formation_rate: float = 0.001  # Rate of new synapse formation
    synapse_pruning_threshold: float = 0.01  # Weight below which synapses are pruned
    max_synapses_per_neuron: int = 100

    # Delay parameters
    min_delay: float = 0.001  # Minimum propagation delay (1ms)
    max_delay: float = 0.020  # Maximum propagation delay (20ms)
    delay_scaling: str = "distance"  # "distance", "random", or "fixed"


class PopulationLayer(nn.Module):
    """
    Layer representing a population of neurons with shared properties.
    
    Supports different neuron types and population-level dynamics
    including lateral inhibition and population codes.
    """

    def __init__(
        self,
        population_size: int,
        neuron_type: str = "adaptive_threshold",
        neuron_config: NeuronV3Config | None = None,
        lateral_inhibition: bool = True,
        inhibition_strength: float = 0.1
    ):
        super().__init__()

        self.population_size = population_size
        self.neuron_type = neuron_type
        self.lateral_inhibition = lateral_inhibition
        self.inhibition_strength = inhibition_strength

        if neuron_config is None:
            from ..neuromorphic import NeuromorphicConfig
            base_config = NeuromorphicConfig()
            neuron_config = NeuronV3Config(base_config=base_config)

        self.neuron_config = neuron_config

        # Create neurons based on type
        self.neurons = nn.ModuleList()
        for i in range(population_size):
            if neuron_type == "adaptive_threshold":
                neuron = AdaptiveThresholdNeuron(neuron_config)
            elif neuron_type == "multi_compartment":
                neuron = MultiCompartmentNeuron(neuron_config, num_dendrites=4)
            else:
                raise ValueError(f"Unsupported neuron type: {neuron_type}")

            self.neurons.append(neuron)

        # Lateral inhibition connections
        if lateral_inhibition:
            self.lateral_weights = nn.Parameter(
                torch.randn(population_size, population_size) * inhibition_strength
            )
            # Zero diagonal (no self-inhibition)
            self.register_buffer('inhibition_mask',
                               torch.eye(population_size) == 0)

        # Population state tracking
        self.register_buffer('population_activity', torch.zeros(population_size))
        self.register_buffer('activity_history', torch.zeros(population_size, 100))
        self.register_buffer('history_index', torch.tensor(0, dtype=torch.long))

        logger.debug(f"Created population layer: {population_size} {neuron_type} neurons")

    def forward(
        self,
        external_input: torch.Tensor,
        current_time: float | None = None,
        dt: float | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Process population dynamics.
        
        Args:
            external_input: External input to population [batch_size, population_size]
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Tuple of (population_spikes, population_states)
        """
        if current_time is None:
            current_time = 0.0
        if dt is None:
            dt = self.neuron_config.base_config.dt

        batch_size = external_input.size(0)
        population_spikes = torch.zeros(batch_size, self.population_size)
        neuron_states = []

        # Calculate lateral inhibition
        lateral_input = torch.zeros_like(external_input)
        if self.lateral_inhibition:
            # Use previous activity for lateral inhibition
            prev_activity = self.population_activity.unsqueeze(0).expand(batch_size, -1)
            lateral_inhibition = torch.matmul(prev_activity, self.lateral_weights * self.inhibition_mask)
            lateral_input = -lateral_inhibition  # Negative for inhibition

        # Process each neuron
        total_input = external_input + lateral_input

        for i, neuron in enumerate(self.neurons):
            if self.neuron_type == "multi_compartment":
                # Split input for dendrites and soma
                dendrite_input = total_input[:, i:i+1].expand(-1, 4)  # 4 dendrites
                soma_input = total_input[:, i:i+1]

                spikes, states = neuron(dendrite_input, soma_input, dt)
                population_spikes[:, i:i+1] = spikes
            else:
                # Single compartment neuron
                neuron_input = total_input[:, i:i+1]
                spikes, states = neuron(neuron_input, current_time, dt)
                population_spikes[:, i:i+1] = spikes

            neuron_states.append(states)

        # Update population activity tracking
        current_activity = population_spikes.mean(dim=0)  # Average across batch
        self.population_activity = current_activity

        # Update activity history
        history_idx = self.history_index.item() % self.activity_history.size(1)
        self.activity_history[:, history_idx] = current_activity
        self.history_index += 1

        population_states = {
            'individual_states': neuron_states,
            'population_activity': self.population_activity.clone(),
            'activity_history': self.activity_history.clone(),
            'lateral_input': lateral_input,
            'total_input': total_input
        }

        return population_spikes, population_states


class RealisticDelays(nn.Module):
    """
    Implements realistic axonal and synaptic delays in spike propagation.
    
    Delays can be based on physical distance, connection type, or
    stochastic variation to model biological realism.
    """

    def __init__(
        self,
        source_size: int,
        target_size: int,
        config: TopologyConfig
    ):
        super().__init__()

        self.source_size = source_size
        self.target_size = target_size
        self.config = config

        # Generate delay matrix
        self.delay_matrix = self._generate_delays()
        self.register_buffer('delays', self.delay_matrix)

        # Spike buffer to handle delays
        max_delay_steps = int(config.max_delay / 0.001) + 1  # Assuming 1ms time step
        self.register_buffer('spike_buffer',
                           torch.zeros(max_delay_steps, source_size))
        self.register_buffer('buffer_index', torch.tensor(0, dtype=torch.long))

        logger.debug(f"Created delay matrix: {source_size} -> {target_size}")

    def _generate_delays(self) -> torch.Tensor:
        """Generate delay matrix based on configuration."""
        delays = torch.zeros(self.source_size, self.target_size)

        if self.config.delay_scaling == "distance":
            # Distance-based delays (assuming 2D layout)
            source_positions = self._generate_positions(self.source_size)
            target_positions = self._generate_positions(self.target_size)

            for i in range(self.source_size):
                for j in range(self.target_size):
                    distance = torch.norm(source_positions[i] - target_positions[j])
                    # Scale distance to delay range
                    normalized_distance = distance / torch.sqrt(torch.tensor(2.0))  # Max distance in unit square
                    delay = self.config.min_delay + normalized_distance * (
                        self.config.max_delay - self.config.min_delay
                    )
                    delays[i, j] = delay

        elif self.config.delay_scaling == "random":
            # Random delays within range
            delays = torch.rand(self.source_size, self.target_size) * (
                self.config.max_delay - self.config.min_delay
            ) + self.config.min_delay

        else:  # "fixed"
            # Fixed delay for all connections
            avg_delay = (self.config.min_delay + self.config.max_delay) / 2
            delays.fill_(avg_delay)

        return delays

    def _generate_positions(self, num_neurons: int) -> torch.Tensor:
        """Generate 2D positions for neurons (for distance-based delays)."""
        # Arrange neurons in a rough square grid
        grid_size = int(np.ceil(np.sqrt(num_neurons)))
        positions = torch.zeros(num_neurons, 2)

        for i in range(num_neurons):
            row = i // grid_size
            col = i % grid_size
            positions[i, 0] = row / grid_size
            positions[i, 1] = col / grid_size

        return positions

    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float | None = None
    ) -> torch.Tensor:
        """
        Apply delays to spike propagation.
        
        Args:
            input_spikes: Input spikes [batch_size, source_size]
            dt: Time step
            
        Returns:
            Delayed spikes [batch_size, target_size]
        """
        if dt is None:
            dt = 0.001  # Default 1ms

        batch_size = input_spikes.size(0)

        # Add current spikes to buffer
        current_idx = self.buffer_index.item() % self.spike_buffer.size(0)
        self.spike_buffer[current_idx] = input_spikes[0]  # Use first batch item for simplicity

        # Calculate delayed outputs
        delayed_spikes = torch.zeros(batch_size, self.target_size)

        for i in range(self.source_size):
            for j in range(self.target_size):
                delay = self.delays[i, j]
                delay_steps = int(delay / dt)

                if delay_steps < self.spike_buffer.size(0):
                    # Get spike from appropriate time in buffer
                    buffer_idx = (current_idx - delay_steps) % self.spike_buffer.size(0)
                    delayed_spike = self.spike_buffer[buffer_idx, i]
                    delayed_spikes[:, j] += delayed_spike  # Add contribution

        # Update buffer index
        self.buffer_index += 1

        return delayed_spikes


class DynamicConnectivity(nn.Module):
    """
    Implements dynamic synaptic connectivity with formation and pruning.
    
    Synapses can be formed based on activity correlation and pruned
    based on weak weights or lack of use.
    """

    def __init__(
        self,
        pre_size: int,
        post_size: int,
        config: TopologyConfig,
        initial_connectivity: torch.Tensor | None = None
    ):
        super().__init__()

        self.pre_size = pre_size
        self.post_size = post_size
        self.config = config

        # Connectivity mask (which connections exist)
        if initial_connectivity is None:
            initial_connectivity = self._generate_initial_connectivity()

        self.register_buffer('connectivity_mask', initial_connectivity)

        # Synaptic weights (only for existing connections)
        initial_weights = torch.randn(pre_size, post_size) * 0.1
        initial_weights = initial_weights * self.connectivity_mask
        self.register_parameter('synaptic_weights', nn.Parameter(initial_weights))

        # Activity tracking for formation/pruning decisions
        self.register_buffer('pre_activity', torch.zeros(pre_size))
        self.register_buffer('post_activity', torch.zeros(post_size))
        self.register_buffer('correlation_history', torch.zeros(pre_size, post_size, 50))
        self.register_buffer('correlation_index', torch.tensor(0, dtype=torch.long))

        # Connection statistics
        self.register_buffer('formation_events', torch.tensor(0, dtype=torch.long))
        self.register_buffer('pruning_events', torch.tensor(0, dtype=torch.long))

        logger.debug(f"Created dynamic connectivity: {pre_size} -> {post_size}")

    def _generate_initial_connectivity(self) -> torch.Tensor:
        """Generate initial connectivity pattern."""
        # Start with random connectivity
        connectivity = torch.rand(self.pre_size, self.post_size) < self.config.connection_probability

        # Ensure each neuron has some connections
        for i in range(self.post_size):
            if not connectivity[:, i].any():
                # Add at least one connection
                random_pre = torch.randint(0, self.pre_size, (1,))
                connectivity[random_pre, i] = True

        return connectivity.float()

    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_time: float,
        dt: float | None = None,
        plasticity: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Process dynamic connectivity updates.
        
        Args:
            pre_spikes: Presynaptic spikes [batch_size, pre_size]
            post_spikes: Postsynaptic spikes [batch_size, post_size]
            current_time: Current simulation time
            dt: Time step
            plasticity: Whether to update connectivity
            
        Returns:
            Tuple of (synaptic_current, connectivity_info)
        """
        if dt is None:
            dt = 0.001

        batch_size = pre_spikes.size(0)

        # Update activity tracking
        self.pre_activity = 0.99 * self.pre_activity + 0.01 * pre_spikes.mean(dim=0)
        self.post_activity = 0.99 * self.post_activity + 0.01 * post_spikes.mean(dim=0)

        # Update correlation history
        corr_idx = self.correlation_index.item() % self.correlation_history.size(-1)
        # Outer product of activities for correlation
        current_correlation = torch.outer(
            pre_spikes.mean(dim=0),
            post_spikes.mean(dim=0)
        )
        self.correlation_history[:, :, corr_idx] = current_correlation
        self.correlation_index += 1

        # Calculate synaptic transmission (only through existing connections)
        effective_weights = self.synaptic_weights * self.connectivity_mask
        synaptic_current = torch.matmul(pre_spikes, effective_weights)

        if plasticity and current_time > 1.0:  # Wait 1 second before structural changes
            # Synapse formation based on correlation
            self._form_synapses()

            # Synapse pruning based on weight magnitude
            self._prune_synapses()

        connectivity_info = {
            'connectivity_matrix': self.connectivity_mask.clone(),
            'effective_weights': effective_weights,
            'num_connections': self.connectivity_mask.sum().item(),
            'formation_events': self.formation_events.item(),
            'pruning_events': self.pruning_events.item(),
            'average_correlation': self.correlation_history.mean(dim=-1)
        }

        return synaptic_current, connectivity_info

    def _form_synapses(self):
        """Form new synapses based on activity correlation."""
        # Calculate average correlation over recent history
        avg_correlation = self.correlation_history.mean(dim=-1)

        # Find highly correlated but unconnected pairs
        correlation_threshold = 0.1  # Minimum correlation for formation
        formation_candidates = (avg_correlation > correlation_threshold) & (self.connectivity_mask == 0)

        # Apply formation rate constraint
        num_candidates = formation_candidates.sum().item()
        if num_candidates > 0:
            formation_probability = self.config.synapse_formation_rate
            num_to_form = int(num_candidates * formation_probability)

            if num_to_form > 0:
                # Randomly select candidates to form
                candidate_indices = torch.where(formation_candidates)
                selected_indices = torch.randperm(len(candidate_indices[0]))[:num_to_form]

                for idx in selected_indices:
                    i, j = candidate_indices[0][idx], candidate_indices[1][idx]

                    # Check connection limit per neuron
                    current_connections = self.connectivity_mask[:, j].sum()
                    if current_connections < self.config.max_synapses_per_neuron:
                        self.connectivity_mask[i, j] = 1.0
                        # Initialize weight for new connection
                        self.synaptic_weights.data[i, j] = torch.randn(1) * 0.1
                        self.formation_events += 1

    def _prune_synapses(self):
        """Prune weak synapses based on weight magnitude."""
        # Find existing connections with weak weights
        weak_connections = (
            (torch.abs(self.synaptic_weights) < self.config.synapse_pruning_threshold) &
            (self.connectivity_mask == 1)
        )

        # Remove weak connections
        num_pruned = weak_connections.sum().item()
        if num_pruned > 0:
            self.connectivity_mask[weak_connections] = 0.0
            self.synaptic_weights.data[weak_connections] = 0.0
            self.pruning_events += num_pruned


class HierarchicalNetwork(nn.Module):
    """
    Multi-layer hierarchical network with cortical-like organization.
    
    Implements feedforward and feedback connections between layers,
    with different processing characteristics at each level.
    """

    def __init__(
        self,
        config: TopologyConfig,
        neuron_configs: list[NeuronV3Config] | None = None,
        layer_types: list[str] | None = None
    ):
        super().__init__()

        self.config = config

        # Set default layer sizes if not provided
        if config.layer_sizes is None:
            config.layer_sizes = [100, 50, 25][:config.num_layers]

        # Set default neuron types if not provided
        if layer_types is None:
            layer_types = ["adaptive_threshold"] * config.num_layers

        # Create population layers
        self.layers = nn.ModuleList()
        for i, (size, layer_type) in enumerate(zip(config.layer_sizes, layer_types, strict=False)):
            neuron_config = neuron_configs[i] if neuron_configs else None
            layer = PopulationLayer(
                population_size=size,
                neuron_type=layer_type,
                neuron_config=neuron_config,
                lateral_inhibition=True
            )
            self.layers.append(layer)

        # Create inter-layer connections
        self.feedforward_connections = nn.ModuleList()
        self.feedback_connections = nn.ModuleList()
        self.delays = nn.ModuleList()

        for i in range(config.num_layers - 1):
            # Feedforward connections (layer i -> layer i+1)
            ff_conn = DynamicConnectivity(
                config.layer_sizes[i],
                config.layer_sizes[i+1],
                config
            )
            self.feedforward_connections.append(ff_conn)

            # Feedback connections (layer i+1 -> layer i)
            fb_conn = DynamicConnectivity(
                config.layer_sizes[i+1],
                config.layer_sizes[i],
                config
            )
            self.feedback_connections.append(fb_conn)

            # Delays for both directions
            ff_delay = RealisticDelays(
                config.layer_sizes[i],
                config.layer_sizes[i+1],
                config
            )
            fb_delay = RealisticDelays(
                config.layer_sizes[i+1],
                config.layer_sizes[i],
                config
            )
            self.delays.append(nn.ModuleList([ff_delay, fb_delay]))

        # Network state tracking
        self.register_buffer('layer_activities', torch.zeros(config.num_layers))
        self.register_buffer('network_coherence', torch.tensor(0.0))

        logger.info(f"Created hierarchical network: {config.num_layers} layers {config.layer_sizes}")

    def forward(
        self,
        input_spikes: torch.Tensor,
        current_time: float | None = None,
        dt: float | None = None,
        top_down_input: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Process hierarchical network dynamics.
        
        Args:
            input_spikes: Input to first layer [batch_size, input_size]
            current_time: Current simulation time
            dt: Time step  
            top_down_input: Optional top-down input to last layer
            
        Returns:
            Tuple of (network_output, network_states)
        """
        if current_time is None:
            current_time = 0.0
        if dt is None:
            dt = 0.001

        batch_size = input_spikes.size(0)

        # Store layer outputs and states
        layer_outputs = []
        layer_states = []

        # Forward pass through layers
        current_input = input_spikes

        for layer_idx, layer in enumerate(self.layers):
            # Add top-down input if provided (to last layer)
            if layer_idx == len(self.layers) - 1 and top_down_input is not None:
                if top_down_input.size(-1) == current_input.size(-1):
                    current_input = current_input + top_down_input

            # Process layer
            layer_output, layer_state = layer(current_input, current_time, dt)
            layer_outputs.append(layer_output)
            layer_states.append(layer_state)

            # Prepare input for next layer (if not last layer)
            if layer_idx < len(self.layers) - 1:
                # Apply feedforward connection and delay
                ff_conn = self.feedforward_connections[layer_idx]
                ff_delay = self.delays[layer_idx][0]  # Feedforward delay

                # Process through connection dynamics
                ff_current, ff_info = ff_conn(
                    layer_output,
                    torch.zeros_like(layer_outputs[layer_idx + 1]) if layer_idx + 1 < len(layer_outputs) else torch.zeros(batch_size, self.config.layer_sizes[layer_idx + 1]),
                    current_time,
                    dt
                )

                # Apply delays
                delayed_input = ff_delay(ff_current, dt)
                current_input = delayed_input

        # Backward pass (feedback connections)
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            fb_conn = self.feedback_connections[layer_idx]
            fb_delay = self.delays[layer_idx][1]  # Feedback delay

            # Get feedback from higher layer
            higher_layer_output = layer_outputs[layer_idx + 1]

            # Process through feedback connection
            fb_current, fb_info = fb_conn(
                higher_layer_output,
                layer_outputs[layer_idx],
                current_time,
                dt
            )

            # Apply feedback delay
            delayed_feedback = fb_delay(fb_current, dt)

            # Add feedback to layer (for next timestep)
            # This would require stateful processing - simplified here
            pass

        # Update network-level metrics
        self.layer_activities = torch.stack([output.mean() for output in layer_outputs])

        # Calculate network coherence (synchronization across layers)
        activity_correlations = []
        for i in range(len(layer_outputs) - 1):
            # Simplified coherence based on activity correlation
            corr = torch.corrcoef(torch.stack([
                layer_outputs[i].mean(dim=0),
                layer_outputs[i+1].mean(dim=0)
            ]))[0, 1]
            activity_correlations.append(corr)

        if activity_correlations:
            self.network_coherence = torch.stack(activity_correlations).mean()

        # Network output is typically the last layer
        network_output = layer_outputs[-1]

        network_states = {
            'layer_outputs': layer_outputs,
            'layer_states': layer_states,
            'layer_activities': self.layer_activities.clone(),
            'network_coherence': self.network_coherence.clone(),
            'feedforward_info': [conn.connectivity_mask.sum().item() for conn in self.feedforward_connections],
            'feedback_info': [conn.connectivity_mask.sum().item() for conn in self.feedback_connections]
        }

        return network_output, network_states
