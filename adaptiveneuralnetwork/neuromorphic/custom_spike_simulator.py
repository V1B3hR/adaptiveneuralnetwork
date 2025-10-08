"""
Custom spike simulator backend for advanced neuromorphic computing.

This module provides a flexible, high-performance spike simulator that can
model various neuron types and plasticity mechanisms for research and
development purposes.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..core.neuromorphic import NeuromorphicConfig
from .hardware_backends import BaseHardwareBackend, HardwareConstraints, HardwareMetrics

logger = logging.getLogger(__name__)


class NeuronModel(Enum):
    """Available neuron models in the simulator."""

    LIF = "leaky_integrate_fire"
    ADEX = "adaptive_exponential"
    IZHIKEVICH = "izhikevich"
    HODGKIN_HUXLEY = "hodgkin_huxley"
    MULTI_COMPARTMENT = "multi_compartment"


class PlasticityRule(Enum):
    """Available plasticity rules."""

    STDP = "spike_timing_dependent_plasticity"
    TRIPLET_STDP = "triplet_stdp"
    HOMEOSTATIC = "homeostatic_scaling"
    METAPLASTICITY = "metaplasticity"
    REWARD_MODULATED = "reward_modulated_stdp"


@dataclass
class NeuronParameters:
    """Parameters for different neuron models."""

    # LIF parameters
    tau_mem: float = 20.0  # membrane time constant (ms)
    tau_syn: float = 5.0   # synaptic time constant (ms)
    v_thresh: float = -50.0  # spike threshold (mV)
    v_reset: float = -70.0   # reset potential (mV)
    v_rest: float = -70.0    # resting potential (mV)

    # AdEx parameters
    delta_t: float = 2.0     # slope factor (mV)
    v_t: float = -50.0       # threshold adaptation (mV)
    tau_w: float = 30.0      # adaptation time constant (ms)
    a: float = 2.0           # sub-threshold adaptation (nS)
    b: float = 0.5           # spike-triggered adaptation (nA)

    # Izhikevich parameters
    a_izh: float = 0.1       # recovery variable time scale
    b_izh: float = 0.2       # recovery variable sensitivity
    c_izh: float = -65.0     # after-spike reset value (mV)
    d_izh: float = 2.0       # after-spike recovery boost

    # Multi-compartment parameters
    compartments: int = 3    # number of compartments
    coupling_strength: float = 0.1  # inter-compartment coupling


@dataclass
class SynapseParameters:
    """Parameters for synaptic connections."""

    weight: float = 1.0
    delay: int = 1           # delay in time steps
    syn_type: str = "excitatory"  # "excitatory" or "inhibitory"

    # STDP parameters
    tau_plus: float = 17.0   # potentiation time constant (ms)
    tau_minus: float = 34.0  # depression time constant (ms)
    a_plus: float = 0.01     # potentiation amplitude
    a_minus: float = 0.012   # depression amplitude

    # Homeostatic parameters
    target_rate: float = 10.0  # target firing rate (Hz)
    scaling_tau: float = 86400.0  # homeostatic time constant (ms)


class CustomSpikeSimulator(BaseHardwareBackend):
    """Advanced custom spike simulator with multiple neuron models and plasticity rules."""

    def __init__(self, device: str = "cpu"):
        from ..core.neuromorphic import NeuromorphicPlatform
        super().__init__(platform=NeuromorphicPlatform.SIMULATION)
        self.device = torch.device(device)
        self.platform_name = "CustomSpikeSimulator"

        # Simulation state
        self.neurons = {}
        self.synapses = {}
        self.plasticity_rules = {}
        self.simulation_time = 0.0
        self.time_step = 0.1  # ms

        # Performance tracking
        self.spike_times = defaultdict(list)
        self.neuron_states = {}
        self.synapse_states = {}

        # Define constraints
        self.constraints = HardwareConstraints(
            max_neurons_per_core=10000,  # Much higher for software simulation
            max_synapses_per_neuron=1000,
            weight_precision_bits=32,    # Full precision in software
            time_step_resolution=1e-4,   # 0.1 ms resolution
            max_delay_steps=1000,
            available_neuron_models=[model.value for model in NeuronModel],
            power_budget_mw=float('inf')  # No power constraint for simulation
        )

    def _setup_hardware_interface(self) -> None:
        """Setup custom spike simulator interface."""
        logger.info("Setting up Custom Spike Simulator interface")
        # Initialize simulation environment
        torch.manual_seed(42)  # For reproducible simulations

        # No actual hardware interface needed for software simulation
        logger.info("Custom Spike Simulator interface ready")

    def initialize(self, config: NeuromorphicConfig) -> None:
        """Initialize the simulator with neuromorphic configuration."""
        logger.info(f"Initializing Custom Spike Simulator on {self.device}")

        self.config = config
        self.time_step = getattr(config, 'time_step', 0.1)

        # Initialize simulation containers
        self.neurons.clear()
        self.synapses.clear()
        self.plasticity_rules.clear()
        self.spike_times.clear()

        logger.info("Custom Spike Simulator initialized successfully")

    def compile_model(self, model: nn.Module) -> str:
        """Compile PyTorch model to simulator format."""
        logger.info("Compiling model for Custom Spike Simulator")

        deployment_id = f"deployment_{len(self.deployments)}"

        # Extract network structure from model
        network_config = self._extract_network_structure(model)

        # Create neurons
        self._create_neurons(network_config)

        # Create synapses
        self._create_synapses(network_config)

        # Setup plasticity rules
        self._setup_plasticity_rules(network_config)

        # Store deployment
        self.deployments[deployment_id] = {
            'model': model,
            'network_config': network_config,
            'compilation_time': torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        }

        logger.info(f"Model compiled successfully: {deployment_id}")
        return deployment_id

    def _extract_network_structure(self, model: nn.Module) -> dict[str, Any]:
        """Extract network structure from PyTorch model."""
        structure = {
            'layers': [],
            'connections': [],
            'neuron_counts': [],
            'total_neurons': 0
        }

        # Analyze model structure
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_info = {
                    'name': name,
                    'type': 'linear',
                    'input_size': module.in_features,
                    'output_size': module.out_features,
                    'neuron_model': NeuronModel.LIF,  # Default to LIF
                    'parameters': NeuronParameters()
                }
                structure['layers'].append(layer_info)
                structure['neuron_counts'].append(module.out_features)
                structure['total_neurons'] += module.out_features

        return structure

    def _create_neurons(self, network_config: dict[str, Any]) -> None:
        """Create neurons in the simulator."""
        neuron_id = 0

        for layer in network_config['layers']:
            layer_neurons = {}

            for i in range(layer['output_size']):
                neuron = self._create_single_neuron(
                    neuron_id,
                    layer['neuron_model'],
                    layer['parameters']
                )
                layer_neurons[neuron_id] = neuron
                neuron_id += 1

            self.neurons[layer['name']] = layer_neurons

    def _create_single_neuron(
        self,
        neuron_id: int,
        model: NeuronModel,
        params: NeuronParameters
    ) -> dict[str, Any]:
        """Create a single neuron with specified model and parameters."""

        if model == NeuronModel.LIF:
            return self._create_lif_neuron(neuron_id, params)
        elif model == NeuronModel.ADEX:
            return self._create_adex_neuron(neuron_id, params)
        elif model == NeuronModel.IZHIKEVICH:
            return self._create_izhikevich_neuron(neuron_id, params)
        elif model == NeuronModel.MULTI_COMPARTMENT:
            return self._create_multicompartment_neuron(neuron_id, params)
        else:
            # Default to LIF
            return self._create_lif_neuron(neuron_id, params)

    def _create_lif_neuron(self, neuron_id: int, params: NeuronParameters) -> dict[str, Any]:
        """Create Leaky Integrate-and-Fire neuron."""
        return {
            'id': neuron_id,
            'model': NeuronModel.LIF,
            'v_mem': params.v_rest,
            'i_syn': 0.0,
            'last_spike_time': -float('inf'),
            'refractory_until': 0.0,
            'parameters': params,
            'spike_history': []
        }

    def _create_adex_neuron(self, neuron_id: int, params: NeuronParameters) -> dict[str, Any]:
        """Create Adaptive Exponential neuron."""
        return {
            'id': neuron_id,
            'model': NeuronModel.ADEX,
            'v_mem': params.v_rest,
            'w': 0.0,  # adaptation variable
            'i_syn': 0.0,
            'last_spike_time': -float('inf'),
            'parameters': params,
            'spike_history': []
        }

    def _create_izhikevich_neuron(self, neuron_id: int, params: NeuronParameters) -> dict[str, Any]:
        """Create Izhikevich neuron."""
        return {
            'id': neuron_id,
            'model': NeuronModel.IZHIKEVICH,
            'v': params.c_izh,  # membrane potential
            'u': params.b_izh * params.c_izh,  # recovery variable
            'i_syn': 0.0,
            'last_spike_time': -float('inf'),
            'parameters': params,
            'spike_history': []
        }

    def _create_multicompartment_neuron(self, neuron_id: int, params: NeuronParameters) -> dict[str, Any]:
        """Create multi-compartment neuron."""
        return {
            'id': neuron_id,
            'model': NeuronModel.MULTI_COMPARTMENT,
            'v_mem': [params.v_rest] * params.compartments,
            'i_syn': [0.0] * params.compartments,
            'last_spike_time': -float('inf'),
            'parameters': params,
            'spike_history': []
        }

    def _create_synapses(self, network_config: dict[str, Any]) -> None:
        """Create synaptic connections between layers."""
        synapse_id = 0

        for i, layer in enumerate(network_config['layers'][:-1]):
            next_layer = network_config['layers'][i + 1]

            # Create all-to-all connections (simplified)
            for pre_id in range(layer['output_size']):
                for post_id in range(next_layer['output_size']):
                    synapse = {
                        'id': synapse_id,
                        'pre_neuron_id': pre_id,
                        'post_neuron_id': post_id + layer['output_size'],
                        'parameters': SynapseParameters(),
                        'trace_pre': 0.0,
                        'trace_post': 0.0,
                        'weight_history': []
                    }
                    self.synapses[synapse_id] = synapse
                    synapse_id += 1

    def _setup_plasticity_rules(self, network_config: dict[str, Any]) -> None:
        """Setup plasticity rules for synapses."""
        # Simple STDP rule for all synapses
        self.plasticity_rules['stdp'] = {
            'rule_type': PlasticityRule.STDP,
            'active': True,
            'update_frequency': 10  # Update every 10 time steps
        }

    def execute(
        self,
        deployment_id: str,
        input_data: torch.Tensor,
        num_timesteps: int
    ) -> tuple[torch.Tensor, HardwareMetrics]:
        """Execute the simulation for specified number of timesteps."""

        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        logger.debug(f"Executing simulation: {deployment_id}, {num_timesteps} timesteps")

        # Initialize metrics tracking
        total_spikes = 0
        total_synaptic_ops = 0
        start_time = self.simulation_time

        # Run simulation
        for timestep in range(num_timesteps):
            self.simulation_time += self.time_step

            # Apply input stimulation
            if timestep == 0:
                self._apply_input_stimulation(input_data)

            # Update all neurons
            spikes_this_step = self._update_neurons()
            total_spikes += spikes_this_step

            # Update synapses and apply plasticity
            synaptic_ops = self._update_synapses()
            total_synaptic_ops += synaptic_ops

            # Apply plasticity rules
            if timestep % 10 == 0:  # Apply every 10 timesteps
                self._apply_plasticity()

        # Generate output
        output_data = self._collect_output(input_data.shape[0])

        # Calculate metrics
        simulation_duration = self.simulation_time - start_time
        avg_spike_rate = total_spikes / (simulation_duration / 1000.0) if simulation_duration > 0 else 0

        metrics = HardwareMetrics(
            power_consumption_mw=self._estimate_power_consumption(total_spikes, total_synaptic_ops),
            core_utilization=min(1.0, len(self.neurons) / 1000.0),
            memory_utilization=0.5,  # Estimated
            spike_rate_hz=avg_spike_rate,
            synaptic_operations_per_second=total_synaptic_ops / (simulation_duration / 1000.0) if simulation_duration > 0 else 0,
            energy_per_synaptic_operation=1.0  # pJ - software simulation
        )

        return output_data, metrics

    def _apply_input_stimulation(self, input_data: torch.Tensor) -> None:
        """Apply input stimulation to first layer neurons."""
        # Simplified: inject current proportional to input
        batch_size, input_size = input_data.shape

        first_layer_name = list(self.neurons.keys())[0]
        first_layer_neurons = self.neurons[first_layer_name]

        for i, (neuron_id, neuron) in enumerate(first_layer_neurons.items()):
            if i < input_size:
                stimulation = input_data[0, i].item() * 10.0  # Scale factor
                neuron['i_syn'] += stimulation

    def _update_neurons(self) -> int:
        """Update all neurons and return number of spikes."""
        total_spikes = 0

        for layer_name, layer_neurons in self.neurons.items():
            for neuron_id, neuron in layer_neurons.items():
                if self._update_single_neuron(neuron):
                    total_spikes += 1
                    self.spike_times[neuron_id].append(self.simulation_time)

        return total_spikes

    def _update_single_neuron(self, neuron: dict[str, Any]) -> bool:
        """Update a single neuron and return True if it spiked."""
        model = neuron['model']

        if model == NeuronModel.LIF:
            return self._update_lif_neuron(neuron)
        elif model == NeuronModel.ADEX:
            return self._update_adex_neuron(neuron)
        elif model == NeuronModel.IZHIKEVICH:
            return self._update_izhikevich_neuron(neuron)
        elif model == NeuronModel.MULTI_COMPARTMENT:
            return self._update_multicompartment_neuron(neuron)
        else:
            return False

    def _update_lif_neuron(self, neuron: dict[str, Any]) -> bool:
        """Update LIF neuron dynamics."""
        params = neuron['parameters']
        dt = self.time_step

        # Membrane potential dynamics
        dv = (-(neuron['v_mem'] - params.v_rest) + neuron['i_syn']) / params.tau_mem
        neuron['v_mem'] += dv * dt

        # Synaptic current decay
        neuron['i_syn'] *= np.exp(-dt / params.tau_syn)

        # Check for spike
        if neuron['v_mem'] >= params.v_thresh:
            neuron['v_mem'] = params.v_reset
            neuron['last_spike_time'] = self.simulation_time
            neuron['spike_history'].append(self.simulation_time)
            return True

        return False

    def _update_adex_neuron(self, neuron: dict[str, Any]) -> bool:
        """Update AdEx neuron dynamics."""
        params = neuron['parameters']
        dt = self.time_step

        v = neuron['v_mem']
        w = neuron['w']

        # Exponential term
        exp_term = params.delta_t * np.exp((v - params.v_t) / params.delta_t)

        # Membrane potential dynamics
        dv = (-(v - params.v_rest) + exp_term - w + neuron['i_syn']) / params.tau_mem
        neuron['v_mem'] += dv * dt

        # Adaptation variable dynamics
        dw = (params.a * (v - params.v_rest) - w) / params.tau_w
        neuron['w'] += dw * dt

        # Synaptic current decay
        neuron['i_syn'] *= np.exp(-dt / params.tau_syn)

        # Check for spike
        if neuron['v_mem'] >= params.v_thresh:
            neuron['v_mem'] = params.v_reset
            neuron['w'] += params.b
            neuron['last_spike_time'] = self.simulation_time
            neuron['spike_history'].append(self.simulation_time)
            return True

        return False

    def _update_izhikevich_neuron(self, neuron: dict[str, Any]) -> bool:
        """Update Izhikevich neuron dynamics."""
        params = neuron['parameters']
        dt = self.time_step

        v = neuron['v']
        u = neuron['u']
        I = neuron['i_syn']

        # Izhikevich equations
        dv = 0.04 * v * v + 5 * v + 140 - u + I
        du = params.a_izh * (params.b_izh * v - u)

        neuron['v'] += dv * dt
        neuron['u'] += du * dt

        # Synaptic current decay
        neuron['i_syn'] *= 0.9  # Simple decay

        # Check for spike
        if neuron['v'] >= 30:  # Spike threshold for Izhikevich
            neuron['v'] = params.c_izh
            neuron['u'] += params.d_izh
            neuron['last_spike_time'] = self.simulation_time
            neuron['spike_history'].append(self.simulation_time)
            return True

        return False

    def _update_multicompartment_neuron(self, neuron: dict[str, Any]) -> bool:
        """Update multi-compartment neuron dynamics."""
        params = neuron['parameters']
        dt = self.time_step

        v_mem = neuron['v_mem']
        i_syn = neuron['i_syn']

        # Update each compartment
        for i in range(params.compartments):
            # Coupling between compartments
            coupling_current = 0.0
            if i > 0:
                coupling_current += params.coupling_strength * (v_mem[i-1] - v_mem[i])
            if i < params.compartments - 1:
                coupling_current += params.coupling_strength * (v_mem[i+1] - v_mem[i])

            # Membrane dynamics
            dv = (-(v_mem[i] - params.v_rest) + i_syn[i] + coupling_current) / params.tau_mem
            v_mem[i] += dv * dt

            # Synaptic current decay
            i_syn[i] *= np.exp(-dt / params.tau_syn)

        # Check for spike in soma (first compartment)
        if v_mem[0] >= params.v_thresh:
            for i in range(params.compartments):
                v_mem[i] = params.v_reset
            neuron['last_spike_time'] = self.simulation_time
            neuron['spike_history'].append(self.simulation_time)
            return True

        return False

    def _update_synapses(self) -> int:
        """Update synaptic connections and return number of operations."""
        synaptic_ops = 0

        for synapse_id, synapse in self.synapses.items():
            pre_id = synapse['pre_neuron_id']
            post_id = synapse['post_neuron_id']

            # Find pre and post neurons
            pre_neuron = self._find_neuron_by_id(pre_id)
            post_neuron = self._find_neuron_by_id(post_id)

            if pre_neuron and post_neuron:
                # Check if pre-synaptic neuron spiked recently
                if (self.simulation_time - pre_neuron['last_spike_time']) < (synapse['parameters'].delay * self.time_step):
                    # Apply synaptic transmission
                    weight = synapse['parameters'].weight
                    post_neuron['i_syn'] += weight
                    synaptic_ops += 1

        return synaptic_ops

    def _find_neuron_by_id(self, neuron_id: int) -> dict[str, Any] | None:
        """Find neuron by ID across all layers."""
        for layer_neurons in self.neurons.values():
            if neuron_id in layer_neurons:
                return layer_neurons[neuron_id]
        return None

    def _apply_plasticity(self) -> None:
        """Apply plasticity rules to synapses."""
        if 'stdp' in self.plasticity_rules and self.plasticity_rules['stdp']['active']:
            self._apply_stdp()

    def _apply_stdp(self) -> None:
        """Apply STDP plasticity rule."""
        for synapse_id, synapse in self.synapses.items():
            pre_neuron = self._find_neuron_by_id(synapse['pre_neuron_id'])
            post_neuron = self._find_neuron_by_id(synapse['post_neuron_id'])

            if pre_neuron and post_neuron:
                # Simple STDP implementation
                dt_spike = post_neuron['last_spike_time'] - pre_neuron['last_spike_time']

                if abs(dt_spike) < 50.0:  # Within STDP window
                    params = synapse['parameters']
                    if dt_spike > 0:  # Post after pre - potentiation
                        delta_w = params.a_plus * np.exp(-dt_spike / params.tau_plus)
                    else:  # Pre after post - depression
                        delta_w = -params.a_minus * np.exp(dt_spike / params.tau_minus)

                    synapse['parameters'].weight = np.clip(
                        synapse['parameters'].weight + delta_w,
                        0.0, 10.0  # Weight bounds
                    )

    def _collect_output(self, batch_size: int) -> torch.Tensor:
        """Collect output from final layer neurons."""
        # Get last layer
        last_layer_name = list(self.neurons.keys())[-1]
        last_layer_neurons = self.neurons[last_layer_name]

        # Count recent spikes as output
        output_size = len(last_layer_neurons)
        output = torch.zeros(batch_size, output_size, device=self.device)

        recent_window = 10.0  # ms
        for i, (neuron_id, neuron) in enumerate(last_layer_neurons.items()):
            # Count spikes in recent window
            recent_spikes = sum(1 for t in neuron['spike_history']
                              if self.simulation_time - t < recent_window)
            output[0, i] = float(recent_spikes)

        return output

    def _estimate_power_consumption(self, total_spikes: int, total_synaptic_ops: int) -> float:
        """Estimate power consumption based on activity."""
        # Simple power model
        spike_power = total_spikes * 0.1  # mW per spike
        synapse_power = total_synaptic_ops * 0.01  # mW per synaptic operation
        base_power = 10.0  # mW baseline

        return base_power + spike_power + synapse_power

    def get_constraints(self) -> HardwareConstraints:
        """Get hardware constraints."""
        return self.constraints

    def get_simulation_state(self) -> dict[str, Any]:
        """Get current simulation state for analysis."""
        return {
            'simulation_time': self.simulation_time,
            'total_neurons': sum(len(layer) for layer in self.neurons.values()),
            'total_synapses': len(self.synapses),
            'spike_counts': {nid: len(spikes) for nid, spikes in self.spike_times.items()},
            'average_firing_rate': np.mean([len(spikes) / (self.simulation_time / 1000.0)
                                          for spikes in self.spike_times.values()]) if self.spike_times else 0.0
        }
