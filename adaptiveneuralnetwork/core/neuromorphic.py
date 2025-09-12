"""
Neuromorphic hardware compatibility layer for adaptive neural networks.

This module provides abstractions and implementations for neuromorphic hardware
compatibility, including spike-based computation, event-driven processing,
and hardware-specific optimizations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Protocol, runtime_checkable
import numpy as np
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NeuromorphicPlatform(Enum):
    """Supported neuromorphic hardware platforms."""
    # 2nd Generation Platforms
    LOIHI = "loihi"
    SPINNAKER = "spinnaker" 
    TRUENORTH = "truenorth"
    AKIDA = "akida"
    GENERIC_SNN = "generic_snn"
    
    # 3rd Generation Platforms
    LOIHI2 = "loihi2"
    SPINNAKER2 = "spinnaker2"
    GENERIC_V3 = "generic_v3"
    
    # Simulation
    SIMULATION = "simulation"


@dataclass
class SpikeEvent:
    """Represents a spike event in neuromorphic processing."""
    neuron_id: int
    timestamp: float
    amplitude: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass  
class NeuromorphicConfig:
    """Configuration for neuromorphic hardware compatibility."""
    platform: NeuromorphicPlatform = NeuromorphicPlatform.SIMULATION
    dt: float = 0.001  # Time step in seconds
    v_threshold: float = 1.0  # Spike threshold
    v_reset: float = 0.0  # Reset potential
    v_rest: float = 0.0  # Resting potential
    tau_mem: float = 0.01  # Membrane time constant
    tau_syn: float = 0.005  # Synaptic time constant
    refractory_period: float = 0.002  # Refractory period
    encoding_window: float = 0.1  # Time window for rate encoding
    max_spike_rate: float = 1000.0  # Maximum spike rate (Hz)
    
    # 3rd Generation Extensions
    generation: int = 2  # Neuromorphic generation (2 or 3)
    
    # Advanced neuron parameters
    enable_multi_compartment: bool = False
    enable_adaptive_threshold: bool = False
    enable_burst_firing: bool = False
    enable_stochastic_dynamics: bool = False
    
    # Plasticity parameters
    enable_stdp: bool = False
    enable_metaplasticity: bool = False
    enable_homeostatic_scaling: bool = False
    
    # Network topology parameters
    enable_hierarchical_structure: bool = False
    enable_dynamic_connectivity: bool = False
    num_hierarchy_levels: int = 3
    
    # Temporal coding parameters
    enable_temporal_patterns: bool = False
    enable_phase_encoding: bool = False
    enable_oscillatory_dynamics: bool = False
    enable_sparse_coding: bool = False
    
    # Input data characteristics (for intelligent configuration)
    input_data_type: Optional[str] = None  # 'sequential', 'high_dimensional', 'temporal_patterns'
    
    # Internal flag to track auto-configuration
    _auto_configure_phase_encoding: bool = True
    
    def __post_init__(self):
        """Configure intelligent defaults after initialization."""
        # Apply intelligent phase encoding configuration only if enabled
        if self._auto_configure_phase_encoding:
            self._configure_phase_encoding()
    
    def _was_phase_encoding_explicitly_set(self, original_value: bool) -> bool:
        """
        Determine if phase encoding was explicitly set by the user.
        
        This is a heuristic: if the user provided any temporal coding parameters
        or set the value to True, we assume it was intentional.
        """
        # If set to True, assume it was intentional
        if original_value:
            return True
            
        # If any temporal coding parameters are set, assume user is configuring manually
        temporal_params_set = any([
            self.enable_temporal_patterns,
            self.enable_oscillatory_dynamics,
            self.enable_sparse_coding
        ])
        
        return temporal_params_set
    
    def _detect_hardware_phase_encoding_support(self) -> Tuple[bool, str]:
        """
        Detect hardware platform's phase encoding capabilities.
        
        Returns:
            (supported: bool, reason: str) - Whether phase encoding is supported and reason
        """
        platform_capabilities = {
            # 3rd generation platforms - full support
            NeuromorphicPlatform.LOIHI2: (True, "Loihi2 has native phase encoding support"),
            NeuromorphicPlatform.SPINNAKER2: (True, "SpiNNaker2 supports advanced temporal coding"),
            NeuromorphicPlatform.GENERIC_V3: (True, "Generic V3 supports all temporal features"),
            
            # 2nd generation platforms - limited support
            NeuromorphicPlatform.LOIHI: (False, "Loihi has limited phase encoding support"),
            NeuromorphicPlatform.SPINNAKER: (False, "SpiNNaker has basic temporal coding only"),
            NeuromorphicPlatform.TRUENORTH: (False, "TrueNorth does not support phase encoding"),
            NeuromorphicPlatform.AKIDA: (False, "Akida has limited temporal coding support"),
            NeuromorphicPlatform.GENERIC_SNN: (False, "Generic SNN has basic spike processing only"),
            
            # Simulation - full support for testing
            NeuromorphicPlatform.SIMULATION: (True, "Simulation supports all features for testing")
        }
        
        return platform_capabilities.get(self.platform, (False, f"Unknown platform: {self.platform}"))
    
    def _analyze_model_requirements(self) -> Tuple[bool, List[str]]:
        """
        Analyze model configuration to determine if phase encoding would be beneficial.
        
        Returns:
            (should_enable: bool, reasons: List[str]) - Whether to enable and reasons
        """
        should_enable = False
        reasons = []
        
        # Check temporal pattern requirements
        if self.enable_temporal_patterns:
            should_enable = True
            reasons.append("Temporal patterns are enabled")
        
        # Check oscillatory dynamics
        if self.enable_oscillatory_dynamics:
            should_enable = True
            reasons.append("Oscillatory dynamics are enabled")
        
        # Check hierarchical structure with multiple levels
        if self.enable_hierarchical_structure and self.num_hierarchy_levels > 1:
            should_enable = True
            reasons.append(f"Hierarchical structure with {self.num_hierarchy_levels} levels")
        
        # Check 3rd generation advanced features
        if self.generation >= 3:
            advanced_features_enabled = any([
                self.enable_multi_compartment,
                self.enable_adaptive_threshold,
                self.enable_burst_firing,
                self.enable_stochastic_dynamics
            ])
            if advanced_features_enabled:
                should_enable = True
                reasons.append("Advanced 3rd generation neuron features are enabled")
        
        return should_enable, reasons
    
    def _analyze_input_data_characteristics(self) -> Tuple[bool, List[str]]:
        """
        Analyze input data characteristics to determine if phase encoding would be beneficial.
        
        Returns:
            (should_enable: bool, reasons: List[str]) - Whether to enable and reasons
        """
        should_enable = False
        reasons = []
        
        if self.input_data_type:
            if self.input_data_type == 'sequential':
                should_enable = True
                reasons.append("Sequential data benefits from phase encoding")
            elif self.input_data_type == 'high_dimensional':
                should_enable = True
                reasons.append("High-dimensional data benefits from temporal multiplexing")
            elif self.input_data_type == 'temporal_patterns':
                should_enable = True
                reasons.append("Data has natural temporal patterns")
        
        return should_enable, reasons
    
    def _configure_phase_encoding(self):
        """
        Intelligently configure phase encoding based on hardware, model, and data characteristics.
        """
        # Store original value
        original_value = self.enable_phase_encoding
        
        # Detect hardware capabilities
        hw_supported, hw_reason = self._detect_hardware_phase_encoding_support()
        
        # Analyze model requirements
        model_should_enable, model_reasons = self._analyze_model_requirements()
        
        # Analyze input data characteristics
        data_should_enable, data_reasons = self._analyze_input_data_characteristics()
        
        # Combine all factors to make decision
        should_enable = False
        all_reasons = []
        
        if hw_supported:
            if model_should_enable or data_should_enable:
                should_enable = True
                all_reasons.append(hw_reason)
                all_reasons.extend(model_reasons)
                all_reasons.extend(data_reasons)
        else:
            # Hardware doesn't support it - provide fallback guidance
            if model_should_enable or data_should_enable:
                logger.warning(
                    f"Phase encoding would be beneficial but {hw_reason}. "
                    f"Consider using simulation mode or upgrading to 3rd generation platform."
                )
        
        # Apply auto-configuration only if user didn't explicitly set it to True
        if not original_value and should_enable:
            self.enable_phase_encoding = True
            if all_reasons:
                logger.info(
                    f"Automatically enabled phase encoding. Reasons: {'; '.join(all_reasons)}"
                )
        elif original_value and not (hw_supported and (model_should_enable or data_should_enable)):
            # User set it but conditions may not warrant it - provide guidance
            if not hw_supported:
                logger.warning(
                    f"Phase encoding is enabled but {hw_reason}. "
                    f"Consider using simulation mode for testing."
                )
            elif not (model_should_enable or data_should_enable):
                logger.info(
                    f"Phase encoding is enabled. Consider enabling temporal patterns or "
                    f"oscillatory dynamics to fully utilize phase encoding capabilities."
                )
    
    @classmethod
    def create_with_explicit_phase_encoding(cls, enable_phase_encoding: bool, **kwargs):
        """
        Create a NeuromorphicConfig with explicit phase encoding setting.
        
        This factory method allows users to explicitly set phase encoding
        and bypass auto-configuration.
        
        Args:
            enable_phase_encoding: Explicit phase encoding setting
            **kwargs: Other configuration parameters
            
        Returns:
            NeuromorphicConfig with explicit phase encoding setting
        """
        config = cls(_auto_configure_phase_encoding=False, 
                    enable_phase_encoding=enable_phase_encoding, 
                    **kwargs)
        return config


@runtime_checkable
class NeuromorphicBackend(Protocol):
    """Protocol for neuromorphic hardware backends."""
    
    def initialize(self, config: NeuromorphicConfig) -> None:
        """Initialize the neuromorphic backend."""
        ...
    
    def encode_spikes(self, data: torch.Tensor) -> List[SpikeEvent]:
        """Encode data as spike events."""
        ...
    
    def decode_spikes(self, spikes: List[SpikeEvent]) -> torch.Tensor:
        """Decode spike events back to tensor data."""
        ...
    
    def run_network(self, spikes: List[SpikeEvent], network_params: Dict[str, Any]) -> List[SpikeEvent]:
        """Run network inference on neuromorphic hardware."""
        ...


class LeakyIntegrateFireNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model for neuromorphic compatibility.
    
    This implements a basic LIF neuron that can be used as a building block
    for neuromorphic-compatible networks.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        super().__init__()
        self.config = config
        
        # Neuron parameters  
        self.register_buffer('v_mem', torch.zeros(1))  # Membrane potential
        self.register_buffer('i_syn', torch.zeros(1))  # Synaptic current
        
        # Decay factors (exponential approximation)
        self.alpha_mem = torch.tensor(np.exp(-config.dt / config.tau_mem))
        self.alpha_syn = torch.tensor(np.exp(-config.dt / config.tau_syn))
        
    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LIF neuron.
        
        Args:
            input_current: Input current at this time step
        
        Returns:
            (spike_output, membrane_potential)
        """
        batch_size = input_current.shape[0]
        
        # Expand state if needed
        if self.v_mem.shape[0] != batch_size:
            self.v_mem = self.v_mem.expand(batch_size, -1).contiguous()
            self.i_syn = self.i_syn.expand(batch_size, -1).contiguous()
        
        # Update synaptic current
        self.i_syn = self.alpha_syn * self.i_syn + input_current
        
        # Update membrane potential
        self.v_mem = self.alpha_mem * self.v_mem + self.i_syn
        
        # Check for spikes
        spikes = (self.v_mem >= self.config.v_threshold).float()
        
        # Reset spiked neurons
        self.v_mem = torch.where(spikes.bool(), 
                                torch.tensor(self.config.v_reset), 
                                self.v_mem)
        
        return spikes, self.v_mem
    
    def reset_state(self, batch_size: Optional[int] = None):
        """Reset neuron state."""
        if batch_size is not None:
            self.v_mem = torch.full((batch_size, 1), self.config.v_rest)
            self.i_syn = torch.zeros(batch_size, 1)
        else:
            self.v_mem.fill_(self.config.v_rest)
            self.i_syn.fill_(0.0)


class RateToSpikeEncoder:
    """
    Convert continuous rate values to spike trains for neuromorphic processing.
    
    Supports different encoding schemes:
    - Poisson encoding
    - Rate encoding
    - Temporal encoding
    """
    
    def __init__(self, config: NeuromorphicConfig, encoding_type: str = "poisson"):
        self.config = config
        self.encoding_type = encoding_type
        
    def encode(self, rates: torch.Tensor, duration: Optional[float] = None) -> List[SpikeEvent]:
        """
        Encode rate values as spike events.
        
        Args:
            rates: Rate values to encode [batch_size, num_neurons]
            duration: Encoding duration (uses config.encoding_window if None)
        
        Returns:
            List of spike events
        """
        if duration is None:
            duration = self.config.encoding_window
        
        num_steps = int(duration / self.config.dt)
        batch_size, num_neurons = rates.shape
        
        spikes = []
        
        if self.encoding_type == "poisson":
            # Poisson encoding - spikes follow Poisson process
            for batch_idx in range(batch_size):
                for neuron_idx in range(num_neurons):
                    rate = rates[batch_idx, neuron_idx].item()
                    
                    # Clamp rate to reasonable bounds
                    rate = max(0, min(rate * self.config.max_spike_rate, self.config.max_spike_rate))
                    
                    # Generate Poisson spike times
                    if rate > 0:
                        # Expected number of spikes in duration
                        lambda_param = rate * duration
                        num_spikes = np.random.poisson(lambda_param)
                        
                        # Generate random spike times
                        if num_spikes > 0:
                            spike_times = np.sort(np.random.uniform(0, duration, num_spikes))
                            
                            for spike_time in spike_times:
                                spikes.append(SpikeEvent(
                                    neuron_id=batch_idx * num_neurons + neuron_idx,
                                    timestamp=spike_time,
                                    amplitude=1.0,
                                    metadata={'batch_idx': batch_idx, 'neuron_idx': neuron_idx}
                                ))
        
        elif self.encoding_type == "rate":
            # Rate encoding - constant inter-spike interval based on rate
            for batch_idx in range(batch_size):
                for neuron_idx in range(num_neurons):
                    rate = rates[batch_idx, neuron_idx].item()
                    
                    if rate > 0:
                        # Calculate inter-spike interval
                        isi = 1.0 / (rate * self.config.max_spike_rate)
                        
                        # Generate regularly spaced spikes
                        current_time = isi
                        while current_time < duration:
                            spikes.append(SpikeEvent(
                                neuron_id=batch_idx * num_neurons + neuron_idx,
                                timestamp=current_time,
                                amplitude=1.0,
                                metadata={'batch_idx': batch_idx, 'neuron_idx': neuron_idx}
                            ))
                            current_time += isi
        
        return spikes
    
    def decode(self, spikes: List[SpikeEvent], num_neurons: int, 
              duration: Optional[float] = None) -> torch.Tensor:
        """
        Decode spike events back to rate values.
        
        Args:
            spikes: List of spike events
            num_neurons: Total number of neurons
            duration: Decoding duration
        
        Returns:
            Decoded rates [batch_size, num_neurons]
        """
        if duration is None:
            duration = self.config.encoding_window
        
        # Group spikes by neuron
        spike_counts = {}
        max_neuron_id = 0
        
        for spike in spikes:
            neuron_id = spike.neuron_id
            max_neuron_id = max(max_neuron_id, neuron_id)
            
            if neuron_id not in spike_counts:
                spike_counts[neuron_id] = 0
            spike_counts[neuron_id] += 1
        
        # Determine batch size
        batch_size = (max_neuron_id // num_neurons) + 1 if spikes else 1
        
        # Create rate tensor
        rates = torch.zeros(batch_size, num_neurons)
        
        for neuron_id, count in spike_counts.items():
            batch_idx = neuron_id // num_neurons
            neuron_idx = neuron_id % num_neurons
            
            # Convert spike count to rate
            rate = count / duration / self.config.max_spike_rate
            rates[batch_idx, neuron_idx] = rate
        
        return rates


class SpikeBasedAdaptiveLayer(nn.Module):
    """
    Adaptive neural network layer using spike-based processing.
    
    This layer converts the standard adaptive dynamics to spike-based
    computation for neuromorphic compatibility.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        config: NeuromorphicConfig,
        num_timesteps: int = 100
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.config = config
        self.num_timesteps = num_timesteps
        
        # LIF neurons for processing
        self.lif_neurons = nn.ModuleList([
            LeakyIntegrateFireNeuron(config) for _ in range(hidden_size)
        ])
        
        # Synaptic weights
        self.input_weights = nn.Linear(input_size, hidden_size, bias=False)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Spike encoder/decoder
        self.encoder = RateToSpikeEncoder(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using spike-based processing.
        
        Args:
            x: Input tensor [batch_size, input_size]
        
        Returns:
            Output tensor [batch_size, hidden_size]
        """
        batch_size = x.shape[0]
        
        # Reset neuron states
        for neuron in self.lif_neurons:
            neuron.reset_state(batch_size)
        
        # Convert input to spikes
        input_spikes = self.encoder.encode(x)
        
        # Process through time steps
        output_spikes = []
        
        for t in range(self.num_timesteps):
            current_time = t * self.config.dt
            
            # Get input spikes for current time step
            input_current = torch.zeros(batch_size, self.input_size)
            
            for spike in input_spikes:
                if abs(spike.timestamp - current_time) < self.config.dt / 2:
                    metadata = spike.metadata or {}
                    batch_idx = metadata.get('batch_idx', 0)
                    neuron_idx = metadata.get('neuron_idx', 0)
                    
                    if batch_idx < batch_size and neuron_idx < self.input_size:
                        input_current[batch_idx, neuron_idx] += spike.amplitude
            
            # Transform input through weights
            weighted_input = self.input_weights(input_current)
            
            # Add recurrent input from previous step
            if t > 0:
                # Get previous spike output
                prev_spikes = torch.zeros(batch_size, self.hidden_size)
                for spike in output_spikes:
                    if abs(spike.timestamp - (current_time - self.config.dt)) < self.config.dt / 2:
                        metadata = spike.metadata or {}
                        batch_idx = metadata.get('batch_idx', 0)
                        neuron_idx = metadata.get('neuron_idx', 0)
                        
                        if batch_idx < batch_size and neuron_idx < self.hidden_size:
                            prev_spikes[batch_idx, neuron_idx] += spike.amplitude
                
                recurrent_input = self.recurrent_weights(prev_spikes)
                total_input = weighted_input + recurrent_input
            else:
                total_input = weighted_input
            
            # Process through LIF neurons
            for neuron_idx, neuron in enumerate(self.lif_neurons):
                current_input = total_input[:, neuron_idx:neuron_idx+1]
                spikes, _ = neuron(current_input)
                
                # Record output spikes
                for batch_idx in range(batch_size):
                    if spikes[batch_idx, 0] > 0:
                        output_spikes.append(SpikeEvent(
                            neuron_id=batch_idx * self.hidden_size + neuron_idx,
                            timestamp=current_time,
                            amplitude=1.0,
                            metadata={'batch_idx': batch_idx, 'neuron_idx': neuron_idx}
                        ))
        
        # Decode output spikes back to rates
        output_rates = self.encoder.decode(
            output_spikes, 
            self.hidden_size,
            duration=self.num_timesteps * self.config.dt
        )
        
        return output_rates


class NeuromorphicAdaptiveModel(nn.Module):
    """
    Complete adaptive neural network model with neuromorphic compatibility.
    
    This model can run on neuromorphic hardware or simulate neuromorphic
    processing on conventional hardware.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        config: Optional[NeuromorphicConfig] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.config = config or NeuromorphicConfig()
        
        # Build layers
        layers = []
        
        # First layer
        layers.append(SpikeBasedAdaptiveLayer(
            input_dim, hidden_dim, self.config
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(SpikeBasedAdaptiveLayer(
                hidden_dim, hidden_dim, self.config
            ))
        
        # Output layer (if more than 1 layer)
        if num_layers > 1:
            layers.append(SpikeBasedAdaptiveLayer(
                hidden_dim, output_dim, self.config
            ))
        
        self.layers = nn.ModuleList(layers)
        
        # Final projection if needed
        if num_layers == 1:
            self.output_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through neuromorphic model."""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final projection if needed
        if self.output_projection is not None:
            x = self.output_projection(x)
        
        return x


class SimulationBackend:
    """
    Simulation backend for neuromorphic processing.
    
    This provides a software simulation of neuromorphic hardware
    for development and testing purposes.
    """
    
    def __init__(self):
        self.config = None
        self.encoder = None
    
    def initialize(self, config: NeuromorphicConfig) -> None:
        """Initialize simulation backend."""
        self.config = config
        self.encoder = RateToSpikeEncoder(config)
        logger.info("Initialized neuromorphic simulation backend")
    
    def encode_spikes(self, data: torch.Tensor) -> List[SpikeEvent]:
        """Encode data as spikes."""
        if self.encoder is None:
            raise RuntimeError("Backend not initialized")
        return self.encoder.encode(data)
    
    def decode_spikes(self, spikes: List[SpikeEvent]) -> torch.Tensor:
        """Decode spikes back to tensor."""
        if self.encoder is None:
            raise RuntimeError("Backend not initialized")
        
        # Determine tensor shape from spikes
        if not spikes:
            return torch.zeros(1, 1)
        
        max_neuron = max(spike.neuron_id for spike in spikes)
        return self.encoder.decode(spikes, max_neuron + 1)
    
    def run_network(self, spikes: List[SpikeEvent], network_params: Dict[str, Any]) -> List[SpikeEvent]:
        """Run network simulation."""
        # This is a placeholder - in practice would run full SNN simulation
        logger.info(f"Simulating neuromorphic network with {len(spikes)} input spikes")
        
        # Simple passthrough for now
        return spikes


def create_neuromorphic_model(
    input_dim: int,
    output_dim: int,
    platform: NeuromorphicPlatform = NeuromorphicPlatform.SIMULATION,
    config: Optional[NeuromorphicConfig] = None
) -> NeuromorphicAdaptiveModel:
    """
    Factory function to create neuromorphic adaptive model.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        platform: Target neuromorphic platform
        config: Neuromorphic configuration
    
    Returns:
        Configured neuromorphic model
    """
    if config is None:
        config = NeuromorphicConfig(platform=platform)
    
    model = NeuromorphicAdaptiveModel(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config
    )
    
    logger.info(f"Created neuromorphic model for platform: {platform.value}")
    
    return model


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create neuromorphic model
    config = NeuromorphicConfig(
        platform=NeuromorphicPlatform.SIMULATION,
        dt=0.001,
        v_threshold=1.0
    )
    
    model = create_neuromorphic_model(
        input_dim=784,  # MNIST
        output_dim=10,
        config=config
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 784)
    
    print(f"Input shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("Neuromorphic model test completed successfully!")