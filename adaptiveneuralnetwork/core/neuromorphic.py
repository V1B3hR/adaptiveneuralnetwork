"""
Neuromorphic hardware compatibility layer for adaptive neural networks.

This module provides abstractions and implementations for neuromorphic hardware
compatibility, including spike-based computation, event-driven processing,
hardware-specific optimizations, and advanced real-time adaptation mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Protocol, runtime_checkable, Callable, Union
import numpy as np
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import deque, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

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
    
    # Advanced Platforms (4th Gen)
    MEMRISTIVE_CROSSBAR = "memristive_crossbar"
    PHOTONIC_SNN = "photonic_snn"
    QUANTUM_NEUROMORPHIC = "quantum_neuromorphic"
    
    # Simulation
    SIMULATION = "simulation"


class PlasticityType(Enum):
    """Types of synaptic plasticity mechanisms."""
    STDP = "spike_timing_dependent_plasticity"
    BCM = "bienenstock_cooper_munro"
    HOMEOSTATIC = "homeostatic_scaling" 
    METAPLASTICITY = "metaplasticity"
    TRIPLET_STDP = "triplet_stdp"
    CALCIUM_DEPENDENT = "calcium_dependent"
    DOPAMINE_MODULATED = "dopamine_modulated"
    VOLTAGE_DEPENDENT = "voltage_dependent"
    STRUCTURAL_PLASTICITY = "structural_plasticity"


class AdaptationMode(Enum):
    """Real-time adaptation modes."""
    CONTINUOUS = "continuous"
    EPISODIC = "episodic"
    TRIGGERED = "triggered"
    LEARNING_BASED = "learning_based"
    HYBRID = "hybrid"


class NeuronType(Enum):
    """Advanced neuron model types."""
    LIF = "leaky_integrate_fire"
    ADAPTIVE_LIF = "adaptive_lif"
    IZHIKEVICH = "izhikevich"
    HODGKIN_HUXLEY = "hodgkin_huxley" 
    MULTI_COMPARTMENT = "multi_compartment"
    STOCHASTIC_LIF = "stochastic_lif"
    FRACTIONAL_LIF = "fractional_lif"
    RESONATOR = "resonator_neuron"


@dataclass
class SpikeEvent:
    """Represents a spike event in neuromorphic processing."""
    neuron_id: int
    timestamp: float
    amplitude: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    # Advanced spike properties
    phase: Optional[float] = None
    burst_index: Optional[int] = None
    dendrite_id: Optional[int] = None
    axon_delay: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PlasticityRule:
    """Configuration for plasticity rules."""
    rule_type: PlasticityType
    learning_rate: float = 0.01
    time_window: float = 0.02  # ms
    
    # STDP parameters
    tau_plus: float = 0.02
    tau_minus: float = 0.02
    a_plus: float = 1.0
    a_minus: float = 1.0
    
    # BCM parameters
    tau_bcm: float = 1.0
    theta_0: float = 1.0
    
    # Homeostatic parameters
    target_rate: float = 10.0  # Hz
    tau_homeostatic: float = 10.0  # s
    
    # Metaplasticity parameters
    meta_learning_rate: float = 0.001
    sliding_threshold: bool = True
    
    # Modulation parameters
    modulation_factor: float = 1.0
    modulator_type: Optional[str] = None  # 'dopamine', 'acetylcholine', etc.
    
    # Real-time adaptation
    adaptive_learning_rate: bool = False
    adaptation_window: float = 1.0
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 0.1


@dataclass
class RealTimeAdaptationConfig:
    """Configuration for real-time adaptation mechanisms."""
    mode: AdaptationMode = AdaptationMode.CONTINUOUS
    update_frequency: float = 0.1  # seconds
    
    # Performance monitoring
    performance_window: float = 1.0
    performance_threshold: float = 0.8
    adaptation_sensitivity: float = 0.1
    
    # Resource monitoring
    monitor_energy: bool = True
    monitor_latency: bool = True
    monitor_accuracy: bool = True
    energy_budget: Optional[float] = None  # Joules
    latency_budget: Optional[float] = None  # seconds
    
    # Adaptation strategies
    parameter_scaling: bool = True
    topology_adaptation: bool = False
    plasticity_modulation: bool = True
    
    # Learning rate schedules
    lr_schedule_type: str = "exponential"  # exponential, cosine, linear, adaptive
    lr_decay_rate: float = 0.95
    lr_min: float = 1e-6
    lr_max: float = 0.1
    
    # Network topology adaptation
    pruning_threshold: float = 0.01
    growth_threshold: float = 0.8
    max_connections_per_neuron: int = 100
    
    # Environmental adaptation
    temperature_compensation: bool = False
    noise_adaptation: bool = False
    power_scaling: bool = False


@dataclass  
class NeuromorphicConfig:
    """Enhanced configuration for neuromorphic hardware compatibility."""
    platform: NeuromorphicPlatform = NeuromorphicPlatform.SIMULATION
    
    # Basic neuron parameters
    dt: float = 0.001  # Time step in seconds
    v_threshold: float = 1.0  # Spike threshold
    v_reset: float = 0.0  # Reset potential
    v_rest: float = 0.0  # Resting potential
    tau_mem: float = 0.01  # Membrane time constant
    tau_syn: float = 0.005  # Synaptic time constant
    refractory_period: float = 0.002  # Refractory period
    
    # Advanced neuron parameters
    neuron_type: NeuronType = NeuronType.LIF
    v_spike: float = 1.0  # Spike amplitude
    tau_adaptation: float = 0.1  # Adaptation time constant
    adaptation_strength: float = 0.1  # Adaptation coupling
    noise_amplitude: float = 0.0  # Background noise
    
    # Izhikevich parameters (for Izhikevich neuron type)
    izhikevich_a: float = 0.02
    izhikevich_b: float = 0.2
    izhikevich_c: float = -65.0
    izhikevich_d: float = 2.0
    
    # Multi-compartment parameters
    num_compartments: int = 1
    compartment_coupling: float = 0.1
    dendritic_delay: float = 0.001
    
    # Encoding parameters
    encoding_window: float = 0.1  # Time window for rate encoding
    max_spike_rate: float = 1000.0  # Maximum spike rate (Hz)
    temporal_resolution: float = 0.0001  # Temporal resolution
    
    # Generation and features
    generation: int = 3  # Neuromorphic generation (2, 3, or 4)
    
    # Advanced neuron features
    enable_multi_compartment: bool = False
    enable_adaptive_threshold: bool = True
    enable_burst_firing: bool = False
    enable_stochastic_dynamics: bool = False
    enable_calcium_dynamics: bool = False
    enable_ion_channels: bool = False
    
    # Plasticity configuration
    plasticity_rules: List[PlasticityRule] = field(default_factory=list)
    enable_stdp: bool = True
    enable_metaplasticity: bool = False
    enable_homeostatic_scaling: bool = True
    enable_structural_plasticity: bool = False
    
    # Real-time adaptation
    real_time_adaptation: RealTimeAdaptationConfig = field(default_factory=RealTimeAdaptationConfig)
    
    # Network topology parameters
    enable_hierarchical_structure: bool = False
    enable_dynamic_connectivity: bool = True
    num_hierarchy_levels: int = 3
    connectivity_density: float = 0.1
    
    # Temporal coding parameters
    enable_temporal_patterns: bool = True
    enable_phase_encoding: bool = False
    enable_oscillatory_dynamics: bool = False
    enable_sparse_coding: bool = True
    enable_population_coding: bool = False
    
    # Advanced temporal features
    enable_gamma_oscillations: bool = False
    enable_theta_rhythms: bool = False
    enable_delta_waves: bool = False
    oscillation_frequency: float = 40.0  # Hz for gamma
    phase_coupling_strength: float = 0.1
    
    # Hardware-specific parameters
    bit_precision: int = 16  # Bit precision for weights/states
    quantization_levels: int = 256
    enable_analog_compute: bool = False
    enable_in_memory_compute: bool = False
    
    # Energy and timing
    energy_per_spike: float = 1e-12  # Joules per spike
    synaptic_delay_mean: float = 0.001
    synaptic_delay_std: float = 0.0005
    axonal_delay_mean: float = 0.002
    axonal_delay_std: float = 0.001
    
    # Memristive parameters (for memristive platforms)
    memristor_conductance_min: float = 1e-6
    memristor_conductance_max: float = 1e-3
    memristor_retention_time: float = 100.0  # seconds
    memristor_switching_energy: float = 1e-15  # Joules
    
    # Input data characteristics
    input_data_type: Optional[str] = None
    expected_input_rate: Optional[float] = None
    input_sparsity: Optional[float] = None
    
    # Monitoring and logging
    enable_performance_monitoring: bool = True
    enable_energy_monitoring: bool = False
    enable_spike_monitoring: bool = False
    monitoring_resolution: float = 0.01  # seconds
    
    # Real-time constraints
    max_processing_latency: Optional[float] = None  # seconds
    real_time_factor: float = 1.0  # 1.0 = real-time, >1 = faster than real-time
    
    # Safety and robustness
    enable_fault_tolerance: bool = False
    redundancy_factor: int = 1
    error_correction: bool = False
    
    # Internal flags
    _auto_configure_phase_encoding: bool = True
    _parameter_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Enhanced post-initialization with intelligent configuration."""
        # Initialize default plasticity rules if none provided
        if not self.plasticity_rules and self.enable_stdp:
            self.plasticity_rules.append(
                PlasticityRule(
                    rule_type=PlasticityType.STDP,
                    learning_rate=0.01,
                    adaptive_learning_rate=True
                )
            )
        
        if self.enable_homeostatic_scaling:
            self.plasticity_rules.append(
                PlasticityRule(
                    rule_type=PlasticityType.HOMEOSTATIC,
                    learning_rate=0.001,
                    target_rate=10.0
                )
            )
        
        # Configure platform-specific parameters
        self._configure_platform_specifics()
        
        # Apply intelligent configuration
        if self._auto_configure_phase_encoding:
            self._configure_phase_encoding()
        
        # Initialize parameter history
        self._log_parameter_state("initialization")
    
    def _configure_platform_specifics(self):
        """Configure platform-specific optimizations."""
        platform_configs = {
            NeuromorphicPlatform.LOIHI2: {
                'bit_precision': 8,
                'enable_in_memory_compute': True,
                'max_spike_rate': 1000.0,
                'energy_per_spike': 23e-12
            },
            NeuromorphicPlatform.SPINNAKER2: {
                'bit_precision': 16,
                'enable_analog_compute': False,
                'max_spike_rate': 10000.0,
                'energy_per_spike': 45e-12
            },
            NeuromorphicPlatform.MEMRISTIVE_CROSSBAR: {
                'bit_precision': 4,
                'enable_analog_compute': True,
                'enable_in_memory_compute': True,
                'energy_per_spike': 0.1e-12
            },
            NeuromorphicPlatform.PHOTONIC_SNN: {
                'bit_precision': 32,
                'enable_analog_compute': True,
                'max_spike_rate': 100000.0,
                'energy_per_spike': 0.01e-12,
                'temporal_resolution': 1e-6
            }
        }
        
        if self.platform in platform_configs:
            config = platform_configs[self.platform]
            for param, value in config.items():
                if hasattr(self, param):
                    setattr(self, param, value)
            logger.info(f"Applied {self.platform.value} specific configuration")
    
    def update_parameters_realtime(self, parameter_updates: Dict[str, Any], 
                                 source: str = "manual") -> bool:
        """
        Update neuromorphic parameters in real-time.
        
        Args:
            parameter_updates: Dictionary of parameter names and new values
            source: Source of the update (manual, adaptive, performance, etc.)
            
        Returns:
            bool: Success status of parameter update
        """
        try:
            # Validate parameter updates
            valid_updates = self._validate_parameter_updates(parameter_updates)
            
            # Apply updates
            for param_name, new_value in valid_updates.items():
                old_value = getattr(self, param_name)
                setattr(self, param_name, new_value)
                logger.debug(f"Updated {param_name}: {old_value} -> {new_value}")
            
            # Log parameter change
            self._log_parameter_state(f"update_{source}")
            
            # Trigger reconfiguration if needed
            if any(param in ['platform', 'neuron_type', 'generation'] 
                   for param in valid_updates.keys()):
                self._configure_platform_specifics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update parameters: {e}")
            return False
    
    def _validate_parameter_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameter updates for safety and compatibility."""
        valid_updates = {}
        
        for param_name, new_value in updates.items():
            if not hasattr(self, param_name):
                logger.warning(f"Unknown parameter: {param_name}")
                continue
            
            # Type checking
            current_value = getattr(self, param_name)
            if not isinstance(new_value, type(current_value)):
                logger.warning(f"Type mismatch for {param_name}: {type(new_value)} vs {type(current_value)}")
                continue
            
            # Range validation
            if param_name in ['dt', 'tau_mem', 'tau_syn'] and new_value <= 0:
                logger.warning(f"Invalid value for {param_name}: must be positive")
                continue
            
            if param_name == 'max_spike_rate' and new_value > 100000:
                logger.warning(f"Spike rate {new_value} Hz may be too high for platform {self.platform.value}")
            
            valid_updates[param_name] = new_value
        
        return valid_updates
    
    def _log_parameter_state(self, event: str):
        """Log current parameter state for history tracking."""
        state = {
            'timestamp': time.time(),
            'event': event,
            'dt': self.dt,
            'v_threshold': self.v_threshold,
            'tau_mem': self.tau_mem,
            'learning_rates': [rule.learning_rate for rule in self.plasticity_rules],
            'platform': self.platform.value
        }
        self._parameter_history.append(state)
        
        # Keep only recent history (last 1000 entries)
        if len(self._parameter_history) > 1000:
            self._parameter_history = self._parameter_history[-1000:]
    
    def get_adaptive_learning_rate(self, rule_index: int = 0, 
                                 performance_metric: Optional[float] = None) -> float:
        """
        Calculate adaptive learning rate based on performance and time.
        
        Args:
            rule_index: Index of plasticity rule
            performance_metric: Current performance metric (0-1, higher is better)
            
        Returns:
            Adaptive learning rate
        """
        if rule_index >= len(self.plasticity_rules):
            return 0.01  # Default learning rate
        
        rule = self.plasticity_rules[rule_index]
        base_lr = rule.learning_rate
        
        if not rule.adaptive_learning_rate:
            return base_lr
        
        # Time-based decay
        if self.real_time_adaptation.lr_schedule_type == "exponential":
            time_factor = self.real_time_adaptation.lr_decay_rate ** (
                len(self._parameter_history) / 100
            )
        else:
            time_factor = 1.0
        
        # Performance-based adaptation
        if performance_metric is not None:
            if performance_metric < self.real_time_adaptation.performance_threshold:
                # Poor performance - increase learning rate
                perf_factor = 1.0 + (self.real_time_adaptation.adaptation_sensitivity * 
                                   (self.real_time_adaptation.performance_threshold - performance_metric))
            else:
                # Good performance - maintain or slightly decrease learning rate
                perf_factor = 1.0 - (0.1 * self.real_time_adaptation.adaptation_sensitivity)
        else:
            perf_factor = 1.0
        
        # Combine factors
        adaptive_lr = base_lr * time_factor * perf_factor
        
        # Clamp to bounds
        adaptive_lr = max(rule.min_learning_rate, 
                         min(rule.max_learning_rate, adaptive_lr))
        
        return adaptive_lr
    
    def adapt_to_environment(self, environmental_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Adapt neuromorphic parameters based on environmental conditions.
        
        Args:
            environmental_data: Dictionary with environmental measurements
            
        Returns:
            Dictionary of parameter adaptations made
        """
        adaptations = {}
        
        # Temperature compensation
        if ('temperature' in environmental_data and 
            self.real_time_adaptation.temperature_compensation):
            
            temp_celsius = environmental_data['temperature']
            # Typical Q10 factor for biological systems
            q10_factor = 2.0 ** ((temp_celsius - 25.0) / 10.0)
            
            # Adapt time constants
            new_tau_mem = self.tau_mem / q10_factor
            new_tau_syn = self.tau_syn / q10_factor
            
            adaptations.update({
                'tau_mem': new_tau_mem,
                'tau_syn': new_tau_syn
            })
        
        # Noise adaptation
        if ('noise_level' in environmental_data and 
            self.real_time_adaptation.noise_adaptation):
            
            noise_level = environmental_data['noise_level']
            # Increase threshold with higher noise
            threshold_adjustment = 1.0 + (noise_level * 0.1)
            new_threshold = self.v_threshold * threshold_adjustment
            
            adaptations['v_threshold'] = new_threshold
        
        # Power scaling
        if ('available_power' in environmental_data and 
            self.real_time_adaptation.power_scaling):
            
            power_ratio = environmental_data['available_power'] / 1.0  # Normalized
            # Scale spike rate and precision based on available power
            new_max_rate = self.max_spike_rate * power_ratio
            
            adaptations['max_spike_rate'] = new_max_rate
        
        # Apply adaptations
        if adaptations:
            success = self.update_parameters_realtime(adaptations, "environmental")
            if success:
                logger.info(f"Applied environmental adaptations: {list(adaptations.keys())}")
        
        return adaptations
    
    def export_config(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Export configuration to dictionary or file."""
        config_dict = {
            'platform': self.platform.value,
            'neuron_type': self.neuron_type.value,
            'basic_params': {
                'dt': self.dt,
                'v_threshold': self.v_threshold,
                'v_reset': self.v_reset,
                'v_rest': self.v_rest,
                'tau_mem': self.tau_mem,
                'tau_syn': self.tau_syn
            },
            'plasticity_rules': [
                {
                    'type': rule.rule_type.value,
                    'learning_rate': rule.learning_rate,
                    'adaptive': rule.adaptive_learning_rate
                } for rule in self.plasticity_rules
            ],
            'real_time_adaptation': {
                'mode': self.real_time_adaptation.mode.value,
                'update_frequency': self.real_time_adaptation.update_frequency,
                'parameter_scaling': self.real_time_adaptation.parameter_scaling
            },
            'parameter_history_length': len(self._parameter_history)
        }
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Exported configuration to {filename}")
        
        return config_dict


class RealTimeParameterManager:
    """
    Manages real-time parameter adaptation for neuromorphic systems.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.adaptation_config = config.real_time_adaptation
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.energy_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        
        # Adaptation state
        self.is_adapting = False
        self.last_adaptation_time = 0.0
        self.adaptation_thread = None
        self.stop_adaptation = threading.Event()
        
        # Callbacks for parameter updates
        self.update_callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def start_adaptation(self):
        """Start real-time parameter adaptation."""
        if self.is_adapting:
            return
        
        self.is_adapting = True
        self.stop_adaptation.clear()
        
        if self.adaptation_config.mode == AdaptationMode.CONTINUOUS:
            self.adaptation_thread = threading.Thread(
                target=self._continuous_adaptation_loop
            )
            self.adaptation_thread.start()
            logger.info("Started continuous parameter adaptation")
    
    def stop_adaptation_process(self):
        """Stop real-time parameter adaptation."""
        self.is_adapting = False
        self.stop_adaptation.set()
        
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            self.adaptation_thread.join()
        
        logger.info("Stopped parameter adaptation")
    
    def _continuous_adaptation_loop(self):
        """Continuous adaptation loop running in separate thread."""
        while not self.stop_adaptation.wait(self.adaptation_config.update_frequency):
            try:
                self._perform_adaptation_step()
            except Exception as e:
                logger.error(f"Error in adaptation step: {e}")
    
    def _perform_adaptation_step(self):
        """Perform a single adaptation step."""
        current_time = time.time()
        
        # Skip if not enough time has passed
        if (current_time - self.last_adaptation_time < 
            self.adaptation_config.update_frequency):
            return
        
        # Analyze current performance
        performance_metrics = self._analyze_performance()
        
        # Determine necessary adaptations
        adaptations = self._calculate_adaptations(performance_metrics)
        
        # Apply adaptations
        if adaptations:
            success = self.config.update_parameters_realtime(adaptations, "adaptive")
            if success:
                # Notify callbacks
                for callback in self.update_callbacks:
                    callback(adaptations)
        
        self.last_adaptation_time = current_time
    
    def _analyze_performance(self) -> Dict[str, float]:
        """Analyze current system performance."""
        metrics = {}
        
        # Calculate average performance over window
        if self.performance_history:
            recent_performance = list(self.performance_history)[-int(
                self.adaptation_config.performance_window / 
                self.adaptation_config.update_frequency
            ):]
            metrics['avg_performance'] = np.mean(recent_performance)
            metrics['performance_trend'] = self._calculate_trend(recent_performance)
        
        # Energy efficiency
        if self.energy_history and self.adaptation_config.monitor_energy:
            recent_energy = list(self.energy_history)[-10:]
            metrics['avg_energy'] = np.mean(recent_energy)
            metrics['energy_trend'] = self._calculate_trend(recent_energy)
        
        # Latency analysis
        if self.latency_history and self.adaptation_config.monitor_latency:
            recent_latency = list(self.latency_history)[-10:]
            metrics['avg_latency'] = np.mean(recent_latency)
            metrics['latency_trend'] = self._calculate_trend(recent_latency)
        
        return metrics
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend direction (-1 to 1) for a data series."""
        if len(data) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        return np.tanh(coeffs[0])  # Normalize to [-1, 1]
    
    def _calculate_adaptations(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate parameter adaptations based on performance metrics."""
        adaptations = {}
        
        # Performance-based learning rate adaptation
        if 'avg_performance' in metrics:
            perf = metrics['avg_performance']
            
            if perf < self.adaptation_config.performance_threshold:
                # Poor performance - adapt more aggressively
                if self.adaptation_config.parameter_scaling:
                    # Increase learning rates
                    for i, rule in enumerate(self.config.plasticity_rules):
                        if rule.adaptive_learning_rate:
                            new_lr = rule.learning_rate * 1.1
                            new_lr = min(new_lr, rule.max_learning_rate)
                            rule.learning_rate = new_lr
                
                # Adjust temporal resolution for better precision
                if metrics.get('latency_trend', 0) < 0:  # Latency improving
                    new_dt = self.config.dt * 0.95  # Finer time steps
                    adaptations['dt'] = max(new_dt, 1e-5)
        
        # Energy-based adaptations
        if ('avg_energy' in metrics and 
            self.adaptation_config.energy_budget is not None):
            
            energy_ratio = metrics['avg_energy'] / self.adaptation_config.energy_budget
            
            if energy_ratio > 1.1:  # Over budget
                # Reduce spike rate to save energy
                new_rate = self.config.max_spike_rate * 0.9
                adaptations['max_spike_rate'] = new_rate
                
                # Increase refractory period
                new_refract = self.config.refractory_period * 1.05
                adaptations['refractory_period'] = new_refract
        
        # Latency-based adaptations
        if ('avg_latency' in metrics and 
            self.adaptation_config.latency_budget is not None):
            
            if metrics['avg_latency'] > self.adaptation_config.latency_budget:
                # Reduce precision to improve speed
                if self.config.bit_precision > 4:
                    adaptations['bit_precision'] = self.config.bit_precision - 1
                
                # Increase time step for faster processing
                new_dt = self.config.dt * 1.05
                adaptations['dt'] = min(new_dt, 0.01)
        
        return adaptations
    
    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback to be notified of parameter updates."""
        self.update_callbacks.append(callback)
    
    def log_performance(self, accuracy: float, energy: Optional[float] = None, 
                       latency: Optional[float] = None):
        """Log performance metrics for adaptation."""
        self.performance_history.append(accuracy)
        
        if energy is not None:
            self.energy_history.append(energy)
        
        if latency is not None:
            self.latency_history.append(latency)


class EnhancedLeakyIntegrateFireNeuron(nn.Module):
    """
    Enhanced Leaky Integrate-and-Fire neuron with advanced features.
    """
    
    def __init__(self, config: NeuromorphicConfig, neuron_id: int = 0):
        super().__init__()
        self.config = config
        self.neuron_id = neuron_id
        
        # Basic LIF parameters
        self.register_buffer('v_mem', torch.zeros(1))
        self.register_buffer('i_syn', torch.zeros(1))
        
        # Advanced features
        if config.enable_adaptive_threshold:
            self.register_buffer('v_th_adapt', torch.full((1,), config.v_threshold))
            self.register_buffer('spike_count', torch.zeros(1))
        
        if config.enable_calcium_dynamics:
            self.register_buffer('calcium', torch.zeros(1))
            self.tau_calcium = 0.05  # Calcium time constant
        
        # Stochastic dynamics
        if config.enable_stochastic_dynamics:
            self.noise_amplitude = config.noise_amplitude
        
        # Multi-compartment
        if config.enable_multi_compartment and config.num_compartments > 1:
            self.register_buffer('v_compartments', 
                               torch.zeros(config.num_compartments))
            self.compartment_weights = nn.Parameter(
                torch.ones(config.num_compartments) / config.num_compartments
            )
        
        # Decay factors
        self.alpha_mem = torch.tensor(np.exp(-config.dt / config.tau_mem))
        self.alpha_syn = torch.tensor(np.exp(-config.dt / config.tau_syn))
        
        if config.enable_adaptive_threshold:
            self.alpha_adapt = torch.tensor(np.exp(-config.dt / config.tau_adaptation))
    
    def forward(self, input_current: torch.Tensor, 
                modulation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with multiple neuron features.
        
        Args:
            input_current: Input current
            modulation: Neuromodulation signal (dopamine, etc.)
            
        Returns:
            (spike_output, neuron_states)
        """
        batch_size = input_current.shape[0]
        
        # Expand state buffers
        if self.v_mem.shape[0] != batch_size:
            self._expand_states(batch_size)
        
        # Add stochastic noise
        if self.config.enable_stochastic_dynamics:
            noise = torch.randn_like(input_current) * self.noise_amplitude
            input_current = input_current + noise
        
        # Apply neuromodulation
        if modulation is not None:
            input_current = input_current * (1.0 + modulation)
        
        # Update synaptic current
        self.i_syn = self.alpha_syn * self.i_syn + input_current
        
        # Multi-compartment processing
        if self.config.enable_multi_compartment and hasattr(self, 'v_compartments'):
            # Update each compartment
            for i in range(self.config.num_compartments):
                compartment_input = self.i_syn * self.compartment_weights[i]
                self.v_compartments[i] = (self.alpha_mem * self.v_compartments[i] + 
                                        compartment_input.squeeze())
            
            # Weighted sum for membrane potential
            self.v_mem = torch.sum(self.v_compartments * self.compartment_weights.view(1, -1), 
                                 dim=1, keepdim=True)
        else:
            # Standard single compartment
            self.v_mem = self.alpha_mem * self.v_mem + self.i_syn
        
        # Determine spike threshold
        if self.config.enable_adaptive_threshold:
            threshold = self.v_th_adapt
        else:
            threshold = self.config.v_threshold
        
        # Check for spikes
        spikes = (self.v_mem >= threshold).float()
        
        # Update adaptive threshold
        if self.config.enable_adaptive_threshold:
            # Increase threshold after spike
            self.v_th_adapt = torch.where(
                spikes.bool(),
                self.v_th_adapt + self.config.adaptation_strength,
                self.alpha_adapt * self.v_th_adapt + (1 - self.alpha_adapt) * self.config.v_threshold
            )
            self.spike_count += spikes
        
        # Update calcium dynamics
        if self.config.enable_calcium_dynamics:
            alpha_ca = torch.tensor(np.exp(-self.config.dt / self.tau_calcium))
            self.calcium = alpha_ca * self.calcium + spikes
        
        # Reset spiked neurons
        self.v_mem = torch.where(spikes.bool(), 
                               torch.tensor(self.config.v_reset), 
                               self.v_mem)
        
        # Prepare output states
        states = {
            'v_mem': self.v_mem,
            'i_syn': self.i_syn,
        }
        
        if self.config.enable_adaptive_threshold:
            states['v_threshold'] = self.v_th_adapt
            states['spike_count'] = self.spike_count
        
        if self.config.enable_calcium_dynamics:
            states['calcium'] = self.calcium
        
        return spikes, states
    
    def _expand_states(self, batch_size: int):
        """Expand state tensors for batch processing."""
        self.v_mem = self.v_mem.expand(batch_size, -1).contiguous()
        self.i_syn = self.i_syn.expand(batch_size, -1).contiguous()
        
        if self.config.enable_adaptive_threshold:
            self.v_th_adapt = self.v_th_adapt.expand(batch_size, -1).contiguous()
            self.spike_count = self.spike_count.expand(batch_size, -1).contiguous()
        
        if self.config.enable_calcium_dynamics:
            self.calcium = self.calcium.expand(batch_size, -1).contiguous()


class AdaptivePlasticityManager:
    """
    Manages multiple plasticity rules with real-time adaptation.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.plasticity_rules = config.plasticity_rules
        
        # Plasticity state tracking
        self.pre_spike_history = defaultdict(deque)
        self.post_spike_history = defaultdict(deque)
        self.weight_traces = {}
        
        # Homeostatic variables
        self.firing_rates = defaultdict(float)
        self.rate_targets = defaultdict(lambda: 10.0)  # Hz
        
        # Metaplasticity
        self.plasticity_thresholds = defaultdict(lambda: 1.0)
        
    def update_synaptic_weights(self, pre_neurons: List[int], post_neurons: List[int],
                              spike_times: Dict[int, List[float]], 
                              weights: torch.Tensor) -> torch.Tensor:
        """
        Update synaptic weights based on spike timing and plasticity rules.
        
        Args:
            pre_neurons: List of presynaptic neuron IDs
            post_neurons: List of postsynaptic neuron IDs  
            spike_times: Dictionary mapping neuron ID to spike times
            weights: Current synaptic weight matrix
            
        Returns:
            Updated weight matrix
        """
        updated_weights = weights.clone()
        
        # Apply each plasticity rule
        for rule in self.plasticity_rules:
            if rule.rule_type == PlasticityType.STDP:
                updated_weights = self._apply_stdp(
                    updated_weights, pre_neurons, post_neurons, spike_times, rule
                )
            elif rule.rule_type == PlasticityType.HOMEOSTATIC:
                updated_weights = self._apply_homeostatic_scaling(
                    updated_weights, post_neurons, spike_times, rule
                )
            elif rule.rule_type == PlasticityType.METAPLASTICITY:
                updated_weights = self._apply_metaplasticity(
                    updated_weights, pre_neurons, post_neurons, spike_times, rule
                )
        
        return updated_weights
    
    def _apply_stdp(self, weights: torch.Tensor, pre_neurons: List[int], 
                   post_neurons: List[int], spike_times: Dict[int, List[float]], 
                   rule: PlasticityRule) -> torch.Tensor:
        """Apply Spike-Timing Dependent Plasticity (STDP)."""
        learning_rate = self.config.get_adaptive_learning_rate(0)
        
        for i, pre_id in enumerate(pre_neurons):
            for j, post_id in enumerate(post_neurons):
                if pre_id in spike_times and post_id in spike_times:
                    pre_spikes = spike_times[pre_id]
                    post_spikes = spike_times[post_id]
                    
                    # Calculate weight changes for all spike pairs
                    for pre_time in pre_spikes:
                        for post_time in post_spikes:
                            dt = post_time - pre_time
                            
                            if abs(dt) <= rule.time_window:
                                if dt > 0:  # Pre before post - potentiation
                                    dw = (learning_rate * rule.a_plus * 
                                         np.exp(-dt / rule.tau_plus))
                                else:  # Post before pre - depression  
                                    dw = (-learning_rate * rule.a_minus * 
                                         np.exp(dt / rule.tau_minus))
                                
                                # Apply modulation
                                dw *= rule.modulation_factor
                                
                                # Update weight
                                weights[i, j] += dw
        
        return weights
    
    def _apply_homeostatic_scaling(self, weights: torch.Tensor, post_neurons: List[int],
                                 spike_times: Dict[int, List[float]], 
                                 rule: PlasticityRule) -> torch.Tensor:
        """Apply homeostatic scaling to maintain target firing rates."""
        current_time = max(max(times) if times else 0 for times in spike_times.values())
        time_window = rule.adaptation_window
        
        for j, post_id in enumerate(post_neurons):
            if post_id in spike_times:
                # Calculate recent firing rate
                recent_spikes = [t for t in spike_times[post_id] 
                               if t > current_time - time_window]
                current_rate = len(recent_spikes) / time_window
                
                # Update running average
                alpha = 1.0 - np.exp(-self.config.dt / rule.tau_homeostatic)
                self.firing_rates[post_id] = (alpha * current_rate + 
                                            (1 - alpha) * self.firing_rates[post_id])
                
                # Calculate scaling factor
                rate_error = rule.target_rate - self.firing_rates[post_id]
                scaling_factor = 1.0 + rule.learning_rate * rate_error / rule.target_rate
                
                # Apply scaling to all incoming weights
                weights[:, j] *= scaling_factor
        
        return weights
    
    def _apply_metaplasticity(self, weights: torch.Tensor, pre_neurons: List[int],
                            post_neurons: List[int], spike_times: Dict[int, List[float]],
                            rule: PlasticityRule) -> torch.Tensor:
        """Apply metaplasticity - plasticity of plasticity."""
        for i, pre_id in enumerate(pre_neurons):
            for j, post_id in enumerate(post_neurons):
                # Calculate recent activity correlation
                if pre_id in spike_times and post_id in spike_times:
                    pre_rate = len(spike_times[pre_id]) / rule.adaptation_window
                    post_rate = len(spike_times[post_id]) / rule.adaptation_window
                    
                    # Update plasticity threshold based on activity
                    activity_product = pre_rate * post_rate
                    threshold_change = rule.meta_learning_rate * (
                        activity_product - self.plasticity_thresholds[(pre_id, post_id)]
                    )
                    
                    self.plasticity_thresholds[(pre_id, post_id)] += threshold_change
                    
                    # Modulate future plasticity based on threshold
                    if rule.sliding_threshold:
                        modulation = 1.0 / (1.0 + self.plasticity_thresholds[(pre_id, post_id)])
                        rule.modulation_factor = modulation
        
        return weights


# Factory function for creating enhanced neuromorphic models
def create_enhanced_neuromorphic_model(
    input_dim: int,
    output_dim: int,
    platform: NeuromorphicPlatform = NeuromorphicPlatform.SIMULATION,
    config: Optional[NeuromorphicConfig] = None,
    enable_real_time_adaptation: bool = True
) -> Tuple[nn.Module, RealTimeParameterManager]:
    """
    Create enhanced neuromorphic model with real-time adaptation capabilities.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension  
        platform: Target neuromorphic platform
        config: Neuromorphic configuration
        enable_real_time_adaptation: Whether to enable real-time parameter adaptation
    
    Returns:
        (model, parameter_manager): Model and parameter adaptation manager
    """
    if config is None:
        config = NeuromorphicConfig(platform=platform)
    
    # Create parameter manager
    param_manager = None
    if enable_real_time_adaptation:
        param_manager = RealTimeParameterManager(config)
    
    # Import and use the original model class (assuming it exists)
    from . import NeuromorphicAdaptiveModel  # This would import from the existing code
    
    model = NeuromorphicAdaptiveModel(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config
    )
    
    logger.info(f"Created enhanced neuromorphic model for platform: {platform.value}")
    
    return model, param_manager


if __name__ == "__main__":
    # Example usage with enhanced features
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced configuration
    config = NeuromorphicConfig(
        platform=NeuromorphicPlatform.LOIHI2,
        neuron_type=NeuronType.ADAPTIVE_LIF,
        enable_adaptive_threshold=True,
        enable_homeostatic_scaling=True,
        enable_real_time_adaptation=True
    )
    
    # Add custom plasticity rule
    config.plasticity_rules.append(
        PlasticityRule(
            rule_type=PlasticityType.STDP,
            learning_rate=0.01,
            adaptive_learning_rate=True,
            tau_plus=0.02,
            tau_minus=0.02
        )
    )
    
    # Create model with parameter manager
    model, param_manager = create_enhanced_neuromorphic_model(
        input_dim=784,
        output_dim=10,
        config=config
    )
    
    # Start real-time adaptation
    if param_manager:
        param_manager.start_adaptation()
        
        # Add callback for parameter updates
        def on_parameter_update(updates):
            print(f"Parameters updated: {list(updates.keys())}")
        
        param_manager.add_update_callback(on_parameter_update)
    
    # Test model
    batch_size = 4
    x = torch.randn(batch_size, 784)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Platform: {config.platform.value}")
    print(f"Neuron type: {config.neuron_type.value}")
    
    # Simulate environmental adaptation
    environmental_data = {
        'temperature': 30.0,  # Celsius
        'noise_level': 0.1,   # Normalized
        'available_power': 0.8  # Normalized
    }
    
    adaptations = config.adapt_to_environment(environmental_data)
    print(f"Environmental adaptations: {adaptations}")
    
    # Export configuration
    config_export = config.export_config("neuromorphic_config.json")
    print(f"Configuration exported with {len(config_export)} sections")
    
    # Stop adaptation when done
    if param_manager:
        param_manager.stop_adaptation_process()
    
    print("Enhanced neuromorphic model test completed successfully!")
