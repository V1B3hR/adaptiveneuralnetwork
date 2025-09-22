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
    enable_alpha_oscillations: bool = False  # NEW: 8-12 Hz
    enable_beta_oscillations: bool = False   # NEW: 13-30 Hz
    oscillation_frequency: float = 40.0  # Hz for gamma
    phase_coupling_strength: float = 0.1
    
    # Brain wave frequency parameters
    delta_frequency: float = 2.0   # 0.5-4 Hz
    theta_frequency: float = 6.0   # 4-8 Hz
    alpha_frequency: float = 10.0  # 8-12 Hz
    beta_frequency: float = 20.0   # 13-30 Hz
    gamma_frequency: float = 40.0  # 30-100 Hz
    
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
    
    def _configure_phase_encoding(self):
        """Configure intelligent phase encoding based on platform and application."""
        if self.platform in [NeuromorphicPlatform.PHOTONIC_SNN, NeuromorphicPlatform.LOIHI2]:
            self.enable_phase_encoding = True
            self.enable_oscillatory_dynamics = True
            
            # Enable multiple oscillation bands for advanced platforms
            self.enable_gamma_oscillations = True
            self.enable_alpha_oscillations = True
            self.enable_beta_oscillations = True
            
            if self.generation >= 4:  # 4th generation platforms
                self.enable_theta_rhythms = True
                self.enable_delta_waves = True
        
        # Configure based on generation
        if self.generation >= 3:
            self.enable_alpha_oscillations = True
            self.enable_beta_oscillations = True
            self.enable_gamma_oscillations = True
            
            if self.generation >= 4:
                self.enable_theta_rhythms = True
                self.enable_delta_waves = True
        
        logger.debug(f"Configured phase encoding for {self.platform.value}")
    
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
        else:
            self.noise_amplitude = 0.0
        
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
            # Expand compartment states if needed
            if self.v_compartments.shape[0] != batch_size:
                self.v_compartments = self.v_compartments.unsqueeze(0).expand(batch_size, -1)
            
            # Update each compartment
            for i in range(self.config.num_compartments):
                compartment_input = self.i_syn * self.compartment_weights[i]
                self.v_compartments[:, i] = (self.alpha_mem * self.v_compartments[:, i] + 
                                           compartment_input.squeeze())
            
            # Weighted sum for membrane potential
            self.v_mem = torch.sum(self.v_compartments * self.compartment_weights.unsqueeze(0), 
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
            # Ensure spike_count has the same shape as spikes
            if self.spike_count.shape != spikes.shape:
                self.spike_count = self.spike_count.expand_as(spikes)
            self.spike_count = self.spike_count + spikes
        
        # Update calcium dynamics
        if self.config.enable_calcium_dynamics and hasattr(self, 'calcium'):
            alpha_ca = torch.tensor(np.exp(-self.config.dt / self.tau_calcium))
            # Ensure calcium has the same shape as spikes
            if self.calcium.shape != spikes.shape:
                self.calcium = self.calcium.expand_as(spikes)
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
        
        if self.config.enable_calcium_dynamics and hasattr(self, 'calcium'):
            states['calcium'] = self.calcium
        
        return spikes, states
    
    def _expand_states(self, batch_size: int):
        """Expand state tensors for batch processing."""
        self.v_mem = self.v_mem.expand(batch_size, -1).contiguous()
        self.i_syn = self.i_syn.expand(batch_size, -1).contiguous()
        
        if self.config.enable_adaptive_threshold:
            self.v_th_adapt = self.v_th_adapt.expand(batch_size, -1).contiguous()
            self.spike_count = self.spike_count.expand(batch_size, -1).contiguous()
        
        if self.config.enable_calcium_dynamics and hasattr(self, 'calcium'):
            self.calcium = self.calcium.expand(batch_size, -1).contiguous()
        
        if self.config.enable_multi_compartment and hasattr(self, 'v_compartments'):
            self.v_compartments = self.v_compartments.expand(batch_size, -1).contiguous()


class BrainWaveOscillator:
    """
    Advanced brain wave oscillation generator supporting multiple frequency bands.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.dt = config.dt
        
        # Oscillation parameters for different frequency bands
        self.oscillators = {
            'delta': {'freq_range': (0.5, 4.0), 'phase': 0.0, 'amplitude': 1.0},
            'theta': {'freq_range': (4.0, 8.0), 'phase': 0.0, 'amplitude': 1.0},
            'alpha': {'freq_range': (8.0, 12.0), 'phase': 0.0, 'amplitude': 1.0},
            'beta': {'freq_range': (13.0, 30.0), 'phase': 0.0, 'amplitude': 1.0},
            'gamma': {'freq_range': (30.0, 100.0), 'phase': 0.0, 'amplitude': 1.0}
        }
        
        # Current time and phase tracking
        self.current_time = 0.0
        self.phase_coupling_matrix = {}
        
        # Neural rhythm synchronization parameters
        self.sync_strength = config.phase_coupling_strength
        self.enable_phase_locking = config.enable_phase_encoding
        
        # Circadian rhythm integration
        self.circadian_period = 24 * 60 * 60  # 24 hours in seconds
        self.circadian_phase = 0.0
        self.circadian_amplitude = 0.3  # Modulation strength
        self.sleep_wake_cycle_active = True
        
        # Circadian modulation of different frequency bands
        self.circadian_modulation = {
            'delta': 2.0,    # Enhanced during sleep
            'theta': 0.8,    # Reduced during sleep
            'alpha': 1.2,    # Peak during relaxed wakefulness
            'beta': 0.6,     # Reduced during sleep
            'gamma': 0.4     # Minimal during sleep
        }
        
    def generate_oscillation(self, band: str, frequency: Optional[float] = None) -> float:
        """Generate oscillation for specific frequency band."""
        if band not in self.oscillators:
            raise ValueError(f"Unknown oscillation band: {band}")
        
        osc = self.oscillators[band]
        
        # Use provided frequency or middle of range
        if frequency is None:
            freq = (osc['freq_range'][0] + osc['freq_range'][1]) / 2
        else:
            freq = np.clip(frequency, osc['freq_range'][0], osc['freq_range'][1])
        
        # Generate oscillation
        oscillation = osc['amplitude'] * np.sin(2 * np.pi * freq * self.current_time + osc['phase'])
        
        return oscillation
    
    def update_phase_coupling(self, band1: str, band2: str, coupling_strength: float):
        """Update phase coupling between frequency bands."""
        self.phase_coupling_matrix[(band1, band2)] = coupling_strength
        self.phase_coupling_matrix[(band2, band1)] = coupling_strength
    
    def generate_synchronized_oscillations(self) -> Dict[str, float]:
        """Generate synchronized oscillations across all bands."""
        oscillations = {}
        
        for band in self.oscillators.keys():
            oscillations[band] = self.generate_oscillation(band)
        
        # Apply phase coupling
        if self.enable_phase_locking:
            for (band1, band2), coupling in self.phase_coupling_matrix.items():
                phase_diff = self.oscillators[band1]['phase'] - self.oscillators[band2]['phase']
                coupling_force = coupling * np.sin(phase_diff)
                
                # Adjust phases based on coupling
                self.oscillators[band1]['phase'] -= coupling_force * self.dt * 0.1
                self.oscillators[band2]['phase'] += coupling_force * self.dt * 0.1
        
        self.current_time += self.dt
        return oscillations
    
    def update_circadian_phase(self, real_time_hours: Optional[float] = None):
        """Update circadian phase based on real time or simulation time"""
        if real_time_hours is not None:
            # Use real time of day (0-24 hours)
            self.circadian_phase = (real_time_hours / 24.0) * 2 * np.pi
        else:
            # Use simulation time
            self.circadian_phase = (self.current_time % self.circadian_period) / self.circadian_period * 2 * np.pi
    
    def get_circadian_modulation(self, band: str) -> float:
        """Get circadian modulation factor for specific frequency band"""
        if not self.sleep_wake_cycle_active:
            return 1.0
            
        # Calculate base circadian influence (0.5 to 1.5)
        base_circadian = 1.0 + self.circadian_amplitude * np.cos(self.circadian_phase)
        
        # Apply band-specific modulation
        band_modulation = self.circadian_modulation.get(band, 1.0)
        
        # During sleep phase (circadian_phase around π), enhance delta and reduce others
        sleep_factor = 0.5 * (1 + np.cos(self.circadian_phase))  # 1 during day, 0 during night
        
        if band == 'delta':
            # Delta waves enhanced during sleep
            return base_circadian * (band_modulation * (1 - sleep_factor) + sleep_factor)
        else:
            # Other bands reduced during sleep
            return base_circadian * (band_modulation * sleep_factor + (1 - sleep_factor) * 0.3)
    
    def get_sleep_wake_state(self) -> str:
        """Determine current sleep/wake state based on circadian phase"""
        # Sleep phase roughly from 22:00 to 06:00 (5.76 to 1.57 in phase, wrapping around)
        wake_phase_start = 0.25 * 2 * np.pi  # 06:00
        sleep_phase_start = 0.917 * 2 * np.pi  # 22:00 (adjusted)
        
        if wake_phase_start <= self.circadian_phase <= sleep_phase_start:
            return "wake"
        else:
            return "sleep"
    
    def apply_circadian_oscillation_modulation(self) -> Dict[str, float]:
        """Apply circadian modulation to all oscillation bands"""
        modulated_oscillations = {}
        current_state = self.get_sleep_wake_state()
        
        for band in self.oscillators.keys():
            base_oscillation = self.generate_oscillation(band)
            circadian_mod = self.get_circadian_modulation(band)
            modulated_oscillations[band] = base_oscillation * circadian_mod
            
        return modulated_oscillations



class NeuromodulationSystem:
    """
    Advanced neuromodulation system supporting multiple neurotransmitters.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        
        # Neurotransmitter concentrations and dynamics
        self.neurotransmitters = {
            'dopamine': {'concentration': 0.0, 'decay_rate': 0.1, 'baseline': 0.1},
            'acetylcholine': {'concentration': 0.0, 'decay_rate': 0.05, 'baseline': 0.05},
            'serotonin': {'concentration': 0.0, 'decay_rate': 0.08, 'baseline': 0.08},
            'oxytocin': {'concentration': 0.0, 'decay_rate': 0.12, 'baseline': 0.02},
            'norepinephrine': {'concentration': 0.0, 'decay_rate': 0.15, 'baseline': 0.03},
            'gaba': {'concentration': 0.0, 'decay_rate': 0.2, 'baseline': 0.1}
        }
        
        # Receptor sensitivity and dynamics
        self.receptor_sensitivity = defaultdict(lambda: 1.0)
        self.modulation_effects = {}
        
        # Stress response system
        self.stress_level = 0.0  # Current stress level (0.0 to 1.0)
        self.stress_threshold = 0.6  # Threshold for stress response activation
        self.stress_adaptation_rate = 0.1  # How quickly to adapt to stress
        self.baseline_stress = 0.1  # Baseline stress level
        
        # Stress-responsive neurotransmitter mapping
        self.stress_responses = {
            'cortisol': {'concentration': 0.0, 'decay_rate': 0.05, 'baseline': 0.05},
            'adrenaline': {'concentration': 0.0, 'decay_rate': 0.3, 'baseline': 0.01}
        }
        
        # Add stress hormones to neurotransmitter system
        self.neurotransmitters.update(self.stress_responses)
        
    def release_neurotransmitter(self, nt_type: str, amount: float):
        """Release neurotransmitter with specified amount."""
        if nt_type in self.neurotransmitters:
            # Ensure we don't go below zero
            if amount < 0:
                current = self.neurotransmitters[nt_type]['concentration']
                amount = max(amount, -current)  # Don't allow negative concentration
            self.neurotransmitters[nt_type]['concentration'] += amount
            logger.debug(f"Released {amount:.3f} {nt_type}")
    
    def update_concentrations(self):
        """Update neurotransmitter concentrations with decay."""
        for nt_type, params in self.neurotransmitters.items():
            # Exponential decay towards baseline
            decay_factor = np.exp(-params['decay_rate'] * self.config.dt)
            current = params['concentration']
            baseline = params['baseline']
            
            params['concentration'] = baseline + (current - baseline) * decay_factor
    
    def get_modulation_factor(self, nt_type: str, effect_type: str = 'excitatory') -> float:
        """Get modulation factor for specific neurotransmitter and effect."""
        if nt_type not in self.neurotransmitters:
            return 1.0
        
        concentration = self.neurotransmitters[nt_type]['concentration']
        sensitivity = self.receptor_sensitivity[nt_type]
        
        # Different neurotransmitters have different effect profiles
        if nt_type == 'dopamine':
            if effect_type == 'plasticity':
                return 1.0 + concentration * sensitivity * 2.0  # Enhances plasticity
            else:
                return 1.0 + concentration * sensitivity
        elif nt_type == 'acetylcholine':
            if effect_type == 'attention':
                return 1.0 + concentration * sensitivity * 1.5  # Enhances attention
            else:
                return 1.0 + concentration * sensitivity * 0.8
        elif nt_type == 'serotonin':
            if effect_type == 'mood':
                return 1.0 + concentration * sensitivity * 1.2  # Affects mood regulation
            else:
                return 1.0 + concentration * sensitivity * 0.6
        elif nt_type == 'gaba':
            return 1.0 - concentration * sensitivity * 0.8  # Inhibitory
        else:
            return 1.0 + concentration * sensitivity
    
    def update_stress_level(self, stressor_intensity: float, stressor_type: str = "general"):
        """Update stress level based on environmental stressors"""
        # Different stressor types have different impacts
        stressor_multipliers = {
            "energy_attack": 2.0,
            "trust_violation": 1.5,
            "communication_failure": 1.2,
            "general": 1.0
        }
        
        multiplier = stressor_multipliers.get(stressor_type, 1.0)
        stress_increase = stressor_intensity * multiplier * self.stress_adaptation_rate
        
        # Update stress level with saturation
        self.stress_level = min(1.0, max(0.0, self.stress_level + stress_increase))
        
        # Trigger stress response if threshold exceeded
        if self.stress_level > self.stress_threshold:
            self._activate_stress_response()
    
    def _activate_stress_response(self):
        """Activate neuromodulatory stress response"""
        stress_intensity = self.stress_level
        
        # Release stress-related neurotransmitters
        self.release_neurotransmitter('cortisol', stress_intensity * 0.3)
        self.release_neurotransmitter('adrenaline', stress_intensity * 0.5)
        
        # Reduce calming neurotransmitters
        self.neurotransmitters['serotonin']['concentration'] *= (1.0 - stress_intensity * 0.2)
        self.release_neurotransmitter('gaba', -stress_intensity * 0.1)  # Reduce inhibition
        
        # Increase attention-related neurotransmitters
        self.release_neurotransmitter('norepinephrine', stress_intensity * 0.4)
        self.release_neurotransmitter('acetylcholine', stress_intensity * 0.3)
        
        logger.debug(f"Stress response activated: level={stress_intensity:.3f}")
    
    def apply_stress_modulation(self, base_activity: float, neuron_type: str = "excitatory") -> float:
        """Apply stress-based modulation to neural activity"""
        stress_factor = 1.0
        
        # Get stress-related neurotransmitter concentrations
        cortisol = self.neurotransmitters['cortisol']['concentration']
        adrenaline = self.neurotransmitters['adrenaline']['concentration']
        
        if neuron_type == "excitatory":
            # Stress increases excitatory activity
            stress_factor = 1.0 + (cortisol * 0.5 + adrenaline * 0.8)
        elif neuron_type == "inhibitory":
            # Stress reduces inhibitory activity initially
            stress_factor = 1.0 - (cortisol * 0.2 + adrenaline * 0.3)
        
        return base_activity * stress_factor
    
    def get_stress_recovery_rate(self) -> float:
        """Calculate stress recovery rate based on current neurotransmitter balance"""
        # Recovery enhanced by serotonin, GABA, and oxytocin
        serotonin = self.neurotransmitters['serotonin']['concentration']
        gaba = self.neurotransmitters['gaba']['concentration']
        oxytocin = self.neurotransmitters['oxytocin']['concentration']
        
        recovery_factors = serotonin + gaba * 0.8 + oxytocin * 1.2
        base_recovery = 0.05  # Base recovery rate
        
        return base_recovery * (1.0 + recovery_factors)
    
    def update_stress_recovery(self):
        """Update stress level with natural recovery"""
        recovery_rate = self.get_stress_recovery_rate()
        
        # Exponential decay towards baseline
        self.stress_level = self.baseline_stress + (self.stress_level - self.baseline_stress) * (1.0 - recovery_rate)
        
        # Ensure stress level stays within bounds
        self.stress_level = max(0.0, min(1.0, self.stress_level))


class EnvironmentalAdaptationEngine:
    """
    Advanced environmental adaptation engine with multi-parameter optimization.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.adaptation_config = config.real_time_adaptation
        
        # Environmental parameter tracking
        self.environmental_history = defaultdict(deque)
        self.adaptation_strategies = {}
        
        # Temperature adaptation parameters
        self.temperature_q10 = 2.0  # Q10 factor for temperature effects
        self.reference_temp = 25.0  # Reference temperature in Celsius
        
        # Humidity adaptation parameters
        self.humidity_effects = {
            'membrane_resistance': lambda h: 1.0 + 0.1 * (h - 0.5),
            'ionic_conductance': lambda h: 1.0 - 0.05 * (h - 0.5)
        }
        
        # Noise adaptation parameters
        self.noise_threshold_scaling = 1.2
        self.noise_adaptation_rate = 0.1
        
        # Power scaling parameters
        self.power_efficiency_curves = {
            'low_power': {'threshold_scale': 1.5, 'rate_scale': 0.7},
            'normal_power': {'threshold_scale': 1.0, 'rate_scale': 1.0},
            'high_power': {'threshold_scale': 0.8, 'rate_scale': 1.3}
        }
    
    def adapt_to_temperature(self, temperature: float) -> Dict[str, float]:
        """Adapt neural parameters based on temperature."""
        adaptations = {}
        
        # Q10 temperature scaling
        temp_factor = self.temperature_q10 ** ((temperature - self.reference_temp) / 10.0)
        
        # Adapt time constants (faster at higher temperatures)
        adaptations['tau_mem'] = self.config.tau_mem / temp_factor
        adaptations['tau_syn'] = self.config.tau_syn / temp_factor
        adaptations['refractory_period'] = self.config.refractory_period / temp_factor
        
        # Adapt spike threshold (slightly higher at high temperatures)
        thermal_noise_factor = 1.0 + 0.01 * (temperature - self.reference_temp)
        adaptations['v_threshold'] = self.config.v_threshold * thermal_noise_factor
        
        logger.debug(f"Temperature adaptation at {temperature}°C: {list(adaptations.keys())}")
        return adaptations
    
    def adapt_to_humidity(self, humidity: float) -> Dict[str, float]:
        """Adapt neural parameters based on humidity (0-1 normalized)."""
        adaptations = {}
        
        # Membrane resistance changes with humidity
        resistance_factor = self.humidity_effects['membrane_resistance'](humidity)
        adaptations['tau_mem'] = self.config.tau_mem * resistance_factor
        
        # Ionic conductance effects
        conductance_factor = self.humidity_effects['ionic_conductance'](humidity)
        adaptations['v_threshold'] = self.config.v_threshold / conductance_factor
        
        logger.debug(f"Humidity adaptation at {humidity:.2f}: {list(adaptations.keys())}")
        return adaptations
    
    def adapt_to_noise(self, noise_level: float) -> Dict[str, float]:
        """Adapt neural parameters based on environmental noise."""
        adaptations = {}
        
        # Increase threshold to maintain signal-to-noise ratio
        threshold_adjustment = 1.0 + noise_level * self.noise_threshold_scaling
        adaptations['v_threshold'] = self.config.v_threshold * threshold_adjustment
        
        # Adjust noise amplitude in neurons
        adaptations['noise_amplitude'] = self.config.noise_amplitude + noise_level * 0.1
        
        # Adapt integration time for better noise filtering
        if noise_level > 0.3:  # High noise environment
            adaptations['tau_mem'] = self.config.tau_mem * 1.2  # Longer integration
        
        logger.debug(f"Noise adaptation at level {noise_level:.2f}: {list(adaptations.keys())}")
        return adaptations
    
    def adapt_to_power_constraints(self, available_power: float) -> Dict[str, float]:
        """Adapt neural parameters based on available power (0-1 normalized)."""
        adaptations = {}
        
        # Determine power mode
        if available_power < 0.3:
            power_mode = 'low_power'
        elif available_power > 0.8:
            power_mode = 'high_power'
        else:
            power_mode = 'normal_power'
        
        power_params = self.power_efficiency_curves[power_mode]
        
        # Scale threshold and spike rates
        adaptations['v_threshold'] = self.config.v_threshold * power_params['threshold_scale']
        adaptations['max_spike_rate'] = self.config.max_spike_rate * power_params['rate_scale']
        
        # Adjust bit precision for power efficiency
        if power_mode == 'low_power':
            adaptations['bit_precision'] = max(4, self.config.bit_precision - 2)
        elif power_mode == 'high_power':
            adaptations['bit_precision'] = min(32, self.config.bit_precision + 2)
        
        logger.debug(f"Power adaptation in {power_mode} mode: {list(adaptations.keys())}")
        return adaptations
    
    def comprehensive_environmental_adaptation(self, env_data: Dict[str, float]) -> Dict[str, float]:
        """Perform comprehensive adaptation based on multiple environmental factors."""
        all_adaptations = {}
        
        # Temperature adaptation
        if 'temperature' in env_data:
            temp_adaptations = self.adapt_to_temperature(env_data['temperature'])
            all_adaptations.update(temp_adaptations)
        
        # Humidity adaptation
        if 'humidity' in env_data:
            humid_adaptations = self.adapt_to_humidity(env_data['humidity'])
            all_adaptations.update(humid_adaptations)
        
        # Noise adaptation
        if 'noise_level' in env_data:
            noise_adaptations = self.adapt_to_noise(env_data['noise_level'])
            all_adaptations.update(noise_adaptations)
        
        # Power adaptation
        if 'available_power' in env_data:
            power_adaptations = self.adapt_to_power_constraints(env_data['available_power'])
            all_adaptations.update(power_adaptations)
        
        # Store environmental history
        for param, value in env_data.items():
            self.environmental_history[param].append(value)
            if len(self.environmental_history[param]) > 1000:  # Keep recent history
                self.environmental_history[param].popleft()
        
        return all_adaptations


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
        
        # Advanced plasticity features
        self.meta_learning_rates = defaultdict(lambda: 0.001)
        self.structural_plasticity_enabled = config.enable_structural_plasticity
        self.connectivity_changes = defaultdict(list)
        
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
        """Apply enhanced metaplasticity - plasticity of plasticity with learning rule adaptation."""
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
                    
                    # Adaptive learning rate based on meta-plasticity
                    meta_lr = self.meta_learning_rates[(pre_id, post_id)]
                    activity_variance = abs(activity_product - self.plasticity_thresholds[(pre_id, post_id)])
                    
                    # Adjust meta-learning rate based on stability
                    if activity_variance < 0.1:  # Stable, reduce meta-learning
                        meta_lr *= 0.99
                    else:  # Unstable, increase meta-learning
                        meta_lr *= 1.01
                    
                    self.meta_learning_rates[(pre_id, post_id)] = np.clip(meta_lr, 1e-6, 0.1)
                    
                    # Modulate future plasticity based on threshold
                    if rule.sliding_threshold:
                        modulation = 1.0 / (1.0 + self.plasticity_thresholds[(pre_id, post_id)])
                        rule.modulation_factor = modulation
                    
                    # Learning rule adaptation based on performance
                    if activity_product > 2.0 * rule.target_rate:  # High activity
                        # Switch to more selective learning
                        rule.a_plus *= 0.95
                        rule.a_minus *= 1.05
                    elif activity_product < 0.5 * rule.target_rate:  # Low activity
                        # Switch to more permissive learning
                        rule.a_plus *= 1.05
                        rule.a_minus *= 0.95
        
        return weights
    
    def apply_structural_plasticity(self, weights: torch.Tensor, 
                                  connection_strengths: torch.Tensor) -> torch.Tensor:
        """Apply structural plasticity - dynamic connectivity adaptation."""
        if not self.structural_plasticity_enabled:
            return weights
        
        # Prune weak connections
        pruning_mask = torch.abs(weights) > self.config.real_time_adaptation.pruning_threshold
        weights = weights * pruning_mask.float()
        
        # Grow new connections based on activity patterns
        # This is a simplified version - in practice, would use more sophisticated rules
        growth_probability = torch.sigmoid(connection_strengths - self.config.real_time_adaptation.growth_threshold)
        new_connections = torch.bernoulli(growth_probability * 0.01)  # Low probability of new connections
        
        # Initialize new connections with small random weights
        new_weights = torch.randn_like(weights) * 0.01 * new_connections
        weights = weights + new_weights
        
        # Track connectivity changes
        pruned_connections = (~pruning_mask).sum().item()
        grown_connections = new_connections.sum().item()
        
        self.connectivity_changes['pruned'].append(pruned_connections)
        self.connectivity_changes['grown'].append(grown_connections)
        
        return weights
    
    def apply_neuromodulated_plasticity(self, weights: torch.Tensor, 
                                      neuromodulation: NeuromodulationSystem,
                                      rule: PlasticityRule) -> torch.Tensor:
        """Apply plasticity with neuromodulator influence."""
        # Get modulation factors for different neurotransmitters
        dopamine_factor = neuromodulation.get_modulation_factor('dopamine', 'plasticity')
        acetylcholine_factor = neuromodulation.get_modulation_factor('acetylcholine', 'attention')
        
        # Modulate learning rates
        modulated_lr = rule.learning_rate * dopamine_factor * acetylcholine_factor
        
        # Apply to weight updates (this would be integrated with STDP in practice)
        rule.learning_rate = np.clip(modulated_lr, rule.min_learning_rate, rule.max_learning_rate)
        
        return weights


class RealTimeMonitoringSystem:
    """
    Advanced real-time monitoring and performance assessment system.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.monitoring_config = config.real_time_adaptation
        
        # Performance metrics tracking
        self.metrics_history = {
            'accuracy': deque(maxlen=1000),
            'convergence_rate': deque(maxlen=1000),
            'energy_efficiency': deque(maxlen=1000),
            'latency': deque(maxlen=1000),
            'spike_rate': deque(maxlen=1000),
            'weight_stability': deque(maxlen=1000),
            'connectivity_density': deque(maxlen=1000)
        }
        
        # Quality assessment metrics
        self.quality_metrics = {
            'signal_to_noise_ratio': 0.0,
            'network_health_score': 1.0,
            'plasticity_efficiency': 1.0,
            'temporal_coherence': 1.0
        }
        
        # Energy monitoring
        self.energy_consumption = {
            'spike_energy': 0.0,
            'synaptic_energy': 0.0,
            'plasticity_energy': 0.0,
            'total_energy': 0.0
        }
        
        # Latency tracking
        self.latency_measurements = {
            'processing_latency': deque(maxlen=100),
            'adaptation_latency': deque(maxlen=100),
            'response_latency': deque(maxlen=100)
        }
        
        # Adaptive thresholds
        self.performance_thresholds = {
            'accuracy_threshold': 0.8,
            'energy_threshold': 1e-9,  # Joules
            'latency_threshold': 0.1,  # seconds
            'convergence_threshold': 0.95
        }
    
    def update_performance_metrics(self, accuracy: float, convergence_rate: float, 
                                 energy_used: float, latency: float):
        """Update core performance metrics."""
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['convergence_rate'].append(convergence_rate)
        self.metrics_history['energy_efficiency'].append(1.0 / (energy_used + 1e-12))
        self.metrics_history['latency'].append(latency)
        
        # Update energy consumption
        self.energy_consumption['total_energy'] += energy_used
    
    def calculate_network_health_score(self, spike_rates: List[float], 
                                     weight_matrix: torch.Tensor) -> float:
        """Calculate comprehensive network health score."""
        # Spike rate diversity (avoid silent or over-active neurons)
        spike_rate_cv = np.std(spike_rates) / (np.mean(spike_rates) + 1e-6)
        rate_health = 1.0 / (1.0 + spike_rate_cv)  # Lower CV is better
        
        # Weight distribution health
        weight_std = torch.std(weight_matrix).item()
        weight_mean = torch.mean(torch.abs(weight_matrix)).item()
        weight_cv = weight_std / (weight_mean + 1e-6)
        weight_health = 1.0 / (1.0 + weight_cv)
        
        # Connectivity health (avoid too sparse or too dense)
        non_zero_weights = (torch.abs(weight_matrix) > 1e-6).float().mean().item()
        connectivity_health = 1.0 - abs(non_zero_weights - 0.3)  # Optimal around 30%
        
        # Combine health metrics
        overall_health = (rate_health + weight_health + connectivity_health) / 3.0
        self.quality_metrics['network_health_score'] = overall_health
        
        return overall_health
    
    def calculate_signal_to_noise_ratio(self, signal: torch.Tensor, 
                                      noise_level: float) -> float:
        """Calculate signal-to-noise ratio."""
        signal_power = torch.mean(signal ** 2).item()
        noise_power = noise_level ** 2
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
        
        self.quality_metrics['signal_to_noise_ratio'] = snr
        return snr
    
    def assess_temporal_coherence(self, spike_times: Dict[int, List[float]]) -> float:
        """Assess temporal coherence of neural activity."""
        if not spike_times:
            return 0.0
        
        # Calculate inter-spike interval variability
        isi_cvs = []
        for neuron_id, times in spike_times.items():
            if len(times) > 2:
                isis = np.diff(times)
                if len(isis) > 1:
                    cv = np.std(isis) / (np.mean(isis) + 1e-6)
                    isi_cvs.append(cv)
        
        if isi_cvs:
            temporal_coherence = 1.0 / (1.0 + np.mean(isi_cvs))
        else:
            temporal_coherence = 0.5
        
        self.quality_metrics['temporal_coherence'] = temporal_coherence
        return temporal_coherence
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        # Current metrics
        for metric, history in self.metrics_history.items():
            if history:
                summary[f'{metric}_current'] = history[-1]
                summary[f'{metric}_mean'] = np.mean(history)
                summary[f'{metric}_std'] = np.std(history)
        
        # Quality metrics
        summary['quality_metrics'] = self.quality_metrics.copy()
        
        # Energy breakdown
        summary['energy_consumption'] = self.energy_consumption.copy()
        
        # Performance flags
        summary['performance_flags'] = self._check_performance_flags()
        
        return summary
    
    def _check_performance_flags(self) -> Dict[str, bool]:
        """Check if performance metrics meet thresholds."""
        flags = {}
        
        if self.metrics_history['accuracy']:
            flags['accuracy_ok'] = self.metrics_history['accuracy'][-1] >= self.performance_thresholds['accuracy_threshold']
        
        if self.metrics_history['latency']:
            flags['latency_ok'] = self.metrics_history['latency'][-1] <= self.performance_thresholds['latency_threshold']
        
        if self.metrics_history['convergence_rate']:
            flags['convergence_ok'] = self.metrics_history['convergence_rate'][-1] >= self.performance_thresholds['convergence_threshold']
        
        flags['energy_ok'] = self.energy_consumption['total_energy'] <= self.performance_thresholds['energy_threshold']
        flags['network_health_ok'] = self.quality_metrics['network_health_score'] >= 0.7
        
        return flags


class StochasticProcessingEngine:
    """
    Advanced stochastic processing engine with adaptive filtering.
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        
        # Stochastic parameters
        self.noise_models = {
            'gaussian': {'std': 0.1, 'mean': 0.0},
            'poisson': {'rate': 10.0},
            'uniform': {'low': -0.1, 'high': 0.1},
            'pink': {'exponent': -1.0, 'std': 0.1}
        }
        
        # Adaptive filtering parameters
        self.filter_coefficients = torch.ones(5) / 5  # Simple moving average
        self.adaptive_learning_rate = 0.01
        self.noise_estimation_window = 100
        
        # Probabilistic spike generation
        self.spike_probability_model = 'exponential'  # exponential, sigmoid, linear
        self.refractory_noise = True
        
        # Noise history for adaptation
        self.noise_history = deque(maxlen=self.noise_estimation_window)
        self.signal_history = deque(maxlen=self.noise_estimation_window)
    
    def generate_stochastic_noise(self, shape: Tuple[int, ...], 
                                noise_type: str = 'gaussian') -> torch.Tensor:
        """Generate stochastic noise with specified characteristics."""
        if noise_type == 'gaussian':
            params = self.noise_models['gaussian']
            noise = torch.normal(params['mean'], params['std'], size=shape)
        elif noise_type == 'poisson':
            params = self.noise_models['poisson']
            noise = torch.poisson(torch.full(shape, params['rate'])) - params['rate']
        elif noise_type == 'uniform':
            params = self.noise_models['uniform']
            noise = torch.rand(shape) * (params['high'] - params['low']) + params['low']
        elif noise_type == 'pink':
            # Simplified pink noise (would need proper implementation for real use)
            white_noise = torch.randn(shape)
            noise = self._apply_pink_filter(white_noise)
        else:
            noise = torch.zeros(shape)
        
        return noise
    
    def _apply_pink_filter(self, white_noise: torch.Tensor) -> torch.Tensor:
        """Apply pink noise filtering (simplified implementation)."""
        # This is a simplified version - proper pink noise requires frequency domain filtering
        # For multi-dimensional inputs, flatten and process
        original_shape = white_noise.shape
        flattened = white_noise.flatten()
        
        if len(flattened) < len(self.filter_coefficients):
            return white_noise  # Too small to filter
        
        # Simple convolution-based filtering
        padded = torch.nn.functional.pad(flattened.unsqueeze(0).unsqueeze(0), 
                                       (len(self.filter_coefficients)//2, len(self.filter_coefficients)//2))
        filtered = torch.nn.functional.conv1d(
            padded,
            self.filter_coefficients.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        
        # Reshape back to original
        return filtered[:len(flattened)].view(original_shape)
    
    def probabilistic_spike_generation(self, membrane_potential: torch.Tensor, 
                                     threshold: float, temperature: float = 1.0) -> torch.Tensor:
        """Generate spikes probabilistically based on membrane potential."""
        if self.spike_probability_model == 'exponential':
            # Exponential probability model
            prob = 1.0 - torch.exp(-(membrane_potential - threshold) / temperature)
        elif self.spike_probability_model == 'sigmoid':
            # Sigmoid probability model
            prob = torch.sigmoid((membrane_potential - threshold) / temperature)
        elif self.spike_probability_model == 'linear':
            # Linear probability model with clipping
            prob = torch.clamp((membrane_potential - threshold) / temperature, 0, 1)
        else:
            # Default: threshold-based
            prob = (membrane_potential >= threshold).float()
        
        # Add refractory noise if enabled
        if self.refractory_noise:
            noise = self.generate_stochastic_noise(prob.shape, 'gaussian') * 0.01
            prob = torch.clamp(prob + noise, 0, 1)
        
        # Generate spikes based on probability
        spikes = torch.bernoulli(prob)
        return spikes
    
    def adaptive_noise_filtering(self, signal: torch.Tensor, 
                               estimated_noise_level: float) -> torch.Tensor:
        """Apply adaptive noise filtering based on estimated noise level."""
        # Update noise estimation
        self.noise_history.append(estimated_noise_level)
        self.signal_history.append(torch.mean(torch.abs(signal)).item())
        
        if len(self.noise_history) >= 10:
            # Adapt filter based on signal-to-noise ratio
            avg_signal = np.mean(list(self.signal_history)[-10:])
            avg_noise = np.mean(list(self.noise_history)[-10:])
            snr = avg_signal / (avg_noise + 1e-12)
            
            # Adjust filter strength based on SNR
            if snr < 1.0:  # High noise
                filter_strength = 0.8
            elif snr > 10.0:  # Low noise
                filter_strength = 0.2
            else:
                filter_strength = 0.5
            
            # Apply adaptive filtering
            filtered_signal = signal * (1 - filter_strength) + \
                            torch.mean(signal) * filter_strength
            
            return filtered_signal
        
        return signal
    
    def estimate_noise_level(self, signal: torch.Tensor, 
                           method: str = 'mad') -> float:
        """Estimate noise level in the signal."""
        if method == 'mad':  # Median Absolute Deviation
            median = torch.median(signal)
            mad = torch.median(torch.abs(signal - median))
            noise_level = mad * 1.4826  # Scale factor for Gaussian noise
        elif method == 'std':  # Standard deviation (assumes zero-mean noise)
            noise_level = torch.std(signal - torch.mean(signal))
        else:
            noise_level = torch.tensor(0.1)  # Default estimate
        
        return noise_level.item()


class NeuromorphicAdaptiveModel(nn.Module):
    """
    Main neuromorphic adaptive model with advanced real-time adaptation capabilities.
    """
    
    def __init__(self, input_dim: int, output_dim: int, config: Optional[NeuromorphicConfig] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or NeuromorphicConfig()
        
        # Core neural components
        self.neurons = nn.ModuleList([
            EnhancedLeakyIntegrateFireNeuron(self.config, i) 
            for i in range(output_dim)
        ])
        
        # Advanced systems
        self.oscillator = BrainWaveOscillator(self.config)
        self.neuromodulation = NeuromodulationSystem(self.config)
        self.environmental_adapter = EnvironmentalAdaptationEngine(self.config)
        self.plasticity_manager = AdaptivePlasticityManager(self.config)
        self.monitoring_system = RealTimeMonitoringSystem(self.config)
        self.stochastic_engine = StochasticProcessingEngine(self.config)
        
        # Parameter manager
        self.parameter_manager = RealTimeParameterManager(self.config)
        
        # Synaptic weights
        self.input_weights = nn.Parameter(torch.randn(input_dim, output_dim, dtype=torch.float32) * 0.1)
        self.recurrent_weights = nn.Parameter(torch.randn(output_dim, output_dim, dtype=torch.float32) * 0.05)
        
        # Encoding and decoding layers
        self.encoder = nn.Linear(input_dim, input_dim)  # Optional input preprocessing
        self.decoder = nn.Linear(output_dim, output_dim)  # Optional output postprocessing
        
        # State tracking
        self.time_step = 0
        self.spike_history = defaultdict(list)
        self.performance_history = []
        
        logger.info(f"Initialized NeuromorphicAdaptiveModel with {len(self.neurons)} neurons")
    
    def forward(self, x: torch.Tensor, environmental_data: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Forward pass with comprehensive neuromorphic processing.
        
        Args:
            x: Input tensor
            environmental_data: Optional environmental parameters
            
        Returns:
            Output spikes and neural states
        """
        batch_size = x.shape[0]
        start_time = time.time()
        
        # Environmental adaptation
        if environmental_data is not None:
            adaptations = self.environmental_adapter.comprehensive_environmental_adaptation(environmental_data)
            if adaptations:
                self.config.update_parameters_realtime(adaptations, "environmental")
        
        # Brain wave oscillations with circadian modulation
        self.oscillator.update_circadian_phase()  # Update circadian phase
        oscillations = self.oscillator.apply_circadian_oscillation_modulation()
        
        # Update stress response and neuromodulation
        self.neuromodulation.update_stress_recovery()
        if environmental_data and 'stress_level' in environmental_data:
            self.neuromodulation.update_stress_level(
                environmental_data['stress_level'], 
                environmental_data.get('stressor_type', 'general')
            )
        
        # Input encoding with stochastic processing
        encoded_input = self.encoder(x)
        if self.config.enable_stochastic_dynamics:
            noise = self.stochastic_engine.generate_stochastic_noise(
                encoded_input.shape, 'gaussian'
            )
            encoded_input = encoded_input + noise
        
        # Neural processing
        outputs = []
        all_states = []
        
        for i, neuron in enumerate(self.neurons):
            # Calculate input current
            input_current = torch.matmul(encoded_input, self.input_weights[:, i:i+1])
            
            # Add recurrent input
            if len(outputs) > 0:
                # Stack previous outputs and compute recurrent input
                prev_outputs = torch.stack(outputs, dim=1)  # [batch, num_prev_neurons, 1]
                recurrent_weights_slice = self.recurrent_weights[:len(outputs), i:i+1].float()  # [num_prev_neurons, 1]
                recurrent_input = torch.matmul(
                    prev_outputs.squeeze(-1),  # [batch, num_prev_neurons]
                    recurrent_weights_slice.squeeze(-1)  # [num_prev_neurons]
                ).unsqueeze(-1)  # [batch, 1]
                input_current = input_current + recurrent_input
            
            # Add oscillatory modulation (now includes circadian rhythm)
            osc_modulation = sum(oscillations.values()) / len(oscillations)
            input_current = input_current * (1.0 + 0.1 * osc_modulation)
            
            # Enhanced neuromodulation with stress response
            dopamine_mod = self.neuromodulation.get_modulation_factor('dopamine')
            
            # Apply stress modulation
            stress_base_current = self.neuromodulation.apply_stress_modulation(
                input_current.item() if input_current.numel() == 1 else torch.mean(input_current).item(),
                neuron_type="excitatory"  # Assume excitatory for simplicity
            )
            stress_mod_tensor = torch.tensor(stress_base_current / (torch.mean(input_current).item() + 1e-8))
            
            neuromod_signal = torch.tensor(dopamine_mod - 1.0 + stress_mod_tensor - 1.0).unsqueeze(0).expand(batch_size, 1)
            
            # Process through neuron
            spike_output, neuron_states = neuron(input_current, neuromod_signal)
            
            # Stochastic spike generation if enabled
            if self.config.enable_stochastic_dynamics:
                spike_output = self.stochastic_engine.probabilistic_spike_generation(
                    neuron_states['v_mem'], 
                    self.config.v_threshold,
                    temperature=0.1
                )
            
            outputs.append(spike_output)
            all_states.append(neuron_states)
            
            # Track spikes for plasticity
            spike_times = [self.time_step * self.config.dt] * int(spike_output.sum().item())
            self.spike_history[i].extend(spike_times)
        
        # Stack outputs
        output_tensor = torch.stack(outputs, dim=1).squeeze(-1)
        
        # Apply plasticity
        if self.training:
            self._update_synaptic_weights()
        
        # Update neuromodulator concentrations
        self.neuromodulation.update_concentrations()
        
        # Performance monitoring
        processing_latency = time.time() - start_time
        self._update_performance_metrics(output_tensor, processing_latency)
        
        # Decode output
        decoded_output = self.decoder(output_tensor)
        
        self.time_step += 1
        
        return decoded_output
    
    def _update_synaptic_weights(self):
        """Update synaptic weights using plasticity rules."""
        pre_neurons = list(range(self.input_dim))
        post_neurons = list(range(self.output_dim))
        
        # Apply plasticity to input weights
        updated_input_weights = self.plasticity_manager.update_synaptic_weights(
            pre_neurons, post_neurons, self.spike_history, self.input_weights
        )
        self.input_weights.data = updated_input_weights
        
        # Apply plasticity to recurrent weights
        updated_recurrent_weights = self.plasticity_manager.update_synaptic_weights(
            post_neurons, post_neurons, self.spike_history, self.recurrent_weights
        )
        self.recurrent_weights.data = updated_recurrent_weights
        
        # Apply structural plasticity if enabled
        if self.config.enable_structural_plasticity:
            connection_strengths = torch.abs(self.input_weights) + torch.abs(self.recurrent_weights.mean(dim=0, keepdim=True).T)
            self.input_weights.data = self.plasticity_manager.apply_structural_plasticity(
                self.input_weights, connection_strengths
            )
    
    def _update_performance_metrics(self, output: torch.Tensor, latency: float):
        """Update performance monitoring metrics."""
        # Calculate basic metrics
        spike_rate = output.float().mean().item()
        energy_estimate = spike_rate * self.config.energy_per_spike * len(self.neurons)
        
        # Estimate accuracy (this would be based on actual task performance)
        accuracy_estimate = 0.8  # Placeholder - would be computed based on task
        convergence_rate = 0.9  # Placeholder - would track learning convergence
        
        # Update monitoring system
        self.monitoring_system.update_performance_metrics(
            accuracy_estimate, convergence_rate, energy_estimate, latency
        )
        
        # Calculate network health
        spike_rates = [len(spikes) for spikes in self.spike_history.values()]
        weight_matrix = torch.cat([self.input_weights.flatten(), self.recurrent_weights.flatten()])
        health_score = self.monitoring_system.calculate_network_health_score(
            spike_rates, weight_matrix.unsqueeze(0)
        )
    
    def adapt_to_environment(self, environmental_data: Dict[str, float]) -> Dict[str, Any]:
        """Adapt the model to environmental conditions."""
        return self.environmental_adapter.comprehensive_environmental_adaptation(environmental_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return self.monitoring_system.get_performance_summary()
    
    def start_real_time_adaptation(self):
        """Start real-time parameter adaptation."""
        self.parameter_manager.start_adaptation()
        
        # Add callback to update model when parameters change
        def on_parameter_update(updates):
            logger.info(f"Real-time parameter update: {list(updates.keys())}")
        
        self.parameter_manager.add_update_callback(on_parameter_update)
    
    def stop_real_time_adaptation(self):
        """Stop real-time parameter adaptation."""
        self.parameter_manager.stop_adaptation_process()
    
    def configure_brain_waves(self, wave_config: Dict[str, Dict[str, float]]):
        """Configure brain wave oscillations."""
        for band, params in wave_config.items():
            if band in self.oscillator.oscillators:
                self.oscillator.oscillators[band].update(params)
        
        logger.info(f"Configured brain waves: {list(wave_config.keys())}")
    
    def release_neurotransmitter(self, nt_type: str, amount: float):
        """Release neurotransmitter for neuromodulation."""
        self.neuromodulation.release_neurotransmitter(nt_type, amount)
    
    def reset_state(self):
        """Reset model state for new sequence."""
        for neuron in self.neurons:
            neuron.v_mem.zero_()
            neuron.i_syn.zero_()
            if hasattr(neuron, 'calcium'):
                neuron.calcium.zero_()
        
        self.spike_history.clear()
        self.time_step = 0
        
        logger.debug("Model state reset")


# Enhanced configuration additions
@dataclass
class BrainWaveConfig:
    """Configuration for brain wave oscillations."""
    enable_delta: bool = False  # 0.5-4 Hz
    enable_theta: bool = False  # 4-8 Hz  
    enable_alpha: bool = False  # 8-12 Hz
    enable_beta: bool = False   # 13-30 Hz
    enable_gamma: bool = False  # 30-100 Hz
    
    # Frequency parameters
    delta_freq: float = 2.0
    theta_freq: float = 6.0
    alpha_freq: float = 10.0
    beta_freq: float = 20.0
    gamma_freq: float = 40.0
    
    # Phase coupling
    enable_cross_frequency_coupling: bool = False
    theta_gamma_coupling: float = 0.1
    alpha_beta_coupling: float = 0.05
    
    # Amplitude modulation
    enable_amplitude_modulation: bool = False
    modulation_depth: float = 0.2


# Factory function for creating enhanced neuromorphic models
def create_enhanced_neuromorphic_model(
    input_dim: int,
    output_dim: int,
    platform: NeuromorphicPlatform = NeuromorphicPlatform.SIMULATION,
    config: Optional[NeuromorphicConfig] = None,
    enable_real_time_adaptation: bool = True
) -> Tuple[NeuromorphicAdaptiveModel, Optional[RealTimeParameterManager]]:
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
    
    # Enhance config with brain wave support
    _update_neuromorphic_config_with_brain_waves(config)
    
    # Create model
    model = NeuromorphicAdaptiveModel(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config
    )
    
    # Create parameter manager
    param_manager = None
    if enable_real_time_adaptation:
        param_manager = RealTimeParameterManager(config)
        
        # Connect parameter manager to model
        def on_parameter_update(updates):
            logger.info(f"Applied parameter updates: {list(updates.keys())}")
        
        param_manager.add_update_callback(on_parameter_update)
    
    logger.info(f"Created enhanced neuromorphic model for platform: {platform.value}")
    
    return model, param_manager


def _update_neuromorphic_config_with_brain_waves(config: NeuromorphicConfig):
    """Update neuromorphic config with enhanced brain wave support."""
    # Enable advanced oscillation features based on generation
    if config.generation >= 3:
        config.enable_alpha_oscillations = True
        config.enable_beta_oscillations = True
        config.enable_gamma_oscillations = True
        
        if config.generation >= 4:
            config.enable_theta_rhythms = True
            config.enable_delta_waves = True
        
        # Configure phase coupling
        config.phase_coupling_strength = 0.1
        config.enable_phase_encoding = True
        config.enable_oscillatory_dynamics = True


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
