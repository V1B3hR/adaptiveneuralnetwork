"""
Advanced 3rd generation neuron models with enhanced biological realism.

This module implements sophisticated neuron models including multi-compartment
neurons, adaptive thresholds, burst firing patterns, stochastic dynamics, and
organ-inspired data preprocessing (liver and lung functions).
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..neuromorphic import NeuromorphicConfig

logger = logging.getLogger(__name__)


# --- Organ-Inspired Preprocessing Modules ---

class LiverFilter(nn.Module):
    """
    Simulates a biological 'liver' by filtering and cleaning input data before neural processing.
    """
    def __init__(self, method='gaussian', kernel_size=5, std=1.0):
        super().__init__()
        self.method = method
        self.kernel_size = kernel_size
        self.std = std

    def forward(self, x):
        # Gaussian filtering
        if self.method == 'gaussian':
            kernel = torch.exp(-0.5 * (torch.arange(self.kernel_size) - self.kernel_size // 2)**2 / self.std**2)
            kernel = kernel / kernel.sum()
            kernel = kernel.to(x.device)
            x_padded = torch.nn.functional.pad(x, (self.kernel_size // 2, self.kernel_size // 2), mode='reflect')
            filtered = torch.nn.functional.conv1d(x_padded.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0))
            return filtered.squeeze(1)
        # Simple normalization
        elif self.method == 'normalize':
            return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        else:
            return x  # No filtering

class LungIntake(nn.Module):
    """
    Simulates a biological 'lung' by controlling and preprocessing incoming data.
    """
    def __init__(self, amplify_factor=1.0, sample_rate=1.0):
        super().__init__()
        self.amplify_factor = amplify_factor
        self.sample_rate = sample_rate

    def forward(self, x):
        # Amplify weak signals
        x_amplified = x * self.amplify_factor
        # Downsample if sample_rate < 1.0
        if self.sample_rate < 1.0 and x.size(0) > 1:
            idx = torch.arange(0, x.size(0), int(1/self.sample_rate)).long()
            x_sampled = x_amplified[idx]
            return x_sampled
        return x_amplified


@dataclass
class CompartmentConfig:
    """Configuration for neural compartments."""
    capacitance: float = 1e-12  # Membrane capacitance (F)
    leak_conductance: float = 1e-9  # Leak conductance (S)
    leak_reversal: float = -70e-3  # Leak reversal potential (V)
    threshold: float = -50e-3  # Spike threshold (V)
    reset: float = -70e-3  # Reset potential (V)
    refractory_period: float = 2e-3  # Refractory period (s)
    coupling_conductance: float = 5e-9  # Inter-compartment coupling (S)


@dataclass
class NeuronV3Config:
    """Extended configuration for 3rd generation neurons."""
    # Base configuration
    base_config: NeuromorphicConfig

    # Adaptive threshold parameters
    threshold_adaptation_rate: float = 0.1
    threshold_decay_rate: float = 0.01
    max_threshold_shift: float = 0.2

    # Homeostatic parameters
    target_spike_rate: float = 10.0  # Hz
    homeostatic_timescale: float = 100.0  # seconds
    scaling_factor: float = 0.001

    # Burst parameters
    burst_threshold_factor: float = 0.8
    afterhyperpolarization_strength: float = 2.0
    burst_adaptation_rate: float = 0.05

    # Stochastic parameters
    noise_amplitude: float = 0.01
    channel_noise: bool = True
    thermal_noise: bool = True

    # Organ settings (optional)
    liver_params: dict = None
    lung_params: dict = None


class MultiCompartmentNeuron(nn.Module):
    """
    Multi-compartment neuron model with dendritic processing.
    Combines with optional organ-inspired preprocessing modules.
    """

    def __init__(
        self,
        config: NeuronV3Config,
        num_dendrites: int = 4,
        compartment_configs: list[CompartmentConfig] | None = None
    ):
        super().__init__()

        self.config = config
        self.num_dendrites = num_dendrites
        self.num_compartments = num_dendrites + 1  # dendrites + soma

        # Organ modules
        self.lung = LungIntake(**(config.lung_params or {}))
        self.liver = LiverFilter(**(config.liver_params or {}))

        # Set up compartment configurations
        if compartment_configs is None:
            self.compartment_configs = [CompartmentConfig() for _ in range(self.num_compartments)]
        else:
            self.compartment_configs = compartment_configs

        # Initialize compartment states
        self.register_buffer('v_mem', torch.zeros(self.num_compartments))
        self.register_buffer('spike_times', torch.full((self.num_compartments,), -float('inf')))

        # Coupling matrix (dendrites connect to soma)
        coupling_matrix = torch.zeros(self.num_compartments, self.num_compartments)
        for i in range(self.num_dendrites):
            # Dendrite i connects to soma (last compartment)
            coupling_matrix[i, -1] = self.compartment_configs[i].coupling_conductance
            coupling_matrix[-1, i] = self.compartment_configs[i].coupling_conductance

        self.register_buffer('coupling_matrix', coupling_matrix)

        logger.debug(f"Created multi-compartment neuron with {self.num_compartments} compartments")

    def forward(
        self,
        dendritic_inputs: torch.Tensor,
        somatic_input: torch.Tensor,
        dt: float | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Simulate multi-compartment neuron dynamics.
        Preprocess inputs using organ modules before neural processing.
        """
        if dt is None:
            dt = self.config.base_config.dt

        batch_size = dendritic_inputs.size(0)
        device = dendritic_inputs.device

        # Organ preprocessing
        dendritic_inputs = self.lung(dendritic_inputs)
        dendritic_inputs = self.liver(dendritic_inputs)
        somatic_input = self.lung(somatic_input)
        somatic_input = self.liver(somatic_input)

        # Initialize states for batch if needed
        if self.v_mem.size(0) != batch_size:
            self.v_mem = torch.zeros(batch_size, self.num_compartments, device=device)
            self.spike_times = torch.full((batch_size, self.num_compartments), -float('inf'), device=device)

        # Combine inputs
        inputs = torch.cat([dendritic_inputs, somatic_input], dim=1)

        # Calculate coupling currents
        coupling_currents = torch.zeros_like(self.v_mem)
        for i in range(self.num_compartments):
            for j in range(self.num_compartments):
                if i != j and self.coupling_matrix[i, j] > 0:
                    coupling_currents[:, i] += (
                        self.coupling_matrix[i, j] * (self.v_mem[:, j] - self.v_mem[:, i])
                    )

        # Update compartment voltages
        spikes = torch.zeros_like(self.v_mem)

        for comp_idx in range(self.num_compartments):
            config = self.compartment_configs[comp_idx]

            # Check refractory period (simplified - assume not refractory for now)
            refractory_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

            # Membrane dynamics (only for non-refractory neurons)
            leak_current = config.leak_conductance * (config.leak_reversal - self.v_mem[:, comp_idx])
            total_current = inputs[:, comp_idx] + coupling_currents[:, comp_idx] + leak_current

            dv_dt = total_current / config.capacitance
            self.v_mem[:, comp_idx] = torch.where(
                refractory_mask,
                self.v_mem[:, comp_idx] + dv_dt * dt,
                torch.tensor(config.reset, device=device)
            )

            # Check for spikes
            spike_mask = (self.v_mem[:, comp_idx] > config.threshold) & refractory_mask
            spikes[:, comp_idx] = spike_mask.float()

            # Reset spiking neurons
            self.v_mem[:, comp_idx] = torch.where(
                spike_mask,
                torch.tensor(config.reset, device=device),
                self.v_mem[:, comp_idx]
            )

        # Return only somatic spikes as main output
        somatic_spikes = spikes[:, -1:]

        states = {
            'compartment_voltages': self.v_mem.clone(),
            'all_spikes': spikes,
            'coupling_currents': coupling_currents,
            'input_after_lung': self.lung(dendritic_inputs).clone(),
            'input_after_liver': self.liver(dendritic_inputs).clone(),
        }

        return somatic_spikes, states


class AdaptiveThresholdNeuron(nn.Module):
    """
    Leaky integrate-and-fire neuron with adaptive spike threshold.
    Now supports optional organ-inspired preprocessing modules.
    """

    def __init__(self, config: NeuronV3Config):
        super().__init__()

        self.config = config

        # Organ modules
        self.lung = LungIntake(**(config.lung_params or {}))
        self.liver = LiverFilter(**(config.liver_params or {}))

        # Neuron state variables
        self.register_buffer('v_mem', torch.tensor(config.base_config.v_rest))
        self.register_buffer('threshold', torch.tensor(config.base_config.v_threshold))
        self.register_buffer('spike_count', torch.tensor(0.0))
        self.register_buffer('time_window_start', torch.tensor(0.0))
        self.register_buffer('last_spike_time', torch.tensor(-float('inf')))

        logger.debug("Created adaptive threshold neuron")

    def forward(
        self,
        input_current: torch.Tensor,
        current_time: float | None = None,
        dt: float | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Simulate adaptive threshold neuron dynamics.
        Preprocess input using organ modules before neural processing.
        """
        if dt is None:
            dt = self.config.base_config.dt
        if current_time is None:
            current_time = 0.0

        batch_size = input_current.size(0)

        # Organ preprocessing
        input_current = self.lung(input_current)
        input_current = self.liver(input_current)

        # Expand state for batch processing
        if self.v_mem.dim() == 0:
            self.v_mem = self.v_mem.unsqueeze(0).expand(batch_size).clone()
            self.threshold = self.threshold.unsqueeze(0).expand(batch_size).clone()
            self.spike_count = self.spike_count.unsqueeze(0).expand(batch_size).clone()
            self.time_window_start = self.time_window_start.unsqueeze(0).expand(batch_size).clone()
            self.last_spike_time = self.last_spike_time.unsqueeze(0).expand(batch_size).clone()

        # Check refractory period
        time_since_spike = current_time - self.last_spike_time
        refractory_mask = time_since_spike > self.config.base_config.refractory_period

        # Membrane dynamics
        tau_mem = self.config.base_config.tau_mem
        leak_current = -(self.v_mem - self.config.base_config.v_rest) / tau_mem

        dv_dt = (input_current.squeeze() + leak_current) / tau_mem
        self.v_mem = torch.where(
            refractory_mask,
            self.v_mem + dv_dt * dt,
            self.config.base_config.v_reset
        )

        # Spike detection
        spike_mask = (self.v_mem > self.threshold) & refractory_mask
        spikes = spike_mask.float().unsqueeze(-1)

        # Reset membrane potential and update spike times
        self.v_mem = torch.where(
            spike_mask,
            self.config.base_config.v_reset,
            self.v_mem
        )

        self.last_spike_time = torch.where(
            spike_mask,
            current_time,
            self.last_spike_time
        )

        # Update spike count for homeostatic adaptation
        self.spike_count += spike_mask.float()

        # Homeostatic threshold adaptation
        window_duration = current_time - self.time_window_start
        adaptation_window = self.config.homeostatic_timescale

        # Check if we should update thresholds (element-wise comparison)
        should_adapt = window_duration > adaptation_window

        if should_adapt.any():
            # Calculate current firing rate
            current_rate = self.spike_count / (window_duration + 1e-6)
            target_rate = self.config.target_spike_rate

            # Adapt threshold to maintain target rate
            rate_error = current_rate - target_rate
            threshold_change = -self.config.threshold_adaptation_rate * rate_error * dt

            # Clamp threshold adaptation
            max_change = self.config.max_threshold_shift * self.config.base_config.v_threshold
            threshold_change = torch.clamp(threshold_change, -max_change, max_change)

            # Only update for neurons that should adapt
            self.threshold = torch.where(
                should_adapt,
                self.threshold + threshold_change,
                self.threshold
            )

            # Reset counting window for adapted neurons
            self.spike_count = torch.where(should_adapt, torch.zeros_like(self.spike_count), self.spike_count)
            self.time_window_start = torch.where(should_adapt, torch.full_like(self.time_window_start, current_time), self.time_window_start)

        # Threshold decay towards baseline
        baseline_threshold = self.config.base_config.v_threshold
        threshold_decay = self.config.threshold_decay_rate * (baseline_threshold - self.threshold) * dt
        self.threshold = self.threshold + threshold_decay

        states = {
            'membrane_potential': self.v_mem.clone(),
            'threshold': self.threshold.clone(),
            'spike_count': self.spike_count.clone(),
            'firing_rate': self.spike_count / (window_duration.clamp(min=1e-6)),
            'input_after_lung': self.lung(input_current).clone(),
            'input_after_liver': self.liver(input_current).clone(),
        }

        return spikes, states


class BurstingNeuron(nn.Module):
    """
    Neuron model capable of generating burst firing patterns.
    Implements burst generation through afterhyperpolarization
    and adaptive burst threshold mechanisms.
    Supports organ-inspired preprocessing.
    """

    def __init__(self, config: NeuronV3Config):
        super().__init__()

        self.config = config

        # Organ modules
        self.lung = LungIntake(**(config.lung_params or {}))
        self.liver = LiverFilter(**(config.liver_params or {}))

        # Neuron state variables
        self.register_buffer('v_mem', torch.tensor(config.base_config.v_rest))
        self.register_buffer('ahp_current', torch.tensor(0.0))  # Afterhyperpolarization
        self.register_buffer('burst_threshold', torch.tensor(config.base_config.v_threshold * config.burst_threshold_factor))
        self.register_buffer('in_burst', torch.tensor(False))
        self.register_buffer('burst_spike_count', torch.tensor(0.0))
        self.register_buffer('last_spike_time', torch.tensor(-float('inf')))

        logger.debug("Created bursting neuron")

    def forward(
        self,
        input_current: torch.Tensor,
        current_time: float | None = None,
        dt: float | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Simulate bursting neuron dynamics.
        Preprocess input using organ modules before neural processing.
        """
        if dt is None:
            dt = self.config.base_config.dt
        if current_time is None:
            current_time = 0.0

        batch_size = input_current.size(0)

        # Organ preprocessing
        input_current = self.lung(input_current)
        input_current = self.liver(input_current)

        # Expand states for batch processing
        if self.v_mem.dim() == 0:
            self.v_mem = self.v_mem.unsqueeze(0).expand(batch_size).clone()
            self.ahp_current = self.ahp_current.unsqueeze(0).expand(batch_size).clone()
            self.burst_threshold = self.burst_threshold.unsqueeze(0).expand(batch_size).clone()
            self.in_burst = self.in_burst.unsqueeze(0).expand(batch_size).clone()
            self.burst_spike_count = self.burst_spike_count.unsqueeze(0).expand(batch_size).clone()
            self.last_spike_time = self.last_spike_time.unsqueeze(0).expand(batch_size).clone()

        # Check refractory period
        time_since_spike = current_time - self.last_spike_time
        refractory_mask = time_since_spike > self.config.base_config.refractory_period

        # Membrane dynamics with AHP
        tau_mem = self.config.base_config.tau_mem
        leak_current = -(self.v_mem - self.config.base_config.v_rest) / tau_mem

        total_current = input_current.squeeze() + leak_current - self.ahp_current
        dv_dt = total_current / tau_mem

        self.v_mem = torch.where(
            refractory_mask,
            self.v_mem + dv_dt * dt,
            self.config.base_config.v_reset
        )

        # Determine effective threshold (burst vs normal)
        effective_threshold = torch.where(
            self.in_burst,
            self.burst_threshold,
            self.config.base_config.v_threshold
        )

        # Spike detection
        spike_mask = (self.v_mem > effective_threshold) & refractory_mask
        spikes = spike_mask.float().unsqueeze(-1)

        # Handle burst dynamics
        burst_initiation_mask = spike_mask & ~self.in_burst
        burst_continuation_mask = spike_mask & self.in_burst

        # Start burst
        self.in_burst = torch.where(
            burst_initiation_mask,
            True,
            self.in_burst
        )

        self.burst_spike_count = torch.where(
            burst_initiation_mask,
            torch.ones_like(self.burst_spike_count),
            self.burst_spike_count
        )

        # Continue burst
        self.burst_spike_count = torch.where(
            burst_continuation_mask,
            self.burst_spike_count + 1,
            self.burst_spike_count
        )

        # End burst after certain number of spikes or time
        burst_end_mask = (self.burst_spike_count > 3) | (time_since_spike > 0.01)  # 10ms max burst
        self.in_burst = torch.where(
            burst_end_mask,
            False,
            self.in_burst
        )

        self.burst_spike_count = torch.where(
            burst_end_mask,
            torch.zeros_like(self.burst_spike_count),
            self.burst_spike_count
        )

        # Reset membrane potential
        self.v_mem = torch.where(
            spike_mask,
            self.config.base_config.v_reset,
            self.v_mem
        )

        # Update spike times
        self.last_spike_time = torch.where(
            spike_mask,
            current_time,
            self.last_spike_time
        )

        # Update AHP current (increases with spikes, decays over time)
        ahp_increment = spike_mask.float() * self.config.afterhyperpolarization_strength
        ahp_decay = self.ahp_current * 0.95  # Exponential decay
        self.ahp_current = ahp_decay + ahp_increment

        # Adapt burst threshold based on recent activity
        threshold_adaptation = self.config.burst_adaptation_rate * (
            self.burst_spike_count / 5.0 - 0.5
        ) * dt

        self.burst_threshold = torch.clamp(
            self.burst_threshold + threshold_adaptation,
            self.config.base_config.v_threshold * 0.5,
            self.config.base_config.v_threshold * 1.5
        )

        states = {
            'membrane_potential': self.v_mem.clone(),
            'ahp_current': self.ahp_current.clone(),
            'in_burst': self.in_burst.clone(),
            'burst_spike_count': self.burst_spike_count.clone(),
            'burst_threshold': self.burst_threshold.clone(),
            'input_after_lung': self.lung(input_current).clone(),
            'input_after_liver': self.liver(input_current).clone(),
        }

        return spikes, states


class StochasticNeuron(nn.Module):
    """
    Leaky integrate-and-fire neuron with stochastic dynamics.
    Includes various noise sources for robustness:
    - Thermal noise in membrane potential
    - Channel noise in conductances  
    - Stochastic threshold
    Supports organ-inspired preprocessing.
    """

    def __init__(self, config: NeuronV3Config):
        super().__init__()

        self.config = config

        # Organ modules
        self.lung = LungIntake(**(config.lung_params or {}))
        self.liver = LiverFilter(**(config.liver_params or {}))

        # Neuron state
        self.register_buffer('v_mem', torch.tensor(config.base_config.v_rest))
        self.register_buffer('last_spike_time', torch.tensor(-float('inf')))

        # Noise parameters
        self.thermal_noise_std = config.noise_amplitude
        self.channel_noise_std = config.noise_amplitude * 0.1

        logger.debug("Created stochastic neuron with noise")

    def forward(
        self,
        input_current: torch.Tensor,
        current_time: float | None = None,
        dt: float | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Simulate stochastic neuron dynamics.
        Preprocess input using organ modules before neural processing.
        """
        if dt is None:
            dt = self.config.base_config.dt
        if current_time is None:
            current_time = 0.0

        batch_size = input_current.size(0)
        device = input_current.device

        # Organ preprocessing
        input_current = self.lung(input_current)
        input_current = self.liver(input_current)

        # Expand states for batch
        if self.v_mem.dim() == 0:
            self.v_mem = self.v_mem.unsqueeze(0).expand(batch_size).clone()
            self.last_spike_time = self.last_spike_time.unsqueeze(0).expand(batch_size).clone()

        # Check refractory period
        time_since_spike = current_time - self.last_spike_time
        refractory_mask = time_since_spike > self.config.base_config.refractory_period

        # Add noise to input current (channel noise)
        if self.config.channel_noise:
            current_noise = torch.randn_like(input_current) * self.channel_noise_std
            noisy_input = input_current + current_noise
        else:
            noisy_input = input_current

        # Membrane dynamics
        tau_mem = self.config.base_config.tau_mem
        leak_current = -(self.v_mem - self.config.base_config.v_rest) / tau_mem

        dv_dt = (noisy_input.squeeze() + leak_current) / tau_mem

        # Add thermal noise to membrane potential
        if self.config.thermal_noise:
            thermal_noise = torch.randn(batch_size, device=device) * self.thermal_noise_std * np.sqrt(dt)
            dv_dt = dv_dt + thermal_noise / tau_mem

        # Update membrane potential
        self.v_mem = torch.where(
            refractory_mask,
            self.v_mem + dv_dt * dt,
            self.config.base_config.v_reset
        )

        # Stochastic threshold - add small random variation
        base_threshold = self.config.base_config.v_threshold
        threshold_noise = torch.randn(batch_size, device=device) * self.thermal_noise_std * 0.1
        stochastic_threshold = base_threshold + threshold_noise

        # Spike detection with stochastic threshold
        spike_mask = (self.v_mem > stochastic_threshold) & refractory_mask
        spikes = spike_mask.float().unsqueeze(-1)

        # Reset membrane potential
        self.v_mem = torch.where(
            spike_mask,
            self.config.base_config.v_reset,
            self.v_mem
        )

        # Update spike times
        self.last_spike_time = torch.where(
            spike_mask,
            current_time,
            self.last_spike_time
        )

        states = {
            'membrane_potential': self.v_mem.clone(),
            'stochastic_threshold': stochastic_threshold,
            'thermal_noise_level': self.thermal_noise_std,
            'channel_noise_level': self.channel_noise_std,
            'input_after_lung': self.lung(input_current).clone(),
            'input_after_liver': self.liver(input_current).clone(),
        }

        return spikes, states


class OrganismNeuron(nn.Module):
    """
    A 'single cell' organism that integrates lung (data intake), liver (filtering), and neuron (processing) functions.
    Simplified LIF neuron with organ-inspired preprocessing.
    """
    def __init__(self, 
                 neuron_type='lif', 
                 lung_params=None, 
                 liver_params=None, 
                 neuron_params=None):
        super().__init__()
        # Organs
        self.lung = LungIntake(**(lung_params or {}))
        self.liver = LiverFilter(**(liver_params or {}))
        # Neuron params
        self.v_rest = neuron_params.get('v_rest', -70e-3) if neuron_params else -70e-3
        self.v_reset = neuron_params.get('v_reset', -70e-3) if neuron_params else -70e-3
        self.v_threshold = neuron_params.get('v_threshold', -50e-3) if neuron_params else -50e-3
        self.tau_mem = neuron_params.get('tau_mem', 20e-3) if neuron_params else 20e-3
        self.refractory_period = neuron_params.get('refractory_period', 2e-3) if neuron_params else 2e-3
        self.neuron_type = neuron_type
        # State
        self.register_buffer('v_mem', torch.tensor(self.v_rest))
        self.register_buffer('last_spike_time', torch.tensor(-float('inf')))

    def forward(self, input_data, current_time=0.0, dt=1e-3):
        """
        Simulate the full organism's signal processing.
        Args:
            input_data: Tensor [batch, 1] or [time, 1]
            current_time: Simulation time
            dt: Time step
        Returns:
            spikes, states
        """
        # Lungs: Intake & amplify
        x = self.lung(input_data)
        # Liver: Filter/clean
        x = self.liver(x)
        batch_size = x.size(0)
        device = x.device
        # Expand state if needed
        if self.v_mem.dim() == 0:
            self.v_mem = self.v_mem.unsqueeze(0).expand(batch_size).clone()
            self.last_spike_time = self.last_spike_time.unsqueeze(0).expand(batch_size).clone()
        # Check refractory mask
        time_since_spike = current_time - self.last_spike_time
        refractory_mask = time_since_spike > self.refractory_period
        # Neuron membrane dynamics (LIF)
        leak_current = -(self.v_mem - self.v_rest) / self.tau_mem
        dv_dt = (x.squeeze() + leak_current) / self.tau_mem
        self.v_mem = torch.where(
            refractory_mask,
            self.v_mem + dv_dt * dt,
            torch.tensor(self.v_reset, device=device)
        )
        # Spike detection
        spike_mask = (self.v_mem > self.v_threshold) & refractory_mask
        spikes = spike_mask.float().unsqueeze(-1)
        # Reset membrane on spike
        self.v_mem = torch.where(
            spike_mask,
            torch.tensor(self.v_reset, device=device),
            self.v_mem
        )
        # Update spike time
        self.last_spike_time = torch.where(
            spike_mask,
            torch.tensor(current_time, device=device),
            self.last_spike_time
        )
        states = {
            'membrane_potential': self.v_mem.clone(),
            'spikes': spikes.clone(),
            'input_after_lung': self.lung(input_data).clone(),
            'input_after_liver': x.clone(),
        }
        return spikes, states

# Example usage:
if __name__ == "__main__":
    # Simulate time series input data for one cell
    time_steps = 100
    dt = 1e-3
    t = torch.arange(0, time_steps * dt, dt)
    input_current = 0.04 * torch.sin(2 * 3.1415 * t * 5) + 0.05  # simple oscillatory input
    input_current = input_current.unsqueeze(1)  # [time, 1]

    cell = OrganismNeuron(
        neuron_type='lif',
        lung_params={'amplify_factor': 2.0, 'sample_rate': 1.0},
        liver_params={'method': 'gaussian', 'kernel_size': 7, 'std': 1.5},
        neuron_params={
            'v_rest': -70e-3,
            'v_reset': -70e-3,
            'v_threshold': -50e-3,
            'tau_mem': 20e-3,
            'refractory_period': 2e-3
        }
    )
    spikes = []
    v_mems = []
    for i in range(time_steps):
        spike, state = cell(input_current[i].unsqueeze(0), current_time=i*dt, dt=dt)
        spikes.append(spike.item())
        v_mems.append(state['membrane_potential'].item())
    print("Spike times:", [i*dt for i, s in enumerate(spikes) if s > 0])
    print("Final membrane potential:", v_mems[-1])
