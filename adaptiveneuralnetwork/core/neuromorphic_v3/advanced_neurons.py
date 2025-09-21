"""
Advanced 3rd generation neuron models with enhanced biological realism.

This module implements sophisticated neuron models including multi-compartment
neurons, adaptive thresholds, burst firing patterns, and stochastic dynamics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..neuromorphic import NeuromorphicConfig

logger = logging.getLogger(__name__)


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


class MultiCompartmentNeuron(nn.Module):
    """
    Multi-compartment neuron model with dendritic processing.

    Implements a neuron with separate soma and dendritic compartments,
    allowing for complex dendritic computation and integration.
    """

    def __init__(
        self,
        config: NeuronV3Config,
        num_dendrites: int = 4,
        compartment_configs: Optional[List[CompartmentConfig]] = None,
    ):
        super().__init__()

        self.config = config
        self.num_dendrites = num_dendrites
        self.num_compartments = num_dendrites + 1  # dendrites + soma

        # Set up compartment configurations
        if compartment_configs is None:
            self.compartment_configs = [CompartmentConfig() for _ in range(self.num_compartments)]
        else:
            self.compartment_configs = compartment_configs

        # Initialize compartment states
        self.register_buffer("v_mem", torch.zeros(self.num_compartments))
        self.register_buffer("spike_times", torch.full((self.num_compartments,), -float("inf")))

        # Coupling matrix (dendrites connect to soma)
        coupling_matrix = torch.zeros(self.num_compartments, self.num_compartments)
        for i in range(self.num_dendrites):
            # Dendrite i connects to soma (last compartment)
            coupling_matrix[i, -1] = self.compartment_configs[i].coupling_conductance
            coupling_matrix[-1, i] = self.compartment_configs[i].coupling_conductance

        self.register_buffer("coupling_matrix", coupling_matrix)

        logger.debug(f"Created multi-compartment neuron with {self.num_compartments} compartments")

    def forward(
        self,
        dendritic_inputs: torch.Tensor,
        somatic_input: torch.Tensor,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simulate multi-compartment neuron dynamics.

        Args:
            dendritic_inputs: Input currents to dendrites [batch_size, num_dendrites]
            somatic_input: Input current to soma [batch_size, 1]
            dt: Time step (uses config default if None)

        Returns:
            Tuple of (spike_output, compartment_states)
        """
        if dt is None:
            dt = self.config.base_config.dt

        batch_size = dendritic_inputs.size(0)
        device = dendritic_inputs.device

        # Initialize states for batch if needed
        if self.v_mem.size(0) != batch_size:
            self.v_mem = torch.zeros(batch_size, self.num_compartments, device=device)
            self.spike_times = torch.full(
                (batch_size, self.num_compartments), -float("inf"), device=device
            )

        # Combine inputs
        inputs = torch.cat([dendritic_inputs, somatic_input], dim=1)

        # Calculate coupling currents
        coupling_currents = torch.zeros_like(self.v_mem)
        for i in range(self.num_compartments):
            for j in range(self.num_compartments):
                if i != j and self.coupling_matrix[i, j] > 0:
                    coupling_currents[:, i] += self.coupling_matrix[i, j] * (
                        self.v_mem[:, j] - self.v_mem[:, i]
                    )

        # Update compartment voltages
        spikes = torch.zeros_like(self.v_mem)

        for comp_idx in range(self.num_compartments):
            config = self.compartment_configs[comp_idx]

            # Check refractory period (simplified - assume not refractory for now)
            refractory_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

            # Membrane dynamics (only for non-refractory neurons)
            leak_current = config.leak_conductance * (
                config.leak_reversal - self.v_mem[:, comp_idx]
            )
            total_current = inputs[:, comp_idx] + coupling_currents[:, comp_idx] + leak_current

            dv_dt = total_current / config.capacitance
            self.v_mem[:, comp_idx] = torch.where(
                refractory_mask,
                self.v_mem[:, comp_idx] + dv_dt * dt,
                torch.tensor(config.reset, device=device),
            )

            # Check for spikes
            spike_mask = (self.v_mem[:, comp_idx] > config.threshold) & refractory_mask
            spikes[:, comp_idx] = spike_mask.float()

            # Reset spiking neurons
            self.v_mem[:, comp_idx] = torch.where(
                spike_mask, torch.tensor(config.reset, device=device), self.v_mem[:, comp_idx]
            )

        # Return only somatic spikes as main output
        somatic_spikes = spikes[:, -1:]

        states = {
            "compartment_voltages": self.v_mem.clone(),
            "all_spikes": spikes,
            "coupling_currents": coupling_currents,
        }

        return somatic_spikes, states


class AdaptiveThresholdNeuron(nn.Module):
    """
    Leaky integrate-and-fire neuron with adaptive spike threshold.

    Implements homeostatic plasticity through threshold adaptation,
    maintaining target firing rates.
    """

    def __init__(self, config: NeuronV3Config):
        super().__init__()

        self.config = config

        # Neuron state variables
        self.register_buffer("v_mem", torch.tensor(config.base_config.v_rest))
        self.register_buffer("threshold", torch.tensor(config.base_config.v_threshold))
        self.register_buffer("spike_count", torch.tensor(0.0))
        self.register_buffer("time_window_start", torch.tensor(0.0))
        self.register_buffer("last_spike_time", torch.tensor(-float("inf")))

        logger.debug("Created adaptive threshold neuron")

    def forward(
        self,
        input_current: torch.Tensor,
        current_time: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simulate adaptive threshold neuron dynamics.

        Args:
            input_current: Input current [batch_size, 1]
            current_time: Current simulation time
            dt: Time step

        Returns:
            Tuple of (spike_output, neuron_states)
        """
        if dt is None:
            dt = self.config.base_config.dt
        if current_time is None:
            current_time = 0.0

        batch_size = input_current.size(0)

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
            refractory_mask, self.v_mem + dv_dt * dt, self.config.base_config.v_reset
        )

        # Spike detection
        spike_mask = (self.v_mem > self.threshold) & refractory_mask
        spikes = spike_mask.float().unsqueeze(-1)

        # Reset membrane potential and update spike times
        self.v_mem = torch.where(spike_mask, self.config.base_config.v_reset, self.v_mem)

        self.last_spike_time = torch.where(spike_mask, current_time, self.last_spike_time)

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
                should_adapt, self.threshold + threshold_change, self.threshold
            )

            # Reset counting window for adapted neurons
            self.spike_count = torch.where(
                should_adapt, torch.zeros_like(self.spike_count), self.spike_count
            )
            self.time_window_start = torch.where(
                should_adapt,
                torch.full_like(self.time_window_start, current_time),
                self.time_window_start,
            )

        # Threshold decay towards baseline
        baseline_threshold = self.config.base_config.v_threshold
        threshold_decay = (
            self.config.threshold_decay_rate * (baseline_threshold - self.threshold) * dt
        )
        self.threshold = self.threshold + threshold_decay

        states = {
            "membrane_potential": self.v_mem.clone(),
            "threshold": self.threshold.clone(),
            "spike_count": self.spike_count.clone(),
            "firing_rate": self.spike_count / (window_duration.clamp(min=1e-6)),
        }

        return spikes, states


class BurstingNeuron(nn.Module):
    """
    Neuron model capable of generating burst firing patterns.

    Implements burst generation through afterhyperpolarization
    and adaptive burst threshold mechanisms.
    """

    def __init__(self, config: NeuronV3Config):
        super().__init__()

        self.config = config

        # Neuron state variables
        self.register_buffer("v_mem", torch.tensor(config.base_config.v_rest))
        self.register_buffer("ahp_current", torch.tensor(0.0))  # Afterhyperpolarization
        self.register_buffer(
            "burst_threshold",
            torch.tensor(config.base_config.v_threshold * config.burst_threshold_factor),
        )
        self.register_buffer("in_burst", torch.tensor(False))
        self.register_buffer("burst_spike_count", torch.tensor(0.0))
        self.register_buffer("last_spike_time", torch.tensor(-float("inf")))

        logger.debug("Created bursting neuron")

    def forward(
        self,
        input_current: torch.Tensor,
        current_time: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simulate bursting neuron dynamics.

        Args:
            input_current: Input current [batch_size, 1]
            current_time: Current simulation time
            dt: Time step

        Returns:
            Tuple of (spike_output, neuron_states)
        """
        if dt is None:
            dt = self.config.base_config.dt
        if current_time is None:
            current_time = 0.0

        batch_size = input_current.size(0)

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
            refractory_mask, self.v_mem + dv_dt * dt, self.config.base_config.v_reset
        )

        # Determine effective threshold (burst vs normal)
        effective_threshold = torch.where(
            self.in_burst, self.burst_threshold, self.config.base_config.v_threshold
        )

        # Spike detection
        spike_mask = (self.v_mem > effective_threshold) & refractory_mask
        spikes = spike_mask.float().unsqueeze(-1)

        # Handle burst dynamics
        burst_initiation_mask = spike_mask & ~self.in_burst
        burst_continuation_mask = spike_mask & self.in_burst

        # Start burst
        self.in_burst = torch.where(burst_initiation_mask, True, self.in_burst)

        self.burst_spike_count = torch.where(
            burst_initiation_mask, torch.ones_like(self.burst_spike_count), self.burst_spike_count
        )

        # Continue burst
        self.burst_spike_count = torch.where(
            burst_continuation_mask, self.burst_spike_count + 1, self.burst_spike_count
        )

        # End burst after certain number of spikes or time
        burst_end_mask = (self.burst_spike_count > 3) | (time_since_spike > 0.01)  # 10ms max burst
        self.in_burst = torch.where(burst_end_mask, False, self.in_burst)

        self.burst_spike_count = torch.where(
            burst_end_mask, torch.zeros_like(self.burst_spike_count), self.burst_spike_count
        )

        # Reset membrane potential
        self.v_mem = torch.where(spike_mask, self.config.base_config.v_reset, self.v_mem)

        # Update spike times
        self.last_spike_time = torch.where(spike_mask, current_time, self.last_spike_time)

        # Update AHP current (increases with spikes, decays over time)
        ahp_increment = spike_mask.float() * self.config.afterhyperpolarization_strength
        ahp_decay = self.ahp_current * 0.95  # Exponential decay
        self.ahp_current = ahp_decay + ahp_increment

        # Adapt burst threshold based on recent activity
        threshold_adaptation = (
            self.config.burst_adaptation_rate * (self.burst_spike_count / 5.0 - 0.5) * dt
        )

        self.burst_threshold = torch.clamp(
            self.burst_threshold + threshold_adaptation,
            self.config.base_config.v_threshold * 0.5,
            self.config.base_config.v_threshold * 1.5,
        )

        states = {
            "membrane_potential": self.v_mem.clone(),
            "ahp_current": self.ahp_current.clone(),
            "in_burst": self.in_burst.clone(),
            "burst_spike_count": self.burst_spike_count.clone(),
            "burst_threshold": self.burst_threshold.clone(),
        }

        return spikes, states


class StochasticNeuron(nn.Module):
    """
    Leaky integrate-and-fire neuron with stochastic dynamics.

    Includes various noise sources for robustness:
    - Thermal noise in membrane potential
    - Channel noise in conductances
    - Stochastic threshold
    """

    def __init__(self, config: NeuronV3Config):
        super().__init__()

        self.config = config

        # Neuron state
        self.register_buffer("v_mem", torch.tensor(config.base_config.v_rest))
        self.register_buffer("last_spike_time", torch.tensor(-float("inf")))

        # Noise parameters
        self.thermal_noise_std = config.noise_amplitude
        self.channel_noise_std = config.noise_amplitude * 0.1

        logger.debug("Created stochastic neuron with noise")

    def forward(
        self,
        input_current: torch.Tensor,
        current_time: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simulate stochastic neuron dynamics.

        Args:
            input_current: Input current [batch_size, 1]
            current_time: Current simulation time
            dt: Time step

        Returns:
            Tuple of (spike_output, neuron_states)
        """
        if dt is None:
            dt = self.config.base_config.dt
        if current_time is None:
            current_time = 0.0

        batch_size = input_current.size(0)
        device = input_current.device

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
            thermal_noise = (
                torch.randn(batch_size, device=device) * self.thermal_noise_std * np.sqrt(dt)
            )
            dv_dt = dv_dt + thermal_noise / tau_mem

        # Update membrane potential
        self.v_mem = torch.where(
            refractory_mask, self.v_mem + dv_dt * dt, self.config.base_config.v_reset
        )

        # Stochastic threshold - add small random variation
        base_threshold = self.config.base_config.v_threshold
        threshold_noise = torch.randn(batch_size, device=device) * self.thermal_noise_std * 0.1
        stochastic_threshold = base_threshold + threshold_noise

        # Spike detection with stochastic threshold
        spike_mask = (self.v_mem > stochastic_threshold) & refractory_mask
        spikes = spike_mask.float().unsqueeze(-1)

        # Reset membrane potential
        self.v_mem = torch.where(spike_mask, self.config.base_config.v_reset, self.v_mem)

        # Update spike times
        self.last_spike_time = torch.where(spike_mask, current_time, self.last_spike_time)

        states = {
            "membrane_potential": self.v_mem.clone(),
            "stochastic_threshold": stochastic_threshold,
            "thermal_noise_level": self.thermal_noise_std,
            "channel_noise_level": self.channel_noise_std,
        }

        return spikes, states
