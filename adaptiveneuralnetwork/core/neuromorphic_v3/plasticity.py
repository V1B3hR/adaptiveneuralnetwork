"""
Advanced synaptic plasticity mechanisms for 3rd generation neuromorphic computing.

This module implements sophisticated plasticity rules including STDP, metaplasticity,
homeostatic scaling, and multi-timescale learning mechanisms.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from ..neuromorphic import NeuromorphicConfig, SpikeEvent
import logging

logger = logging.getLogger(__name__)


@dataclass
class STDPConfig:
    """Configuration for Spike-Timing Dependent Plasticity."""
    # Learning rates
    a_plus: float = 0.01  # Potentiation amplitude
    a_minus: float = 0.012  # Depression amplitude (typically > a_plus)
    
    # Time constants
    tau_plus: float = 0.020  # Potentiation time constant (20ms)
    tau_minus: float = 0.020  # Depression time constant (20ms)
    
    # Weight bounds
    w_min: float = 0.0  # Minimum weight
    w_max: float = 1.0  # Maximum weight
    
    # Multiplicative vs additive plasticity
    multiplicative: bool = True
    
    # Spike pairing requirements
    max_dt: float = 0.1  # Maximum time difference for pairing (100ms)


@dataclass
class MetaplasticityConfig:
    """Configuration for metaplasticity (plasticity of plasticity)."""
    # Metaplastic state parameters
    theta_plus: float = 0.1  # Potentiation threshold
    theta_minus: float = 0.05  # Depression threshold
    
    # Metaplastic time constants
    tau_theta: float = 10.0  # Threshold time constant (10s)
    tau_x: float = 0.1  # Activity variable time constant (100ms)
    
    # Sliding threshold parameters
    sliding_window: float = 1.0  # Sliding window for activity (1s)
    target_activity: float = 0.1  # Target activity level


@dataclass
class HomeostaticConfig:
    """Configuration for homeostatic plasticity mechanisms."""
    # Homeostatic scaling
    target_rate: float = 10.0  # Target firing rate (Hz)
    scaling_factor: float = 0.001  # Scaling rate
    scaling_window: float = 100.0  # Time window for rate calculation (s)
    
    # Intrinsic plasticity
    adapt_threshold: bool = True
    threshold_timescale: float = 100.0  # Threshold adaptation timescale
    
    # Synaptic scaling bounds
    min_scaling: float = 0.1  # Minimum scaling factor
    max_scaling: float = 10.0  # Maximum scaling factor


class STDPSynapse(nn.Module):
    """
    Synapse implementing Spike-Timing Dependent Plasticity.
    
    Classic STDP rule where pre-before-post leads to potentiation (LTP)
    and post-before-pre leads to depression (LTD).
    """
    
    def __init__(
        self,
        pre_size: int,
        post_size: int,
        config: STDPConfig,
        initial_weight: Optional[float] = None
    ):
        super().__init__()
        
        self.config = config
        self.pre_size = pre_size
        self.post_size = post_size
        
        # Initialize synaptic weights
        if initial_weight is None:
            initial_weight = (config.w_min + config.w_max) / 2
        
        self.register_parameter('weights', nn.Parameter(
            torch.full((pre_size, post_size), initial_weight)
        ))
        
        # Spike timing traces
        self.register_buffer('pre_trace', torch.zeros(pre_size))
        self.register_buffer('post_trace', torch.zeros(post_size))
        
        # Last spike times for pairing detection
        self.register_buffer('last_pre_spike', torch.full((pre_size,), -float('inf')))
        self.register_buffer('last_post_spike', torch.full((post_size,), -float('inf')))
        
        logger.debug(f"Created STDP synapse: {pre_size} -> {post_size}")
    
    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_time: float,
        dt: Optional[float] = None,
        learning: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process synaptic transmission and plasticity.
        
        Args:
            pre_spikes: Presynaptic spikes [batch_size, pre_size]
            post_spikes: Postsynaptic spikes [batch_size, post_size]  
            current_time: Current simulation time
            dt: Time step
            learning: Whether to apply plasticity
            
        Returns:
            Tuple of (synaptic_current, plasticity_info)
        """
        if dt is None:
            dt = 0.001  # Default 1ms
            
        batch_size = pre_spikes.size(0)
        
        # Expand traces for batch if needed
        if self.pre_trace.dim() == 1:
            self.pre_trace = self.pre_trace.unsqueeze(0).expand(batch_size, -1).clone()
            self.post_trace = self.post_trace.unsqueeze(0).expand(batch_size, -1).clone()
            self.last_pre_spike = self.last_pre_spike.unsqueeze(0).expand(batch_size, -1).clone()
            self.last_post_spike = self.last_post_spike.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Update spike times
        pre_spike_mask = pre_spikes > 0
        post_spike_mask = post_spikes > 0
        
        self.last_pre_spike = torch.where(
            pre_spike_mask,
            current_time,
            self.last_pre_spike
        )
        
        self.last_post_spike = torch.where(
            post_spike_mask,
            current_time,
            self.last_post_spike
        )
        
        # Update eligibility traces
        self.pre_trace *= torch.exp(torch.tensor(-dt / self.config.tau_plus))
        self.post_trace *= torch.exp(torch.tensor(-dt / self.config.tau_minus))
        
        # Add spike contributions to traces
        self.pre_trace += pre_spikes
        self.post_trace += post_spikes
        
        # Calculate synaptic current (pre-synaptic spikes * weights)
        synaptic_current = torch.matmul(pre_spikes, self.weights)
        
        if learning:
            # STDP weight updates
            weight_changes = self._calculate_stdp_updates(
                pre_spikes, post_spikes, current_time, dt
            )
            
            # Apply weight changes
            if self.config.multiplicative:
                # Multiplicative STDP
                potentiation_mask = weight_changes > 0
                depression_mask = weight_changes < 0
                
                pot_change = weight_changes * (self.config.w_max - self.weights)
                dep_change = weight_changes * (self.weights - self.config.w_min)
                
                weight_update = torch.where(potentiation_mask, pot_change, dep_change)
            else:
                # Additive STDP
                weight_update = weight_changes
            
            # Update weights with bounds
            new_weights = self.weights + weight_update
            self.weights.data = torch.clamp(new_weights, self.config.w_min, self.config.w_max)
        
        plasticity_info = {
            'weight_changes': weight_changes if learning else torch.zeros_like(self.weights),
            'pre_trace': self.pre_trace.clone(),
            'post_trace': self.post_trace.clone(),
            'synaptic_weights': self.weights.clone()
        }
        
        return synaptic_current, plasticity_info
    
    def _calculate_stdp_updates(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_time: float,
        dt: float
    ) -> torch.Tensor:
        """Calculate STDP weight updates based on spike timing."""
        batch_size = pre_spikes.size(0)
        weight_changes = torch.zeros_like(self.weights).unsqueeze(0).expand(batch_size, -1, -1)
        
        # For each batch item, calculate pairwise interactions
        for b in range(batch_size):
            pre_spike_indices = torch.where(pre_spikes[b] > 0)[0]
            post_spike_indices = torch.where(post_spikes[b] > 0)[0]
            
            # LTP: pre before post (Δt = t_post - t_pre > 0)
            for post_idx in post_spike_indices:
                post_time = self.last_post_spike[b, post_idx]
                
                for pre_idx in pre_spike_indices:
                    pre_time = self.last_pre_spike[b, pre_idx]
                    dt_spike = post_time - pre_time
                    
                    if 0 < dt_spike <= self.config.max_dt:
                        # Potentiation
                        ltp_amount = self.config.a_plus * torch.exp(torch.tensor(-dt_spike / self.config.tau_plus))
                        weight_changes[b, pre_idx, post_idx] += ltp_amount
            
            # LTD: post before pre (Δt = t_post - t_pre < 0)  
            for pre_idx in pre_spike_indices:
                pre_time = self.last_pre_spike[b, pre_idx]
                
                for post_idx in post_spike_indices:
                    post_time = self.last_post_spike[b, post_idx]
                    dt_spike = pre_time - post_time  # Note: reversed for LTD
                    
                    if 0 < dt_spike <= self.config.max_dt:
                        # Depression
                        ltd_amount = -self.config.a_minus * torch.exp(torch.tensor(-dt_spike / self.config.tau_minus))
                        weight_changes[b, pre_idx, post_idx] += ltd_amount
        
        # Average over batch
        return weight_changes.mean(dim=0)


class MetaplasticitySynapse(nn.Module):
    """
    Synapse with metaplasticity - plasticity rules that themselves change.
    
    Implements sliding threshold metaplasticity where the plasticity
    threshold adapts based on postsynaptic activity history.
    """
    
    def __init__(
        self,
        pre_size: int,
        post_size: int,
        stdp_config: STDPConfig,
        meta_config: MetaplasticityConfig
    ):
        super().__init__()
        
        self.stdp_config = stdp_config
        self.meta_config = meta_config
        self.pre_size = pre_size
        self.post_size = post_size
        
        # Base STDP synapse
        self.stdp_synapse = STDPSynapse(pre_size, post_size, stdp_config)
        
        # Metaplastic state variables
        self.register_buffer('theta', torch.full((post_size,), meta_config.theta_plus))  # Sliding threshold
        self.register_buffer('x', torch.zeros(post_size))  # Activity variable
        self.register_buffer('activity_history', torch.zeros(post_size, 100))  # Sliding window
        self.register_buffer('history_index', torch.zeros(1, dtype=torch.long))
        
        logger.debug(f"Created metaplastic synapse: {pre_size} -> {post_size}")
    
    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_time: float,
        dt: Optional[float] = None,
        learning: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process synaptic transmission with metaplasticity.
        
        Args:
            pre_spikes: Presynaptic spikes [batch_size, pre_size]
            post_spikes: Postsynaptic spikes [batch_size, post_size]
            current_time: Current simulation time
            dt: Time step
            learning: Whether to apply plasticity
            
        Returns:
            Tuple of (synaptic_current, plasticity_info)
        """
        if dt is None:
            dt = 0.001
            
        batch_size = pre_spikes.size(0)
        
        # Expand metaplastic variables for batch
        if self.theta.dim() == 1:
            self.theta = self.theta.unsqueeze(0).expand(batch_size, -1).clone()
            self.x = self.x.unsqueeze(0).expand(batch_size, -1).clone()
            self.activity_history = self.activity_history.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Update activity variable
        self.x *= torch.exp(-dt / self.meta_config.tau_x)
        self.x += post_spikes
        
        # Update activity history (sliding window)
        window_size = self.activity_history.size(-1)
        current_idx = self.history_index.item() % window_size
        self.activity_history[:, :, current_idx] = post_spikes
        self.history_index += 1
        
        if learning:
            # Update sliding threshold based on recent activity
            recent_activity = self.activity_history.mean(dim=-1)  # Average over window
            
            # Homeostatic adjustment of threshold
            activity_error = recent_activity - self.meta_config.target_activity
            theta_update = activity_error / self.meta_config.tau_theta * dt
            
            self.theta = self.theta + theta_update
            self.theta = torch.clamp(
                self.theta,
                self.meta_config.theta_minus,
                self.meta_config.theta_plus * 2
            )
        
        # Get base STDP output
        synaptic_current, stdp_info = self.stdp_synapse(
            pre_spikes, post_spikes, current_time, dt, learning=False
        )
        
        if learning:
            # Modulate STDP based on metaplastic state
            base_weight_changes = stdp_info['weight_changes']
            
            # Apply metaplastic modulation
            # If activity > threshold: reduce plasticity (saturation)
            # If activity < threshold: increase plasticity (priming)
            activity_factor = torch.sigmoid(-(self.x - self.theta.unsqueeze(-1)))
            
            modulated_changes = base_weight_changes * activity_factor.mean(dim=0)
            
            # Apply modulated weight changes
            new_weights = self.stdp_synapse.weights + modulated_changes
            self.stdp_synapse.weights.data = torch.clamp(
                new_weights,
                self.stdp_config.w_min,
                self.stdp_config.w_max
            )
        
        plasticity_info = {
            'weight_changes': modulated_changes if learning else torch.zeros_like(self.stdp_synapse.weights),
            'metaplastic_threshold': self.theta.clone(),
            'activity_variable': self.x.clone(),
            'recent_activity': recent_activity.clone() if learning else torch.zeros_like(self.x),
            'synaptic_weights': self.stdp_synapse.weights.clone()
        }
        plasticity_info.update(stdp_info)
        
        return synaptic_current, plasticity_info


class HomeostaticScaling(nn.Module):
    """
    Homeostatic synaptic scaling mechanism.
    
    Multiplicatively scales all synapses to maintain target firing rates,
    preserving relative weight relationships while ensuring network stability.
    """
    
    def __init__(
        self,
        num_neurons: int,
        config: HomeostaticConfig
    ):
        super().__init__()
        
        self.config = config
        self.num_neurons = num_neurons
        
        # Scaling factors for each neuron
        self.register_buffer('scaling_factors', torch.ones(num_neurons))
        
        # Activity monitoring
        self.register_buffer('spike_counts', torch.zeros(num_neurons))
        self.register_buffer('time_window_start', torch.tensor(0.0))
        self.register_buffer('last_scaling_time', torch.tensor(0.0))
        
        logger.debug(f"Created homeostatic scaling for {num_neurons} neurons")
    
    def forward(
        self,
        post_spikes: torch.Tensor,
        synaptic_weights: torch.Tensor,
        current_time: float,
        dt: Optional[float] = None,
        apply_scaling: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply homeostatic scaling to synaptic weights.
        
        Args:
            post_spikes: Postsynaptic spikes [batch_size, num_neurons]
            synaptic_weights: Current synaptic weights to scale
            current_time: Current simulation time
            dt: Time step
            apply_scaling: Whether to update scaling factors
            
        Returns:
            Tuple of (scaled_weights, homeostatic_info)
        """
        if dt is None:
            dt = 0.001
            
        batch_size = post_spikes.size(0)
        
        # Expand buffers for batch processing
        if self.spike_counts.dim() == 1:
            self.spike_counts = self.spike_counts.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Update spike counts
        self.spike_counts += post_spikes
        
        # Check if it's time to update scaling
        window_duration = current_time - self.time_window_start
        
        if apply_scaling and window_duration >= self.config.scaling_window:
            # Calculate firing rates
            firing_rates = self.spike_counts / window_duration
            
            # Calculate desired scaling factors
            rate_ratio = self.config.target_rate / (firing_rates + 1e-6)  # Avoid division by zero
            
            # Smooth scaling factor updates
            scaling_update = self.config.scaling_factor * (rate_ratio - 1.0) * dt
            new_scaling = self.scaling_factors + scaling_update
            
            # Apply bounds to scaling factors
            self.scaling_factors = torch.clamp(
                new_scaling,
                self.config.min_scaling,
                self.config.max_scaling
            )
            
            # Reset counters
            self.spike_counts.zero_()
            self.time_window_start = current_time
            self.last_scaling_time = current_time
        
        # Apply scaling to weights
        # Broadcast scaling factors over weight dimensions
        if synaptic_weights.dim() == 2:  # [pre, post]
            scaled_weights = synaptic_weights * self.scaling_factors.unsqueeze(0)
        elif synaptic_weights.dim() == 3:  # [batch, pre, post] 
            scaled_weights = synaptic_weights * self.scaling_factors.unsqueeze(0).unsqueeze(0)
        else:
            scaled_weights = synaptic_weights
        
        homeostatic_info = {
            'scaling_factors': self.scaling_factors.clone(),
            'firing_rates': self.spike_counts / (window_duration + 1e-6),
            'target_rate': self.config.target_rate,
            'time_since_scaling': current_time - self.last_scaling_time
        }
        
        return scaled_weights, homeostatic_info


class MultiTimescalePlasticity(nn.Module):
    """
    Multi-timescale plasticity combining short-term and long-term mechanisms.
    
    Implements multiple plasticity processes operating on different timescales:
    - Fast STDP (milliseconds to seconds)  
    - Slow homeostatic scaling (minutes to hours)
    - Intermediate metaplasticity (seconds to minutes)
    """
    
    def __init__(
        self,
        pre_size: int,
        post_size: int,
        stdp_config: STDPConfig,
        meta_config: MetaplasticityConfig,
        homeostatic_config: HomeostaticConfig
    ):
        super().__init__()
        
        self.pre_size = pre_size
        self.post_size = post_size
        
        # Multiple plasticity mechanisms
        self.stdp_synapse = STDPSynapse(pre_size, post_size, stdp_config)
        self.metaplastic_synapse = MetaplasticitySynapse(pre_size, post_size, stdp_config, meta_config)
        self.homeostatic_scaling = HomeostaticScaling(post_size, homeostatic_config)
        
        # Mechanism weights
        self.register_buffer('stdp_weight', torch.tensor(1.0))
        self.register_buffer('meta_weight', torch.tensor(0.5))
        self.register_buffer('homeostatic_weight', torch.tensor(0.1))
        
        logger.debug(f"Created multi-timescale plasticity: {pre_size} -> {post_size}")
    
    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor, 
        current_time: float,
        dt: Optional[float] = None,
        learning: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply multi-timescale plasticity.
        
        Args:
            pre_spikes: Presynaptic spikes [batch_size, pre_size]
            post_spikes: Postsynaptic spikes [batch_size, post_size]
            current_time: Current simulation time
            dt: Time step
            learning: Whether to apply plasticity
            
        Returns:
            Tuple of (synaptic_current, plasticity_info)
        """
        if dt is None:
            dt = 0.001
        
        # Fast STDP (always active during learning)
        stdp_current, stdp_info = self.stdp_synapse(
            pre_spikes, post_spikes, current_time, dt, learning
        )
        
        # Intermediate metaplasticity (modulates STDP)
        meta_current, meta_info = self.metaplastic_synapse(
            pre_spikes, post_spikes, current_time, dt, learning
        )
        
        # Get current weights from metaplastic synapse (it includes STDP)
        current_weights = self.metaplastic_synapse.stdp_synapse.weights
        
        # Slow homeostatic scaling
        scaled_weights, homeostatic_info = self.homeostatic_scaling(
            post_spikes, current_weights, current_time, dt, learning
        )
        
        # Use scaled weights for synaptic transmission
        synaptic_current = torch.matmul(pre_spikes, scaled_weights)
        
        # Update the actual stored weights with scaling
        if learning:
            self.metaplastic_synapse.stdp_synapse.weights.data = scaled_weights
        
        # Combine plasticity information
        plasticity_info = {
            'stdp': stdp_info,
            'metaplasticity': meta_info, 
            'homeostatic': homeostatic_info,
            'final_weights': scaled_weights.clone(),
            'mechanism_weights': {
                'stdp': self.stdp_weight.item(),
                'meta': self.meta_weight.item(),
                'homeostatic': self.homeostatic_weight.item()
            }
        }
        
        return synaptic_current, plasticity_info