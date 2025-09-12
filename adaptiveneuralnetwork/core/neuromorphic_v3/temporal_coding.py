"""
Advanced temporal coding and pattern encoding for 3rd generation neuromorphic computing.

This module implements sophisticated temporal pattern encoding beyond simple rate coding,
including phase encoding, oscillatory dynamics, and sparse distributed representations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import math
from ..neuromorphic import NeuromorphicConfig, SpikeEvent
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalConfig:
    """Configuration for temporal coding mechanisms."""
    # Pattern encoding parameters
    pattern_window: float = 0.1  # Time window for pattern detection (100ms)
    min_pattern_length: int = 3  # Minimum spikes in a pattern
    max_pattern_length: int = 10  # Maximum spikes in a pattern
    
    # Phase encoding parameters  
    reference_frequency: float = 40.0  # Reference oscillation (40Hz gamma)
    phase_resolution: int = 8  # Number of discrete phase bins
    phase_window: float = 0.025  # Phase encoding window (25ms)
    
    # Oscillatory dynamics
    oscillation_frequencies: List[float] = None  # Will default to [8, 13, 30, 80] Hz
    coupling_strength: float = 0.1  # Oscillation coupling strength
    
    # Sparse coding parameters
    sparsity_target: float = 0.05  # Target activation level (5%)
    lateral_inhibition: float = 0.2  # Lateral inhibition strength
    adaptation_rate: float = 0.01  # Sparsity adaptation rate


class TemporalPatternEncoder(nn.Module):
    """
    Encoder for temporal spike patterns extending beyond rate coding.
    
    Detects and encodes precise temporal sequences and correlations
    in spike trains for pattern-based information processing.
    """
    
    def __init__(
        self,
        input_size: int,
        pattern_size: int,
        config: TemporalConfig
    ):
        super().__init__()
        
        self.input_size = input_size
        self.pattern_size = pattern_size
        self.config = config
        
        # Pattern memory storage
        self.register_buffer('pattern_templates', 
                           torch.randn(pattern_size, config.max_pattern_length, input_size))
        
        # Spike history buffer for pattern detection
        history_length = int(config.pattern_window / 0.001) + 1  # Assume 1ms resolution
        self.register_buffer('spike_history', 
                           torch.zeros(history_length, input_size))
        self.register_buffer('time_history', 
                           torch.zeros(history_length))
        self.register_buffer('history_index', torch.tensor(0, dtype=torch.long))
        
        # Pattern detection weights
        self.pattern_weights = nn.Parameter(torch.randn(pattern_size, input_size))
        
        # Temporal correlation kernels
        self.register_buffer('correlation_kernels', self._create_correlation_kernels())
        
        logger.debug(f"Created temporal pattern encoder: {input_size} -> {pattern_size}")
    
    def _create_correlation_kernels(self) -> torch.Tensor:
        """Create kernels for detecting temporal correlations."""
        kernel_size = min(20, self.config.max_pattern_length)
        num_kernels = 5
        
        kernels = torch.zeros(num_kernels, kernel_size)
        
        # Different temporal kernels
        for i in range(num_kernels):
            if i == 0:  # Exponential decay
                kernels[i] = torch.exp(-torch.arange(kernel_size, dtype=torch.float) / 5)
            elif i == 1:  # Gaussian
                center = kernel_size // 2
                sigma = kernel_size // 4
                kernels[i] = torch.exp(-((torch.arange(kernel_size, dtype=torch.float) - center) ** 2) / (2 * sigma ** 2))
            elif i == 2:  # Oscillatory
                freq = 0.5
                kernels[i] = torch.sin(2 * math.pi * freq * torch.arange(kernel_size, dtype=torch.float) / kernel_size)
            elif i == 3:  # Ramp
                kernels[i] = torch.arange(kernel_size, dtype=torch.float) / kernel_size
            else:  # Delta (immediate)
                kernels[i, 0] = 1.0
        
        return kernels
    
    def forward(
        self,
        input_spikes: torch.Tensor,
        current_time: float,
        dt: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode temporal patterns in spike trains.
        
        Args:
            input_spikes: Input spike trains [batch_size, input_size]
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Tuple of (pattern_activations, encoding_info)
        """
        if dt is None:
            dt = 0.001
            
        batch_size = input_spikes.size(0)
        
        # Update spike history
        hist_idx = self.history_index.item() % self.spike_history.size(0)
        self.spike_history[hist_idx] = input_spikes[0]  # Use first batch for simplicity
        self.time_history[hist_idx] = current_time
        self.history_index += 1
        
        # Extract recent spike patterns
        pattern_activations = torch.zeros(batch_size, self.pattern_size)
        
        # Simple pattern matching based on recent activity
        recent_window = min(self.config.max_pattern_length, self.spike_history.size(0))
        recent_spikes = self.spike_history[-recent_window:]  # [window, input_size]
        
        # Calculate pattern similarities using correlation kernels
        for pattern_idx in range(self.pattern_size):
            pattern_template = self.pattern_templates[pattern_idx, :recent_window]
            
            # Calculate correlation across different temporal kernels
            correlations = []
            for kernel_idx in range(self.correlation_kernels.size(0)):
                kernel = self.correlation_kernels[kernel_idx, :recent_window]
                
                # Weighted correlation with kernel
                weighted_recent = recent_spikes * kernel.unsqueeze(-1)
                weighted_template = pattern_template * kernel.unsqueeze(-1)
                
                # Cosine similarity
                recent_norm = torch.norm(weighted_recent, dim=0) + 1e-6
                template_norm = torch.norm(weighted_template, dim=0) + 1e-6
                
                correlation = torch.sum(weighted_recent * weighted_template, dim=0) / (recent_norm * template_norm)
                correlations.append(correlation.mean())  # Average across input dimensions
            
            # Combine correlations from different kernels
            pattern_strength = torch.stack(correlations).mean()
            pattern_activations[:, pattern_idx] = pattern_strength
        
        # Apply pattern weights for learned responses
        weighted_patterns = torch.matmul(input_spikes, self.pattern_weights.t())
        pattern_activations = pattern_activations + weighted_patterns
        
        # Apply nonlinearity and normalization
        pattern_activations = torch.sigmoid(pattern_activations)
        
        encoding_info = {
            'recent_spikes': recent_spikes.clone(),
            'pattern_strengths': pattern_activations.clone(),
            'correlation_values': correlations,
            'spike_history_length': (self.history_index % self.spike_history.size(0)).item()
        }
        
        return pattern_activations, encoding_info


class PhaseEncoder(nn.Module):
    """
    Phase-based encoding using oscillatory reference signals.
    
    Encodes information in the phase relationship between spikes
    and underlying oscillations (e.g., gamma, theta rhythms).
    """
    
    def __init__(
        self,
        input_size: int,
        config: TemporalConfig
    ):
        super().__init__()
        
        self.input_size = input_size
        self.config = config
        
        # Phase tracking for each input
        self.register_buffer('current_phases', torch.zeros(input_size))
        self.register_buffer('phase_bins', torch.zeros(input_size, config.phase_resolution))
        self.register_buffer('reference_oscillator', torch.tensor(0.0))
        
        # Spike timing relative to phase
        self.register_buffer('last_spike_phases', torch.full((input_size,), -float('inf')))
        
        logger.debug(f"Created phase encoder for {input_size} inputs")
    
    def forward(
        self,
        input_spikes: torch.Tensor,
        current_time: float,
        dt: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode spike timing relative to phase.
        
        Args:
            input_spikes: Input spikes [batch_size, input_size]
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Tuple of (phase_encoded_output, phase_info)
        """
        if dt is None:
            dt = 0.001
            
        batch_size = input_spikes.size(0)
        
        # Update reference oscillator
        omega = 2 * math.pi * self.config.reference_frequency
        self.reference_oscillator = (self.reference_oscillator + omega * dt) % (2 * math.pi)
        
        # Calculate current phase for each input (with slight variations)
        phase_offsets = torch.linspace(0, math.pi/4, self.input_size)  # Slight phase differences
        self.current_phases = (self.reference_oscillator + phase_offsets) % (2 * math.pi)
        
        # Process spikes and their phases
        spike_mask = input_spikes[0] > 0  # Use first batch
        
        # Update phase bins for spiking inputs
        for i in range(self.input_size):
            if spike_mask[i]:
                current_phase = self.current_phases[i]
                
                # Determine phase bin
                phase_bin = int((current_phase / (2 * math.pi)) * self.config.phase_resolution)
                phase_bin = min(phase_bin, self.config.phase_resolution - 1)
                
                # Update phase bin activation
                self.phase_bins[i, phase_bin] += 1.0
                self.last_spike_phases[i] = current_phase
        
        # Decay phase bins
        self.phase_bins *= 0.95
        
        # Generate phase-encoded output
        phase_encoded = torch.zeros(batch_size, self.input_size * self.config.phase_resolution)
        
        # Flatten phase bins for output
        for b in range(batch_size):
            phase_encoded[b] = self.phase_bins.flatten()
        
        # Apply phase preference weights
        phase_weights = torch.cos(self.current_phases.unsqueeze(-1) - 
                                torch.linspace(0, 2*math.pi, self.config.phase_resolution))
        
        weighted_output = phase_weights * self.phase_bins
        
        phase_info = {
            'current_phases': self.current_phases.clone(),
            'phase_bins': self.phase_bins.clone(),
            'reference_phase': self.reference_oscillator.clone(),
            'last_spike_phases': self.last_spike_phases.clone(),
            'phase_weights': phase_weights.clone()
        }
        
        return phase_encoded, phase_info


class OscillatoryDynamics(nn.Module):
    """
    Multi-frequency oscillatory dynamics for neural synchronization.
    
    Implements multiple coupled oscillators representing different
    frequency bands (theta, alpha, beta, gamma) with cross-frequency coupling.
    """
    
    def __init__(
        self,
        num_oscillators: int,
        config: TemporalConfig
    ):
        super().__init__()
        
        self.num_oscillators = num_oscillators
        self.config = config
        
        # Set default frequencies if not provided
        if config.oscillation_frequencies is None:
            config.oscillation_frequencies = [8.0, 13.0, 30.0, 80.0]  # Theta, Alpha, Beta, Gamma
        
        self.frequencies = config.oscillation_frequencies[:num_oscillators]
        
        # Oscillator states (phase and amplitude)
        self.register_buffer('phases', torch.zeros(num_oscillators))
        self.register_buffer('amplitudes', torch.ones(num_oscillators))
        
        # Coupling matrix between oscillators
        self.register_parameter('coupling_matrix', 
                              nn.Parameter(torch.randn(num_oscillators, num_oscillators) * config.coupling_strength))
        
        # Oscillator outputs
        self.register_buffer('oscillator_outputs', torch.zeros(num_oscillators))
        
        logger.debug(f"Created oscillatory dynamics with {num_oscillators} oscillators")
    
    def forward(
        self,
        external_input: Optional[torch.Tensor] = None,
        current_time: float = 0.0,
        dt: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Update oscillatory dynamics.
        
        Args:
            external_input: External input to oscillators [batch_size, num_oscillators]
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Tuple of (oscillator_outputs, oscillatory_info)
        """
        if dt is None:
            dt = 0.001
            
        batch_size = 1
        if external_input is not None:
            batch_size = external_input.size(0)
            external_input = external_input[0]  # Use first batch
        else:
            external_input = torch.zeros(self.num_oscillators)
        
        # Update phases based on natural frequencies
        for i, freq in enumerate(self.frequencies):
            omega = 2 * math.pi * freq
            self.phases[i] = (self.phases[i] + omega * dt) % (2 * math.pi)
        
        # Calculate coupling effects
        coupling_effects = torch.zeros(self.num_oscillators)
        for i in range(self.num_oscillators):
            for j in range(self.num_oscillators):
                if i != j:
                    # Phase difference coupling
                    phase_diff = self.phases[j] - self.phases[i]
                    coupling_force = self.coupling_matrix[i, j] * torch.sin(phase_diff)
                    coupling_effects[i] += coupling_force
        
        # Update phases with coupling
        self.phases += coupling_effects * dt
        self.phases = self.phases % (2 * math.pi)
        
        # Update amplitudes based on external input
        amplitude_decay = 0.01  # Natural decay
        self.amplitudes = self.amplitudes * (1 - amplitude_decay * dt) + external_input * dt
        self.amplitudes = torch.clamp(self.amplitudes, 0.1, 2.0)  # Bounds
        
        # Generate oscillator outputs
        for i in range(self.num_oscillators):
            self.oscillator_outputs[i] = self.amplitudes[i] * torch.sin(self.phases[i])
        
        # Cross-frequency phase coupling (phase-amplitude coupling)
        pac_outputs = torch.zeros(self.num_oscillators)
        for i in range(self.num_oscillators - 1):
            # Higher frequency amplitude modulated by lower frequency phase
            low_freq_phase = self.phases[i]
            high_freq_amp = self.amplitudes[i + 1]
            
            # Phase-amplitude coupling strength
            pac_strength = 0.1 * torch.cos(low_freq_phase) 
            pac_outputs[i + 1] = high_freq_amp * (1 + pac_strength)
        
        # Combine outputs
        combined_output = self.oscillator_outputs + pac_outputs
        
        # Expand for batch
        batch_output = combined_output.unsqueeze(0).expand(batch_size, -1)
        
        oscillatory_info = {
            'phases': self.phases.clone(),
            'amplitudes': self.amplitudes.clone(),
            'frequencies': torch.tensor(self.frequencies),
            'coupling_effects': coupling_effects.clone(),
            'pac_outputs': pac_outputs.clone(),
            'raw_outputs': self.oscillator_outputs.clone()
        }
        
        return batch_output, oscillatory_info


class SparseDistributedRepresentation(nn.Module):
    """
    Sparse distributed representation with competitive dynamics.
    
    Implements sparse coding principles with lateral inhibition
    to create distributed but sparse neural representations.
    """
    
    def __init__(
        self,
        input_size: int,
        representation_size: int,
        config: TemporalConfig
    ):
        super().__init__()
        
        self.input_size = input_size
        self.representation_size = representation_size
        self.config = config
        
        # Encoding weights (input -> representation)
        self.encoding_weights = nn.Parameter(torch.randn(input_size, representation_size) * 0.1)
        
        # Lateral inhibition weights
        self.lateral_weights = nn.Parameter(
            torch.randn(representation_size, representation_size) * config.lateral_inhibition
        )
        # Zero diagonal (no self-inhibition)
        self.register_buffer('inhibition_mask', torch.eye(representation_size) == 0)
        
        # Adaptive thresholds for sparsity control
        self.register_buffer('adaptive_thresholds', torch.ones(representation_size))
        
        # Activity tracking for sparsity adaptation
        self.register_buffer('activity_history', torch.zeros(representation_size, 100))
        self.register_buffer('history_index', torch.tensor(0, dtype=torch.long))
        
        logger.debug(f"Created sparse distributed representation: {input_size} -> {representation_size}")
    
    def forward(
        self,
        input_activation: torch.Tensor,
        dt: Optional[float] = None,
        adapt_sparsity: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate sparse distributed representation.
        
        Args:
            input_activation: Input activations [batch_size, input_size]
            dt: Time step
            adapt_sparsity: Whether to adapt sparsity levels
            
        Returns:
            Tuple of (sparse_representation, sparsity_info)
        """
        if dt is None:
            dt = 0.001
            
        batch_size = input_activation.size(0)
        
        # Forward encoding
        raw_activations = torch.matmul(input_activation, self.encoding_weights)
        
        # Apply lateral inhibition iteratively
        current_activations = raw_activations.clone()
        
        # Competitive dynamics (multiple iterations)
        num_iterations = 5
        for _ in range(num_iterations):
            # Calculate lateral inhibition
            inhibition = torch.matmul(current_activations, 
                                    self.lateral_weights * self.inhibition_mask)
            
            # Update activations with inhibition
            current_activations = raw_activations - inhibition
            
            # Apply adaptive thresholds
            current_activations = torch.relu(current_activations - self.adaptive_thresholds)
        
        # Apply winner-take-all or k-winners-take-all for sparsity
        target_active = int(self.representation_size * self.config.sparsity_target)
        
        sparse_representation = torch.zeros_like(current_activations)
        for b in range(batch_size):
            # Find top-k activations
            _, top_indices = torch.topk(current_activations[b], target_active)
            sparse_representation[b, top_indices] = current_activations[b, top_indices]
        
        if adapt_sparsity:
            # Update activity history
            current_activity = sparse_representation.mean(dim=0)  # Average across batch
            hist_idx = self.history_index.item() % self.activity_history.size(1)
            self.activity_history[:, hist_idx] = current_activity
            self.history_index += 1
            
            # Adapt thresholds to maintain target sparsity
            recent_activity = self.activity_history.mean(dim=1)
            target_activity = self.config.sparsity_target
            
            # Homeostatic threshold adaptation
            activity_error = recent_activity - target_activity
            threshold_update = self.config.adaptation_rate * activity_error * dt
            
            self.adaptive_thresholds = self.adaptive_thresholds + threshold_update
            self.adaptive_thresholds = torch.clamp(self.adaptive_thresholds, 0.01, 2.0)
        
        # Calculate actual sparsity
        actual_sparsity = (sparse_representation > 0).float().mean()
        
        sparsity_info = {
            'raw_activations': raw_activations.clone(),
            'sparse_representation': sparse_representation.clone(),
            'adaptive_thresholds': self.adaptive_thresholds.clone(),
            'actual_sparsity': actual_sparsity,
            'target_sparsity': self.config.sparsity_target,
            'activity_history': self.activity_history.clone(),
            'num_active': (sparse_representation > 0).sum(dim=-1).float()
        }
        
        return sparse_representation, sparsity_info