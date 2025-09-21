"""
Real-time sensory processing pipeline using 3rd generation neuromorphic principles.

This module implements multi-modal sensory processing pipelines with temporal
coding, oscillatory dynamics, and adaptive processing for real-time applications.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..core.neuromorphic import NeuromorphicConfig
from ..core.neuromorphic_v3 import (
    OscillatoryDynamics,
    PhaseEncoder,
    PopulationLayer,
    SparseDistributedRepresentation,
    TemporalPatternEncoder,
)
from ..core.neuromorphic_v3.advanced_neurons import NeuronV3Config
from ..core.neuromorphic_v3.temporal_coding import TemporalConfig

logger = logging.getLogger(__name__)


@dataclass
class SensoryConfig:
    """Configuration for sensory processing pipeline."""

    # Sensory modalities
    modalities: List[str] = None  # ['vision', 'audio', 'tactile']

    # Input dimensions per modality
    vision_input_size: int = 784  # 28x28 for MNIST-like
    audio_input_size: int = 256  # Frequency bins
    tactile_input_size: int = 64  # Tactile sensors

    # Processing parameters
    temporal_window: float = 0.1  # 100ms processing window
    sampling_rate: float = 1000.0  # Hz
    real_time_buffer_size: int = 100

    # Neuromorphic parameters
    enable_oscillatory_processing: bool = True
    enable_temporal_encoding: bool = True
    enable_cross_modal_binding: bool = True
    enable_adaptive_filtering: bool = True

    # Oscillatory frequencies for different modalities
    vision_frequencies: List[float] = None  # Alpha, Beta, Gamma
    audio_frequencies: List[float] = None  # Theta, Alpha, Beta, Gamma
    tactile_frequencies: List[float] = None  # Delta, Theta, Alpha

    # Sparsity and efficiency
    sparse_coding_target: float = 0.05
    adaptation_rate: float = 0.01

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["vision", "audio"]

        if self.vision_frequencies is None:
            self.vision_frequencies = [10.0, 20.0, 40.0, 80.0]  # Alpha, Beta, Gamma

        if self.audio_frequencies is None:
            self.audio_frequencies = [4.0, 8.0, 16.0, 32.0, 64.0]  # Theta to Gamma

        if self.tactile_frequencies is None:
            self.tactile_frequencies = [2.0, 6.0, 12.0]  # Delta, Theta, Alpha


class SensoryPreprocessor(nn.Module):
    """
    Neuromorphic sensory preprocessing with adaptive filtering.

    Converts raw sensory input to spike trains with adaptive
    thresholding and temporal dynamics.
    """

    def __init__(self, input_size: int, modality: str, config: SensoryConfig):
        super().__init__()

        self.input_size = input_size
        self.modality = modality
        self.config = config

        # Adaptive threshold neurons for preprocessing
        base_config = NeuromorphicConfig(generation=3)
        neuron_config = NeuronV3Config(
            base_config=base_config,
            threshold_adaptation_rate=config.adaptation_rate,
            target_spike_rate=50.0,  # Moderate activity for preprocessing
        )

        # Population of adaptive neurons
        self.adaptive_population = PopulationLayer(
            population_size=input_size,
            neuron_type="adaptive_threshold",
            neuron_config=neuron_config,
            lateral_inhibition=True,
            inhibition_strength=0.1,
        )

        # Temporal buffer for real-time processing
        self.register_buffer("input_history", torch.zeros(config.real_time_buffer_size, input_size))
        self.register_buffer("buffer_index", torch.tensor(0, dtype=torch.long))

        # Adaptive gain control
        self.register_buffer("gain_factors", torch.ones(input_size))
        self.register_buffer("activity_tracker", torch.zeros(input_size))

        logger.debug(f"Initialized {modality} preprocessor: {input_size} channels")

    def forward(
        self, raw_input: torch.Tensor, current_time: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Preprocess raw sensory input to spike trains."""
        batch_size = raw_input.size(0)

        # Update input history buffer
        buffer_idx = self.buffer_index.item() % self.config.real_time_buffer_size
        self.input_history[buffer_idx] = raw_input[0]  # Store first batch item
        self.buffer_index += 1

        # Adaptive gain control
        if self.config.enable_adaptive_filtering:
            # Track activity levels
            activity = torch.abs(raw_input).mean(dim=0)
            self.activity_tracker = 0.9 * self.activity_tracker + 0.1 * activity

            # Adjust gain to maintain target activity
            target_activity = 0.5
            gain_adjustment = target_activity / (self.activity_tracker + 1e-6)
            self.gain_factors = torch.clamp(gain_adjustment, 0.1, 10.0)

            # Apply adaptive gain
            processed_input = raw_input * self.gain_factors.unsqueeze(0)
        else:
            processed_input = raw_input

        # Convert to spikes through adaptive population
        spike_output, population_states = self.adaptive_population(processed_input, current_time)

        # Add temporal dynamics (edge detection)
        if self.buffer_index > 1:
            prev_input = self.input_history[(buffer_idx - 1) % self.config.real_time_buffer_size]
            temporal_diff = processed_input[0] - prev_input

            # Enhance spikes for temporal changes
            temporal_enhancement = torch.sigmoid(temporal_diff * 5.0)
            spike_output = spike_output + temporal_enhancement.unsqueeze(0) * 0.3

        preprocessing_info = {
            "gain_factors": self.gain_factors.clone(),
            "activity_levels": self.activity_tracker.clone(),
            "spike_rate": spike_output.mean(),
            "temporal_enhancement": (
                temporal_enhancement if self.buffer_index > 1 else torch.zeros_like(spike_output[0])
            ),
        }

        return spike_output, preprocessing_info


class ModalityProcessor(nn.Module):
    """
    Modality-specific processor with oscillatory dynamics and temporal encoding.

    Processes preprocessed spikes through modality-specific neural populations
    with appropriate oscillatory frequencies and temporal patterns.
    """

    def __init__(self, input_size: int, feature_size: int, modality: str, config: SensoryConfig):
        super().__init__()

        self.input_size = input_size
        self.feature_size = feature_size
        self.modality = modality
        self.config = config

        # Get modality-specific frequencies
        if modality == "vision":
            frequencies = config.vision_frequencies
        elif modality == "audio":
            frequencies = config.audio_frequencies
        elif modality == "tactile":
            frequencies = config.tactile_frequencies
        else:
            frequencies = [10.0, 20.0, 40.0]  # Default frequencies

        # Oscillatory dynamics for modality
        if config.enable_oscillatory_processing:
            temporal_config = TemporalConfig(
                oscillation_frequencies=frequencies, coupling_strength=0.1
            )

            self.oscillatory_system = OscillatoryDynamics(
                num_oscillators=len(frequencies), config=temporal_config
            )
        else:
            self.oscillatory_system = None

        # Temporal pattern encoder
        if config.enable_temporal_encoding:
            temporal_config = TemporalConfig(
                pattern_window=config.temporal_window, max_pattern_length=10
            )

            self.temporal_encoder = TemporalPatternEncoder(
                input_size, feature_size // 2, temporal_config
            )
        else:
            self.temporal_encoder = None

        # Phase encoder for oscillatory alignment
        if config.enable_oscillatory_processing:
            self.phase_encoder = PhaseEncoder(input_size, temporal_config)
        else:
            self.phase_encoder = None

        # Feature extraction layers
        base_config = NeuromorphicConfig(generation=3)
        neuron_config = NeuronV3Config(
            base_config=base_config, threshold_adaptation_rate=0.05, target_spike_rate=30.0
        )

        self.feature_layers = nn.ModuleList(
            [
                PopulationLayer(
                    population_size=feature_size,
                    neuron_type="adaptive_threshold",
                    neuron_config=neuron_config,
                    lateral_inhibition=True,
                    inhibition_strength=0.2,
                )
            ]
        )

        # Feature projection
        self.feature_projection = nn.Linear(input_size, feature_size)

        # Sparse coding for efficiency
        temporal_config = TemporalConfig(sparsity_target=config.sparse_coding_target)
        self.sparse_encoder = SparseDistributedRepresentation(
            feature_size, feature_size, temporal_config
        )

        logger.debug(f"Initialized {modality} processor: {input_size} -> {feature_size}")

    def forward(
        self, spike_input: torch.Tensor, current_time: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process modality-specific spike patterns."""
        if current_time is None:
            current_time = 0.001

        batch_size = spike_input.size(0)
        processing_info = {}

        # Oscillatory modulation
        if self.oscillatory_system is not None:
            # Use input activity to drive oscillations
            oscillatory_input = spike_input.mean(dim=0).unsqueeze(0)
            oscillations, osc_info = self.oscillatory_system(
                external_input=oscillatory_input, current_time=current_time
            )

            # Modulate input with oscillatory activity
            osc_modulation = oscillations.unsqueeze(0).expand(batch_size, -1)
            modulated_input = spike_input + osc_modulation[:, : spike_input.size(1)] * 0.1

            processing_info["oscillatory_info"] = osc_info
        else:
            modulated_input = spike_input

        # Temporal pattern encoding
        if self.temporal_encoder is not None:
            temporal_patterns, temporal_info = self.temporal_encoder(modulated_input, current_time)
            processing_info["temporal_info"] = temporal_info
        else:
            temporal_patterns = modulated_input

        # Phase encoding for oscillatory alignment
        if self.phase_encoder is not None:
            phase_encoded, phase_info = self.phase_encoder(modulated_input, current_time)
            processing_info["phase_info"] = phase_info

            # Combine temporal and phase information
            if self.temporal_encoder is not None:
                combined_features = torch.cat(
                    [temporal_patterns, phase_encoded[:, : temporal_patterns.size(1)]], dim=1
                )
            else:
                combined_features = phase_encoded
        else:
            combined_features = temporal_patterns

        # Project to feature space
        projected_features = torch.relu(self.feature_projection(combined_features))

        # Process through feature extraction layers
        for layer in self.feature_layers:
            projected_features, layer_states = layer(projected_features, current_time)

        # Apply sparse coding
        sparse_features, sparsity_info = self.sparse_encoder(projected_features)
        processing_info["sparsity_info"] = sparsity_info

        return sparse_features, processing_info


class CrossModalIntegration(nn.Module):
    """
    Cross-modal integration using temporal binding and attention mechanisms.

    Integrates information from multiple sensory modalities using
    temporal synchronization and attention-based binding.
    """

    def __init__(self, modality_dims: Dict[str, int], integration_dim: int, config: SensoryConfig):
        super().__init__()

        self.modality_dims = modality_dims
        self.integration_dim = integration_dim
        self.config = config

        # Cross-modal attention mechanism
        self.cross_attention = nn.ModuleDict()

        for modality in modality_dims:
            self.cross_attention[modality] = nn.MultiheadAttention(
                embed_dim=modality_dims[modality], num_heads=4, batch_first=True
            )

        # Temporal synchronization
        if config.enable_cross_modal_binding:
            # Realistic delays between modalities
            self.modality_delays = nn.ModuleDict()

            for modality in modality_dims:
                # Audio is typically faster than vision
                if modality == "audio":
                    delay_range = (0.001, 0.005)  # 1-5ms
                elif modality == "vision":
                    delay_range = (0.020, 0.050)  # 20-50ms
                elif modality == "tactile":
                    delay_range = (0.010, 0.030)  # 10-30ms
                else:
                    delay_range = (0.005, 0.020)  # Default

                # Simple delay implementation
                self.modality_delays[modality] = nn.Parameter(
                    torch.tensor((delay_range[0] + delay_range[1]) / 2), requires_grad=False
                )

        # Integration network
        total_input_dim = sum(modality_dims.values())

        self.integration_network = nn.Sequential(
            nn.Linear(total_input_dim, integration_dim * 2),
            nn.ReLU(),
            nn.Linear(integration_dim * 2, integration_dim),
            nn.ReLU(),
        )

        # Temporal binding through oscillatory synchronization
        if config.enable_oscillatory_processing:
            temporal_config = TemporalConfig(
                oscillation_frequencies=[4.0, 8.0, 16.0, 32.0],  # Cross-modal frequencies
                coupling_strength=0.2,
            )

            self.binding_oscillations = OscillatoryDynamics(
                num_oscillators=4, config=temporal_config
            )
        else:
            self.binding_oscillations = None

        # Attention weights for modality importance
        self.modality_attention = nn.Parameter(torch.ones(len(modality_dims)) / len(modality_dims))

        logger.debug(f"Initialized cross-modal integration: {modality_dims} -> {integration_dim}")

    def forward(
        self, modality_features: Dict[str, torch.Tensor], current_time: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Integrate features from multiple sensory modalities."""
        if current_time is None:
            current_time = 0.001

        integration_info = {}

        # Apply cross-modal attention
        attended_features = {}
        attention_weights = {}

        modality_list = list(modality_features.keys())

        for i, source_modality in enumerate(modality_list):
            source_features = modality_features[source_modality]

            # Compute attention with other modalities
            attended_feature_list = []
            attention_weight_list = []

            for target_modality in modality_list:
                if source_modality != target_modality:
                    target_features = modality_features[target_modality]

                    # Cross-attention between modalities
                    attended, attn_weights = self.cross_attention[source_modality](
                        source_features.unsqueeze(1),  # Add sequence dimension
                        target_features.unsqueeze(1),
                        target_features.unsqueeze(1),
                    )

                    attended_feature_list.append(attended.squeeze(1))
                    attention_weight_list.append(attn_weights.squeeze(1))

            if attended_feature_list:
                # Combine attended features
                attended_features[source_modality] = torch.stack(attended_feature_list).mean(dim=0)
                attention_weights[source_modality] = torch.stack(attention_weight_list).mean(dim=0)
            else:
                attended_features[source_modality] = source_features
                attention_weights[source_modality] = torch.ones_like(source_features[:, :1])

        integration_info["attention_weights"] = attention_weights

        # Temporal synchronization through binding oscillations
        if self.binding_oscillations is not None:
            # Use combined modality activity to drive binding oscillations
            combined_activity = torch.stack(
                [feat.mean(dim=1) for feat in attended_features.values()]
            ).mean(dim=0)

            binding_signals, binding_info = self.binding_oscillations(
                external_input=combined_activity.unsqueeze(1), current_time=current_time
            )

            # Modulate features with binding signals
            for modality in attended_features:
                binding_modulation = binding_signals[0, : attended_features[modality].size(1)]
                attended_features[modality] = attended_features[modality] + binding_modulation * 0.1

            integration_info["binding_info"] = binding_info

        # Apply modality importance weights
        normalized_attention = torch.softmax(self.modality_attention, dim=0)

        weighted_features = []
        for i, modality in enumerate(modality_list):
            weight = normalized_attention[i]
            weighted_feature = attended_features[modality] * weight
            weighted_features.append(weighted_feature)

        # Concatenate all modality features
        concatenated_features = torch.cat(weighted_features, dim=1)

        # Final integration through network
        integrated_features = self.integration_network(concatenated_features)

        integration_info.update(
            {
                "modality_weights": normalized_attention.detach(),
                "integration_sparsity": (integrated_features > 0).float().mean().item(),
                "cross_modal_coherence": self._compute_coherence(attended_features),
            }
        )

        return integrated_features, integration_info

    def _compute_coherence(self, modality_features: Dict[str, torch.Tensor]) -> float:
        """Compute cross-modal coherence measure."""
        if len(modality_features) < 2:
            return 1.0

        modality_list = list(modality_features.keys())
        coherence_scores = []

        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                feat_i = modality_features[modality_list[i]]
                feat_j = modality_features[modality_list[j]]

                # Compute normalized cross-correlation
                feat_i_norm = feat_i / (torch.norm(feat_i, dim=1, keepdim=True) + 1e-6)
                feat_j_norm = feat_j / (torch.norm(feat_j, dim=1, keepdim=True) + 1e-6)

                # Truncate to same size
                min_size = min(feat_i_norm.size(1), feat_j_norm.size(1))
                correlation = torch.sum(
                    feat_i_norm[:, :min_size] * feat_j_norm[:, :min_size], dim=1
                )
                coherence_scores.append(correlation.mean().item())

        return np.mean(coherence_scores) if coherence_scores else 0.0


class SensoryProcessingPipeline(nn.Module):
    """
    Complete real-time sensory processing pipeline using 3rd generation neuromorphic principles.

    Integrates preprocessing, modality-specific processing, and cross-modal integration
    for real-time multi-modal sensory processing applications.
    """

    def __init__(self, config: SensoryConfig):
        super().__init__()

        self.config = config
        self.modalities = config.modalities

        # Modality-specific preprocessors
        self.preprocessors = nn.ModuleDict()
        self.processors = nn.ModuleDict()

        modality_dims = {}

        for modality in config.modalities:
            if modality == "vision":
                input_size = config.vision_input_size
                feature_size = 128
            elif modality == "audio":
                input_size = config.audio_input_size
                feature_size = 96
            elif modality == "tactile":
                input_size = config.tactile_input_size
                feature_size = 64
            else:
                input_size = 100  # Default
                feature_size = 80

            self.preprocessors[modality] = SensoryPreprocessor(input_size, modality, config)

            self.processors[modality] = ModalityProcessor(
                input_size, feature_size, modality, config
            )

            modality_dims[modality] = feature_size

        # Cross-modal integration
        if config.enable_cross_modal_binding and len(config.modalities) > 1:
            integration_dim = sum(modality_dims.values()) // 2

            self.cross_modal_integration = CrossModalIntegration(
                modality_dims, integration_dim, config
            )
        else:
            self.cross_modal_integration = None

        # Processing statistics
        self.processing_stats = {
            "frame_count": 0,
            "total_latency": 0.0,
            "modality_activities": {mod: deque(maxlen=100) for mod in config.modalities},
        }

        logger.info(f"Initialized sensory processing pipeline: {config.modalities}")

    def forward(
        self, sensory_inputs: Dict[str, torch.Tensor], current_time: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process multi-modal sensory inputs in real-time."""
        if current_time is None:
            current_time = self.processing_stats["frame_count"] / self.config.sampling_rate

        processing_info = {
            "preprocessing": {},
            "modality_processing": {},
            "cross_modal": {},
            "timing": {},
        }

        # Process each modality
        modality_features = {}

        for modality in self.modalities:
            if modality in sensory_inputs:
                raw_input = sensory_inputs[modality]

                # Preprocessing
                spikes, preprocess_info = self.preprocessors[modality](raw_input, current_time)
                processing_info["preprocessing"][modality] = preprocess_info

                # Modality-specific processing
                features, modality_info = self.processors[modality](spikes, current_time)
                modality_features[modality] = features
                processing_info["modality_processing"][modality] = modality_info

                # Update activity tracking
                activity = features.abs().mean().item()
                self.processing_stats["modality_activities"][modality].append(activity)

        # Cross-modal integration
        if self.cross_modal_integration is not None and len(modality_features) > 1:
            integrated_features, integration_info = self.cross_modal_integration(
                modality_features, current_time
            )
            processing_info["cross_modal"] = integration_info
        else:
            # Simple concatenation if no cross-modal integration
            if modality_features:
                integrated_features = torch.cat(list(modality_features.values()), dim=1)
            else:
                integrated_features = torch.zeros(1, 100)  # Fallback

        # Update processing statistics
        self.processing_stats["frame_count"] += 1

        # Compute processing latency (simulated)
        processing_latency = (
            len(self.modalities) * 0.005 + 0.002
        )  # 5ms per modality + 2ms integration
        self.processing_stats["total_latency"] += processing_latency

        processing_info["timing"] = {
            "frame_count": self.processing_stats["frame_count"],
            "processing_latency_ms": processing_latency * 1000,
            "average_latency_ms": (
                self.processing_stats["total_latency"] / self.processing_stats["frame_count"]
            )
            * 1000,
            "real_time_factor": (current_time / (self.processing_stats["total_latency"] + 1e-6)),
        }

        return integrated_features, processing_info

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        stats = {
            "frame_count": self.processing_stats["frame_count"],
            "average_latency_ms": 0.0,
            "modality_activities": {},
            "processing_efficiency": {},
        }

        if self.processing_stats["frame_count"] > 0:
            stats["average_latency_ms"] = (
                self.processing_stats["total_latency"] / self.processing_stats["frame_count"]
            ) * 1000

        # Modality activity statistics
        for modality, activities in self.processing_stats["modality_activities"].items():
            if activities:
                stats["modality_activities"][modality] = {
                    "mean": np.mean(activities),
                    "std": np.std(activities),
                    "min": np.min(activities),
                    "max": np.max(activities),
                }

        # Processing efficiency metrics
        target_framerate = self.config.sampling_rate
        actual_framerate = self.processing_stats["frame_count"] / (
            self.processing_stats["total_latency"] + 1e-6
        )

        stats["processing_efficiency"] = {
            "target_framerate_hz": target_framerate,
            "actual_framerate_hz": actual_framerate,
            "efficiency_ratio": actual_framerate / target_framerate,
            "real_time_capable": actual_framerate >= target_framerate * 0.9,
        }

        return stats

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "frame_count": 0,
            "total_latency": 0.0,
            "modality_activities": {mod: deque(maxlen=100) for mod in self.config.modalities},
        }

    def configure_for_real_time(self, target_latency_ms: float = 10.0) -> None:
        """Configure pipeline for real-time processing constraints."""
        logger.info(f"Configuring for real-time processing: target latency {target_latency_ms}ms")

        # Reduce buffer sizes for lower latency
        reduced_buffer_size = max(10, self.config.real_time_buffer_size // 4)

        for preprocessor in self.preprocessors.values():
            if hasattr(preprocessor, "input_history"):
                preprocessor.input_history = preprocessor.input_history[:reduced_buffer_size]

        # Reduce sparse coding targets for faster processing
        for processor in self.processors.values():
            if hasattr(processor, "sparse_encoder"):
                processor.sparse_encoder.config.sparsity_target *= 0.5  # More aggressive sparsity

        logger.info("Real-time configuration applied")
