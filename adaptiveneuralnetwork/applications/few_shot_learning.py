"""
Few-shot learning implementation using 3rd generation neuromorphic principles.

This module implements few-shot learning using neuromorphic mechanisms like
rapid synaptic plasticity, temporal pattern encoding, and sparse representations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging

from ..core.neuromorphic_v3 import (
    STDPSynapse, MetaplasticitySynapse, TemporalPatternEncoder,
    AdaptiveThresholdNeuron, SparseDistributedRepresentation,
    PopulationLayer
)
from ..core.neuromorphic_v3.advanced_neurons import NeuronV3Config
from ..core.neuromorphic_v3.plasticity import STDPConfig, MetaplasticityConfig
from ..core.neuromorphic_v3.temporal_coding import TemporalConfig
from ..core.neuromorphic import NeuromorphicConfig

logger = logging.getLogger(__name__)


@dataclass
class FewShotLearningConfig:
    """Configuration for few-shot learning system."""
    # Problem setup
    n_way: int = 5  # Number of classes per episode
    k_shot: int = 1  # Number of examples per class
    query_size: int = 15  # Number of query examples per class
    
    # Network architecture
    input_size: int = 784
    feature_dim: int = 128
    memory_dim: int = 256
    
    # Learning parameters
    fast_adaptation_steps: int = 5
    fast_learning_rate: float = 0.01
    meta_learning_rate: float = 0.001
    
    # Neuromorphic parameters
    enable_rapid_plasticity: bool = True
    enable_temporal_encoding: bool = True
    enable_sparse_memory: bool = True
    
    # STDP parameters for rapid adaptation
    stdp_strength: float = 0.1
    stdp_window: float = 0.02  # 20ms window
    
    # Memory parameters
    memory_capacity: int = 1000
    memory_sparsity: float = 0.1


class PrototypicalMemory(nn.Module):
    """
    Prototypical memory using sparse distributed representations.
    
    Stores class prototypes in a neuromorphic memory system with
    sparse activation patterns for efficient storage and retrieval.
    """
    
    def __init__(
        self,
        feature_dim: int,
        memory_dim: int,
        temporal_config: TemporalConfig
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        
        # Sparse memory encoder
        self.memory_encoder = SparseDistributedRepresentation(
            feature_dim, memory_dim, temporal_config
        )
        
        # Prototype storage
        self.register_buffer('prototypes', torch.zeros(0, memory_dim))
        self.register_buffer('prototype_labels', torch.zeros(0, dtype=torch.long))
        
        # Attention mechanism for prototype retrieval
        self.attention = nn.MultiheadAttention(memory_dim, num_heads=8, batch_first=True)
        
        logger.debug(f"Initialized prototypical memory: {feature_dim} -> {memory_dim}")
    
    def encode_prototype(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features into sparse prototypical representation."""
        sparse_features, _ = self.memory_encoder(features)
        return sparse_features.mean(dim=0, keepdim=True)  # Average to create prototype
    
    def store_prototype(self, features: torch.Tensor, label: int) -> None:
        """Store a new prototype for a class."""
        prototype = self.encode_prototype(features)
        
        # Check if prototype for this label already exists
        if label in self.prototype_labels:
            # Update existing prototype (running average)
            label_mask = (self.prototype_labels == label)
            existing_prototype = self.prototypes[label_mask]
            updated_prototype = 0.8 * existing_prototype + 0.2 * prototype
            self.prototypes[label_mask] = updated_prototype
        else:
            # Add new prototype
            self.prototypes = torch.cat([self.prototypes, prototype], dim=0)
            new_label = torch.tensor([label], dtype=torch.long, device=self.prototype_labels.device)
            self.prototype_labels = torch.cat([self.prototype_labels, new_label], dim=0)
    
    def retrieve_similar_prototypes(
        self, 
        query_features: torch.Tensor, 
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve most similar prototypes using attention mechanism."""
        if self.prototypes.size(0) == 0:
            return torch.empty(0, self.memory_dim), torch.empty(0, dtype=torch.long), torch.empty(0)
        
        # Encode query
        query_sparse, _ = self.memory_encoder(query_features)
        query_prototype = query_sparse.mean(dim=0, keepdim=True)
        
        # Compute attention weights
        query_expanded = query_prototype.unsqueeze(0)  # [1, 1, memory_dim]
        prototypes_expanded = self.prototypes.unsqueeze(0)  # [1, n_prototypes, memory_dim]
        
        attended_prototypes, attention_weights = self.attention(
            query_expanded, prototypes_expanded, prototypes_expanded
        )
        
        # Get top-k similar prototypes
        attention_scores = attention_weights.squeeze(0).squeeze(0)  # [n_prototypes]
        top_k = min(top_k, self.prototypes.size(0))
        
        _, top_indices = torch.topk(attention_scores, top_k)
        
        similar_prototypes = self.prototypes[top_indices]
        similar_labels = self.prototype_labels[top_indices]
        similarity_scores = attention_scores[top_indices]
        
        return similar_prototypes, similar_labels, similarity_scores
    
    def clear_prototypes(self) -> None:
        """Clear all stored prototypes."""
        self.prototypes = torch.zeros(0, self.memory_dim, device=self.prototypes.device)
        self.prototype_labels = torch.zeros(0, dtype=torch.long, device=self.prototype_labels.device)


class RapidPlasticityModule(nn.Module):
    """
    Rapid synaptic plasticity for fast adaptation using STDP.
    
    Implements fast synaptic changes that enable few-shot learning
    through spike-timing dependent plasticity mechanisms.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: FewShotLearningConfig
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
        # STDP synaptic connections
        stdp_config = STDPConfig(
            a_plus=config.stdp_strength,
            a_minus=config.stdp_strength * 1.2,
            tau_plus=config.stdp_window,
            tau_minus=config.stdp_window,
            multiplicative=True
        )
        
        self.stdp_synapses = STDPSynapse(input_size, output_size, stdp_config)
        
        # Metaplasticity for controlling learning rate
        meta_config = MetaplasticityConfig(
            theta_plus=0.1,
            tau_theta=1.0,  # Fast metaplastic changes
            target_activity=0.2
        )
        
        self.meta_synapses = MetaplasticitySynapse(
            input_size, output_size, stdp_config, meta_config
        )
        
        # Fast adaptation parameters
        self.register_buffer('adaptation_rate', torch.ones(output_size))
        
        logger.debug(f"Initialized rapid plasticity: {input_size} -> {output_size}")
    
    def forward(
        self, 
        input_spikes: torch.Tensor, 
        target_spikes: torch.Tensor,
        learning: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process inputs through rapid plasticity synapses."""
        current_time = 0.001  # Simplified time
        
        # Process through metaplastic synapses
        synaptic_current, plasticity_info = self.meta_synapses(
            input_spikes, target_spikes, current_time, learning=learning
        )
        
        # Update adaptation rates based on metaplastic state
        if learning:
            activity_level = plasticity_info['activity_variable'].mean(dim=0)
            self.adaptation_rate = 1.0 + activity_level  # Boost learning for active neurons
        
        plasticity_info['adaptation_rate'] = self.adaptation_rate.clone()
        
        return synaptic_current, plasticity_info
    
    def rapid_adapt(
        self,
        support_inputs: torch.Tensor,
        support_targets: torch.Tensor,
        num_steps: int = 5
    ) -> Dict[str, float]:
        """Perform rapid adaptation on support set using fast plasticity."""
        adaptation_stats = {
            'weight_changes': 0.0,
            'adaptation_strength': 0.0
        }
        
        initial_weights = self.meta_synapses.stdp_synapse.weights.clone()
        
        # Convert targets to spike patterns
        target_spikes = torch.zeros_like(support_targets).float()
        target_spikes.scatter_(1, support_targets.unsqueeze(1), 1.0)
        
        for step in range(num_steps):
            # Generate input spikes (rate coding)
            input_spikes = (torch.rand_like(support_inputs) < torch.sigmoid(support_inputs)).float()
            
            # Fast plasticity update
            _, plasticity_info = self.forward(input_spikes, target_spikes, learning=True)
            
            # Boost learning rate for rapid adaptation
            weight_update = plasticity_info['weight_changes'] * self.adaptation_rate.unsqueeze(0)
            
            # Apply boosted weight update
            self.meta_synapses.stdp_synapse.weights.data += weight_update * self.config.fast_learning_rate
        
        final_weights = self.meta_synapses.stdp_synapse.weights
        weight_change = torch.norm(final_weights - initial_weights).item()
        
        adaptation_stats['weight_changes'] = weight_change
        adaptation_stats['adaptation_strength'] = self.adaptation_rate.mean().item()
        
        return adaptation_stats


class FewShotLearningSystem(nn.Module):
    """
    Complete few-shot learning system using 3rd generation neuromorphic principles.
    
    Combines temporal pattern encoding, prototypical memory, and rapid plasticity
    for efficient few-shot learning.
    """
    
    def __init__(self, config: FewShotLearningConfig):
        super().__init__()
        
        self.config = config
        
        # Base neuromorphic configuration
        base_config = NeuromorphicConfig(
            generation=3,
            enable_stdp=config.enable_rapid_plasticity,
            enable_temporal_patterns=config.enable_temporal_encoding
        )
        
        # Build feature extraction network
        self._build_feature_extractor(base_config)
        
        # Temporal pattern encoder
        if config.enable_temporal_encoding:
            temporal_config = TemporalConfig(
                pattern_window=0.1,
                max_pattern_length=10
            )
            
            self.temporal_encoder = TemporalPatternEncoder(
                config.input_size, config.feature_dim, temporal_config
            )
        else:
            self.temporal_encoder = None
        
        # Prototypical memory system
        if config.enable_sparse_memory:
            temporal_config = TemporalConfig(sparsity_target=config.memory_sparsity)
            
            self.prototypical_memory = PrototypicalMemory(
                config.feature_dim, config.memory_dim, temporal_config
            )
        else:
            self.prototypical_memory = None
        
        # Rapid plasticity module
        if config.enable_rapid_plasticity:
            self.rapid_plasticity = RapidPlasticityModule(
                config.feature_dim, config.n_way, config
            )
        else:
            self.rapid_plasticity = None
        
        # Classification head
        self.classifier = nn.Linear(config.memory_dim, config.n_way)
        
        logger.info(f"Initialized few-shot learning system: {config.n_way}-way, {config.k_shot}-shot")
    
    def _build_feature_extractor(self, base_config: NeuromorphicConfig) -> None:
        """Build neuromorphic feature extraction network."""
        neuron_config = NeuronV3Config(
            base_config=base_config,
            threshold_adaptation_rate=0.2,  # Fast adaptation
            target_spike_rate=30.0  # Higher activity for feature extraction
        )
        
        # Population layers for feature extraction
        self.feature_layers = nn.ModuleList([
            PopulationLayer(
                population_size=256,
                neuron_type="adaptive_threshold",
                neuron_config=neuron_config,
                lateral_inhibition=True,
                inhibition_strength=0.1
            ),
            PopulationLayer(
                population_size=self.config.feature_dim,
                neuron_type="adaptive_threshold", 
                neuron_config=neuron_config,
                lateral_inhibition=True,
                inhibition_strength=0.15
            )
        ])
        
        # Input projection
        self.input_projection = nn.Linear(self.config.input_size, 256)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using neuromorphic feature extractor."""
        # Project input
        features = torch.relu(self.input_projection(x))
        
        # Process through population layers
        for layer in self.feature_layers:
            features, _ = layer(features)
        
        return features
    
    def forward(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor,
        query_x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for few-shot learning episode."""
        batch_size = query_x.size(0)
        
        # Extract features for support and query sets
        support_features = self.extract_features(support_x)
        query_features = self.extract_features(query_x)
        
        # Temporal pattern encoding if enabled
        if self.temporal_encoder is not None:
            support_patterns, _ = self.temporal_encoder(support_features, current_time=0.001)
            query_patterns, _ = self.temporal_encoder(query_features, current_time=0.001)
        else:
            support_patterns = support_features
            query_patterns = query_features
        
        # Build prototypes from support set
        class_prototypes = []
        unique_classes = torch.unique(support_y)
        
        for class_idx in unique_classes:
            class_mask = (support_y == class_idx)
            class_features = support_patterns[class_mask]
            
            # Create class prototype
            if self.prototypical_memory is not None:
                prototype = self.prototypical_memory.encode_prototype(class_features)
                class_prototypes.append(prototype)
            else:
                # Simple averaging
                prototype = class_features.mean(dim=0, keepdim=True)
                class_prototypes.append(prototype)
        
        class_prototypes = torch.cat(class_prototypes, dim=0)
        
        # Rapid adaptation if enabled
        if self.rapid_plasticity is not None:
            # Convert to spike patterns for STDP learning
            support_spikes = (torch.rand_like(support_patterns) < torch.sigmoid(support_patterns)).float()
            
            adaptation_stats = self.rapid_plasticity.rapid_adapt(
                support_spikes, support_y, self.config.fast_adaptation_steps
            )
            
            # Apply plasticity to query processing
            query_spikes = (torch.rand_like(query_patterns) < torch.sigmoid(query_patterns)).float()
            adapted_features, _ = self.rapid_plasticity(
                query_spikes, torch.zeros(query_spikes.size(0), self.config.n_way), learning=False
            )
            query_patterns = query_patterns + adapted_features * 0.1  # Small plasticity influence
        
        # Compute distances to prototypes
        query_expanded = query_patterns.unsqueeze(1)  # [batch_size, 1, feature_dim]
        prototypes_expanded = class_prototypes.unsqueeze(0)  # [1, n_classes, feature_dim]
        
        # Euclidean distance in feature space
        distances = torch.norm(query_expanded - prototypes_expanded, dim=2)  # [batch_size, n_classes]
        
        # Convert distances to logits (negative distance)
        logits = -distances
        
        return logits
    
    def meta_learn_episode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform meta-learning update on a single episode."""
        # Forward pass
        logits = self(support_x, support_y, query_x)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, query_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute accuracy
        pred = logits.argmax(dim=1)
        accuracy = (pred == query_y).float().mean().item()
        
        episode_stats = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'avg_confidence': torch.softmax(logits, dim=1).max(dim=1)[0].mean().item()
        }
        
        return episode_stats
    
    def evaluate_episode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate performance on a single episode."""
        self.eval()
        
        with torch.no_grad():
            logits = self(support_x, support_y, query_x)
            
            # Compute metrics
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, query_y)
            
            pred = logits.argmax(dim=1)
            accuracy = (pred == query_y).float().mean().item()
            
            # Per-class accuracy
            class_accuracies = {}
            unique_classes = torch.unique(query_y)
            for class_idx in unique_classes:
                class_mask = (query_y == class_idx)
                if class_mask.sum() > 0:
                    class_acc = (pred[class_mask] == query_y[class_mask]).float().mean().item()
                    class_accuracies[f'class_{class_idx.item()}_acc'] = class_acc
            
            episode_stats = {
                'loss': loss.item(),
                'accuracy': accuracy,
                'confidence': torch.softmax(logits, dim=1).max(dim=1)[0].mean().item(),
                'entropy': -torch.sum(torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1).mean().item()
            }
            
            episode_stats.update(class_accuracies)
        
        return episode_stats
    
    def adaptation_analysis(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze adaptation mechanisms during few-shot learning."""
        analysis = {}
        
        # Feature adaptation analysis
        initial_features = self.extract_features(support_x)
        
        if self.rapid_plasticity is not None:
            # Analyze rapid plasticity changes
            initial_weights = self.rapid_plasticity.meta_synapses.stdp_synapse.weights.clone()
            
            # Perform adaptation
            support_spikes = (torch.rand_like(initial_features) < torch.sigmoid(initial_features)).float()
            adaptation_stats = self.rapid_plasticity.rapid_adapt(
                support_spikes, support_y, self.config.fast_adaptation_steps
            )
            
            final_weights = self.rapid_plasticity.meta_synapses.stdp_synapse.weights
            
            analysis.update({
                'weight_change_magnitude': torch.norm(final_weights - initial_weights).item(),
                'weight_change_ratio': (torch.norm(final_weights - initial_weights) / torch.norm(initial_weights)).item(),
                'adaptation_stats': adaptation_stats
            })
        
        # Memory analysis
        if self.prototypical_memory is not None:
            unique_classes = torch.unique(support_y)
            memory_stats = {
                'num_classes': len(unique_classes),
                'prototype_similarity': {}
            }
            
            # Analyze prototype similarities
            for i, class_i in enumerate(unique_classes):
                class_i_features = initial_features[support_y == class_i]
                prototype_i = self.prototypical_memory.encode_prototype(class_i_features)
                
                for j, class_j in enumerate(unique_classes):
                    if i < j:  # Only compute upper triangle
                        class_j_features = initial_features[support_y == class_j]
                        prototype_j = self.prototypical_memory.encode_prototype(class_j_features)
                        
                        similarity = torch.cosine_similarity(prototype_i, prototype_j, dim=1).item()
                        memory_stats['prototype_similarity'][f'{class_i.item()}_{class_j.item()}'] = similarity
            
            analysis['memory_stats'] = memory_stats
        
        return analysis