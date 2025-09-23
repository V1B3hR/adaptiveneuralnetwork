"""
Continual learning implementation using 3rd generation neuromorphic principles.

This module implements continual learning without catastrophic forgetting using
advanced neuromorphic mechanisms like metaplasticity and homeostatic scaling.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
import time

from ..core.neuromorphic_v3 import (
    HierarchicalNetwork, MetaplasticitySynapse, HomeostaticScaling,
    AdaptiveThresholdNeuron, SparseDistributedRepresentation
)
from ..core.neuromorphic_v3.advanced_neurons import NeuronV3Config
from ..core.neuromorphic_v3.plasticity import STDPConfig, MetaplasticityConfig, HomeostaticConfig
from ..core.neuromorphic_v3.network_topology import TopologyConfig  
from ..core.neuromorphic_v3.temporal_coding import TemporalConfig
from ..core.neuromorphic import NeuromorphicConfig

logger = logging.getLogger(__name__)


@dataclass
class ContinualLearningConfig:
    """Configuration for continual learning system."""
    # Network architecture
    num_tasks: int = 10
    input_size: int = 784
    output_size: int = 10
    hidden_layers: List[int] = None
    
    # Continual learning parameters
    consolidation_strength: float = 0.1  # EWC-like regularization
    memory_replay_ratio: float = 0.2  # Fraction of memory replay
    catastrophic_threshold: float = 0.05  # Performance drop threshold
    
    # Non-stationary data handling
    distribution_shift_detection: bool = True
    adaptation_window_size: int = 1000  # Samples to consider for shift detection
    shift_threshold: float = 0.1  # Threshold for detecting distribution shift
    rapid_adaptation_rate: float = 0.01  # Learning rate for rapid adaptation
    concept_drift_buffer_size: int = 5000  # Buffer for concept drift
    
    # Neuromorphic parameters
    enable_metaplasticity: bool = True
    enable_homeostatic_scaling: bool = True
    enable_sparse_coding: bool = True
    sparse_activity_target: float = 0.05
    
    # Adaptation parameters
    learning_rate_adaptation: bool = True
    threshold_adaptation: bool = True
    synaptic_consolidation: bool = True
    
    # Advanced learning paradigms
    enable_progressive_networks: bool = False  # Progressive neural networks
    enable_advanced_image_processing: bool = False  # Advanced visual features
    enable_memory_augmentation: bool = True  # Enhanced memory architectures
    enable_lifelong_benchmarking: bool = False  # Benchmark tracking
    
    # Image processing parameters
    enable_spatial_attention: bool = True
    enable_temporal_pooling: bool = True
    visual_feature_dim: int = 2048
    
    # Memory augmentation parameters
    memory_consolidation_strength: float = 0.5
    memory_retrieval_temperature: float = 1.0
    memory_update_frequency: int = 100
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256, 128]


class NonStationaryDataHandler:
    """Handles non-stationary data streams with distribution shift detection."""
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.data_buffer = []
        self.statistics_buffer = []
        self.current_statistics = None
        self.shift_detected = False
        self.adaptation_mode = False
        
        # Distribution tracking
        self.running_mean = None
        self.running_var = None
        self.sample_count = 0
        
        # Concept drift detection
        self.concept_drift_buffer = []
        self.performance_history = []
        
    def update_statistics(self, data: torch.Tensor) -> None:
        """Update running statistics for distribution shift detection."""
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_size = data.size(0)
        
        if self.running_mean is None:
            self.running_mean = batch_mean.clone()
            self.running_var = batch_var.clone()
            self.sample_count = batch_size
        else:
            # Online mean and variance update
            total_count = self.sample_count + batch_size
            delta = batch_mean - self.running_mean
            
            self.running_mean += delta * batch_size / total_count
            self.running_var = (
                (self.running_var * self.sample_count + batch_var * batch_size + delta**2 * self.sample_count * batch_size / total_count) 
                / total_count
            )
            self.sample_count = total_count
            
        # Store statistics in buffer
        current_stats = {
            'mean': batch_mean.clone(),
            'var': batch_var.clone(),
            'sample_count': batch_size
        }
        
        self.statistics_buffer.append(current_stats)
        if len(self.statistics_buffer) > self.config.adaptation_window_size:
            self.statistics_buffer.pop(0)
            
    def detect_distribution_shift(self) -> bool:
        """Detect if there's a significant distribution shift."""
        if not self.config.distribution_shift_detection or len(self.statistics_buffer) < 2:
            return False
            
        # Compare recent statistics with historical statistics
        recent_window = min(100, len(self.statistics_buffer) // 4)
        if len(self.statistics_buffer) < recent_window * 2:
            return False
            
        # Get recent and historical statistics
        recent_stats = self.statistics_buffer[-recent_window:]
        historical_stats = self.statistics_buffer[:-recent_window]
        
        # Calculate average statistics
        recent_mean = torch.stack([s['mean'] for s in recent_stats]).mean(dim=0)
        historical_mean = torch.stack([s['mean'] for s in historical_stats]).mean(dim=0)
        
        recent_var = torch.stack([s['var'] for s in recent_stats]).mean(dim=0)
        historical_var = torch.stack([s['var'] for s in historical_stats]).mean(dim=0)
        
        # Detect shift using KL divergence approximation
        mean_shift = torch.norm(recent_mean - historical_mean).item()
        var_shift = torch.norm(recent_var - historical_var).item()
        
        # Normalize by historical variance
        mean_shift_normalized = mean_shift / (torch.norm(historical_var).item() + 1e-8)
        var_shift_normalized = var_shift / (torch.norm(historical_var).item() + 1e-8)
        
        total_shift = mean_shift_normalized + var_shift_normalized
        
        self.shift_detected = total_shift > self.config.shift_threshold
        return self.shift_detected
        
    def handle_concept_drift(self, data: torch.Tensor, labels: torch.Tensor, performance: float) -> Dict[str, Any]:
        """Handle concept drift by updating the drift buffer and adaptation strategy."""
        # Store data in concept drift buffer
        sample = {'data': data.clone(), 'labels': labels.clone(), 'performance': performance}
        self.concept_drift_buffer.append(sample)
        
        if len(self.concept_drift_buffer) > self.config.concept_drift_buffer_size:
            self.concept_drift_buffer.pop(0)
            
        # Track performance history
        self.performance_history.append(performance)
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
            
        # Determine adaptation strategy
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            historical_performance = np.mean(self.performance_history[:-10]) if len(self.performance_history) > 10 else recent_performance
            
            performance_drop = historical_performance - recent_performance
            
            if performance_drop > 0.1:  # Significant performance drop
                self.adaptation_mode = True
                adaptation_strategy = 'rapid_adaptation'
            elif self.shift_detected:
                self.adaptation_mode = True
                adaptation_strategy = 'gradual_adaptation'
            else:
                self.adaptation_mode = False
                adaptation_strategy = 'normal_learning'
        else:
            adaptation_strategy = 'normal_learning'
            
        return {
            'adaptation_strategy': adaptation_strategy,
            'shift_detected': self.shift_detected,
            'adaptation_mode': self.adaptation_mode,
            'buffer_size': len(self.concept_drift_buffer),
            'performance_trend': np.mean(self.performance_history[-5:]) if len(self.performance_history) >= 5 else performance
        }
        
    def get_adaptation_samples(self, num_samples: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get samples from concept drift buffer for adaptation."""
        if len(self.concept_drift_buffer) < num_samples:
            return None
            
        # Sample recent high-performance examples
        sorted_buffer = sorted(self.concept_drift_buffer, key=lambda x: x['performance'], reverse=True)
        selected_samples = sorted_buffer[:num_samples]
        
        data_batch = torch.stack([s['data'] for s in selected_samples])
        label_batch = torch.stack([s['labels'] for s in selected_samples])
        
        return data_batch, label_batch


class EpisodicMemory(nn.Module):
    """
    Episodic memory buffer for experience replay in continual learning.
    
    Uses sparse distributed representations to efficiently store and
    retrieve task-relevant memories.
    """
    
    def __init__(
        self,
        memory_size: int,
        feature_size: int,
        temporal_config: TemporalConfig
    ):
        super().__init__()
        
        self.memory_size = memory_size
        self.feature_size = feature_size
        
        # Sparse memory representations
        self.memory_encoder = SparseDistributedRepresentation(
            feature_size, memory_size, temporal_config
        )
        
        # Memory buffers
        self.register_buffer('memory_features', torch.zeros(memory_size, feature_size))
        self.register_buffer('memory_labels', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('memory_tasks', torch.zeros(memory_size, dtype=torch.long))
        self.register_buffer('memory_importance', torch.zeros(memory_size))
        self.register_buffer('write_pointer', torch.tensor(0, dtype=torch.long))
        
        logger.debug(f"Initialized episodic memory: {memory_size} samples")
    
    def store(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        task_id: int,
        importance: Optional[torch.Tensor] = None
    ) -> None:
        """Store new experiences in episodic memory."""
        batch_size = features.size(0)
        
        if importance is None:
            importance = torch.ones(batch_size)
        
        for i in range(batch_size):
            idx = self.write_pointer.item() % self.memory_size
            
            # Store experience
            self.memory_features[idx] = features[i]
            self.memory_labels[idx] = labels[i]
            self.memory_tasks[idx] = task_id
            self.memory_importance[idx] = importance[i]
            
            self.write_pointer += 1
    
    def sample(self, num_samples: int, task_bias: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample experiences from episodic memory."""
        # Determine available samples
        available_samples = min(self.write_pointer.item(), self.memory_size)
        
        if available_samples == 0:
            return torch.empty(0, self.feature_size), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        num_samples = min(num_samples, available_samples)
        
        # Importance-weighted sampling
        importance_weights = self.memory_importance[:available_samples]
        
        # Add task bias if specified
        if task_bias is not None:
            task_mask = (self.memory_tasks[:available_samples] == task_bias).float()
            importance_weights = importance_weights + task_mask * 2.0  # Boost task relevance
        
        # Sample indices
        indices = torch.multinomial(importance_weights, num_samples, replacement=True)
        
        sampled_features = self.memory_features[indices]
        sampled_labels = self.memory_labels[indices]
        sampled_tasks = self.memory_tasks[indices]
        
        return sampled_features, sampled_labels, sampled_tasks
    
    def get_task_memories(self, task_id: int, max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve all memories from a specific task."""
        available_samples = min(self.write_pointer.item(), self.memory_size)
        
        if available_samples == 0:
            return torch.empty(0, self.feature_size), torch.empty(0, dtype=torch.long)
        
        # Find task-specific memories
        task_mask = self.memory_tasks[:available_samples] == task_id
        task_indices = torch.where(task_mask)[0]
        
        if len(task_indices) == 0:
            return torch.empty(0, self.feature_size), torch.empty(0, dtype=torch.long)
        
        # Limit samples if requested
        if max_samples is not None and len(task_indices) > max_samples:
            # Sample based on importance
            task_importance = self.memory_importance[task_indices]
            selected_indices = torch.multinomial(task_importance, max_samples, replacement=False)
            task_indices = task_indices[selected_indices]
        
        return self.memory_features[task_indices], self.memory_labels[task_indices]


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Network for continual learning without catastrophic forgetting.
    
    Creates lateral connections between tasks while freezing previous task columns,
    enabling knowledge transfer while preserving old knowledge.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_tasks = 0
        
        # Store task-specific columns
        self.task_columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()  # Connections between columns
        self.output_heads = nn.ModuleList()
        
        logger.debug(f"Initialized progressive network: {input_size} -> {hidden_sizes} -> {output_size}")
    
    def add_task(self, task_id: int) -> None:
        """Add a new task column to the progressive network."""
        if task_id != self.num_tasks:
            raise ValueError(f"Expected task_id {self.num_tasks}, got {task_id}")
        
        # Create new column for this task
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        column = nn.Sequential(*layers)
        self.task_columns.append(column)
        
        # Create lateral connections from previous columns
        if self.num_tasks > 0:
            lateral_conn = nn.ModuleList()
            for layer_idx in range(len(self.hidden_sizes)):
                # Create lateral connections from all previous columns at this layer
                conn_layers = nn.ModuleList()
                for prev_task in range(self.num_tasks):
                    # Lateral connections map from same hidden size to same hidden size
                    input_size = self.hidden_sizes[layer_idx]
                    output_size = self.hidden_sizes[layer_idx]
                    conn_layers.append(nn.Linear(input_size, output_size))
                lateral_conn.append(conn_layers)
            self.lateral_connections.append(lateral_conn)
        else:
            self.lateral_connections.append(nn.ModuleList())
        
        # Create output head for this task
        output_head = nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.output_heads.append(output_head)
        
        # Freeze previous task parameters
        if self.num_tasks > 0:
            for prev_task_idx in range(self.num_tasks):
                for param in self.task_columns[prev_task_idx].parameters():
                    param.requires_grad = False
                for param in self.output_heads[prev_task_idx].parameters():
                    param.requires_grad = False
        
        self.num_tasks += 1
        logger.info(f"Added task {task_id}, total tasks: {self.num_tasks}")
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass through progressive network for specific task."""
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} not available, only {self.num_tasks} tasks")
        
        # Store all layer activations for lateral connections
        all_activations = []  # [task][layer] = activation
        
        # Process each column up to and including the target task
        for col_idx in range(task_id + 1):
            current_input = x
            layer_activations = []
            
            layer_idx = 0
            for layer in self.task_columns[col_idx]:
                if isinstance(layer, nn.Linear):
                    # Apply linear layer first
                    linear_output = layer(current_input)
                    
                    # Add lateral connections from previous columns at same layer
                    lateral_input = torch.zeros_like(linear_output)
                    
                    if (col_idx > 0 and 
                        col_idx < len(self.lateral_connections) and 
                        layer_idx < len(self.lateral_connections[col_idx])):
                        
                        connections = self.lateral_connections[col_idx][layer_idx]
                        
                        for prev_col_idx in range(col_idx):
                            if (prev_col_idx < len(all_activations) and 
                                layer_idx < len(all_activations[prev_col_idx]) and
                                prev_col_idx < len(connections)):
                                
                                prev_activation = all_activations[prev_col_idx][layer_idx]
                                lateral_conn = connections[prev_col_idx]
                                lateral_contribution = lateral_conn(prev_activation)
                                lateral_input += lateral_contribution
                    
                    # Combine linear output with lateral input
                    current_input = linear_output + lateral_input
                    layer_activations.append(current_input.clone())
                    layer_idx += 1
                else:
                    # Non-linear activation
                    current_input = layer(current_input)
            
            all_activations.append(layer_activations)
        
        # Use the final activation from the target task column
        if all_activations[task_id]:
            final_activation = all_activations[task_id][-1]
        else:
            # Fallback to input if no activations recorded
            final_activation = x
        
        output = self.output_heads[task_id](final_activation)
        return output


class AdvancedImageProcessor(nn.Module):
    """
    Advanced image processing module for continual learning.
    
    Combines spatial attention, temporal pooling, and multi-scale features
    for robust visual representation learning.
    """
    
    def __init__(self, input_channels: int, feature_dim: int, config: ContinualLearningConfig):
        super().__init__()
        
        self.config = config
        self.feature_dim = feature_dim
        
        # Multi-scale convolutional features
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
        ])
        
        # Spatial attention mechanism
        if config.enable_spatial_attention:
            self.spatial_attention = nn.ModuleList([
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Conv2d(128, 1, kernel_size=1),
                nn.Conv2d(256, 1, kernel_size=1),
                nn.Conv2d(512, 1, kernel_size=1),
            ])
        
        # Temporal pooling for video sequences
        if config.enable_temporal_pooling:
            self.temporal_pooling = nn.LSTM(feature_dim, 256, batch_first=True)  # Use feature_dim not 512
        else:
            self.temporal_pooling = None
        
        # Feature fusion and projection
        self.feature_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific batch normalization layers
        self.task_bn_layers = nn.ModuleList()
        
        logger.debug(f"Initialized advanced image processor: {input_channels} -> {feature_dim}")
    
    def add_task_adaptation(self, task_id: int) -> None:
        """Add task-specific batch normalization for domain adaptation."""
        task_bn = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128), 
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(512),
        ])
        self.task_bn_layers.append(task_bn)
        logger.info(f"Added task-specific BN for task {task_id}")
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None, temporal_sequence: bool = False) -> torch.Tensor:
        """
        Forward pass through advanced image processor.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, C, H, W) for temporal
            task_id: Optional task ID for task-specific adaptation
            temporal_sequence: Whether input is a temporal sequence
        """
        if temporal_sequence and len(x.shape) == 5:
            # Handle temporal sequences: (B, T, C, H, W)
            batch_size, seq_len = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # Reshape to (B*T, C, H, W)
            process_temporal = True
        else:
            process_temporal = False
            batch_size, seq_len = x.shape[0], 1
        
        # Multi-scale feature extraction with attention
        features = []
        current_x = x
        
        for i, conv_layer in enumerate(self.conv_layers):
            current_x = conv_layer(current_x)
            current_x = nn.functional.relu(current_x)
            
            # Apply task-specific batch normalization if available
            if task_id is not None and task_id < len(self.task_bn_layers):
                current_x = self.task_bn_layers[task_id][i](current_x)
            
            # Spatial attention
            if self.config.enable_spatial_attention and hasattr(self, 'spatial_attention'):
                attention_weights = torch.sigmoid(self.spatial_attention[i](current_x))
                current_x = current_x * attention_weights
            
            features.append(current_x)
            
            # Max pooling except for last layer
            if i < len(self.conv_layers) - 1:
                current_x = nn.functional.max_pool2d(current_x, kernel_size=2)
        
        # Feature fusion
        final_features = self.feature_fusion(current_x)
        
        # Temporal processing if needed
        if process_temporal and self.config.enable_temporal_pooling and self.temporal_pooling is not None:
            # Reshape back to temporal format
            final_features = final_features.view(batch_size, seq_len, -1)
            
            # Apply LSTM for temporal modeling
            temporal_features, _ = self.temporal_pooling(final_features)
            # Take final timestep
            final_features = temporal_features[:, -1, :]
        
        return final_features


class MemoryAugmentedArchitecture(nn.Module):
    """
    Memory-augmented architecture for enhanced continual learning.
    
    Combines multiple memory systems: episodic, semantic, and working memory
    for comprehensive knowledge retention and transfer.
    """
    
    def __init__(self, feature_dim: int, memory_dim: int, config: ContinualLearningConfig):
        super().__init__()
        
        self.config = config
        self.feature_dim = feature_dim
        self.memory_dim = memory_dim
        
        # Working memory for current task processing
        self.working_memory = nn.LSTM(feature_dim, memory_dim, batch_first=True)
        
        # Semantic memory for long-term knowledge
        self.semantic_memory = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(memory_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        
        # Memory consolidation mechanism
        self.memory_consolidation = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Tanh()
        )
        
        # Attention mechanism for memory retrieval
        self.memory_attention = nn.MultiheadAttention(
            memory_dim, num_heads=8, batch_first=True
        )
        
        # Memory buffers
        self.register_buffer('semantic_memory_bank', torch.randn(1000, memory_dim) * 0.1)
        self.register_buffer('memory_importance', torch.ones(1000))
        self.register_buffer('memory_write_pointer', torch.tensor(0, dtype=torch.long))
        
        logger.debug(f"Initialized memory-augmented architecture: {feature_dim} -> {memory_dim}")
    
    def update_semantic_memory(self, features: torch.Tensor, importance: torch.Tensor) -> None:
        """Update semantic memory with consolidated features."""
        batch_size = features.size(0)
        
        for i in range(batch_size):
            idx = self.memory_write_pointer.item() % self.semantic_memory_bank.size(0)
            
            # Store consolidated memory
            self.semantic_memory_bank[idx] = features[i]
            self.memory_importance[idx] = importance[i]
            
            self.memory_write_pointer += 1
    
    def retrieve_relevant_memories(self, query: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """Retrieve most relevant memories using attention mechanism."""
        batch_size = query.size(0)
        
        # Expand memory bank to match batch size
        memory_bank = self.semantic_memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        query_expanded = query.unsqueeze(1)  # Add sequence dimension
        
        # Use attention to find relevant memories
        attended_memories, attention_weights = self.memory_attention(
            query_expanded, memory_bank, memory_bank
        )
        
        return attended_memories.squeeze(1)  # Remove sequence dimension
    
    def forward(self, features: torch.Tensor, retrieve_memories: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through memory-augmented architecture.
        
        Returns:
            enhanced_features: Features enhanced with memory
            memory_state: Current memory state for consolidation
        """
        batch_size = features.size(0)
        
        # Process through working memory
        features_seq = features.unsqueeze(1)  # Add sequence dimension
        working_output, (hidden, cell) = self.working_memory(features_seq)
        working_features = working_output.squeeze(1)
        
        # Retrieve and integrate relevant memories
        if retrieve_memories:
            relevant_memories = self.retrieve_relevant_memories(working_features)
            
            # Consolidate working memory with retrieved memories
            combined_features = torch.cat([working_features, relevant_memories], dim=-1)
            consolidated_features = self.memory_consolidation(combined_features)
        else:
            consolidated_features = working_features
        
        # Process through semantic memory transformer
        consolidated_seq = consolidated_features.unsqueeze(1)
        semantic_output = self.semantic_memory(consolidated_seq)
        enhanced_features = semantic_output.squeeze(1)
        
        return enhanced_features, consolidated_features


class SynapticConsolidation(nn.Module):
    """
    Synaptic consolidation mechanism to protect important connections.
    
    Implements Elastic Weight Consolidation (EWC) using neuromorphic
    metaplasticity principles.
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        
        self.model = model
        
        # Fisher information for parameter importance
        self.fisher_information = {}
        self.optimal_params = {}
        
        # Initialize Fisher information storage
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)
                self.optimal_params[name] = param.clone().detach()
    
    def estimate_fisher_information(
        self, 
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 1000
    ) -> None:
        """Estimate Fisher information matrix for current task."""
        self.model.eval()
        
        # Reset Fisher information
        for name in self.fisher_information:
            self.fisher_information[name].zero_()
        
        sample_count = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if sample_count >= num_samples:
                break
                
            batch_size = data.size(0)
            
            # Forward pass
            output = self.model(data)
            
            # Sample from posterior (use predicted class)
            pred_class = output.argmax(dim=1)
            
            # Compute gradients for sampled classes
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                self.model.zero_grad()
                
                # Compute log probability for predicted class
                log_prob = torch.nn.functional.log_softmax(output[i:i+1], dim=1)
                loss = -log_prob[0, pred_class[i]]
                
                loss.backward(retain_graph=True)
                
                # Accumulate squared gradients (Fisher information)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.fisher_information[name] += param.grad.pow(2)
                
                sample_count += 1
        
        # Normalize Fisher information
        for name in self.fisher_information:
            self.fisher_information[name] /= sample_count
        
        logger.debug(f"Estimated Fisher information from {sample_count} samples")
    
    def consolidation_loss(self, consolidation_strength: float = 1.0) -> torch.Tensor:
        """Compute consolidation loss to prevent catastrophic forgetting."""
        loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                # EWC penalty term
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                
                penalty = fisher * (param - optimal).pow(2)
                loss += penalty.sum()
        
        return consolidation_strength * loss
    
    def update_optimal_params(self) -> None:
        """Update optimal parameters after learning new task."""
        for name, param in self.model.named_parameters():
            if name in self.optimal_params:
                self.optimal_params[name] = param.clone().detach()


class ContinualLearningSystem(nn.Module):
    """
    Complete continual learning system using 3rd generation neuromorphic principles.
    
    Combines hierarchical networks, metaplasticity, homeostatic scaling,
    and episodic memory for catastrophic forgetting prevention.
    """
    
    def __init__(self, config: ContinualLearningConfig):
        super().__init__()
        
        self.config = config
        
        # Base neuromorphic configuration
        base_config = NeuromorphicConfig(
            generation=3,
            enable_metaplasticity=config.enable_metaplasticity,
            enable_homeostatic_scaling=config.enable_homeostatic_scaling
        )
        
        # Build neuromorphic network
        self._build_network(base_config)
        
        # Episodic memory for experience replay
        temporal_config = TemporalConfig(sparsity_target=config.sparse_activity_target)
        self.episodic_memory = EpisodicMemory(
            memory_size=10000,  # Large memory buffer
            feature_size=config.hidden_layers[-1],
            temporal_config=temporal_config
        )
        
        # Non-stationary data handler
        self.data_handler = NonStationaryDataHandler(config)
        
        # Synaptic consolidation
        self.synaptic_consolidation = SynapticConsolidation(self.network) if config.synaptic_consolidation else None
        
        # Advanced learning paradigm components
        if config.enable_progressive_networks:
            self.progressive_network = ProgressiveNeuralNetwork(
                config.input_size, config.hidden_layers, config.output_size
            )
        else:
            self.progressive_network = None
        
        if config.enable_advanced_image_processing:
            input_channels = 3 if len(str(config.input_size)) > 3 else 1  # Heuristic for image channels
            self.image_processor = AdvancedImageProcessor(
                input_channels, config.visual_feature_dim, config
            )
        else:
            self.image_processor = None
        
        if config.enable_memory_augmentation:
            self.memory_augmented_arch = MemoryAugmentedArchitecture(
                config.hidden_layers[-1], config.hidden_layers[-1], config
            )
        else:
            self.memory_augmented_arch = None
        
        if config.enable_lifelong_benchmarking:
            self.benchmark_system = LifelongLearningBenchmark(config)
        else:
            self.benchmark_system = None
        
        # Task management
        self.current_task = 0
        self.task_performance = {}
        
        # Adaptation tracking
        self.adaptation_history = []
        self.current_learning_rate = 0.001
        
        logger.info(f"Initialized continual learning system for {config.num_tasks} tasks")
        if config.enable_progressive_networks:
            logger.info("- Progressive neural networks enabled")
        if config.enable_advanced_image_processing:
            logger.info("- Advanced image processing enabled")
        if config.enable_memory_augmentation:
            logger.info("- Memory-augmented architecture enabled")
        if config.enable_lifelong_benchmarking:
            logger.info("- Lifelong learning benchmarking enabled")
    
    def _build_network(self, base_config: NeuromorphicConfig) -> None:
        """Build the neuromorphic network architecture."""
        # Neuron configuration
        neuron_config = NeuronV3Config(
            base_config=base_config,
            threshold_adaptation_rate=0.1 if self.config.threshold_adaptation else 0.0,
            target_spike_rate=20.0
        )
        
        # Topology configuration
        layer_sizes = [self.config.input_size] + self.config.hidden_layers + [self.config.output_size]
        topology_config = TopologyConfig(
            num_layers=len(layer_sizes),
            layer_sizes=layer_sizes,
            connection_probability=0.1
        )
        
        # Hierarchical network with adaptive neurons
        self.network = HierarchicalNetwork(
            config=topology_config,
            neuron_configs=[neuron_config] * len(layer_sizes),
            layer_types=["adaptive_threshold"] * len(layer_sizes)
        )
        
        # Sparse coding for efficient representations
        if self.config.enable_sparse_coding:
            temporal_config = TemporalConfig(sparsity_target=self.config.sparse_activity_target)
            self.sparse_encoder = SparseDistributedRepresentation(
                self.config.hidden_layers[-1],
                self.config.hidden_layers[-1] * 2,
                temporal_config
            )
        else:
            self.sparse_encoder = None
        
        # Output projection
        self.output_projection = nn.Linear(layer_sizes[-1], self.config.output_size)
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the continual learning system."""
        # Process through hierarchical network
        network_output, network_states = self.network(x)
        
        # Apply sparse coding if enabled
        if self.sparse_encoder is not None:
            sparse_output, _ = self.sparse_encoder(network_output)
            features = sparse_output
        else:
            features = network_output
        
        # Final output projection
        output = self.output_projection(features)
        
        return output
    
    def learn_task(
        self,
        train_loader: torch.utils.data.DataLoader,
        task_id: int,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """
        Learn a new task while preventing catastrophic forgetting.
        
        Args:
            train_loader: DataLoader for training data
            task_id: Unique identifier for the task
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            val_loader: Optional DataLoader for validation data. If provided,
                       performance metrics will use validation data for more
                       accurate generalization estimates.
        
        Returns:
            Dictionary containing training statistics including initial and
            final performance on validation set (if provided) or training set.
        """
        logger.info(f"Learning task {task_id}")
        
        self.current_task = task_id
        self.current_learning_rate = learning_rate
        
        # Record initial performance on validation set if available
        initial_performance = 0.0
        if val_loader is not None:
            logger.info(f"Recording initial performance on validation set")
            initial_performance = self.evaluate_task(val_loader)
            logger.info(f"Initial validation performance: {initial_performance:.3f}")
        else:
            logger.debug("No validation loader provided - initial performance will be 0.0")
        
        # Optimizer with adaptive learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Learning statistics
        task_stats = {
            'initial_performance': initial_performance,
            'final_performance': 0.0,
            'consolidation_loss': 0.0,
            'memory_replay_ratio': 0.0,
            'distribution_shifts_detected': 0,
            'adaptation_events': 0,
            'average_learning_rate': learning_rate
        }
        
        self.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_consolidation_loss = 0.0
            total_samples = 0
            memory_samples = 0
            distribution_shifts = 0
            adaptation_events = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_size = data.size(0)
                
                # Update data statistics for shift detection
                self.data_handler.update_statistics(data)
                
                # Detect distribution shift
                shift_detected = self.data_handler.detect_distribution_shift()
                if shift_detected:
                    distribution_shifts += 1
                
                # Forward pass on current task
                output = self(data, task_id)
                task_loss = criterion(output, target)
                
                # Calculate current performance for adaptation
                with torch.no_grad():
                    predictions = output.argmax(dim=1)
                    current_performance = (predictions == target).float().mean().item()
                
                # Handle concept drift and adaptation
                adaptation_info = self.data_handler.handle_concept_drift(data, target, current_performance)
                
                # Adapt learning rate based on adaptation strategy
                if adaptation_info['adaptation_strategy'] == 'rapid_adaptation':
                    adapted_lr = self.config.rapid_adaptation_rate
                    adaptation_events += 1
                elif adaptation_info['adaptation_strategy'] == 'gradual_adaptation':
                    adapted_lr = learning_rate * 0.5
                    adaptation_events += 1
                else:
                    adapted_lr = learning_rate
                    
                # Update optimizer learning rate if needed
                if adapted_lr != self.current_learning_rate:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = adapted_lr
                    self.current_learning_rate = adapted_lr
                
                # Store experiences in episodic memory
                with torch.no_grad():
                    # Extract features from network
                    network_output, _ = self.network(data)
                    if self.sparse_encoder is not None:
                        features, _ = self.sparse_encoder(network_output)
                    else:
                        features = network_output
                    
                    # Compute importance (loss magnitude as proxy)
                    importance = task_loss.item() * torch.ones(batch_size)
                    
                    # Increase importance for samples during distribution shifts
                    if shift_detected:
                        importance *= 2.0
                        
                    self.episodic_memory.store(features, target, task_id, importance)
                
                total_loss = task_loss
                
                # Add consolidation loss for previous tasks
                if self.synaptic_consolidation is not None and task_id > 0:
                    consolidation_loss = self.synaptic_consolidation.consolidation_loss(
                        self.config.consolidation_strength
                    )
                    total_loss += consolidation_loss
                    epoch_consolidation_loss += consolidation_loss.item()
                
                # Experience replay from episodic memory
                replay_ratio = self.config.memory_replay_ratio
                # Increase replay during adaptation
                if adaptation_info['adaptation_mode']:
                    replay_ratio *= 2.0
                    
                if task_id > 0 and np.random.random() < replay_ratio:
                    replay_samples = int(batch_size * replay_ratio)
                    memory_features, memory_labels, memory_tasks = self.episodic_memory.sample(
                        replay_samples
                    )
                    
                    if memory_features.size(0) > 0:
                        # Forward pass on memory
                        memory_output = self.output_projection(memory_features)
                        memory_loss = criterion(memory_output, memory_labels)
                        total_loss += memory_loss * 0.5  # Reduced weight for replay
                        
                        memory_samples += memory_features.size(0)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += task_loss.item()
                total_samples += batch_size
            
            # Log epoch statistics with more comprehensive information
            avg_loss = epoch_loss / len(train_loader)
            avg_consolidation = epoch_consolidation_loss / len(train_loader)
            memory_ratio = memory_samples / total_samples if total_samples > 0 else 0.0
            
            # Store accumulated metrics for task stats
            task_stats['consolidation_loss'] += avg_consolidation
            task_stats['memory_replay_ratio'] += memory_ratio
            task_stats['distribution_shifts_detected'] += distribution_shifts
            task_stats['adaptation_events'] += adaptation_events
            task_stats['average_learning_rate'] = self.current_learning_rate
            
            # Enhanced epoch logging (every 2 epochs to reduce verbosity)
            if epoch % 2 == 0 or epoch == num_epochs - 1:
                logger.debug(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, "
                           f"Consolidation={avg_consolidation:.4f}, "
                           f"Memory_replay={memory_ratio:.2%}, "
                           f"Shifts={distribution_shifts}, "
                           f"Adaptations={adaptation_events}, "
                           f"LR={self.current_learning_rate:.5f}")
        
        # Average out accumulated statistics
        task_stats['consolidation_loss'] /= num_epochs
        task_stats['memory_replay_ratio'] /= num_epochs
        
        # Update synaptic consolidation for this task
        if self.synaptic_consolidation is not None:
            self.synaptic_consolidation.estimate_fisher_information(train_loader)
            self.synaptic_consolidation.update_optimal_params()
        
        # Evaluate performance on validation set if available, otherwise use training set
        if val_loader is not None:
            logger.info("Evaluating final performance on validation set")
            final_performance = self.evaluate_task(val_loader)
            evaluation_set = "validation"
        else:
            logger.warning("⚠️ No validation loader - evaluating on training set!")
            final_performance = self.evaluate_task(train_loader)
            evaluation_set = "training"
        
        task_stats['final_performance'] = final_performance
        self.task_performance[task_id] = final_performance
        
        # Enhanced logging with initial -> final performance summary
        if val_loader is not None and initial_performance > 0:
            improvement = final_performance - initial_performance
            logger.info(f"Task {task_id} learning completed on {evaluation_set} set. "
                       f"Performance: {initial_performance:.3f} -> {final_performance:.3f} "
                       f"(Δ{improvement:+.3f})")
        else:
            logger.info(f"Task {task_id} learning completed on {evaluation_set} set. "
                       f"Performance: {final_performance:.3f}")
        
        return task_stats
    
    def evaluate_task(self, data_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate performance on a specific task."""
        self.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                output = self(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def evaluate_continual_performance(self, task_loaders: Dict[int, torch.utils.data.DataLoader]) -> Dict[str, float]:
        """Evaluate continual learning performance across all tasks."""
        metrics = {}
        
        # Evaluate each task
        task_accuracies = []
        for task_id, loader in task_loaders.items():
            accuracy = self.evaluate_task(loader)
            metrics[f'task_{task_id}_accuracy'] = accuracy
            task_accuracies.append(accuracy)
        
        # Compute continual learning metrics
        metrics['average_accuracy'] = np.mean(task_accuracies) if task_accuracies else 0.0
        
        # Backward transfer (improvement on previous tasks)
        if len(self.task_performance) > 1:
            backward_transfer = []
            for task_id in range(self.current_task):
                if task_id in self.task_performance and task_id in task_loaders:
                    current_perf = self.evaluate_task(task_loaders[task_id])
                    original_perf = self.task_performance[task_id]
                    backward_transfer.append(current_perf - original_perf)
            
            metrics['backward_transfer'] = np.mean(backward_transfer) if backward_transfer else 0.0
        
        # Forgetting measure
        if len(self.task_performance) > 1:
            forgetting = []
            for task_id in range(self.current_task):
                if task_id in self.task_performance and task_id in task_loaders:
                    current_perf = self.evaluate_task(task_loaders[task_id])
                    best_perf = self.task_performance[task_id]
                    forgetting.append(max(0, best_perf - current_perf))
            
            metrics['average_forgetting'] = np.mean(forgetting) if forgetting else 0.0
        
        return metrics
    
    def get_task_representations(self, task_loaders: Dict[int, torch.utils.data.DataLoader]) -> Dict[int, torch.Tensor]:
        """Extract learned representations for each task."""
        self.eval()
        
        task_representations = {}
        
        with torch.no_grad():
            for task_id, loader in task_loaders.items():
                representations = []
                
                for data, _ in loader:
                    # Extract features from network
                    network_output, _ = self.network(data)
                    if self.sparse_encoder is not None:
                        features, _ = self.sparse_encoder(network_output)
                    else:
                        features = network_output
                    
                    representations.append(features)
                
                if representations:
                    task_representations[task_id] = torch.cat(representations, dim=0)
        
        return task_representations


class LifelongLearningBenchmark:
    """
    Comprehensive benchmark system for lifelong learning performance evaluation.
    
    Tracks multiple metrics including catastrophic forgetting, transfer learning,
    adaptation speed, and memory efficiency.
    """
    
    def __init__(self, config: ContinualLearningConfig):
        self.config = config
        self.metrics_history = []
        self.task_timings = {}
        self.memory_usage = {}
        self.adaptation_events = {}
        
        # Benchmark metrics
        self.metric_names = [
            'average_accuracy', 'backward_transfer', 'forward_transfer',
            'average_forgetting', 'learning_efficiency', 'memory_stability',
            'adaptation_speed', 'knowledge_retention'
        ]
        
        logger.info("Initialized lifelong learning benchmark system")
    
    def start_task_timing(self, task_id: int) -> None:
        """Start timing for a task."""
        self.task_timings[task_id] = {'start': time.time()}
    
    def end_task_timing(self, task_id: int) -> None:
        """End timing for a task."""
        if task_id in self.task_timings:
            self.task_timings[task_id]['end'] = time.time()
            self.task_timings[task_id]['duration'] = (
                self.task_timings[task_id]['end'] - self.task_timings[task_id]['start']
            )
    
    def record_memory_usage(self, task_id: int, model: nn.Module) -> None:
        """Record memory usage for a task."""
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.memory_usage[task_id] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_ratio': trainable_params / total_params if total_params > 0 else 0
        }
    
    def record_adaptation_event(self, task_id: int, event_type: str, value: float) -> None:
        """Record adaptation events during learning."""
        if task_id not in self.adaptation_events:
            self.adaptation_events[task_id] = []
        
        self.adaptation_events[task_id].append({
            'type': event_type,
            'value': value,
            'timestamp': time.time()
        })
    
    def evaluate_comprehensive_metrics(
        self, 
        model: ContinualLearningSystem,
        task_loaders: Dict[int, torch.utils.data.DataLoader],
        baseline_performances: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """Evaluate comprehensive lifelong learning metrics."""
        
        # Get basic continual learning metrics
        base_metrics = model.evaluate_continual_performance(task_loaders)
        
        # Calculate additional benchmark metrics
        benchmark_metrics = {}
        
        # 1. Learning Efficiency (accuracy gain per unit time)
        if self.task_timings:
            total_accuracy = base_metrics.get('average_accuracy', 0)
            total_time = sum(timing.get('duration', 0) for timing in self.task_timings.values())
            benchmark_metrics['learning_efficiency'] = total_accuracy / max(total_time, 1e-6)
        
        # 2. Memory Stability (consistency of memory usage)
        if self.memory_usage:
            memory_ratios = [usage['memory_ratio'] for usage in self.memory_usage.values()]
            benchmark_metrics['memory_stability'] = 1.0 - np.std(memory_ratios) if memory_ratios else 0
        
        # 3. Adaptation Speed (time to adapt to new tasks)
        adaptation_speeds = []
        for task_id, events in self.adaptation_events.items():
            if events:
                adaptation_events = [e for e in events if e['type'] == 'adaptation']
                if adaptation_events:
                    avg_adaptation_time = np.mean([e['value'] for e in adaptation_events])
                    adaptation_speeds.append(1.0 / max(avg_adaptation_time, 1e-6))
        
        benchmark_metrics['adaptation_speed'] = np.mean(adaptation_speeds) if adaptation_speeds else 0
        
        # 4. Knowledge Retention (stability of learned representations)
        task_representations = model.get_task_representations(task_loaders)
        if len(task_representations) > 1:
            retention_scores = []
            task_ids = sorted(task_representations.keys())
            
            for i in range(len(task_ids) - 1):
                task_i = task_ids[i]
                for j in range(i + 1, len(task_ids)):
                    task_j = task_ids[j]
                    
                    # Calculate representation similarity
                    repr_i = task_representations[task_i].mean(dim=0)
                    repr_j = task_representations[task_j].mean(dim=0)
                    
                    similarity = torch.cosine_similarity(repr_i, repr_j, dim=0)
                    retention_scores.append(similarity.item())
            
            benchmark_metrics['knowledge_retention'] = np.mean(retention_scores)
        else:
            benchmark_metrics['knowledge_retention'] = 1.0
        
        # 5. Forward Transfer (performance on new tasks using previous knowledge)
        if baseline_performances and len(task_loaders) > 1:
            forward_transfers = []
            for task_id, loader in task_loaders.items():
                if task_id in baseline_performances:
                    current_perf = model.evaluate_task(loader)
                    baseline_perf = baseline_performances[task_id]
                    forward_transfers.append(current_perf - baseline_perf)
            
            benchmark_metrics['forward_transfer'] = np.mean(forward_transfers) if forward_transfers else 0
        
        # Combine all metrics
        comprehensive_metrics = {**base_metrics, **benchmark_metrics}
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': time.time(),
            'num_tasks': len(task_loaders),
            'metrics': comprehensive_metrics.copy()
        })
        
        return comprehensive_metrics
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.metrics_history:
            return {'error': 'No metrics recorded'}
        
        # Latest metrics
        latest_metrics = self.metrics_history[-1]['metrics']
        
        # Metric trends over time
        trends = {}
        for metric in self.metric_names:
            values = [entry['metrics'].get(metric, 0) for entry in self.metrics_history]
            if len(values) > 1:
                trends[f'{metric}_trend'] = np.polyfit(range(len(values)), values, 1)[0]  # Linear trend
        
        # Performance summary
        performance_summary = {
            'total_tasks_learned': self.metrics_history[-1]['num_tasks'],
            'total_learning_time': sum(timing.get('duration', 0) for timing in self.task_timings.values()),
            'average_adaptation_events': np.mean([len(events) for events in self.adaptation_events.values()]) if self.adaptation_events else 0,
            'memory_efficiency': np.mean([usage['memory_ratio'] for usage in self.memory_usage.values()]) if self.memory_usage else 0
        }
        
        return {
            'latest_metrics': latest_metrics,
            'metric_trends': trends,
            'performance_summary': performance_summary,
            'benchmark_timestamp': time.time()
        }