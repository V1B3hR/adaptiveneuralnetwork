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
    
    # Neuromorphic parameters
    enable_metaplasticity: bool = True
    enable_homeostatic_scaling: bool = True
    enable_sparse_coding: bool = True
    sparse_activity_target: float = 0.05
    
    # Adaptation parameters
    learning_rate_adaptation: bool = True
    threshold_adaptation: bool = True
    synaptic_consolidation: bool = True
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256, 128]


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
        
        # Synaptic consolidation
        self.synaptic_consolidation = SynapticConsolidation(self.network) if config.synaptic_consolidation else None
        
        # Task management
        self.current_task = 0
        self.task_performance = {}
        
        logger.info(f"Initialized continual learning system for {config.num_tasks} tasks")
    
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
            connection_probability=0.1,
            enable_dynamic_connectivity=True
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
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """Learn a new task while preventing catastrophic forgetting."""
        logger.info(f"Learning task {task_id}")
        
        self.current_task = task_id
        
        # Optimizer with adaptive learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Learning statistics
        task_stats = {
            'initial_performance': 0.0,
            'final_performance': 0.0,
            'consolidation_loss': 0.0,
            'memory_replay_ratio': 0.0
        }
        
        self.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_consolidation_loss = 0.0
            total_samples = 0
            memory_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_size = data.size(0)
                
                # Forward pass on current task
                output = self(data, task_id)
                task_loss = criterion(output, target)
                
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
                if task_id > 0 and np.random.random() < self.config.memory_replay_ratio:
                    replay_samples = int(batch_size * self.config.memory_replay_ratio)
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
            
            # Log epoch statistics
            avg_loss = epoch_loss / len(train_loader)
            avg_consolidation = epoch_consolidation_loss / len(train_loader)
            memory_ratio = memory_samples / total_samples if total_samples > 0 else 0.0
            
            if epoch % 5 == 0:
                logger.debug(f"Epoch {epoch}: Loss {avg_loss:.4f}, "
                           f"Consolidation {avg_consolidation:.4f}, "
                           f"Memory replay {memory_ratio:.2%}")
        
        # Update synaptic consolidation for this task
        if self.synaptic_consolidation is not None:
            self.synaptic_consolidation.estimate_fisher_information(train_loader)
            self.synaptic_consolidation.update_optimal_params()
        
        # Evaluate performance on current task
        final_performance = self.evaluate_task(train_loader)
        task_stats['final_performance'] = final_performance
        
        self.task_performance[task_id] = final_performance
        
        logger.info(f"Task {task_id} learning completed. Performance: {final_performance:.3f}")
        
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