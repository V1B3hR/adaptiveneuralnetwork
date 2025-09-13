"""
Enhanced memory systems for continual learning with dynamic prioritization.

This module extends the existing episodic memory with improved importance-based sampling
and time-series analysis as part of Phase 1: Adaptive Learning & Continual Improvement.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMemoryConfig:
    """Configuration for enhanced memory systems."""
    # Memory parameters
    memory_size: int = 10000
    importance_decay: float = 0.99    # Decay rate for importance scores
    temporal_weight: float = 0.1      # Weight for temporal importance
    
    # Dynamic prioritization
    priority_alpha: float = 0.6       # Priority sampling exponent
    importance_sampling_weight: float = 0.4  # IS weight for bias correction
    
    # Time-series analysis
    rolling_window: int = 1000        # Rolling window for time-series analysis
    event_threshold: float = 0.8      # Threshold for significant events
    trend_sensitivity: float = 0.05   # Sensitivity for trend detection


class TimeSeriesAnalyzer:
    """Analyzes time-series patterns in learning experiences."""
    
    def __init__(self, config: EnhancedMemoryConfig):
        self.config = config
        self.history = deque(maxlen=config.rolling_window)
        self.timestamps = deque(maxlen=config.rolling_window)
        
    def add_sample(self, value: float, timestamp: Optional[float] = None):
        """Add a new sample to the time series."""
        if timestamp is None:
            timestamp = time.time()
        
        self.history.append(value)
        self.timestamps.append(timestamp)
    
    def detect_significant_events(self) -> List[Tuple[int, float, str]]:
        """
        Detect significant events in the time series.
        
        Returns:
            List of (index, value, event_type) tuples
        """
        if len(self.history) < 10:
            return []
        
        events = []
        history_array = np.array(self.history)
        
        # Detect sudden spikes
        rolling_mean = np.convolve(history_array, np.ones(5)/5, mode='valid')
        rolling_std = np.array([np.std(history_array[max(0, i-4):i+1]) 
                               for i in range(len(history_array))])
        
        for i in range(5, len(history_array)):
            if i-5 < len(rolling_mean):
                z_score = abs(history_array[i] - rolling_mean[i-5]) / (rolling_std[i] + 1e-8)
                if z_score > 2.0:
                    events.append((i, history_array[i], 'spike'))
        
        # Detect trend changes
        if len(history_array) >= 20:
            for i in range(10, len(history_array) - 10):
                before_trend = np.polyfit(range(10), history_array[i-10:i], 1)[0]
                after_trend = np.polyfit(range(10), history_array[i:i+10], 1)[0]
                
                if abs(before_trend - after_trend) > self.config.trend_sensitivity:
                    events.append((i, history_array[i], 'trend_change'))
        
        return events
    
    def get_temporal_importance(self, index: int) -> float:
        """Calculate temporal importance for a given index."""
        if index >= len(self.history):
            return 0.0
        
        # Recent samples are more important
        recency_weight = 1.0 - (len(self.history) - index - 1) / len(self.history)
        
        # Samples near significant events are more important
        events = self.detect_significant_events()
        event_proximity = 0.0
        
        for event_idx, _, _ in events:
            distance = abs(index - event_idx)
            if distance < 10:  # Within 10 samples of an event
                event_proximity = max(event_proximity, 1.0 - distance / 10.0)
        
        return self.config.temporal_weight * recency_weight + event_proximity


class DynamicPriorityBuffer:
    """Enhanced experience replay buffer with dynamic prioritization."""
    
    def __init__(self, config: EnhancedMemoryConfig, feature_size: int):
        self.config = config
        self.feature_size = feature_size
        self.memory_size = config.memory_size
        
        # Memory buffers
        self.features = torch.zeros(self.memory_size, feature_size)
        self.labels = torch.zeros(self.memory_size, dtype=torch.long)
        self.tasks = torch.zeros(self.memory_size, dtype=torch.long)
        self.importance_scores = torch.zeros(self.memory_size)
        self.priorities = torch.zeros(self.memory_size)
        self.timestamps = torch.zeros(self.memory_size)
        
        # Priority tree for efficient sampling
        self.priority_tree = SumTree(self.memory_size)
        
        # Time-series analyzer
        self.time_series_analyzer = TimeSeriesAnalyzer(config)
        
        # Tracking
        self.write_pointer = 0
        self.stored_samples = 0
        
        logger.debug(f"Initialized DynamicPriorityBuffer with {self.memory_size} capacity")
    
    def store(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        loss_values: Optional[torch.Tensor] = None
    ):
        """Store experiences with dynamic importance calculation."""
        batch_size = features.size(0)
        current_time = time.time()
        
        for i in range(batch_size):
            idx = self.write_pointer % self.memory_size
            
            # Store basic information
            self.features[idx] = features[i].detach()
            self.labels[idx] = labels[i]
            self.tasks[idx] = task_id
            self.timestamps[idx] = current_time
            
            # Calculate importance score
            if loss_values is not None:
                base_importance = loss_values[i].item()
            else:
                base_importance = 1.0  # Default importance
            
            # Add temporal importance
            temporal_importance = self.time_series_analyzer.get_temporal_importance(idx)
            total_importance = base_importance + temporal_importance
            
            self.importance_scores[idx] = total_importance
            
            # Calculate priority (importance^alpha)
            priority = total_importance ** self.config.priority_alpha
            self.priorities[idx] = priority
            
            # Update priority tree
            self.priority_tree.update(idx, priority)
            
            # Update time-series analyzer
            self.time_series_analyzer.add_sample(total_importance, current_time)
            
            self.write_pointer += 1
            self.stored_samples = min(self.stored_samples + 1, self.memory_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Sample experiences using prioritized sampling.
        
        Returns:
            features, labels, tasks, importance_weights, indices
        """
        if self.stored_samples == 0:
            return (torch.empty(0, self.feature_size), torch.empty(0, dtype=torch.long),
                   torch.empty(0, dtype=torch.long), torch.empty(0), np.array([]))
        
        # Sample from priority tree
        indices = []
        priorities = []
        
        segment_size = self.priority_tree.total() / batch_size
        
        for i in range(batch_size):
            start = segment_size * i
            end = segment_size * (i + 1)
            
            # Sample uniformly within segment
            sample_value = np.random.uniform(start, end)
            idx, priority = self.priority_tree.get_leaf(sample_value)
            
            indices.append(idx)
            priorities.append(priority)
        
        indices = np.array(indices)
        priorities = np.array(priorities)
        
        # Calculate importance sampling weights
        max_priority = self.priority_tree.max_leaf_value()
        sampling_probabilities = priorities / self.priority_tree.total()
        importance_weights = (1.0 / (self.stored_samples * sampling_probabilities)) ** self.config.importance_sampling_weight
        importance_weights = importance_weights / importance_weights.max()  # Normalize
        
        # Extract sampled data
        sampled_features = self.features[indices]
        sampled_labels = self.labels[indices]
        sampled_tasks = self.tasks[indices]
        
        return (sampled_features, sampled_labels, sampled_tasks, 
                torch.tensor(importance_weights, dtype=torch.float32), indices)
    
    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor):
        """Update priorities based on temporal difference errors."""
        for idx, td_error in zip(indices, td_errors):
            # Add small epsilon to prevent zero priorities
            priority = (abs(td_error.item()) + 1e-6) ** self.config.priority_alpha
            self.priorities[idx] = priority
            self.priority_tree.update(idx, priority)
    
    def decay_importance(self):
        """Apply decay to importance scores."""
        self.importance_scores *= self.config.importance_decay
        
        # Recalculate priorities
        for idx in range(self.stored_samples):
            priority = self.importance_scores[idx] ** self.config.priority_alpha
            self.priorities[idx] = priority
            self.priority_tree.update(idx, priority)


class SumTree:
    """Sum tree for efficient prioritized sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_start = capacity - 1
    
    def update(self, data_idx: int, priority: float):
        """Update priority for a specific data index."""
        tree_idx = data_idx + self.data_start
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def get_leaf(self, value: float) -> Tuple[int, float]:
        """Get leaf index and priority for a given cumulative value."""
        idx = 0
        
        while idx < self.data_start:
            left_child = 2 * idx + 1
            right_child = 2 * idx + 2
            
            if value <= self.tree[left_child]:
                idx = left_child
            else:
                value -= self.tree[left_child]
                idx = right_child
        
        data_idx = idx - self.data_start
        return data_idx, self.tree[idx]
    
    def total(self) -> float:
        """Get total sum of all priorities."""
        return self.tree[0]
    
    def max_leaf_value(self) -> float:
        """Get maximum leaf value."""
        return np.max(self.tree[self.data_start:])


class EventDrivenLearningSystem:
    """System for event-driven learning using enhanced memory."""
    
    def __init__(self, config: EnhancedMemoryConfig, model: nn.Module):
        self.config = config
        self.model = model
        
        # Determine feature size more carefully
        feature_size = 64  # default based on the model's input
        if hasattr(model, 'get_feature_size'):
            feature_size = model.get_feature_size()
        elif hasattr(model, 'fc'):
            # Use input features of the first linear layer
            feature_size = model.fc.in_features
        
        self.memory_buffer = DynamicPriorityBuffer(config, feature_size)
        
        # Event tracking
        self.significant_events = []
        self.learning_triggered_count = 0
        
        logger.info("Initialized EventDrivenLearningSystem")
    
    def process_experience(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        loss_values: Optional[torch.Tensor] = None
    ):
        """Process new experience and trigger learning if needed."""
        # Store experience
        self.memory_buffer.store(features, labels, task_id, loss_values)
        
        # Check for significant events
        events = self.memory_buffer.time_series_analyzer.detect_significant_events()
        new_events = len(events) - len(self.significant_events)
        
        if new_events > 0:
            logger.debug(f"Detected {new_events} new significant events")
            self.significant_events = events
            
            # Trigger intensive learning
            self._trigger_event_driven_learning()
    
    def _trigger_event_driven_learning(self):
        """Trigger intensive learning in response to significant events."""
        self.learning_triggered_count += 1
        
        # Sample important experiences
        batch_size = min(64, self.memory_buffer.stored_samples)
        if batch_size == 0:
            return
        
        features, labels, tasks, importance_weights, indices = self.memory_buffer.sample(batch_size)
        
        # Intensive training on important samples
        self.model.train()
        
        for _ in range(5):  # Multiple passes for intensive learning
            if hasattr(self.model, 'train_step'):
                loss_info = self.model.train_step(features, labels)
                
                # Update priorities based on losses
                if 'loss' in loss_info:
                    td_errors = torch.full((batch_size,), loss_info['loss'])
                    self.memory_buffer.update_priorities(indices, td_errors)
            else:
                # Basic training
                predictions = self.model(features)
                losses = nn.CrossEntropyLoss(reduction='none')(predictions, labels)
                weighted_loss = (losses * importance_weights).mean()
                
                if hasattr(self.model, 'optimizer'):
                    self.model.optimizer.zero_grad()
                    weighted_loss.backward()
                    self.model.optimizer.step()
                
                # Update priorities
                self.memory_buffer.update_priorities(indices, losses.detach())
        
        logger.info(f"Event-driven learning triggered (#{self.learning_triggered_count})")
    
    def get_memory_statistics(self) -> Dict[str, float]:
        """Get statistics about the memory system."""
        if self.memory_buffer.stored_samples == 0:
            return {}
        
        return {
            'stored_samples': self.memory_buffer.stored_samples,
            'avg_importance': self.memory_buffer.importance_scores[:self.memory_buffer.stored_samples].mean().item(),
            'max_importance': self.memory_buffer.importance_scores[:self.memory_buffer.stored_samples].max().item(),
            'significant_events': len(self.significant_events),
            'learning_triggers': self.learning_triggered_count,
            'priority_tree_total': self.memory_buffer.priority_tree.total()
        }