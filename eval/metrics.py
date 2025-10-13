"""
Standardized metrics computation for model evaluation.

Provides consistent metric calculation across different model types and tasks.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class StandardMetrics:
    """Container for standardized evaluation metrics."""
    
    accuracy: float = 0.0
    loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Additional metrics
    latency_ms: float = 0.0
    throughput: float = 0.0
    
    # Custom metrics dictionary
    custom_metrics: dict[str, float] = field(default_factory=dict)
    
    # Metadata
    num_samples: int = 0
    evaluation_time: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "accuracy": self.accuracy,
            "loss": self.loss,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "custom_metrics": self.custom_metrics,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time,
        }


def compute_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    compute_detailed: bool = True,
) -> StandardMetrics:
    """
    Compute standardized metrics for a model.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation data
        device: Device to run evaluation on
        loss_fn: Loss function (optional)
        compute_detailed: Whether to compute detailed metrics (precision, recall, F1)
        
    Returns:
        StandardMetrics object with computed metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # For detailed metrics
    all_predictions = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    inputs = batch[0]
                    labels = batch[1] if len(batch) > 1 else None
            else:
                inputs = batch
                labels = None
            
            inputs = inputs.to(device)
            if labels is not None:
                labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss if available
            if loss_fn is not None and labels is not None:
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
            
            # Compute accuracy
            if labels is not None:
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    predictions = outputs.argmax(dim=1)
                else:
                    predictions = (outputs > 0.5).squeeze()
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                if compute_detailed:
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
    
    evaluation_time = time.time() - start_time
    
    # Compute basic metrics
    metrics = StandardMetrics()
    metrics.num_samples = total_samples
    metrics.evaluation_time = evaluation_time
    
    if total_samples > 0:
        metrics.accuracy = 100.0 * total_correct / total_samples
        metrics.throughput = total_samples / evaluation_time if evaluation_time > 0 else 0.0
        metrics.latency_ms = 1000.0 * evaluation_time / len(data_loader) if len(data_loader) > 0 else 0.0
    
    if loss_fn is not None and len(data_loader) > 0:
        metrics.loss = total_loss / len(data_loader)
    
    # Compute detailed metrics
    if compute_detailed and len(all_predictions) > 0:
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Get unique classes
        unique_classes = np.unique(all_labels)
        
        # Compute per-class metrics and average
        precisions = []
        recalls = []
        f1_scores = []
        
        for cls in unique_classes:
            tp = np.sum((all_predictions == cls) & (all_labels == cls))
            fp = np.sum((all_predictions == cls) & (all_labels != cls))
            fn = np.sum((all_predictions != cls) & (all_labels == cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        metrics.precision = np.mean(precisions) if precisions else 0.0
        metrics.recall = np.mean(recalls) if recalls else 0.0
        metrics.f1_score = np.mean(f1_scores) if f1_scores else 0.0
    
    return metrics


def compute_custom_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    metric_functions: dict[str, Any],
) -> dict[str, float]:
    """
    Compute custom metrics defined by user functions.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation data
        device: Device to run evaluation on
        metric_functions: Dictionary of metric name to function
        
    Returns:
        Dictionary of metric name to value
    """
    model.eval()
    
    custom_metrics = {}
    
    with torch.no_grad():
        for metric_name, metric_fn in metric_functions.items():
            try:
                metric_value = metric_fn(model, data_loader, device)
                custom_metrics[metric_name] = float(metric_value)
            except Exception as e:
                print(f"Warning: Failed to compute metric {metric_name}: {e}")
                custom_metrics[metric_name] = 0.0
    
    return custom_metrics
