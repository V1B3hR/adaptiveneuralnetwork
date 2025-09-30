"""
Trainer class for centralized training orchestration.

This module provides a flexible Trainer class that orchestrates the training
process with support for callbacks, AMP, gradient accumulation, and
deterministic execution.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Use newer AMP API if available (PyTorch 1.10+)
try:
    from torch.amp import GradScaler, autocast
    AMP_DEVICE = 'cuda'  # For newer API
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_DEVICE = None  # For older API

from .callbacks import Callback, CallbackList


class Trainer:
    """
    Centralized trainer for model training and evaluation.
    
    Features:
    - Callback system for extensibility
    - Automatic Mixed Precision (AMP) support
    - Gradient accumulation
    - Deterministic seed initialization
    - Flexible fit/evaluate interface
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Trainer.
        
        Args:
            model: Neural network model to train
            optimizer: Optimizer for model parameters
            criterion: Loss function
            device: Device to run training on (default: auto-detect)
            callbacks: List of callbacks to use during training
            use_amp: Enable Automatic Mixed Precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping (None to disable)
            seed: Random seed for reproducibility (None for non-deterministic)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.callbacks = CallbackList(callbacks)
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize AMP scaler if enabled
        if use_amp:
            if AMP_DEVICE:
                self.scaler = GradScaler(AMP_DEVICE)
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Set deterministic seed if provided
        if seed is not None:
            self.set_seed(seed)
        
        # Training state
        self.current_epoch = 0
        self.num_epochs = 0
        self.metrics_history = []
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for deterministic training.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic CUDA operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
    ) -> List[Dict[str, Any]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            val_loader: Optional DataLoader for validation data
        
        Returns:
            List of metrics dictionaries for each epoch
        """
        self.num_epochs = num_epochs
        self.metrics_history = []
        
        # Call on_train_begin callbacks
        self.callbacks.on_train_begin(self)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Call on_epoch_begin callbacks
            self.callbacks.on_epoch_begin(epoch, self)
            
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Evaluate if validation loader provided
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            
            # Combine metrics
            epoch_metrics = {
                'epoch': epoch,
                **train_metrics,
                **val_metrics,
            }
            self.metrics_history.append(epoch_metrics)
            
            # Call on_epoch_end callbacks
            self.callbacks.on_epoch_end(epoch, self, logs=epoch_metrics)
        
        # Call on_train_end callbacks
        self.callbacks.on_train_end(self, logs={'metrics_history': self.metrics_history})
        
        return self.metrics_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Call on_batch_begin callbacks
            batch_logs = {'batch_idx': batch_idx, 'batch_size': data.size(0)}
            self.callbacks.on_batch_begin(batch_idx, self, logs=batch_logs)
            
            # Forward pass with optional AMP
            if self.use_amp:
                if AMP_DEVICE:
                    with autocast(AMP_DEVICE):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                else:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Call on_backward_end callbacks
            self.callbacks.on_backward_end(batch_idx, self, logs=batch_logs)
            
            # Update weights with gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Gradient clipping with AMP
                    if self.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping without AMP
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update batch logs
            batch_logs['loss'] = loss.item() * self.gradient_accumulation_steps
            batch_logs['accuracy'] = correct / total if total > 0 else 0.0
            
            # Call on_batch_end callbacks
            self.callbacks.on_batch_end(batch_idx, self, logs=batch_logs)
        
        # Handle remaining gradients if batch count not divisible by accumulation steps
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': correct / total if total > 0 else 0.0,
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            val_loader: DataLoader for validation/test data
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Call on_evaluate_begin callbacks
        self.callbacks.on_evaluate_begin(self)
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass with optional AMP
                if self.use_amp:
                    if AMP_DEVICE:
                        with autocast(AMP_DEVICE):
                            output = self.model(data)
                            loss = self.criterion(output, target)
                    else:
                        with autocast():
                            output = self.model(data)
                            loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total if total > 0 else 0.0,
        }
        
        # Call on_evaluate_end callbacks
        self.callbacks.on_evaluate_end(self, logs=metrics)
        
        return metrics
    
    def save_checkpoint(self, path: str, **extra_data: Any) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            **extra_data: Additional data to save in checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'metrics_history': self.metrics_history,
            **extra_data,
        }
        
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            Dictionary with checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint
