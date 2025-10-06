"""
Trainer class for training neural networks with callbacks and advanced features.

This module provides a flexible Trainer class that supports:
- Callback system for extensibility
- Automatic Mixed Precision (AMP) training
- Gradient accumulation
- Deterministic seed initialization
- Checkpoint saving and loading
"""

import random
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader

from .callbacks import Callback, CallbackList


class Trainer:
    """
    Trainer class for training neural networks with advanced features.
    
    Features:
    - Callback system for extensible training logic
    - Automatic Mixed Precision (AMP) support
    - Gradient accumulation for larger effective batch sizes
    - Deterministic seed initialization
    - Checkpoint saving and loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: Neural network model to train
            optimizer: Optimizer for updating model parameters
            criterion: Loss function
            device: Device to train on (defaults to CPU)
            callbacks: List of callbacks for training hooks
            use_amp: Enable Automatic Mixed Precision training
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping (None to disable)
            seed: Random seed for reproducibility (None to disable)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cpu')
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize callbacks
        self.callbacks = CallbackList(callbacks)
        
        # Initialize AMP scaler if needed
        self.scaler = None
        if self.use_amp:
            # Create scaler for AMP
            # Try new API first (torch.amp), fall back to old API for compatibility
            try:
                # New API (PyTorch >= 2.0)
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.scaler = torch.amp.GradScaler(device_type)
            except AttributeError:
                # Old API fallback
                self.scaler = torch.cuda.amp.GradScaler()
        
        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)
        
        # Training state
        self.num_epochs = 0
        self.current_epoch = 0
        self.metrics_history = []
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
    ) -> List[Dict[str, Any]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            val_loader: Optional DataLoader for validation data
        
        Returns:
            List of metrics dictionaries, one per epoch
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
            
            # Evaluate on validation set if provided
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(epoch_metrics)
            
            # Call on_epoch_end callbacks
            self.callbacks.on_epoch_end(epoch, self, logs=epoch_metrics)
        
        # Call on_train_end callbacks
        self.callbacks.on_train_end(self)
        
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
            batch_logs = {'batch_size': data.size(0)}
            self.callbacks.on_batch_begin(batch_idx, self, logs=batch_logs)
            
            # Forward pass with optional AMP
            if self.use_amp and self.scaler is not None:
                # Try new API first, fall back to old API
                try:
                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                    with torch.amp.autocast(device_type):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss = loss / self.gradient_accumulation_steps
                except AttributeError:
                    # Old API fallback
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss = loss / self.gradient_accumulation_steps
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Call on_backward_end callbacks
            self.callbacks.on_backward_end(batch_idx, self, logs=batch_logs)
            
            # Update weights after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm is not None:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                
                # Optimizer step
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Zero gradients
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
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': correct / total if total > 0 else 0.0,
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Call on_evaluate_begin callbacks
        self.callbacks.on_evaluate_begin(self)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Track metrics
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total if total > 0 else 0.0,
        }
        
        # Call on_evaluate_end callbacks
        self.callbacks.on_evaluate_end(self, logs=val_metrics)
        
        return val_metrics
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the current training state.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a checkpoint and restore training state.
        
        Args:
            path: Path to the checkpoint file
        
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint
