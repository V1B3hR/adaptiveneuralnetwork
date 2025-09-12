"""
Training loops for adaptive neural networks.

This module provides training and evaluation loops with metrics tracking
and logging capabilities.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..api.model import AdaptiveModel
from ..api.config import AdaptiveConfig


class TrainingLoop:
    """Main training loop for adaptive neural networks."""
    
    def __init__(
        self,
        model: AdaptiveModel,
        config: AdaptiveConfig,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None
    ):
        self.model = model
        self.config = config
        
        # Set up optimizer
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Set up loss criterion
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_epoch = 0
        
        # Setup checkpoint directory
        if config.save_checkpoint:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move to device
            data = data.to(self.config.device)
            target = target.to(self.config.device)
            
            # Flatten input for MNIST
            if data.dim() == 4:  # [batch, channels, height, width]
                data = data.view(data.size(0), -1)
                
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Add adaptive loss component (but don't call model.compute_loss as it calls backward)
            # Just use the standard loss for now - adaptive loss will be added in future versions
            total_loss_batch = loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Track metrics
            total_loss += total_loss_batch.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, '
                      f'Loss: {total_loss_batch.item():.6f}, '
                      f'Acc: {100. * correct / total:.2f}%')
                      
        epoch_time = time.time() - start_time
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': 100. * correct / total,
            'epoch_time': epoch_time
        }
        
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # Move to device
                data = data.to(self.config.device)
                target = target.to(self.config.device)
                
                # Flatten input for MNIST
                if data.dim() == 4:
                    data = data.view(data.size(0), -1)
                    
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Track accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return {
            'val_loss': total_loss / len(test_loader),
            'val_accuracy': 100. * correct / total
        }
        
    def train(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate if test loader provided
            val_metrics = {}
            if test_loader is not None:
                val_metrics = self.evaluate(test_loader)
                
            # Get model-specific metrics
            model_metrics = self.model.get_metrics()
            
            # Combine all metrics
            epoch_metrics = {
                'epoch': epoch,
                'timestamp': time.time(),
                **train_metrics,
                **val_metrics,
                **model_metrics
            }
            
            self.metrics_history.append(epoch_metrics)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"  Train Acc:  {train_metrics['train_accuracy']:.2f}%")
            if val_metrics:
                print(f"  Val Loss:   {val_metrics['val_loss']:.6f}")
                print(f"  Val Acc:    {val_metrics['val_accuracy']:.2f}%")
            print(f"  Active Nodes: {model_metrics['active_node_ratio']:.3f}")
            print(f"  Mean Energy:  {model_metrics['mean_energy']:.3f}")
            print("-" * 50)
            
            # Save checkpoint
            if self.config.save_checkpoint and (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                
        # Save final metrics
        if self.config.metrics_file:
            self.save_metrics()
            
        print("Training completed!")
        return self.metrics_history
        
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.get_state_dict_full(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics_history': self.metrics_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.model.load_state_dict_full(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {checkpoint_path}, epoch {epoch}")
        return epoch
        
    def save_metrics(self) -> None:
        """Save metrics history to JSON file."""
        if self.config.metrics_file:
            with open(self.config.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            print(f"Metrics saved: {self.config.metrics_file}")


def quick_train(
    config: AdaptiveConfig,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    num_epochs: int = 1
) -> Dict[str, Any]:
    """
    Quick training function for testing and development.
    
    Args:
        config: Model configuration
        train_loader: Training data loader
        test_loader: Optional test data loader
        num_epochs: Number of epochs to train
        
    Returns:
        Final metrics dictionary
    """
    # Create model
    model = AdaptiveModel(config)
    
    # Create training loop
    trainer = TrainingLoop(model, config)
    
    # Train
    metrics_history = trainer.train(train_loader, test_loader, num_epochs)
    
    return {
        'model': model,
        'trainer': trainer,
        'metrics_history': metrics_history,
        'final_metrics': metrics_history[-1] if metrics_history else {}
    }