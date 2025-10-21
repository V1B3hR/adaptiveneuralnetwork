"""
Enhanced Trainer class for advanced neural network training with extensibility, monitoring, and robust features.

Key Features:
- Extensible callback system for custom training hooks
- Automatic Mixed Precision (AMP) training with device auto-detection
- Gradient accumulation for effective large-batch training
- Deterministic seed initialization for reproducibility
- Checkpoint saving/loading for resuming training
- Progress bar support via tqdm (optional)
- Early stopping and learning rate scheduling support via callbacks
- Custom metrics extensibility
"""

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # fallback to no progress bar

# Handle relative imports for both module use and direct execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add parent directory to path to enable imports when run as script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from adaptiveneuralnetwork.training.callbacks import Callback, CallbackList
else:
    from .callbacks import Callback, CallbackList


class Trainer:
    """
    Enhanced Trainer for neural network training with advanced features and extensibility.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | None = None,
        callbacks: list[Callback] | None = None,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        seed: int | None = None,
        metrics: dict[str, Callable] | None = None,
        progress_bar: bool = True,
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
            metrics: Custom metrics functions {'name': fn(outputs, targets) -> float}
            progress_bar: Show tqdm progress bar during training/validation
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.progress_bar = progress_bar

        # Move model to device
        self.model.to(self.device)

        # Initialize callbacks
        self.callbacks = CallbackList(callbacks)

        # AMP scaler initialization
        self.scaler = None
        if self.use_amp:
            try:
                # PyTorch >= 2.0
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.scaler = torch.amp.GradScaler(device_type)
            except AttributeError:
                # PyTorch < 2.0 fallback
                self.scaler = torch.cuda.amp.GradScaler()

        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)

        # Training state
        self.num_epochs = 0
        self.current_epoch = 0
        self.metrics_history = []

        # Custom metrics
        self.metrics = metrics or {}

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: DataLoader | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        early_stopping: Callback | None = None,
    ) -> list[dict[str, Any]]:
        """
        Train the model for a specified number of epochs.

        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of epochs to train
            val_loader: Optional DataLoader for validation data
            scheduler: Optional LR scheduler
            early_stopping: Optional early stopping callback

        Returns:
            List of metrics dictionaries, one per epoch
        """
        self.num_epochs = num_epochs
        self.metrics_history = []

        if early_stopping is not None:
            self.callbacks.append(early_stopping)

        # Call on_train_begin callbacks
        self.callbacks.on_train_begin(self)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
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

            # Scheduler step
            if scheduler is not None:
                if val_loader is not None and hasattr(scheduler, "step"):
                    # For ReduceLROnPlateau, pass val_loss
                    if "val_loss" in val_metrics:
                        scheduler.step(val_metrics["val_loss"])
                    else:
                        scheduler.step()
                else:
                    scheduler.step()

            self.callbacks.on_epoch_end(epoch, self, logs=epoch_metrics)

            # Early stopping check
            if early_stopping is not None and getattr(early_stopping, "stop_training", False):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.callbacks.on_train_end(self)
        return self.metrics_history

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
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
        metric_sums = dict.fromkeys(self.metrics, 0.0)

        batch_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch+1}") if self.progress_bar else enumerate(train_loader)
        for batch_idx, (data, target) in batch_iter:
            data = data.to(self.device)
            target = target.to(self.device)

            batch_logs = {'batch_size': data.size(0)}
            self.callbacks.on_batch_begin(batch_idx, self, logs=batch_logs)

            # Forward pass with AMP
            if self.use_amp and self.scaler is not None:
                try:
                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                    with torch.amp.autocast(device_type):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                        loss = loss / self.gradient_accumulation_steps
                except AttributeError:
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

            self.callbacks.on_backward_end(batch_idx, self, logs=batch_logs)

            # Update weights after accumulating gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if self.max_grad_norm is not None:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Custom metrics
            for name, fn in self.metrics.items():
                metric_sums[name] += fn(output, target)

            batch_logs['loss'] = loss.item() * self.gradient_accumulation_steps
            batch_logs['accuracy'] = correct / total if total > 0 else 0.0
            for name, fn in self.metrics.items():
                batch_logs[name] = fn(output, target)

            self.callbacks.on_batch_end(batch_idx, self, logs=batch_logs)

        metrics_dict = {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': correct / total if total > 0 else 0.0,
        }
        for name in self.metrics:
            metrics_dict[f'train_{name}'] = metric_sums[name] / len(train_loader)

        return metrics_dict

    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model on validation data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.callbacks.on_evaluate_begin(self)

        total_loss = 0.0
        correct = 0
        total = 0
        metric_sums = dict.fromkeys(self.metrics, 0.0)

        with torch.no_grad():
            batch_iter = tqdm(val_loader, desc="Validating") if self.progress_bar else val_loader
            for data, target in batch_iter:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                for name, fn in self.metrics.items():
                    metric_sums[name] += fn(output, target)

        val_metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total if total > 0 else 0.0,
        }
        for name in self.metrics:
            val_metrics[f'val_{name}'] = metric_sums[name] / len(val_loader)

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

    def load_checkpoint(self, path: str, strict: bool = True) -> dict[str, Any]:
        """
        Load a checkpoint and restore training state.

        Args:
            path: Path to the checkpoint file
            strict: Strictly enforce that the keys in state_dict match the model

        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.metrics_history = checkpoint.get('metrics_history', [])
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint


if __name__ == "__main__":
    """
    Demonstration of the Trainer class usage.
    This simple example shows how to train a basic neural network using the Trainer.
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from adaptiveneuralnetwork.training.callbacks import LoggingCallback

    print("=" * 70)
    print("Trainer Demo: Simple Neural Network Training")
    print("=" * 70)

    # Create a simple model
    class SimpleClassifier(nn.Module):
        """Simple feedforward classifier for demonstration."""
        def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.network(x)

    # Create dummy dataset
    print("\n1. Creating dummy dataset...")
    num_samples = 500
    input_dim = 784
    num_classes = 10
    batch_size = 32

    X_train = torch.randn(num_samples, input_dim)
    y_train = torch.randint(0, num_classes, (num_samples,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val = torch.randn(num_samples // 5, input_dim)
    y_val = torch.randint(0, num_classes, (num_samples // 5,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"   - Training samples: {num_samples}")
    print(f"   - Validation samples: {num_samples // 5}")
    print(f"   - Batch size: {batch_size}")

    # Create model, optimizer, and loss
    print("\n2. Initializing model and trainer...")
    model = SimpleClassifier(input_dim=input_dim, hidden_dim=128, output_dim=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create trainer with logging
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[LoggingCallback(log_interval=5, verbose=True)],
        seed=42,  # For reproducibility
        progress_bar=True,
    )

    print(f"   - Device: {trainer.device}")
    print("   - Seed: 42")

    # Train the model
    print("\n3. Training model...")
    num_epochs = 3
    metrics_history = trainer.fit(
        train_loader=train_loader,
        num_epochs=num_epochs,
        val_loader=val_loader,
    )

    # Display results
    print("\n" + "=" * 70)
    print("Training Results")
    print("=" * 70)
    for epoch, metrics in enumerate(metrics_history):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {metrics['train_loss']:.4f}, Train Acc: {metrics['train_accuracy']:.2%}")
        print(f"  Val Loss:   {metrics['val_loss']:.4f}, Val Acc:   {metrics['val_accuracy']:.2%}")

    print("\n" + "=" * 70)
    print("Demo completed successfully! ✓")
    print("=" * 70)
    print("\nFor more advanced examples, see:")
    print("  - examples/phase4_trainer_examples.py")
    print("  - tests/test_trainer_callbacks.py")
