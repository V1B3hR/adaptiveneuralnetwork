"""
Example: Using the Trainer class with callbacks.

This script demonstrates how to use the new Phase 4 Trainer class
with various callbacks for logging and profiling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from adaptiveneuralnetwork.training.callbacks import LoggingCallback, ProfilingCallback
from adaptiveneuralnetwork.training.trainer import Trainer


class SimpleClassifier(nn.Module):
    """Simple feedforward classifier for demonstration."""

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def create_dummy_dataset(num_samples=1000, input_dim=784, num_classes=10, batch_size=32):
    """Create a dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def example_basic_training():
    """Example 1: Basic training with logging."""
    print("\n" + "="*70)
    print("Example 1: Basic Training with LoggingCallback")
    print("="*70)

    # Create model, optimizer, and loss
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = create_dummy_dataset(num_samples=1000, batch_size=32)
    val_loader = create_dummy_dataset(num_samples=200, batch_size=32)

    # Create trainer with logging callback
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[LoggingCallback(log_interval=10, verbose=True)],
        seed=42,  # For reproducibility
    )

    # Train
    metrics = trainer.fit(
        train_loader=train_loader,
        num_epochs=3,
        val_loader=val_loader,
    )

    print(f"\nFinal validation accuracy: {metrics[-1]['val_accuracy']:.2%}")


def example_with_profiling():
    """Example 2: Training with profiling."""
    print("\n" + "="*70)
    print("Example 2: Training with ProfilingCallback")
    print("="*70)

    # Create model, optimizer, and loss
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = create_dummy_dataset(num_samples=500, batch_size=32)

    # Create trainer with profiling callback
    profiling_cb = ProfilingCallback(profile_memory=True, profile_cuda=False)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[profiling_cb],
        seed=42,
    )

    # Train
    trainer.fit(train_loader=train_loader, num_epochs=2)

    # Get profiling metrics
    metrics = profiling_cb.get_metrics()
    print(f"\nAverage batch time: {sum(metrics['batch_times']) / len(metrics['batch_times']) * 1000:.2f}ms")


def example_with_amp():
    """Example 3: Training with Automatic Mixed Precision."""
    print("\n" + "="*70)
    print("Example 3: Training with AMP (Automatic Mixed Precision)")
    print("="*70)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model, optimizer, and loss
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = create_dummy_dataset(num_samples=500, batch_size=32)

    # Create trainer with AMP enabled
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=torch.cuda.is_available(),  # Only use AMP if CUDA is available
        callbacks=[LoggingCallback(log_interval=5, verbose=True)],
        seed=42,
    )

    # Train
    metrics = trainer.fit(train_loader=train_loader, num_epochs=2)

    print(f"\nAMP enabled: {trainer.use_amp}")
    print(f"Final training accuracy: {metrics[-1]['train_accuracy']:.2%}")


def example_with_gradient_accumulation():
    """Example 4: Training with gradient accumulation."""
    print("\n" + "="*70)
    print("Example 4: Training with Gradient Accumulation")
    print("="*70)

    # Create model, optimizer, and loss
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders with smaller batch size
    train_loader = create_dummy_dataset(num_samples=500, batch_size=16)

    # Create trainer with gradient accumulation
    # This effectively uses a batch size of 16 * 4 = 64
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        gradient_accumulation_steps=4,
        callbacks=[LoggingCallback(log_interval=5, verbose=True)],
        seed=42,
    )

    # Train
    metrics = trainer.fit(train_loader=train_loader, num_epochs=2)

    print(f"\nGradient accumulation steps: {trainer.gradient_accumulation_steps}")
    print(f"Effective batch size: {16 * trainer.gradient_accumulation_steps}")
    print(f"Final training accuracy: {metrics[-1]['train_accuracy']:.2%}")


def example_multiple_callbacks():
    """Example 5: Training with multiple callbacks."""
    print("\n" + "="*70)
    print("Example 5: Training with Multiple Callbacks")
    print("="*70)

    # Create model, optimizer, and loss
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = create_dummy_dataset(num_samples=1000, batch_size=32)
    val_loader = create_dummy_dataset(num_samples=200, batch_size=32)

    # Create multiple callbacks
    logging_cb = LoggingCallback(log_interval=10, verbose=True)
    profiling_cb = ProfilingCallback(profile_memory=False, profile_cuda=False)

    # Create trainer with multiple callbacks
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[logging_cb, profiling_cb],
        seed=42,
    )

    # Train
    metrics = trainer.fit(
        train_loader=train_loader,
        num_epochs=3,
        val_loader=val_loader,
    )

    # Show profiling summary (automatically printed by profiling callback)
    prof_metrics = profiling_cb.get_metrics()
    print(f"\nCollected {len(prof_metrics['batch_times'])} batch time measurements")
    print(f"Final validation accuracy: {metrics[-1]['val_accuracy']:.2%}")


def example_checkpoint_save_load():
    """Example 6: Saving and loading checkpoints."""
    print("\n" + "="*70)
    print("Example 6: Checkpoint Saving and Loading")
    print("="*70)

    import os
    import tempfile

    # Create model, optimizer, and loss
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Create data loaders
    train_loader = create_dummy_dataset(num_samples=500, batch_size=32)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        callbacks=[LoggingCallback(log_interval=5, verbose=False)],
        seed=42,
    )

    # Train for a few epochs
    print("Training initial model...")
    metrics1 = trainer.fit(train_loader=train_loader, num_epochs=2)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        trainer.save_checkpoint(checkpoint_path, custom_data="Phase 4 Example")
        print(f"Checkpoint saved to {checkpoint_path}")

        # Create new model and trainer
        new_model = SimpleClassifier()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_trainer = Trainer(
            model=new_model,
            optimizer=new_optimizer,
            criterion=criterion,
            seed=42,
        )

        # Load checkpoint
        checkpoint = new_trainer.load_checkpoint(checkpoint_path)
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        print(f"Custom data: {checkpoint.get('custom_data', 'N/A')}")

        # Continue training
        print("\nContinuing training from checkpoint...")
        metrics2 = new_trainer.fit(train_loader=train_loader, num_epochs=1)

        print(f"\nOriginal final loss: {metrics1[-1]['train_loss']:.4f}")
        print(f"Resumed final loss: {metrics2[-1]['train_loss']:.4f}")


if __name__ == "__main__":
    # Run all examples
    example_basic_training()
    example_with_profiling()
    example_with_amp()
    example_with_gradient_accumulation()
    example_multiple_callbacks()
    example_checkpoint_save_load()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
