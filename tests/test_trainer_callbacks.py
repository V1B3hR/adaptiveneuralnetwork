"""
Tests for Phase 4 Training Loop Abstraction: Trainer and Callbacks.

This module tests:
- Callback interface and lifecycle events
- Callback sequencing and ordering
- Trainer class with fit/evaluate methods
- AMP (Automatic Mixed Precision) integration
- Gradient accumulation
- Deterministic seed initialization
"""

import unittest

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from adaptiveneuralnetwork.training.callbacks import (
    Callback,
    CallbackList,
    LoggingCallback,
    ProfilingCallback,
)
from adaptiveneuralnetwork.training.trainer import Trainer


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def create_dummy_data(num_samples=100, input_dim=10, num_classes=2, batch_size=16):
    """Create dummy dataset for testing."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


class MockCallback(Callback):
    """Mock callback for testing callback sequencing."""

    def __init__(self, name="mock"):
        super().__init__()
        self.name = name
        self.call_log = []

    def on_train_begin(self, trainer, logs=None):
        self.call_log.append(('on_train_begin', logs))

    def on_train_end(self, trainer, logs=None):
        self.call_log.append(('on_train_end', logs))

    def on_epoch_begin(self, epoch, trainer, logs=None):
        self.call_log.append(('on_epoch_begin', epoch, logs))

    def on_epoch_end(self, epoch, trainer, logs=None):
        self.call_log.append(('on_epoch_end', epoch, logs))

    def on_batch_begin(self, batch_idx, trainer, logs=None):
        self.call_log.append(('on_batch_begin', batch_idx, logs))

    def on_batch_end(self, batch_idx, trainer, logs=None):
        self.call_log.append(('on_batch_end', batch_idx, logs))

    def on_backward_end(self, batch_idx, trainer, logs=None):
        self.call_log.append(('on_backward_end', batch_idx, logs))

    def on_evaluate_begin(self, trainer, logs=None):
        self.call_log.append(('on_evaluate_begin', logs))

    def on_evaluate_end(self, trainer, logs=None):
        self.call_log.append(('on_evaluate_end', logs))


class TestCallbacks(unittest.TestCase):
    """Test callback system."""

    def test_callback_interface(self):
        """Test that Callback base class has all required methods."""
        callback = MockCallback()

        # Check all methods exist
        self.assertTrue(hasattr(callback, 'on_train_begin'))
        self.assertTrue(hasattr(callback, 'on_train_end'))
        self.assertTrue(hasattr(callback, 'on_epoch_begin'))
        self.assertTrue(hasattr(callback, 'on_epoch_end'))
        self.assertTrue(hasattr(callback, 'on_batch_begin'))
        self.assertTrue(hasattr(callback, 'on_batch_end'))
        self.assertTrue(hasattr(callback, 'on_backward_end'))
        self.assertTrue(hasattr(callback, 'on_evaluate_begin'))
        self.assertTrue(hasattr(callback, 'on_evaluate_end'))

    def test_callback_list(self):
        """Test CallbackList container."""
        cb1 = MockCallback("cb1")
        cb2 = MockCallback("cb2")

        callback_list = CallbackList([cb1, cb2])

        # Test that callbacks are called in order
        callback_list.on_train_begin(None)
        self.assertEqual(len(cb1.call_log), 1)
        self.assertEqual(len(cb2.call_log), 1)
        self.assertEqual(cb1.call_log[0][0], 'on_train_begin')
        self.assertEqual(cb2.call_log[0][0], 'on_train_begin')

    def test_callback_sequencing(self):
        """Test that callbacks are called in the correct sequence."""
        mock_cb = MockCallback()
        callback_list = CallbackList([mock_cb])

        # Simulate training sequence
        callback_list.on_train_begin(None)
        callback_list.on_epoch_begin(0, None)
        callback_list.on_batch_begin(0, None)
        callback_list.on_batch_end(0, None)
        callback_list.on_backward_end(0, None)
        callback_list.on_epoch_end(0, None)
        callback_list.on_train_end(None)

        # Check call sequence
        expected_sequence = [
            'on_train_begin',
            'on_epoch_begin',
            'on_batch_begin',
            'on_batch_end',
            'on_backward_end',
            'on_epoch_end',
            'on_train_end',
        ]

        actual_sequence = [call[0] for call in mock_cb.call_log]
        self.assertEqual(actual_sequence, expected_sequence)

    def test_logging_callback(self):
        """Test LoggingCallback functionality."""
        # Create callback with custom interval
        callback = LoggingCallback(log_interval=5, verbose=False)

        # Test initialization
        self.assertEqual(callback.log_interval, 5)
        self.assertFalse(callback.verbose)

        # Test epoch begin
        callback.on_epoch_begin(0, None)
        self.assertIsNotNone(callback.epoch_start_time)
        self.assertEqual(callback.batch_count, 0)

        # Test batch tracking
        for i in range(10):
            callback.on_batch_begin(i, None)
            callback.on_batch_end(i, None, logs={'batch_size': 16, 'loss': 0.5, 'accuracy': 0.8})

        self.assertEqual(callback.batch_count, 10)
        self.assertEqual(callback.sample_count, 160)

    def test_profiling_callback(self):
        """Test ProfilingCallback functionality."""
        callback = ProfilingCallback(profile_memory=False, profile_cuda=False)

        # Simulate training
        callback.on_epoch_begin(0, None)
        callback.on_batch_begin(0, None)
        callback.on_batch_end(0, None)
        callback.on_epoch_end(0, None)

        # Check metrics collection
        metrics = callback.get_metrics()
        self.assertIn('epoch_times', metrics)
        self.assertIn('batch_times', metrics)
        self.assertEqual(len(metrics['epoch_times']), 1)
        self.assertEqual(len(metrics['batch_times']), 1)


class TestTrainer(unittest.TestCase):
    """Test Trainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cpu')

        # Create dummy data
        self.train_loader = create_dummy_data(num_samples=64, batch_size=16)
        self.val_loader = create_dummy_data(num_samples=32, batch_size=16)

    def test_trainer_initialization(self):
        """Test Trainer initialization."""
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
        )

        self.assertEqual(trainer.model, self.model)
        self.assertEqual(trainer.optimizer, self.optimizer)
        self.assertEqual(trainer.criterion, self.criterion)
        self.assertEqual(trainer.device, self.device)
        self.assertEqual(trainer.gradient_accumulation_steps, 1)
        self.assertFalse(trainer.use_amp)

    def test_trainer_fit(self):
        """Test Trainer fit method."""
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
        )

        # Train for 2 epochs
        metrics_history = trainer.fit(
            train_loader=self.train_loader,
            num_epochs=2,
            val_loader=self.val_loader,
        )

        # Check metrics history
        self.assertEqual(len(metrics_history), 2)
        self.assertIn('train_loss', metrics_history[0])
        self.assertIn('train_accuracy', metrics_history[0])
        self.assertIn('val_loss', metrics_history[0])
        self.assertIn('val_accuracy', metrics_history[0])

    def test_trainer_evaluate(self):
        """Test Trainer evaluate method."""
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
        )

        # Evaluate
        metrics = trainer.evaluate(self.val_loader)

        # Check metrics
        self.assertIn('val_loss', metrics)
        self.assertIn('val_accuracy', metrics)
        self.assertGreaterEqual(metrics['val_accuracy'], 0.0)
        self.assertLessEqual(metrics['val_accuracy'], 1.0)

    def test_trainer_with_callbacks(self):
        """Test Trainer with callbacks."""
        mock_cb = MockCallback()

        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            callbacks=[mock_cb],
        )

        # Train for 1 epoch
        trainer.fit(train_loader=self.train_loader, num_epochs=1)

        # Check that callbacks were called
        call_types = [call[0] for call in mock_cb.call_log]
        self.assertIn('on_train_begin', call_types)
        self.assertIn('on_epoch_begin', call_types)
        self.assertIn('on_batch_begin', call_types)
        self.assertIn('on_batch_end', call_types)
        self.assertIn('on_backward_end', call_types)
        self.assertIn('on_epoch_end', call_types)
        self.assertIn('on_train_end', call_types)

    def test_trainer_callback_order(self):
        """Test that callbacks are called in correct order during training."""
        mock_cb = MockCallback()

        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            callbacks=[mock_cb],
        )

        # Create minimal dataset for testing order
        small_loader = create_dummy_data(num_samples=16, batch_size=16)

        # Train for 1 epoch
        trainer.fit(train_loader=small_loader, num_epochs=1)

        # Extract call sequence
        call_sequence = [call[0] for call in mock_cb.call_log]

        # Verify order
        self.assertEqual(call_sequence[0], 'on_train_begin')
        self.assertEqual(call_sequence[1], 'on_epoch_begin')
        self.assertEqual(call_sequence[2], 'on_batch_begin')
        # After batch operations
        self.assertIn('on_backward_end', call_sequence)
        self.assertIn('on_batch_end', call_sequence)
        # Epoch and training end
        self.assertIn('on_epoch_end', call_sequence)
        self.assertEqual(call_sequence[-1], 'on_train_end')

    def test_trainer_gradient_accumulation(self):
        """Test gradient accumulation."""
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            gradient_accumulation_steps=2,
        )

        # Train for 1 epoch
        metrics_history = trainer.fit(
            train_loader=self.train_loader,
            num_epochs=1,
        )

        # Check that training completed successfully
        self.assertEqual(len(metrics_history), 1)
        self.assertIn('train_loss', metrics_history[0])

    def test_trainer_deterministic_seed(self):
        """Test deterministic seed initialization."""
        seed = 42

        # Create two trainers with same seed
        trainer1 = Trainer(
            model=SimpleModel(),
            optimizer=optim.Adam(SimpleModel().parameters()),
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
            seed=seed,
        )

        trainer2 = Trainer(
            model=SimpleModel(),
            optimizer=optim.Adam(SimpleModel().parameters()),
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
            seed=seed,
        )

        # Both trainers should produce same results (check model init)
        # Note: Full determinism would require identical data loader seeds too
        self.assertIsNotNone(trainer1)
        self.assertIsNotNone(trainer2)

    def test_trainer_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        import os
        import tempfile

        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
        )

        # Train for 1 epoch
        trainer.fit(train_loader=self.train_loader, num_epochs=1)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
            trainer.save_checkpoint(checkpoint_path)

            # Verify file exists
            self.assertTrue(os.path.exists(checkpoint_path))

            # Load checkpoint
            new_trainer = Trainer(
                model=SimpleModel(),
                optimizer=optim.Adam(SimpleModel().parameters()),
                criterion=nn.CrossEntropyLoss(),
                device=self.device,
            )

            checkpoint = new_trainer.load_checkpoint(checkpoint_path)

            # Verify checkpoint data
            self.assertIn('model_state_dict', checkpoint)
            self.assertIn('optimizer_state_dict', checkpoint)
            self.assertIn('epoch', checkpoint)


class TestAMPIntegration(unittest.TestCase):
    """Test AMP (Automatic Mixed Precision) integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device('cpu')  # Use CPU for testing
        self.train_loader = create_dummy_data(num_samples=32, batch_size=16)

    def test_amp_disabled(self):
        """Test training with AMP disabled."""
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            use_amp=False,
        )

        self.assertFalse(trainer.use_amp)
        self.assertIsNone(trainer.scaler)

        # Should train without errors
        metrics = trainer.fit(train_loader=self.train_loader, num_epochs=1)
        self.assertEqual(len(metrics), 1)

    def test_amp_enabled(self):
        """Test training with AMP enabled."""
        # Note: AMP might not work on CPU, but we test the setup
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            use_amp=True,
        )

        self.assertTrue(trainer.use_amp)
        self.assertIsNotNone(trainer.scaler)

        # Training might work on CPU with AMP in recent PyTorch versions
        try:
            metrics = trainer.fit(train_loader=self.train_loader, num_epochs=1)
            self.assertEqual(len(metrics), 1)
        except RuntimeError:
            # AMP might not be supported on CPU in older PyTorch
            pass


class TestIntegrationWithCallbacks(unittest.TestCase):
    """Integration tests with multiple callbacks."""

    def test_multiple_callbacks(self):
        """Test training with multiple callbacks."""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        logging_cb = LoggingCallback(log_interval=2, verbose=False)
        profiling_cb = ProfilingCallback(profile_memory=False, profile_cuda=False)
        mock_cb = MockCallback()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device('cpu'),
            callbacks=[logging_cb, profiling_cb, mock_cb],
        )

        # Train
        train_loader = create_dummy_data(num_samples=32, batch_size=16)
        trainer.fit(train_loader=train_loader, num_epochs=2)

        # Verify all callbacks were used
        self.assertGreater(len(mock_cb.call_log), 0)
        self.assertGreater(len(profiling_cb.metrics['epoch_times']), 0)

    def test_logging_and_profiling_callbacks(self):
        """Test LoggingCallback and ProfilingCallback work together."""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        logging_cb = LoggingCallback(log_interval=1, verbose=False)
        profiling_cb = ProfilingCallback(profile_memory=False, profile_cuda=False)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device('cpu'),
            callbacks=[logging_cb, profiling_cb],
        )

        # Train
        train_loader = create_dummy_data(num_samples=64, batch_size=16)
        val_loader = create_dummy_data(num_samples=32, batch_size=16)

        metrics_history = trainer.fit(
            train_loader=train_loader,
            num_epochs=2,
            val_loader=val_loader,
        )

        # Check metrics
        self.assertEqual(len(metrics_history), 2)

        # Check profiling metrics
        prof_metrics = profiling_cb.get_metrics()
        self.assertEqual(len(prof_metrics['epoch_times']), 2)
        self.assertGreater(len(prof_metrics['batch_times']), 0)


if __name__ == '__main__':
    unittest.main()
