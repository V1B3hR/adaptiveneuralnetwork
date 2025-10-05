# Examples Directory

This directory contains examples demonstrating various features of the Adaptive Neural Network framework.

## Quick Start Example

**New to Adaptive Neural Network?** Start here!

### Running the Quick Start Examples

```bash
# Run all quick start examples
PYTHONPATH=. python examples/quickstart_example.py

# Or from within examples/
cd examples
PYTHONPATH=.. python quickstart_example.py
```

The quick start example demonstrates:
1. **Smoke Test** - Quick validation with default settings
2. **Custom Smoke Test** - Custom parameters and output directory
3. **Benchmark Mode** - Full evaluation with detailed metrics
4. **Load and Predict** - Loading trained models and making predictions
5. **Analyze Results** - Reading and analyzing results JSON files

📖 See [QUICKSTART.md](../QUICKSTART.md) for comprehensive documentation.

---

## Phase 4 Training Loop Abstraction - Examples

This directory contains examples demonstrating the Phase 4 Trainer class and callback system.

## Running the Examples

```bash
# Run all examples
python examples/phase4_trainer_examples.py

# Or run from the repository root with PYTHONPATH
PYTHONPATH=. python examples/phase4_trainer_examples.py
```

## Examples Included

### 1. Basic Training with LoggingCallback
Demonstrates basic training setup with logging of loss, accuracy, and throughput metrics.

### 2. Training with ProfilingCallback
Shows how to use the profiling callback to collect timing and memory metrics.

### 3. Training with AMP (Automatic Mixed Precision)
Demonstrates enabling AMP for faster training on compatible hardware.

### 4. Training with Gradient Accumulation
Shows how to use gradient accumulation to increase effective batch size.

### 5. Training with Multiple Callbacks
Demonstrates using multiple callbacks together for comprehensive monitoring.

### 6. Checkpoint Saving and Loading
Shows how to save and load training checkpoints for resuming training.

## Key Components

### Trainer Class
The `Trainer` class provides a unified interface for training with support for:
- Callbacks at all lifecycle points
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Deterministic seed initialization
- Checkpoint management

### Callbacks
- **LoggingCallback**: Logs training metrics (loss, accuracy, throughput)
- **ProfilingCallback**: Collects performance metrics (timing, memory)
- **Custom Callbacks**: Extend the `Callback` base class for custom behavior

## Creating Custom Callbacks

```python
from adaptiveneuralnetwork.training.callbacks import Callback

class MyCustomCallback(Callback):
    def on_epoch_end(self, epoch, trainer, logs=None):
        # Custom logic after each epoch
        print(f"Epoch {epoch} completed!")
        
    def on_batch_end(self, batch_idx, trainer, logs=None):
        # Custom logic after each batch
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx} batches")
```

## Callback Lifecycle Hooks

Callbacks support the following hooks:
- `on_train_begin`: Called before training starts
- `on_train_end`: Called after training completes
- `on_epoch_begin`: Called at the start of each epoch
- `on_epoch_end`: Called at the end of each epoch
- `on_batch_begin`: Called before processing each batch
- `on_batch_end`: Called after processing each batch
- `on_backward_end`: Called after the backward pass
- `on_evaluate_begin`: Called before evaluation
- `on_evaluate_end`: Called after evaluation

## Integration with Existing Code

The Trainer class is designed to work seamlessly with PyTorch models and data loaders:

```python
import torch
import torch.nn as nn
from adaptiveneuralnetwork.training import Trainer, LoggingCallback

# Your existing PyTorch model
model = nn.Sequential(...)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[LoggingCallback()],
)

# Train
trainer.fit(train_loader, num_epochs=10, val_loader=val_loader)
```
