# Migrating from TrainingLoop to Trainer

This guide helps you migrate from the legacy `TrainingLoop` class to the new Phase 4 `Trainer` class.

## Why Migrate?

The new `Trainer` class offers:
- **Extensible callback system** - Add custom behavior without modifying core code
- **AMP support** - Automatic Mixed Precision for faster training
- **Gradient accumulation** - Train with larger effective batch sizes
- **Deterministic training** - Better reproducibility with seed initialization
- **Better separation of concerns** - Model, optimizer, and loss are separate
- **Checkpoint management** - Save and resume training easily

## Backward Compatibility

The old `TrainingLoop` class is still available and fully functional. You can use both in the same project.

## Migration Examples

### Before (TrainingLoop)

```python
from adaptiveneuralnetwork.training import TrainingLoop
from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel

config = AdaptiveConfig(
    input_dim=784,
    output_dim=10,
    num_epochs=10,
    learning_rate=0.001,
)

model = AdaptiveModel(config)
trainer = TrainingLoop(model, config)
metrics = trainer.train(train_loader, test_loader, num_epochs=10)
```

### After (Trainer)

```python
from adaptiveneuralnetwork.training import Trainer, LoggingCallback
import torch.nn as nn
import torch.optim as optim

# For AdaptiveModel
from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel

config = AdaptiveConfig(input_dim=784, output_dim=10)
model = AdaptiveModel(config)

# Or use any PyTorch model
# model = nn.Sequential(...)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    callbacks=[LoggingCallback(log_interval=10)],
    seed=42,  # For reproducibility
)

metrics = trainer.fit(
    train_loader=train_loader,
    num_epochs=10,
    val_loader=test_loader,
)
```

## Feature Comparison

| Feature | TrainingLoop | Trainer |
|---------|-------------|---------|
| Basic training | ✅ | ✅ |
| Evaluation | ✅ | ✅ |
| Metrics tracking | ✅ | ✅ |
| Checkpoint save/load | ✅ | ✅ |
| Callbacks | ❌ | ✅ |
| AMP support | ❌ | ✅ |
| Gradient accumulation | ❌ | ✅ |
| Deterministic seed | ❌ | ✅ |
| Works with any PyTorch model | ❌ | ✅ |
| Extensible without code changes | ❌ | ✅ |

## Key Differences

### 1. Separate Optimizer and Criterion

**TrainingLoop**: Optimizer and criterion are created internally
```python
trainer = TrainingLoop(model, config)
```

**Trainer**: You provide optimizer and criterion
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion)
```

### 2. Callbacks Instead of Hard-coded Logging

**TrainingLoop**: Logging is built-in and not customizable
```python
# Logging happens automatically
metrics = trainer.train(train_loader, test_loader)
```

**Trainer**: Use callbacks for custom behavior
```python
from adaptiveneuralnetwork.training import LoggingCallback, ProfilingCallback

trainer = Trainer(
    model, optimizer, criterion,
    callbacks=[
        LoggingCallback(log_interval=10, verbose=True),
        ProfilingCallback(profile_memory=True),
    ]
)
```

### 3. Advanced Features

**TrainingLoop**: No built-in support for AMP or gradient accumulation

**Trainer**: Easy to enable
```python
trainer = Trainer(
    model, optimizer, criterion,
    use_amp=True,  # Enable AMP
    gradient_accumulation_steps=4,  # Accumulate gradients
    seed=42,  # Deterministic training
)
```

## Common Migration Patterns

### Pattern 1: Simple Migration

If you just need basic training, the migration is straightforward:

```python
# Old
trainer = TrainingLoop(model, config)
metrics = trainer.train(train_loader, test_loader, num_epochs=10)

# New
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, optimizer, criterion)
metrics = trainer.fit(train_loader, num_epochs=10, val_loader=test_loader)
```

### Pattern 2: With Custom Logging

```python
# Old - logging was hard-coded
trainer = TrainingLoop(model, config)

# New - customizable logging
from adaptiveneuralnetwork.training import LoggingCallback

trainer = Trainer(
    model, optimizer, criterion,
    callbacks=[LoggingCallback(log_interval=100, verbose=True)]
)
```

### Pattern 3: With Profiling

```python
# Old - no built-in profiling
trainer = TrainingLoop(model, config)

# New - easy profiling
from adaptiveneuralnetwork.training import ProfilingCallback

profiling_cb = ProfilingCallback()
trainer = Trainer(
    model, optimizer, criterion,
    callbacks=[profiling_cb]
)
trainer.fit(train_loader, num_epochs=10)

# Get profiling metrics
metrics = profiling_cb.get_metrics()
print(f"Average batch time: {sum(metrics['batch_times']) / len(metrics['batch_times']):.4f}s")
```

## Creating Custom Callbacks

One of the biggest advantages of the new Trainer is extensibility:

```python
from adaptiveneuralnetwork.training.callbacks import Callback

class LearningRateSchedulerCallback(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        self.scheduler.step()
        print(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")

# Use it
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
trainer = Trainer(
    model, optimizer, criterion,
    callbacks=[LearningRateSchedulerCallback(scheduler)]
)
```

## When to Use Each

### Use TrainingLoop When:
- Working with existing code that uses `AdaptiveModel`
- Don't need callbacks or advanced features
- Want minimal code changes

### Use Trainer When:
- Starting a new project
- Need callbacks for custom behavior
- Want AMP or gradient accumulation
- Need deterministic training
- Working with standard PyTorch models
- Want better extensibility

## Best Practices

1. **Start with simple migration**: Get basic training working first
2. **Add callbacks incrementally**: Start with LoggingCallback, add more as needed
3. **Use deterministic seed**: Set `seed=42` for reproducible experiments
4. **Enable AMP on GPU**: Use `use_amp=True` when training on CUDA
5. **Profile first**: Use ProfilingCallback to identify bottlenecks

## Getting Help

- See `examples/phase4_trainer_examples.py` for working examples
- Check `tests/test_trainer_callbacks.py` for detailed usage patterns
- Read the callback docstrings for API details
