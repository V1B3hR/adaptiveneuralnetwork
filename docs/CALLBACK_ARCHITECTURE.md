# Callback Architecture - Phase 4

## Overview

The callback system provides a flexible way to extend training behavior without modifying core training logic. Callbacks are invoked at specific points during training, allowing you to:

- Log metrics and visualizations
- Profile performance
- Save checkpoints
- Adjust learning rates
- Early stopping
- Custom metrics computation
- Integration with experiment tracking tools

## Callback Lifecycle

Callbacks have hooks at the following points in the training process:

```
Training Start
├─ on_train_begin()
│
├─ Epoch Loop (for each epoch)
│  ├─ on_epoch_begin(epoch)
│  │
│  ├─ Training Batches
│  │  ├─ for each batch:
│  │  │  ├─ on_batch_begin(batch_idx)
│  │  │  ├─ forward pass
│  │  │  ├─ backward pass
│  │  │  ├─ on_backward_end(batch_idx)
│  │  │  └─ on_batch_end(batch_idx)
│  │
│  ├─ Validation (optional)
│  │  ├─ on_evaluate_begin()
│  │  ├─ evaluation loop
│  │  └─ on_evaluate_end()
│  │
│  └─ on_epoch_end(epoch)
│
└─ on_train_end()
```

## Hook Specifications

### on_train_begin(trainer, logs=None)
Called once before training starts.

**Use cases:**
- Initialize tracking variables
- Set up logging infrastructure
- Validate configuration

**Example:**
```python
def on_train_begin(self, trainer, logs=None):
    self.start_time = time.time()
    print(f"Starting training with {trainer.num_epochs} epochs")
```

### on_train_end(trainer, logs=None)
Called once after training completes.

**Use cases:**
- Print final summaries
- Save final artifacts
- Clean up resources

**Example:**
```python
def on_train_end(self, trainer, logs=None):
    total_time = time.time() - self.start_time
    print(f"Training completed in {total_time:.2f}s")
```

### on_epoch_begin(epoch, trainer, logs=None)
Called at the start of each epoch.

**Use cases:**
- Reset epoch-level metrics
- Print epoch header
- Adjust hyperparameters

**Example:**
```python
def on_epoch_begin(self, epoch, trainer, logs=None):
    self.epoch_start_time = time.time()
    print(f"Epoch {epoch + 1}/{trainer.num_epochs}")
```

### on_epoch_end(epoch, trainer, logs=None)
Called at the end of each epoch.

**Use cases:**
- Log epoch metrics
- Save checkpoints
- Learning rate scheduling
- Early stopping checks

**Example:**
```python
def on_epoch_end(self, epoch, trainer, logs=None):
    if logs and logs.get('val_loss', float('inf')) < self.best_loss:
        self.best_loss = logs['val_loss']
        trainer.save_checkpoint(f'best_model_epoch_{epoch}.pt')
```

### on_batch_begin(batch_idx, trainer, logs=None)
Called before processing each batch.

**Use cases:**
- Batch-level timing
- Dynamic batch size adjustment

**Example:**
```python
def on_batch_begin(self, batch_idx, trainer, logs=None):
    self.batch_start_time = time.time()
```

### on_batch_end(batch_idx, trainer, logs=None)
Called after processing each batch (after optimizer step).

**Use cases:**
- Log batch metrics
- Track throughput
- Update progress bars

**Example:**
```python
def on_batch_end(self, batch_idx, trainer, logs=None):
    if batch_idx % 10 == 0 and logs:
        print(f"Batch {batch_idx}: loss={logs.get('loss', 0):.4f}")
```

### on_backward_end(batch_idx, trainer, logs=None)
Called after the backward pass but before optimizer step.

**Use cases:**
- Gradient clipping
- Gradient statistics
- Gradient accumulation logic

**Example:**
```python
def on_backward_end(self, batch_idx, trainer, logs=None):
    total_norm = 0
    for p in trainer.model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Gradient norm: {total_norm:.4f}")
```

### on_evaluate_begin(trainer, logs=None)
Called before evaluation starts.

**Use cases:**
- Set up evaluation-specific state
- Switch to evaluation mode

### on_evaluate_end(trainer, logs=None)
Called after evaluation completes.

**Use cases:**
- Log evaluation metrics
- Compare with best metrics

## Creating Custom Callbacks

### Basic Template

```python
from adaptiveneuralnetwork.training.callbacks import Callback

class MyCallback(Callback):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        # Your custom logic here
        pass
```

### Example: Early Stopping

```python
class EarlyStoppingCallback(Callback):
    def __init__(self, patience=5, min_delta=0.001, monitor='val_loss'):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.wait = 0
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        if logs is None:
            return
        
        current_value = logs.get(self.monitor, None)
        if current_value is None:
            return
        
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                trainer.stop_training = True  # Would need to add this flag
```

### Example: Learning Rate Scheduler

```python
class LRSchedulerCallback(Callback):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")
```

### Example: Weights & Biases Integration

```python
class WandbCallback(Callback):
    def __init__(self, project_name, run_name=None):
        super().__init__()
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project_name, name=run_name)
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        if logs:
            self.wandb.log(logs, step=epoch)
    
    def on_train_end(self, trainer, logs=None):
        self.run.finish()
```

### Example: TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardCallback(Callback):
    def __init__(self, log_dir='runs'):
        super().__init__()
        self.writer = SummaryWriter(log_dir)
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        if logs:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, epoch)
    
    def on_train_end(self, trainer, logs=None):
        self.writer.close()
```

### Example: Model Checkpoint

```python
class ModelCheckpointCallback(Callback):
    def __init__(self, filepath='checkpoint_{epoch}.pt', save_best_only=True, monitor='val_loss'):
        super().__init__()
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_value = float('inf')
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        if logs is None:
            return
        
        current_value = logs.get(self.monitor, None)
        if current_value is None:
            return
        
        if not self.save_best_only or current_value < self.best_value:
            self.best_value = current_value
            filepath = self.filepath.format(epoch=epoch)
            trainer.save_checkpoint(filepath)
            print(f"Model saved to {filepath}")
```

## Callback Order

When multiple callbacks are registered, they are executed in the order they were added:

```python
trainer = Trainer(
    model, optimizer, criterion,
    callbacks=[
        callback1,  # Executed first
        callback2,  # Executed second
        callback3,  # Executed third
    ]
)
```

All callbacks complete each hook before moving to the next hook:

```
on_epoch_begin:
  callback1.on_epoch_begin()
  callback2.on_epoch_begin()
  callback3.on_epoch_begin()

on_batch_begin:
  callback1.on_batch_begin()
  callback2.on_batch_begin()
  callback3.on_batch_begin()
```

## Best Practices

1. **Keep callbacks focused**: Each callback should do one thing well
2. **Avoid side effects**: Don't modify trainer state unless necessary
3. **Handle missing logs gracefully**: Always check if `logs` is None or if keys exist
4. **Use appropriate hooks**: Choose the right hook for your use case
5. **Clean up resources**: Use `on_train_end` to close files, connections, etc.
6. **Document your callbacks**: Add docstrings explaining what each callback does
7. **Test independently**: Write unit tests for your callbacks

## Common Pitfalls

1. **Modifying logs dictionary**: Logs are shared across callbacks, be careful
2. **Expensive operations**: Avoid slow operations in frequently called hooks
3. **State management**: Initialize state in `__init__` or `on_train_begin`
4. **Missing error handling**: Wrap risky operations in try/except
5. **Circular dependencies**: Don't have callbacks that depend on each other

## Performance Considerations

- Callbacks are called on every batch/epoch, so keep them fast
- Use `on_batch_end` with intervals for batch-level logging (not every batch)
- Profile your callbacks if training is slow
- Consider async operations for expensive tasks (file I/O, network requests)

## Testing Callbacks

Example test using mock callbacks:

```python
class MockCallback(Callback):
    def __init__(self):
        self.calls = []
    
    def on_epoch_begin(self, epoch, trainer, logs=None):
        self.calls.append(('on_epoch_begin', epoch))
    
    def on_epoch_end(self, epoch, trainer, logs=None):
        self.calls.append(('on_epoch_end', epoch))

# Test
callback = MockCallback()
# ... train with callback ...
assert ('on_epoch_begin', 0) in callback.calls
assert ('on_epoch_end', 0) in callback.calls
```

## Further Reading

- See `adaptiveneuralnetwork/training/callbacks.py` for built-in callbacks
- Check `tests/test_trainer_callbacks.py` for test examples
- Review `examples/phase4_trainer_examples.py` for usage examples
