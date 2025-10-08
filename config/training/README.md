# Training Configuration Files

This directory contains YAML configuration templates for training workflows.

## Available Configurations

### `mnist.yaml`
Training configuration for MNIST handwritten digits dataset.
- Standard image classification setup
- 10 epochs, batch size 64
- Cosine learning rate scheduler
- Good starting point for vision tasks

### `kaggle_default.yaml`
Training configuration for Kaggle datasets (ANNOMI, mental health, etc.).
- Designed for text/NLP datasets
- 20 epochs with gradient accumulation
- Mixed precision training enabled
- Early stopping on validation accuracy

### `quick_test.yaml`
Fast configuration for experimentation and debugging.
- Minimal resources (CPU, small model)
- 3 epochs only
- Small batch size
- Useful for testing changes quickly

## Usage

### Basic Usage
```bash
# Train with a configuration file
python train.py --config config/training/mnist.yaml

# Override specific parameters
python train.py --config config/training/mnist.yaml --epochs 20 --device cpu
```

### Creating Custom Configurations

1. Copy an existing configuration as a template:
```bash
cp config/training/mnist.yaml config/training/my_experiment.yaml
```

2. Edit the configuration file:
```yaml
dataset:
  name: "my_dataset"
  batch_size: 128
  
model:
  hidden_dim: 256
  num_nodes: 128
  
training:
  epochs: 30
  learning_rate: 0.0001
```

3. Run training:
```bash
python train.py --config config/training/my_experiment.yaml
```

## Configuration Structure

All configuration files follow this structure:

```yaml
dataset:        # Dataset loading and preprocessing
  name: str
  data_path: str (optional)
  batch_size: int
  num_workers: int
  # ... more dataset parameters

model:          # Model architecture
  name: str
  input_dim: int
  hidden_dim: int
  output_dim: int
  num_nodes: int
  # ... more model parameters

optimizer:      # Optimizer and scheduler
  name: str
  learning_rate: float
  weight_decay: float
  scheduler: str (optional)
  # ... more optimizer parameters

training:       # Training process
  epochs: int
  device: str
  use_amp: bool
  checkpoint_dir: str
  log_dir: str
  # ... more training parameters

evaluation:     # Evaluation settings
  metrics: list
  batch_size: int
  save_predictions: bool
  output_dir: str
```

## Tips

1. **Version Control**: Commit configuration files with your code to track experiments
2. **Naming**: Use descriptive names like `experiment_01_large_batch.yaml`
3. **Comments**: Add comments to document non-obvious parameter choices
4. **Validation**: Test new configurations with `quick_test.yaml` first
5. **Sharing**: Configuration files make it easy to share exact experimental setups

## See Also

- [Script Consolidation Guide](../../docs/SCRIPT_CONSOLIDATION.md)
- [Main README](../../README.md)
- Configuration class documentation in `adaptiveneuralnetwork/training/config.py`
