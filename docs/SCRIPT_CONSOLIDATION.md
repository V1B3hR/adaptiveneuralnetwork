# Script Consolidation and Refactoring

## Overview

This document describes the consolidated script architecture for the Adaptive Neural Network project. The refactoring reduces redundancy, improves maintainability, and establishes a modular structure.

## New Structure

### Entry Point Scripts

#### `train.py` - Unified Training Entry Point
Located at the repository root, this script replaces multiple scattered training scripts:
- `core/train.py`
- `adaptiveneuralnetwork/training/scripts/train.py`
- `training/scripts/train_new_datasets.py`
- `training/scripts/train_kaggle_datasets.py`
- And other specialized training scripts

**Features:**
- Configuration-driven workflows (YAML/JSON)
- CLI with flexible parameter overrides
- Support for multiple datasets
- Consistent interface across all training tasks

**Usage:**
```bash
# Train with configuration file
python train.py --config config/training/mnist.yaml

# Train with dataset name and custom parameters
python train.py --dataset mnist --epochs 20 --batch-size 128

# List available datasets
python train.py --list-datasets

# Override config parameters
python train.py --config config/training/kaggle_default.yaml --device cpu --epochs 5
```

#### `eval.py` - Unified Evaluation Entry Point
Provides consistent evaluation interface for all trained models.

**Usage:**
```bash
# Evaluate a single checkpoint
python eval.py --checkpoint checkpoints/model.pt --dataset mnist

# Evaluate all checkpoints in a directory
python eval.py --checkpoint-dir checkpoints/ --dataset cifar10

# Use configuration file
python eval.py --config config/training/mnist.yaml --checkpoint checkpoints/model.pt
```

### Configuration System

#### Training Configurations (`adaptiveneuralnetwork/training/config.py`)
Defines structured configuration classes:
- `DatasetConfig` - Dataset loading and preprocessing
- `ModelConfig` - Model architecture parameters
- `OptimizerConfig` - Optimizer and scheduler settings
- `TrainingConfig` - Training process parameters
- `EvaluationConfig` - Evaluation settings
- `WorkflowConfig` - Complete workflow configuration

#### Configuration Files (`config/training/`)
YAML configuration templates for common use cases:
- `mnist.yaml` - MNIST dataset training
- `kaggle_default.yaml` - Kaggle datasets (ANNOMI, etc.)
- `quick_test.yaml` - Fast experimentation and debugging

**Benefits:**
- Version control for experiments
- Easy reproduction of results
- Parameter documentation
- Sharing configurations across team

### Module Organization

```
adaptiveneuralnetwork/
├── data/              # Data loading and preprocessing
│   ├── kaggle_datasets.py
│   ├── optimized_datasets.py
│   └── streaming_datasets.py
├── models/            # Model definitions
│   ├── graph_spatial.py
│   ├── pos_tagger.py
│   └── video_models.py
├── training/          # Training components
│   ├── config.py      # Training configuration (NEW)
│   ├── trainer.py     # Training loop
│   ├── callbacks.py   # Training callbacks
│   └── scripts/       # Legacy scripts (deprecated)
├── utils/             # Utility functions
│   ├── profiling.py
│   ├── reproducibility.py
│   └── drift.py
└── config.py          # Global configuration
```

## Migration Guide

### For Training Scripts

**Old way:**
```bash
python core/train.py --dataset vr_driving --epochs 10
python training/scripts/train_annomi.py --data-path data/
```

**New way:**
```bash
python train.py --dataset vr_driving --epochs 10
python train.py --dataset annomi --data-path data/
```

### For Custom Configurations

**Old way:**
Edit script parameters directly or use command-line flags

**New way:**
Create a YAML configuration file:
```yaml
# config/training/my_experiment.yaml
dataset:
  name: "annomi"
  batch_size: 32
model:
  hidden_dim: 256
training:
  epochs: 20
  learning_rate: 0.0005
```

Then run:
```bash
python train.py --config config/training/my_experiment.yaml
```

## Available Datasets

Run `python train.py --list-datasets` to see all available datasets:

- **mnist** - MNIST handwritten digits
- **cifar10** - CIFAR-10 natural images
- **annomi** - ANNOMI Motivational Interviewing dataset
- **mental_health** - Mental Health dataset
- **vr_driving** - VR Driving simulation dataset
- **autvi** - Automotive Vehicle Inspection dataset
- **digakust** - Digital Acoustic Analysis dataset
- **synthetic** - Synthetic dataset for testing

## Configuration File Structure

```yaml
dataset:
  name: "dataset_name"
  data_path: "path/to/data"
  batch_size: 64
  num_workers: 4
  # ... more dataset parameters

model:
  name: "model_name"
  hidden_dim: 128
  num_nodes: 64
  # ... more model parameters

optimizer:
  name: "adam"
  learning_rate: 0.001
  scheduler: "cosine"
  # ... more optimizer parameters

training:
  epochs: 10
  device: "cuda"
  use_amp: false
  # ... more training parameters

evaluation:
  metrics: ["accuracy", "loss"]
  save_predictions: false
  # ... more evaluation parameters
```

## Extending the System

### Adding a New Dataset

1. Add dataset loader to `adaptiveneuralnetwork/data/`
2. Register in `AVAILABLE_DATASETS` dict in `train.py`
3. Create configuration template in `config/training/`

### Adding a New Model

1. Define model class in `adaptiveneuralnetwork/models/`
2. Update model factory/registry if needed
3. Add model-specific parameters to configuration

### Custom Training Logic

For specialized training requirements:
1. Extend `adaptiveneuralnetwork/training/trainer.py`
2. Add custom callbacks in `adaptiveneuralnetwork/training/callbacks.py`
3. Use configuration to enable/disable custom behavior

## Legacy Scripts

The following scripts are now **deprecated** but retained for compatibility:

- `core/train.py` → Use `train.py` instead
- `adaptiveneuralnetwork/training/scripts/train.py` → Use `train.py` instead
- `training/scripts/train_*.py` → Use `train.py` with appropriate config/dataset

These scripts will be removed in a future release. Please migrate to the new unified interface.

## Benefits of Consolidation

1. **Reduced Redundancy**: Single source of truth for training logic
2. **Improved Maintainability**: Easier to update and extend
3. **Configuration-Driven**: Version control for experiments
4. **Consistent Interface**: Same commands for all datasets
5. **Better Documentation**: Centralized and comprehensive
6. **Easier Testing**: Modular components are easier to test

## Testing

Basic validation of the new entry points:

```bash
# Test configuration loading
python train.py --list-datasets

# Test config file parsing
python train.py --config config/training/quick_test.yaml --save-config /tmp/test_config.yaml

# Test parameter overrides
python train.py --dataset synthetic --epochs 1 --device cpu
```

## Future Enhancements

- [ ] Complete integration with existing training implementations
- [ ] Add distributed training support to configuration
- [ ] Add hyperparameter search configuration
- [ ] Create web UI for configuration management
- [ ] Add configuration validation and schema
- [ ] Automated migration tool for legacy scripts

## Questions and Support

For questions about the new structure, please refer to:
- This documentation
- Configuration examples in `config/training/`
- Code documentation in `adaptiveneuralnetwork/training/config.py`
- Main README.md for general project information
