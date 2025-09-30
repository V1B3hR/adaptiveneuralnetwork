# Phase 3: Model Architecture Modularization

## Overview

Phase 3 introduces a modular architecture system that makes the codebase configurable and extensible without editing core logic. The implementation uses a **layer registry pattern** and **config-driven model assembly** to enable rapid development of new model variants.

## Key Components

### 1. Layer Registry (`adaptiveneuralnetwork/core/layer_registry.py`)

The layer registry provides a string-to-class mapping system for neural network layers.

**Features:**
- Decorator-based registration
- Factory function support
- Type-safe layer instantiation
- Comprehensive error handling

**Example Usage:**

```python
from adaptiveneuralnetwork.core import layer_registry

# Register a custom layer
@layer_registry.register("my_custom_layer")
class MyCustomLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x * self.dim

# Create layer from registry
layer = layer_registry.create("my_custom_layer", dim=10)
```

### 2. Model Builder (`adaptiveneuralnetwork/core/model_builder.py`)

The model builder constructs PyTorch models from configuration dictionaries or files.

**Features:**
- YAML and JSON config support
- Nested layer construction
- Localized seed management
- Built-in layer registration (52 PyTorch layers)

**Example Usage:**

```python
from adaptiveneuralnetwork.core import ModelBuilder

builder = ModelBuilder(seed=42)

# From config dict
config = {
    "name": "mlp",
    "seed": 42,
    "layers": [
        {"type": "linear", "in_features": 784, "out_features": 256},
        {"type": "relu"},
        {"type": "linear", "in_features": 256, "out_features": 10}
    ]
}

model = builder.build_from_config(config)

# From YAML file
model = builder.build_from_file("config/models/simple_mlp.yaml")
```

### 3. Layer Registration (`adaptiveneuralnetwork/models/layer_registration.py`)

Automatic registration of custom model components with the global layer registry.

**Registered Custom Layers:**
- `convlstm_cell` - ConvLSTM cell for spatiotemporal modeling
- `convlstm` - Full ConvLSTM model
- `conv3d_model` - 3D CNN for video analysis
- `video_transformer` - Transformer for video sequences
- `positional_encoding` - Positional encoding for transformers
- `hybrid_video` - Hybrid video model

## Configuration File Format

### YAML Format

```yaml
name: model_name
description: "Optional description"
seed: 42  # Optional, for reproducible initialization

layers:
  - type: linear
    in_features: 784
    out_features: 256
  
  - type: relu
  
  - type: dropout
    p: 0.3
  
  - type: linear
    in_features: 256
    out_features: 10
```

### JSON Format

```json
{
  "name": "model_name",
  "seed": 42,
  "layers": [
    {"type": "linear", "in_features": 784, "out_features": 256},
    {"type": "relu"},
    {"type": "dropout", "p": 0.3},
    {"type": "linear", "in_features": 256, "out_features": 10}
  ]
}
```

## Configuration Examples

Four example configurations are provided in `config/models/`:

### 1. Simple MLP (`simple_mlp.yaml`)
Basic feedforward network for classification tasks.

### 2. Simple CNN (`simple_cnn.yaml`)
Convolutional network for image classification with:
- 3 conv blocks with batch normalization
- Adaptive average pooling
- Classifier head

### 3. Video ConvLSTM (`video_convlstm.yaml`)
Spatiotemporal model for video analysis using ConvLSTM.

### 4. Video Transformer (`video_transformer.yaml`)
Attention-based model for video understanding.

## Adding a New Model Variant

To add a new model variant, create a YAML config file:

```yaml
# config/models/my_model.yaml
name: my_model
seed: 42

layers:
  - type: conv2d
    in_channels: 3
    out_channels: 64
    kernel_size: 7
    stride: 2
    padding: 3
  
  - type: batchnorm2d
    num_features: 64
  
  - type: relu
  
  # ... more layers
```

Then load and use it:

```python
from adaptiveneuralnetwork.core import ModelBuilder

builder = ModelBuilder()
model = builder.build_from_file("config/models/my_model.yaml")
```

**Time to add new layer: < 2 minutes** ✓

## Adding a Custom Layer Type

To add a custom layer type:

```python
from adaptiveneuralnetwork.core import layer_registry
import torch.nn as nn

@layer_registry.register("my_layer")
class MyCustomLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize your layer
    
    def forward(self, x):
        # Implement forward pass
        return x
```

Now it's available in configs:

```yaml
layers:
  - type: my_layer
    param1: value1
    param2: value2
```

## Seed Localization

Random seeds are localized to model construction, preventing global state pollution:

```python
builder1 = ModelBuilder(seed=42)
model1 = builder1.build_from_config(config)

builder2 = ModelBuilder(seed=42)
model2 = builder2.build_from_config(config)

# model1 and model2 have identical weights
assert all(torch.allclose(p1, p2) for p1, p2 in zip(
    model1.parameters(), model2.parameters()
))
```

## Benefits

### 1. Rapid Prototyping
New model architectures can be defined and tested in minutes using config files.

### 2. Experiment Tracking
Model configurations are version-controlled and easily compared.

### 3. Reproducibility
Seed localization ensures reproducible model initialization without global state.

### 4. Extensibility
New layer types can be registered without modifying core code.

### 5. Maintainability
Configuration files are easier to read and modify than code.

## Testing

Comprehensive test suite in `tests/test_phase3_modularization.py`:

- **LayerRegistry**: 7 tests
- **ModelBuilder**: 7 tests
- **BuiltinLayers**: 5 tests
- **ModelVariants**: 3 tests
- **ConfigFiles**: 2 tests
- **SeedLocalization**: 1 test

**Total: 25 tests, all passing** ✓

## Running Tests

```bash
# Run Phase 3 tests
python -m pytest tests/test_phase3_modularization.py -v

# Run demo
python scripts/phase3_demo.py
```

## Metrics

### Success Criteria

✓ **New model variants via config only**: 4 example configs provided  
✓ **Core architecture LOC**: Unchanged at 559 LOC (no increase)  
✓ **Variant config coverage**: 4 model types (MLP, CNN, ConvLSTM, Transformer)  
✓ **Time to add new layer**: < 2 minutes with config file  
✓ **Registered layers**: 52 built-in + custom layer support  

### Code Organization

- `adaptiveneuralnetwork/core/layer_registry.py`: 108 LOC
- `adaptiveneuralnetwork/core/model_builder.py`: 227 LOC
- `adaptiveneuralnetwork/models/layer_registration.py`: 59 LOC
- **Total new code**: 394 LOC

### Test Coverage

- 25 test cases
- 100% pass rate
- Covers registry, builder, configs, and seed localization

## Migration Guide

Existing code continues to work unchanged. To use new features:

### Before (Hardcoded):

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10)
)
```

### After (Config-driven):

```yaml
# config/models/my_model.yaml
name: my_model
layers:
  - type: linear
    in_features: 784
    out_features: 256
  - type: relu
  - type: dropout
    p: 0.3
  - type: linear
    in_features: 256
    out_features: 10
```

```python
from adaptiveneuralnetwork.core import ModelBuilder

builder = ModelBuilder()
model = builder.build_from_file("config/models/my_model.yaml")
```

## Future Enhancements

Potential improvements for future phases:

1. **Config validation**: JSON Schema or dataclass-based validation
2. **Nested configs**: Include and extend other configs
3. **Parameter search**: Grid/random search over config parameters
4. **Auto-documentation**: Generate docs from registered layers
5. **Config templates**: Reusable config components

## Conclusion

Phase 3 successfully implements a modular, config-driven architecture that:

- Makes the codebase extensible without editing core logic
- Enables rapid development of new model variants
- Maintains backward compatibility
- Provides comprehensive testing and documentation

The system is production-ready and sets the foundation for subsequent phases.
