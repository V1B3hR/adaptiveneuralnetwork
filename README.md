# Adaptive Neural Network
[![CI](https://github.com/V1B3hR/adaptiveneuralnetwork/workflows/CI%20-%20Train,%20Test,%20Coverage%20&%20Artifacts/badge.svg)](https://github.com/V1B3hR/adaptiveneuralnetwork/actions)
![Coverage](https://img.shields.io/badge/coverage-71%25-yellow)

A production-ready biologically-inspired neural network with vectorized training capabilities and adaptive learning mechanisms. This library provides a PyTorch-compatible implementation of adaptive neural networks with phase-based dynamics, energy management, and continual learning support.

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With JAX backend support
pip install -e ".[jax]"

# With neuromorphic hardware support
pip install -e ".[neuromorphic]"

# With multi-modal support
pip install -e ".[multimodal]"

# Full installation with all backends
pip install -e ".[jax,neuromorphic,multimodal]"
```

### Basic Usage

```python
from adaptiveneuralnetwork.api import create_adaptive_model, AdaptiveConfig

# Create configuration
config = AdaptiveConfig(
    num_nodes=100,
    hidden_dim=64,
    backend="pytorch"  # or "jax", "neuromorphic"
)

# Create model with backend selection
model = create_adaptive_model(config)

# Or use convenience function
model = create_adaptive_model(
    backend="pytorch",  # Choose your backend
    num_nodes=128,
    hidden_dim=64
)

# Standard PyTorch training loop
import torch
optimizer = torch.optim.Adam(model.parameters())
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch.data)
    loss = criterion(output, batch.target)
    loss.backward()
    optimizer.step()
```

### Run Benchmarks

```bash
# MNIST benchmark (classic)
python scripts/run_benchmark.py --dataset mnist --epochs 10

# CIFAR-10 standard benchmark
python -c "
from adaptiveneuralnetwork.benchmarks.vision.cifar10 import run_cifar10_benchmark
results = run_cifar10_benchmark(epochs=5, batch_size=64)
print(f'CIFAR-10 accuracy: {results[\"standard\"][\"final_test_accuracy\"]:.3f}')
"

# CIFAR-10 robustness benchmark
python -c "
from adaptiveneuralnetwork.benchmarks.vision.cifar10 import CIFAR10Benchmark
from adaptiveneuralnetwork.api import AdaptiveConfig

config = AdaptiveConfig(num_nodes=64, hidden_dim=32)
benchmark = CIFAR10Benchmark(config)

# Run robustness test
results = benchmark.run_robustness_benchmark(
    corruption_types=['gaussian_noise', 'brightness'],
    severities=[1, 3, 5]
)
robustness = results['robustness_results']['relative_robustness']
print(f'Domain shift robustness: {robustness:.3f}')
"

# Multi-modal benchmark
python -c "
from adaptiveneuralnetwork.benchmarks.multimodal import run_multimodal_benchmark
results = run_multimodal_benchmark(
    modalities=['text', 'image'], 
    epochs=3, 
    batch_size=16
)
print(f'Multi-modal accuracy: {results[\"final_test_accuracy\"]:.3f}')
"
```

### Performance Profiling

```bash
python scripts/profile.py --profile-type comprehensive
```

## 🏗 Architecture

### Core Components

- **`adaptiveneuralnetwork/core/`** — Vectorized node states, phase scheduling, and dynamics
  - `nodes.py` — Tensor-based node state management
  - `phases.py` — Phase scheduling (active, sleep, interactive, inspired)
  - `dynamics.py` — Core dynamics engine with energy and activity updates

- **`adaptiveneuralnetwork/training/`** — Training infrastructure
  - `datasets.py` — MNIST and synthetic dataset loaders
  - `loops.py` — Training loops with metrics tracking and checkpointing

- **`adaptiveneuralnetwork/benchmarks/`** — Standardized benchmarks
  - `vision/mnist.py` — MNIST classification benchmark

- **`adaptiveneuralnetwork/api/`** — High-level API
  - `model.py` — Main AdaptiveModel class
  - `config.py` — Configuration management with YAML support

### Key Features

- **Vectorized Operations**: Efficient batch processing for training and inference
- **Phase-Based Dynamics**: Biologically-inspired phases (active, sleep, interactive, inspired)
- **Energy Management**: Node energy levels influence behavior and phase transitions
- **Adaptive Learning**: Dynamic adaptation rates and node connectivity
- **Multi-Backend Support**: PyTorch, JAX, and neuromorphic hardware compatibility
- **Domain Shift Robustness**: CIFAR-10 corrupted benchmarks for testing robustness
- **Multi-Modal Learning**: Text + image processing capabilities
- **Neuromorphic Compatibility**: Spike-based computation and hardware abstraction
- **Production-Ready**: Comprehensive testing, CI/CD, and packaging

## 🔬 Backend Comparison

| Backend | Use Case | Performance | Features |
|---------|----------|-------------|----------|
| **PyTorch** | General purpose, GPU/CPU | High | Dynamic graphs, CUDA, standard ML |
| **JAX** | Research, TPU, functional | Very High | JIT compilation, auto-vectorization, functional programming |
| **Neuromorphic** | Edge computing, low power | Specialized | Spike-based, event-driven, hardware simulation |

## 📊 Benchmark Results

| Model | Dataset | Accuracy | Training Time | Active Nodes |
|-------|---------|----------|---------------|--------------|
| Adaptive-100 | MNIST | ~95%* | ~60s* | ~60%* |
| Adaptive-200 | MNIST | TBD | TBD | TBD |

*Preliminary results - benchmarks in progress

## 🛠 Development

### Running Tests

```bash
# Unit tests
pytest adaptiveneuralnetwork/tests/ -v

# Integration tests
pytest adaptiveneuralnetwork/tests/test_integration.py -v

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Linting
ruff check adaptiveneuralnetwork/

# Formatting
black adaptiveneuralnetwork/

# Type checking  
mypy adaptiveneuralnetwork/core/ adaptiveneuralnetwork/api/
```

## 🗺 Roadmap

### Current Version (0.3.0)
- [x] Vectorized core abstractions
- [x] MNIST benchmark pipeline
- [x] Basic training loops and metrics
- [x] Configuration system
- [x] Profiling utilities
- [x] CI/CD infrastructure
- [x] **Domain shift robustness (CIFAR-10 corrupted)**
- [x] **JAX backend for advanced acceleration**
- [x] **Multi-modal benchmarks (text + image)**
- [x] **Neuromorphic hardware compatibility layer**

### Previous Versions
#### Version 0.2.0
- [x] Continual learning (Split MNIST)
- [x] Advanced phase controllers (anxiety/restorative mechanics)
- [x] Energy/activity sparsity metrics
- [x] Sleep-phase ablation studies

#### Version 0.1.0 
- [x] Basic adaptive neural network implementation
- [x] PyTorch integration
- [x] MNIST benchmarking

## 📖 Documentation

- [API Reference](docs/api/) — Detailed API documentation
- [Configuration Guide](docs/configuration.md) — Configuration options and examples
- [Benchmarking Guide](docs/benchmarking.md) — Running and interpreting benchmarks
- [Contributing Guide](CONTRIBUTING.md) — Development setup and contribution guidelines

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Issue and PR templates

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [GitHub Repository](https://github.com/V1B3hR/adaptiveneuralnetwork)
- [Issue Tracker](https://github.com/V1B3hR/adaptiveneuralnetwork/issues)
- [Changelog](CHANGELOG.md)