# Adaptive Neural Network

A production-ready biologically-inspired neural network with vectorized training capabilities and adaptive learning mechanisms. This library provides a PyTorch-compatible implementation of adaptive neural networks with phase-based dynamics, energy management, and continual learning support.

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from adaptiveneuralnetwork.api import AdaptiveModel, AdaptiveConfig

# Create configuration
config = AdaptiveConfig(
    num_nodes=100,
    hidden_dim=64,
    num_epochs=10,
    learning_rate=0.001
)

# Create and train model
model = AdaptiveModel(config)

# Standard PyTorch training loop
optimizer = torch.optim.Adam(model.parameters())
for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch.data)
    loss = criterion(output, batch.target)
    loss.backward()
    optimizer.step()
```

### Run MNIST Benchmark

```bash
# Quick test
python scripts/run_benchmark.py --quick-test --epochs 1

# Full benchmark
python scripts/run_benchmark.py --epochs 10 --batch-size 128
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
- **Production-Ready**: Comprehensive testing, CI/CD, and packaging

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

### Current Version (0.1.0)
- [x] Vectorized core abstractions
- [x] MNIST benchmark pipeline
- [x] Basic training loops and metrics
- [x] Configuration system
- [x] Profiling utilities
- [x] CI/CD infrastructure

### Planned (0.2.0)
- [ ] Continual learning (Split MNIST)
- [ ] Advanced phase controllers (anxiety/restorative mechanics)
- [ ] Energy/activity sparsity metrics
- [ ] Sleep-phase ablation studies

### Future (0.3.0+)
- [ ] Domain shift robustness (CIFAR-10 corrupted)
- [ ] JAX backend for advanced acceleration
- [ ] Multi-modal benchmarks
- [ ] Neuromorphic hardware compatibility

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