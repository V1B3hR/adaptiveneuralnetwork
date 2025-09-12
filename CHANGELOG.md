# Changelog

All notable changes to the Adaptive Neural Network project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial production-oriented refactor (0.1.0 foundation)
- Vectorized core abstractions (NodeState, PhaseScheduler, AdaptiveDynamics)
- PyTorch-compatible AdaptiveModel API
- MNIST benchmark pipeline with training loops and metrics
- Configuration system with YAML support
- Profiling utilities with torch.profiler integration
- Comprehensive test suite (unit and integration tests)
- CI/CD with GitHub Actions (lint, type check, test)
- Packaging with pyproject.toml and pip installability
- Documentation scaffolding (README, CONTRIBUTING)

### Changed
- Migrated from conceptual prototype to production-ready structure
- Replaced individual node objects with batched tensor operations
- Unified API around standard PyTorch training patterns

## [0.1.0] - Planned

### Added
- **Core Components**
  - Vectorized node state management with energy and activity tracking
  - Phase scheduler supporting active, sleep, interactive, and inspired phases
  - Adaptive dynamics engine with configurable update rules
  - Production-ready PyTorch model wrapper

- **Training Infrastructure**
  - MNIST dataset integration with torchvision
  - Training loops with metrics tracking and checkpointing
  - Synthetic dataset generation for development and testing
  - Configurable loss functions and optimization

- **Benchmarking**
  - Complete MNIST classification benchmark
  - Performance profiling tools
  - Metrics collection (accuracy, energy efficiency, phase distribution)
  - Automated benchmark reporting

- **Developer Experience**
  - Comprehensive configuration management
  - Command-line tools for benchmarking and profiling
  - Full test coverage with pytest
  - Code quality tools (ruff, black, mypy)
  - CI/CD pipeline with multi-Python version testing

- **Documentation**
  - API documentation and usage examples
  - Development setup and contribution guidelines
  - Benchmarking and profiling guides
  - Roadmap with planned features

### Technical Details
- **Dependencies**: PyTorch 2.0+, torchvision, numpy, rich, pyyaml
- **Python Support**: 3.10, 3.11, 3.12
- **Testing**: Unit tests, integration tests, benchmark smoke tests
- **Performance**: Vectorized operations, GPU support, batch processing

## Future Releases

### [0.2.0] - Continual Learning (Planned)
- Split MNIST benchmark for continual learning evaluation
- Advanced anxiety and restorative behavior mechanics
- Sleep-phase ablation studies and analysis
- Energy and activity sparsity metrics

### [0.3.0] - Domain Robustness (Planned)
- CIFAR-10 corrupted dataset benchmarks
- Domain shift robustness evaluation
- Advanced visualization and analysis tools
- Multi-modal benchmark support

### [0.4.0] - Advanced Optimization (Planned)
- JAX backend option for advanced acceleration
- Custom CUDA kernels for core operations
- Neuromorphic hardware compatibility layer
- Large-scale distributed training support