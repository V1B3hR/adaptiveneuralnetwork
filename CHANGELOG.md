# Changelog

All notable changes to the Adaptive Neural Network project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-17

### Added

#### Core Configuration System
- **Centralized Configuration**: New `AdaptiveNeuralNetworkConfig` class with nested configuration for all subsystems
- **Multiple Configuration Sources**: Support for YAML/JSON files, environment variables, and runtime overrides
- **Runtime Toggles**: Enable/disable proactive interventions, attack resilience features, and trend analysis
- **Configuration Integration**: Seamlessly integrated with `AliveLoopNode` with backward compatibility
- **Structured Logging**: Configurable event logging for config changes, interventions, and attacks
- **Example Configurations**: `example_genome.json` and `benchmark_config.json` with different profiles

#### Proactive Interventions Configuration
- Configurable anxiety, calm, energy, joy, grief, hope, anger, and resilience interventions
- Adjustable thresholds for intervention triggers (e.g., `anxiety_threshold`, `help_signal_cooldown`)
- Per-subsystem enable/disable flags for fine-grained control

#### Attack Resilience Configuration  
- **Energy Drain Resistance**: Configurable resistance factor (0.0-1.0) and per-attacker ratio limits
- **Signal Redundancy**: Adjustable redundancy levels and frequency hopping enable/disable
- **Jamming Detection**: Configurable sensitivity for jamming detection
- **Trust Manipulation**: Growth rate limits and rapid trust thresholds for detection

#### Bitext Training Pipeline
- **BitextDatasetLoader**: Kaggle integration via kagglehub with local CSV fallback
- **TextClassificationBaseline**: Scikit-learn TF-IDF + LogisticRegression baseline
- **CLI Interface**: `run_bitext_training.py` with smoke and benchmark modes
- **Synthetic Data Generation**: Fallback data generation when real data unavailable
- **GitHub Actions Integration**: Automated bitext training workflow with configurable parameters

#### Testing Infrastructure
- **Configuration Tests**: 52 comprehensive tests covering config functionality and integration
- **Behavioral Change Tests**: Verification that config flags actually change system behavior
- **Monotonicity Tests**: Ensure parameter changes have expected directional effects
- **Bounds Testing**: Verify configuration limits are respected
- **Integration Tests**: AliveLoopNode integration with configuration system

#### Dependencies and Tooling
- **Core Dependencies**: Updated requirements.txt with pinned runtime dependencies
- **Development Tools**: New dev-requirements.txt with pytest, black, mypy, etc.
- **Optional Dependencies**: New `[nlp]` extra with pandas, scikit-learn, kagglehub
- **Python 3.12 Support**: Now requires Python 3.12+ (removed 3.9-3.11 support)
- **Editor Configuration**: Added .editorconfig for consistent formatting

#### GitHub Actions Workflows
- **Bitext Training Workflow**: Automated training with workflow_dispatch and scheduled runs
- **Enhanced CI**: Updated CI to support Python 3.12+ with bitext smoke tests
- **Artifact Management**: Structured artifact uploads with validation and summaries

#### Documentation
- **Bitext Training Guide**: Complete documentation in `docs/bitext_training.md`
- **Configuration Examples**: Documented configuration options and usage patterns
- **Updated README**: New sections for configuration system and bitext training
- **API Documentation**: Inline documentation for all new modules

### Changed
- **Version**: Bumped from 0.3.0 to 0.1.0 (reset for official release)
- **Python Compatibility**: Now requires Python 3.12+ (removed 3.9-3.11 support)
- **Dependencies**: Reorganized core vs optional dependencies for lighter installs
- **AliveLoopNode**: Enhanced constructor to accept configuration with backward compatibility
- **Rolling History**: Now configurable via `rolling_history.max_len` parameter
- **CI Pipeline**: Extended testing matrix and added bitext training validation

### Enhanced
- **Trend Analysis**: Configurable window sizes and prediction parameters
- **Memory Management**: Configurable history lengths for all emotional state tracking
- **Environment Adaptation**: Configurable stress thresholds and adaptation rates
- **Logging**: Structured event logging with configurable verbosity levels

### Technical Details
- **Lines of Code**: Added ~50,000 lines across configuration, training, and testing modules
- **Test Coverage**: 52 tests covering configuration, behavioral changes, and integration
- **Configuration Parameters**: 25+ configurable parameters across 5 major subsystems
- **Backwards Compatibility**: All existing code works without modification

### Development Infrastructure
- **Validation**: Configuration validation with warnings for out-of-range values
- **Reproducibility**: Deterministic seeds and controlled randomness throughout
- **Error Handling**: Graceful degradation when optional dependencies missing
- **Documentation**: Comprehensive usage examples and troubleshooting guides

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
- **Python Support**: 3.12+
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