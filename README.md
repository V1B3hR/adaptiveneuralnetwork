# Adaptive Neural Network
[![CI](https://github.com/V1B3hR/adaptiveneuralnetwork/workflows/CI%20-%20Train,%20Test,%20Coverage%20&%20Artifacts/badge.svg)](https://github.com/V1B3hR/adaptiveneuralnetwork/actions)
![Coverage](https://img.shields.io/badge/coverage-71%25-yellow)
![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue)
![Python Versions](https://img.shields.io/pypi/pyversions/adaptiveneuralnetwork)

A production‑ready, biologically‑inspired adaptive neural network framework featuring:
- Vectorized phase‑driven dynamics (active / sleep / interactive / inspired).
- Energy & sparsity–aware node regulation.
- Multi‑backend support (PyTorch, JAX, neuromorphic abstraction layer).
- Robustness, adversarial & multimodal benchmarks.
- Structured validation guides (intelligence, robustness, spatial reasoning, production signal processing).

> Why Adaptive vs Conventional Architectures?  
> Adaptive Neural Network focuses on *dynamic internal state evolution* + *phase transitions* and *energy modulation* rather than static feed‑forward passes. This enables emerging behaviors: selective activation, contextual plasticity, and graceful degradation under domain shift.

---

## 🔥 Feature Tiers

| Tier | Capabilities | Files / Guides |
|------|--------------|----------------|
| Basic | Core nodes, phases, PyTorch backend, MNIST benchmark | `core/`, `benchmark_cli.py`, `vision/mnist.py` |
| Intermediate | Energy dynamics, CIFAR-10 standard & corrupted, profiling, config system | `dynamics.py`, `scripts/profile.py`, `config/` |
| Advanced | JAX backend, multimodal (text+image), neuromorphic layer, continual learning, robustness | `benchmarks/multimodal/`, `enhanced_robustness_results.json`, `ROBUSTNESS_VALIDATION_GUIDE.md` |
| Research | Adversarial evaluation, spatial dimension integration, intelligence benchmark readiness | `adversarial_results.json`, `SPATIAL_DIMENSION_IMPLEMENTATION_SUMMARY.md`, `INTELLIGENCE_BENCHMARK_GUIDE.md` |

---

## 🚀 Quick Start

### Installation (Editable Dev)

```bash
git clone https://github.com/V1B3hR/adaptiveneuralnetwork.git
cd adaptiveneuralnetwork
pip install -e .
```

### Extras

| Extra | Purpose | Included |
|-------|---------|----------|
| jax | JAX backend acceleration | jax, jaxlib, flax, optax |
| neuromorphic | Visualization + spike abstractions | scipy, matplotlib |
| multimodal | Text + image fusion | transformers, tokenizers |
| dev | Dev tooling | pytest, coverage, ruff, mypy, black |
| docs | Documentation build | sphinx, myst-parser |

Install with multiple extras:
```bash
pip install -e ".[jax,neuromorphic,multimodal]"
```

### Minimal Runtime Example

```python
from adaptiveneuralnetwork.api import create_adaptive_model, AdaptiveConfig

config = AdaptiveConfig(
    num_nodes=128,
    hidden_dim=64,
    backend="pytorch"   # "jax" or "neuromorphic" also valid
)
model = create_adaptive_model(config)
```

### Training Loop (PyTorch)

```python
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    output = model(batch.data)        # Internally phases may modulate activity
    loss = criterion(output, batch.target)
    loss.backward()
    optimizer.step()
```

### CLI Utilities

```
adaptive-benchmark --help
adaptive-profile  --help
```

Example:
```bash
adaptive-benchmark --dataset mnist --epochs 10 --backend pytorch
adaptive-benchmark --dataset cifar10 --epochs 5 --corruptions gaussian_noise,brightness
```

---

## 🧪 Benchmarks & Evaluation

Benchmark JSON artifacts already in repo:
- `benchmark_results.json` (General classification)
- `enhanced_robustness_results.json` (Corruption / domain shift)
- `adversarial_results.json` (Adversarial / perturbation resilience)
- `final_validation.json` (Aggregated final phase / node stats)

### Example (Load & Print)
```python
import json, pathlib
data = json.loads(pathlib.Path("benchmark_results.json").read_text())
print("Available runs:", list(data.keys()))
```

### (Placeholder) Current Snapshot
| Model Variant | Dataset | Test Acc | Active Node % | Notes |
|---------------|---------|----------|---------------|-------|
| Adaptive‑100  | MNIST   | ~0.95*   | ~60%*         | Preliminary |
| Adaptive‑200  | MNIST   | TBD      | TBD           | Pending |
| Adaptive‑128  | CIFAR10 | (fill)   | (fill)        | From `benchmark_results.json` |
| Adaptive‑JAX  | CIFAR10 | (fill)   | (fill)        | JIT accelerated |
| Multimodal‑Small | Text+Image | (fill) | (fill) | `run_multimodal_benchmark` |

*Replace placeholder values by running:
```bash
python scripts/run_benchmark.py --dataset mnist --epochs 10 --output benchmark_results.json
python scripts/run_benchmark.py --dataset cifar10 --epochs 5 --output benchmark_results.json
```

### Robustness / Corruption Testing
```python
from adaptiveneuralnetwork.benchmarks.vision.cifar10 import CIFAR10Benchmark
from adaptiveneuralnetwork.api import AdaptiveConfig
benchmark = CIFAR10Benchmark(AdaptiveConfig(num_nodes=64, hidden_dim=32))
results = benchmark.run_robustness_benchmark(
    corruption_types=['gaussian_noise','brightness'],
    severities=[1,3,5]
)
print(results['robustness_results']['relative_robustness'])
```

### Adversarial & Stress Tests
See: `adversarial_results.json` + `ROBUSTNESS_VALIDATION_GUIDE.md` for methodology.

---

## 📈 Performance & Profiling

Use:
```bash
python scripts/profile.py --profile-type comprehensive
adaptive-profile --profile-type phases
```

Collect coverage & type safety:
```bash
pytest --cov adaptiveneuralnetwork --cov-report term-missing
mypy adaptiveneuralnetwork/
```

See `docs/performance.md` (proposed) for:
- Phase transition cost breakdown
- Node activation sparsity histograms
- Energy pool convergence curves
- Backend comparison (PyTorch vs JAX wall‑clock)
- Memory footprint of dynamic node sets

---

## 🏗 Architecture Overview

Phases: ACTIVE → (optional INTERACTIVE) → SLEEP → INSPIRED (creative recombination)  
Each phase can adjust:
- Learning rate scaling
- Node recruitment / dropout
- Energy redistribution

Core Modules:
- `core/nodes.py` — Vectorized node state (energy, activity, adaptivity)
- `core/phases.py` — Scheduler & transitions
- `core/dynamics.py` — Energy dynamics, plasticity updates
- `api/model.py` — High-level `AdaptiveModel`
- `api/config.py` — Declarative YAML/obj config
- `benchmarks/` — Standard + robustness + multimodal
- `scripts/` — Benchmarking, profiling

(Consider adding an SVG diagram in `docs/images/architecture.svg`.)

---

## 🛡 Responsible & Ethical AI

This project includes explicit artifacts:
- `AI Ethics Framework` / `ethicsframework.md`
- `ROBUSTNESS_VALIDATION_GUIDE.md`
- `INTELLIGENCE_BENCHMARK_GUIDE.md`
- `PRODUCTION_SIGNAL_PROCESSING.md`
- `SPATIAL_DIMENSION_IMPLEMENTATION_SUMMARY.md`

Recommended next steps:
1. Add automated ethical compliance checklist script.
2. Integrate fairness/stability metrics into benchmark output schema.
3. Add gating CI job that fails if robustness regression > threshold.

---

## 🗺 Roadmap

### Current (0.3.0)
- Vectorized core abstractions
- MNIST + CIFAR-10 + corrupted domain shift
- JAX backend
- Multimodal (text+image) benchmark
- Neuromorphic compatibility layer
- Robustness + adversarial JSON outputs
- Profiling + coverage

### Proposed 0.4.0 (Planned)
- Distributed training (Ray / torch.distributed)
- ONNX export + model introspection
- Automated README benchmark table generation script
- Enhanced continual learning scenarios (e.g., blurred → corrupted → adversarial progression)
- Adaptive pruning & self-healing node lifecycle
- Energy-aware optimizer variants (meta-adaptation)
- Plugin system for custom phases
- Dataset abstraction unification + streaming (WebDataset / HuggingFace Datasets)
- Optional Graph / Spatial reasoning integration module
- Reproducibility harness (seed isolation + determinism report)

### Longer-Term
- Formal intelligence evaluation harness integration
- Neuromorphic hardware backends (Loihi / custom spike simulators)
- Probabilistic phase scheduling (stochastic policy)
- Mixed precision + quantization aware phases

---

## 🧪 Testing & Quality

```bash
# Unit tests
pytest adaptiveneuralnetwork/tests -m "unit"

# Integration
pytest -m "integration"

# Skip slow
pytest -m "not slow"

# Static quality
ruff check adaptiveneuralnetwork/
black --check adaptiveneuralnetwork/
mypy adaptiveneuralnetwork/
```

Suggested pre-commit hooks:
```
black .
ruff check --fix .
mypy adaptiveneuralnetwork/
pytest -q
```

---

## 📖 Documentation

- API Reference: `docs/api/`
- Configuration: `docs/configuration.md`
- Benchmarking: `docs/benchmarking.md`
- Performance: `docs/performance.md` (proposed new)
- Robustness: `ROBUSTNESS_VALIDATION_GUIDE.md`
- Intelligence Benchmarks: `INTELLIGENCE_BENCHMARK_GUIDE.md`

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md):
- Dev environment, style, test matrix
- Issue and PR templates
- Ethical + robustness contribution standards

---

## 🔗 Links
- [Repository](https://github.com/V1B3hR/adaptiveneuralnetwork)
- [Issue Tracker](https://github.com/V1B3hR/adaptiveneuralnetwork/issues)
- [Changelog](CHANGELOG.md)
- [Roadmap](roadmap.md)

---

## ✍️ Citation

If you use this project in research:

```bibtex
@software{adaptive_neural_network,
  title        = {Adaptive Neural Network: Phase-Driven Biologically Inspired Adaptive Learning},
  author       = {{Adaptive Neural Network Contributors}},
  year         = {2025},
  url          = {https://github.com/V1B3hR/adaptiveneuralnetwork},
  version      = {0.3.0}
}
```

---

## 📄 License
GNU General Public License v3.0 – see [LICENSE](LICENSE).
