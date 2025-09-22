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
- **HR Analytics integration for employee attrition prediction.**
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
| nlp | Bitext training & text classification | pandas, scikit-learn, kagglehub |
| pos | Part-of-Speech tagging support | seqeval, scikit-learn |
| dev | Dev tooling | pytest, coverage, ruff, mypy, black |
| docs | Documentation build | sphinx, myst-parser |

Install with multiple extras:
```bash
pip install -e ".[jax,neuromorphic,multimodal]"
pip install -e ".[nlp,dev]"  # NLP + development tools
```

---

## 📊 HR Analytics Integration

The framework now includes integrated support for IBM HR Analytics Employee Attrition dataset analysis and prediction using adaptive neural networks.

### Dataset Setup

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. **Place the CSV file** at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
3. **Or use synthetic data** - the system will automatically generate synthetic HR data if the real dataset is not available

### Running HR Analytics Training

```bash
# Run with default parameters
python runsimulation.py

# Run with custom training parameters
EPOCHS=50 BATCH_SIZE=128 python runsimulation.py

# With reproducible seed
python runsimulation.py 42
```

### Training Outputs

The training process creates several artifacts in the `outputs/` directory:
- `hr_training_results.json` - Training metrics and progress
- `dataset_info.json` - Dataset characteristics and metadata
- `hr_model_weights.json` - Model weights and architecture info

### CI/CD Integration

The HR Analytics training is integrated into the CI pipeline:
- **Automated dataset caching** for efficient builds
- **Configurable training parameters** via environment variables
- **Artifact uploading** for training results and coverage reports
- **Python 3.12 testing** for compatibility validation

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EPOCHS` | Number of training epochs | 10 |
| `BATCH_SIZE` | Training batch size | 32 |

---

## ⚙️ Configuration System

The v0.1.0 release introduces a centralized configuration system for reproducible experiments and runtime parameter control.

### Quick Configuration

```python
from adaptiveneuralnetwork.config import AdaptiveNeuralNetworkConfig
from core.alive_node import AliveLoopNode

# Create configuration
config = AdaptiveNeuralNetworkConfig()
config.proactive_interventions.anxiety_threshold = 6.0
config.attack_resilience.energy_drain_resistance = 0.9
config.rolling_history.max_len = 50

# Use with nodes
node = AliveLoopNode(position=[0, 0], velocity=[0, 0], config=config)
```

### Configuration Sources

**File-based (YAML/JSON):**
```python
# From JSON
config = AdaptiveNeuralNetworkConfig.from_json('config/examples/benchmark_config.json')

# From YAML  
config = AdaptiveNeuralNetworkConfig.from_yaml('config/my_config.yaml')
```

**Environment Variables:**
```bash
export ANN_TREND_WINDOW=10
export ANN_ANXIETY_ENABLED=false
export ANN_ENERGY_DRAIN_RESISTANCE=0.8

python your_script.py  # Automatically loads env vars
```

**Runtime Overrides:**
```python
from adaptiveneuralnetwork.config import load_config

config = load_config(
    'config/base.json',
    **{'trend_analysis.window': 15, 'log_level': 'DEBUG'}
)
```

### Key Configuration Areas

- **Proactive Interventions**: Enable/disable anxiety, calm, energy interventions
- **Attack Resilience**: Energy drain resistance, signal redundancy, jamming detection
- **Trend Analysis**: Window sizes, prediction steps
- **Rolling History**: Memory lengths for trend analysis
- **Environment Adaptation**: Stress thresholds, adaptation rates

See `config/examples/` for complete configuration examples.

---

## 📝 Bitext Training & Text Classification

New lightweight text classification pipeline for demonstrating state-modulated behavior:

### Quick Start

```bash
# Check dependencies
python -m adaptiveneuralnetwork.training.run_bitext_training --check-deps

# Smoke test (quick validation)
python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke

# Benchmark with Kaggle dataset
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
python -m adaptiveneuralnetwork.training.run_bitext_training \
  --mode benchmark \
  --dataset-name username/sentiment-dataset \
  --subset-size 5000
```

### Programmatic Usage

```python
from adaptiveneuralnetwork.training.bitext_dataset import BitextDatasetLoader
from adaptiveneuralnetwork.training.text_baseline import TextClassificationBaseline

# Load dataset
loader = BitextDatasetLoader(
    dataset_name="kaggle-user/dataset",
    sampling_fraction=0.1,
    normalize_text=True
)
train_df, val_df = loader.load_dataset()

# Train baseline model
baseline = TextClassificationBaseline(max_features=10000, random_state=42)
metrics = baseline.fit(
    texts=train_df['text'].tolist(),
    labels=train_df['label'].tolist(),
    validation_texts=val_df['text'].tolist(),
    validation_labels=val_df['label'].tolist()
)

# Make predictions
predictions = baseline.predict(["Sample text to classify"])
```

**Features:**
- Kaggle dataset integration via kagglehub
- Local CSV fallback support
- Smoke testing for CI/CD
- TF-IDF + LogisticRegression baseline
- Deterministic results with configurable seeds
- GitHub Actions workflow integration

See [docs/bitext_training.md](docs/bitext_training.md) for complete documentation.

---

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

### Part-of-Speech Tagging (NEW)
Sequence labeling with adaptive epoch/sample heuristics:

```bash
# Train on Kaggle POS dataset with automatic heuristics
python train_pos_tagging.py --data-path /path/to/pos_dataset --auto

# Quick test with synthetic data
python train_pos_tagging.py --synthetic --epochs 2 --max-sentences 100

# Evaluate trained model
python evaluate_pos_tagging.py --checkpoint pos_tagging_output/best_model.pt --data-path /path/to/test_data
```

**Features:**
- **Dynamic Heuristics**: Automatic epoch (12-40) and batch size selection based on dataset size
- **Model Options**: BiLSTM (default) or Transformer encoder
- **Comprehensive Metrics**: Token accuracy, macro/micro F1, per-tag F1 scores  
- **Flexible Loading**: Auto-detects CSV columns (sentence/word/pos)
- **Memory Optimization**: Gradient accumulation for large datasets

See `POS_TAGGING_GUIDE.md` for detailed usage and expected performance.

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
