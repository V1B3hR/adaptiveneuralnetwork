# Adaptive Neural Network

[![CI](https://github.com/V1B3hR/adaptiveneuralnetwork/workflows/CI%20-%20Train,%20Test,%20Coverage%20&%20Artifacts/badge.svg)](https://github.com/V1B3hR/adaptiveneuralnetwork/actions)
![Coverage](https://img.shields.io/badge/coverage-71%25-yellow)
![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue)
![Python Versions](https://img.shields.io/pypi/pyversions/adaptiveneuralnetwork)

## What is Adaptive Neural Network?

**Adaptive Neural Network** is a production‑ready, biologically‑inspired framework for neural networks that go beyond conventional architectures. It focuses on dynamic internal state evolution, phase transitions, and energy modulation—enabling neural behaviors such as selective activation, sleep/active cycles, adaptive node regulation, and real-time robustness. The framework is designed for research and real-world deployments, supporting PyTorch, JAX, and neuromorphic backends.

Key features:
- **Vectorized phase-driven dynamics** (active, sleep, interactive, inspired)
- **Energy & sparsity–aware node regulation**
- **Multi-backend support** (PyTorch, JAX, neuromorphic abstraction)
- **Robustness, adversarial & multimodal benchmarks**
- **HR Analytics integration for employee attrition prediction**
- **Structured validation guides for intelligence, robustness, spatial reasoning, and production signal processing**

> **Why "Adaptive"?**  
> The framework introduces dynamic phase transitions, energy modulation, and proactive interventions inspired by biological neural systems. Unlike conventional static feed-forward networks, adaptive neural networks can self-modulate, respond to stressors, and demonstrate emergent behaviors in complex environments.

---

## 🏆 Biggest Achievements

### 1. **Biologically Inspired Phase Dynamics**
- Introduces active, sleep, interactive, and inspired phases for nodes—allowing networks to rest, reorganize, and creatively recombine knowledge.

### 2. **Energy & Sparsity Regulation**
- Implements energy pools, node recruitment/dropout, and sparsity-aware learning for improved efficiency and resilience.

### 3. **Multi-Backend & Neuromorphic Support**
- Runs seamlessly on PyTorch and JAX; includes experimental compatibility with neuromorphic hardware abstraction.

### 4. **Robustness & Adversarial Benchmarks**
- Provides out-of-the-box benchmarks for domain shift, corruption, and adversarial attacks—complete with JSON artifacts for reproducibility.

### 5. **HR Analytics Integration**
- Features built-in analysis and prediction for IBM HR Analytics Employee Attrition dataset—including synthetic data fallback, CI/CD integration, and artifact outputs for enterprise validation.

### 6. **Centralized Configuration System**
- Offers unified YAML/JSON/env-based configuration for reproducible experiments, runtime overrides, and proactive interventions.

### 7. **Multimodal & NLP Pipelines**
- Supports multimodal (text+image) fusion, bitext training, and part-of-speech tagging with dynamic heuristic-driven training.

### 8. **Comprehensive Testing & Profiling**
- Integrated test matrix: unit, integration, robustness, and static quality checks. Profiling and coverage utilities for phase transitions, energy costs, and backend comparisons.

### 9. **Ethical & Responsible AI Artifacts**
- Includes explicit frameworks for AI ethics, robustness validation, intelligence benchmarking, and production signal processing.

---

## 🚀 Quick Start

### Installation (Editable Dev)
```bash
git clone https://github.com/V1B3hR/adaptiveneuralnetwork.git
cd adaptiveneuralnetwork
pip install -e .
```

### Extras
Install with multiple extras:
```bash
pip install -e ".[jax,neuromorphic,multimodal]"
pip install -e ".[nlp,dev]"  # NLP + development tools
```

---

## 📊 HR Analytics Integration

The framework supports IBM HR Analytics Employee Attrition dataset analysis and prediction.

**Steps:**
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Place CSV file at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
3. Or use synthetic data (auto-generated if real dataset is absent)

**Run training:**
```bash
python runsimulation.py
EPOCHS=50 BATCH_SIZE=128 python runsimulation.py
python runsimulation.py 42   # Reproducible seed
```

Artifacts in `outputs/`:  
- `hr_training_results.json`  
- `dataset_info.json`  
- `hr_model_weights.json`  

Integrated into CI/CD with automated dataset caching, configurable parameters, and Python 3.12 compatibility.

---

## ⚙️ Configuration System

Centralized configuration for reproducible experiments and runtime parameter control:

**Quick Example:**
```python
from adaptiveneuralnetwork.config import AdaptiveNeuralNetworkConfig
config = AdaptiveNeuralNetworkConfig()
config.proactive_interventions.anxiety_threshold = 6.0
config.attack_resilience.energy_drain_resistance = 0.9
```

Supports JSON/YAML files, environment variables, and runtime overrides.

---

## 📝 Bitext Training & Text Classification

Text classification pipeline demonstrating adaptive state modulation.

**Smoke test:**
```bash
python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke
```

**Benchmark:**
```bash
python -m adaptiveneuralnetwork.training.run_bitext_training --mode benchmark --dataset-name username/sentiment-dataset --subset-size 5000
```

Programmatic usage with TF-IDF + LogisticRegression baseline, Kaggle integration, and deterministic seeds.

---

## 🧪 Benchmarks & Evaluation

Out-of-the-box benchmarks for classification, robustness, adversarial, and multimodal tasks.  
Artifacts:  
- `benchmark_results.json`  
- `enhanced_robustness_results.json`  
- `adversarial_results.json`  
- `final_validation.json`  

Robustness and adversarial guides included.

---

## 🏗 Architecture Overview

Phases: ACTIVE → INTERACTIVE → SLEEP → INSPIRED  
Modules:  
- `core/nodes.py` – Vectorized node state  
- `core/phases.py` – Scheduler & transitions  
- `core/dynamics.py` – Energy dynamics  
- `api/model.py`, `api/config.py` – High-level APIs  
- `benchmarks/`, `scripts/` – Standard & advanced benchmarks  

---

## 🛡 Responsible & Ethical AI

Artifacts:
- `ethicsframework.md`
- `ROBUSTNESS_VALIDATION_GUIDE.md`
- `INTELLIGENCE_BENCHMARK_GUIDE.md`
- `PRODUCTION_SIGNAL_PROCESSING.md`
- `SPATIAL_DIMENSION_IMPLEMENTATION_SUMMARY.md`

---

## 🧪 Intelligence & Behavioral Testing

### Comprehensive Test Coverage

The Adaptive Neural Network includes three specialized test categories validating advanced intelligent behaviors:

#### **Cognitive Intelligence Testing** (17 tests)
- **Adaptive Reasoning**: Context-dependent strategy switching, environmental adaptation
- **Meta-Learning**: Few-shot learning, knowledge transfer, learning-to-learn mechanisms  
- **Creative Problem Solving**: Novel solution generation, divergent thinking, creative insights

#### **Emergent Behavior Testing** (12 tests)
- **Phase Coherence**: Meaningful phase transitions, behavioral consistency
- **Energy-Intelligence Correlation**: Performance optimization based on energy levels

#### **Biological Plausibility Testing** (14 tests) 
- **Neuroplasticity Simulation**: Hebbian learning, synaptic adaptation, LTP modeling
- **Circadian Rhythm Modeling**: Sleep-wake cycles, performance modulation

### Test Results Summary

| Category | Tests | Status | Key Features |
|----------|-------|--------|--------------|
| Cognitive Intelligence | 17 | ✅ 100% | Context adaptation, creativity, meta-learning |
| Emergent Behavior | 12 | ✅ 100% | Phase coherence, energy optimization |
| Biological Plausibility | 14 | ✅ 100% | Neural plasticity, circadian rhythms |
| **Total** | **43** | **✅ 100%** | **Full intelligence validation** |

**Full Results**: See [`tests/TEST_RESULTS.md`](tests/TEST_RESULTS.md) for detailed test outcomes and performance metrics.

### Running Intelligence Tests

```bash
# Cognitive intelligence tests
python -m unittest discover tests/cognitive_intelligence/ -v

# Emergent behavior tests  
python -m unittest discover tests/emergent_behavior/ -v

# Biological plausibility tests
python -m unittest discover tests/biological_plausibility/ -v

# Quick validation (all categories)
python -m unittest discover tests/cognitive_intelligence/ -q
python -m unittest discover tests/emergent_behavior/ -q
python -m unittest discover tests/biological_plausibility/ -q
```

---

## 🧪 Testing & Quality

```bash
pytest adaptiveneuralnetwork/tests -m "unit"
pytest -m "integration"
pytest -m "not slow"
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
