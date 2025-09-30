Refactor phases.

Phase 0 – Inventory & Metrics (Foundation)
Purpose: Understand the current state and quantify baseline performance before changing anything.

Entry Criteria:

Code builds and runs end-to-end.
Core Tasks:

 List all modules (data, model, training, utilities).
 Generate dependency graph (e.g., pydeps).
 Add lightweight timing & memory instrumentation (per batch).
 Profile 1–2 representative training runs (CPU + GPU).
 Capture data loader throughput (samples/sec).
 Record peak GPU memory and host RSS.
 Identify top 5 hotspots (function level).
 Store baseline metrics JSON (benchmarks/baseline.json).
Exit Criteria:

Hotspots ranked.
Baseline reproducible and documented.
Deliverables:

System map diagram
Profiling report
Baseline metrics file
Success Metrics (Baseline Numbers Captured):

Batch latency: ____ ms
Data throughput: ____ samples/sec
GPU util avg: ____ %
Peak GPU memory: ____ GB

Phase 1 – Data Layer Rework
Purpose: Remove I/O and collation bottlenecks; ensure efficient batching & transfer.

**Status: ✅ COMPLETE** - Achieved +949% throughput improvement vs Phase 0 baseline

Entry Criteria:

Baseline metrics established (Phase 0 complete).
Core Tasks:

 Replace per-sample Python loops with vectorized batch collation.
 Introduce pinned memory + async prefetch (e.g., prefetch factor or ring buffer).
 (Optional) Convert raw dataset to memory-mapped or Arrow/Parquet format.
 Implement Dataset / Buffer abstraction returning ready tensors.
 Add index-based sampling (avoid copying large structures).
 Add mini benchmark script for loader only.
 Re-profile data path.
Exit Criteria:

Data loader no longer a top 2 hotspot.
Throughput improved ≥ target (define X%).
Deliverables:

New Dataset/Buffer API
Loader benchmark script + updated metrics snapshot
Success Metrics:

Data throughput: +949% (target: +30%) ✅
Loader CPU time share: < 5% ✅
Prefetch queue idle time: minimal ✅

Phase 2 – Core Tensor Path Optimization

Purpose: Reduce per-batch compute overhead and allocation churn.

Entry Criteria:

Data path stable and not dominant bottleneck.
Core Tasks:

 Audit tensor device transfers (eliminate duplicates).
 Ensure tensor layouts contiguous / channels-last if beneficial.
 Merge redundant elementwise ops (fuse expressions).
 Enable torch.compile or TorchScript (experiment).
 Remove unnecessary dtype casts.
 Evaluate mixed precision trial (forward-only test).
 Profile kernel launch counts pre/post changes.
Exit Criteria:

Step time reduced by target %.
Allocation count per batch decreased.
Deliverables:

Optimized forward/training path
Before/after profiling diff
Success Metrics:

Mean step latency: -X%
Allocations per step: -Y%
Kernel launches: -Z%

Phase 3 – Model Architecture Modularization

Purpose: Make architecture configurable and extensible without editing core logic.

Entry Criteria:

Stable optimized tensor path (Phase 2).
Core Tasks:

 Extract layer classes into separate modules.
 Introduce layer registry (string → class).
 Implement config-driven model assembly (YAML/JSON).
 Remove hidden global state (random seeds localized).
 Add tests: construct model variants from config.
Exit Criteria:

New model variant can be added via config only.
Core model file LOC reduced vs baseline.
Deliverables:

Layer registry
Config examples
Architecture assembly function
Success Metrics:

Core architecture LOC: -X%
Time to add new layer: < Y minutes
Variant config coverage: Z models

Phase 4 – Training Loop Abstraction

Purpose: Centralize training orchestration with hooks/callbacks for extensibility.

Entry Criteria:

Modular model architecture complete.
Core Tasks:

 Implement Trainer class (fit, evaluate).
 Define callback interface (events: epoch start/end, batch start/end, after backward).
 Integrate AMP support toggle.
 Add gradient accumulation option.
 Add deterministic seed initialization.
 Add logging callback (throughput, loss).
 Write unit test with mock callback order assertions.
Exit Criteria:

Existing training logic replaced by Trainer.
At least 2 callbacks functioning (logging, profiling).
Deliverables:

Trainer module
Callback examples
Tests for callback sequencing
Success Metrics:

Lines duplicated across scripts: -X%
Adding new behavior (e.g., LR scheduler logging) requires 0 core edits.

Phase 5 – Parallelization & Hardware Utilization

Purpose: Maximize device utilization and reduce idle periods.

Entry Criteria:

Stable trainer abstraction.
Core Tasks:

 Measure current GPU utilization (profiling tool).
 Enable multi-process Distributed Data Parallel (if multi-GPU available).
 Add gradient checkpointing for memory pressure (if needed).
 Integrate mixed precision fully (training + scaler).
 Optimize batch size (auto-scaling search).
 Overlap data prefetch with compute (verify queue depth).
 Track utilization before/after for 3 runs.
Exit Criteria:

GPU utilization improved to target.
No regression in accuracy/metrics.
Deliverables:

DDP/FSDP setup (conditional)
Mixed precision training path
Utilization report
Success Metrics:

GPU utilization: +X%
Memory footprint: -Y% or batch size +Z%
Training time per epoch: -Q%

Phase 6 – Evaluation & Validation Layer

Purpose: Reliable, reproducible model assessment and benchmark tracking.

Entry Criteria:

Performance path mostly optimized.
Core Tasks:

 Create evaluation module separate from training.
 Add standardized metrics (accuracy, loss, custom metrics).
 Implement deterministic test run script.
 Add microbenchmarks (forward-only latency, data loader throughput, memory).
 Store benchmark results versioned (JSON history).
 Add drift detection (compare last vs median of last N).
Exit Criteria:

One command produces evaluation & benchmark artifacts.
Baseline metrics versioned for >1 run.
Deliverables:

eval/ scripts
benchmarks/ history JSON
Metrics comparison utility
Success Metrics:

Repro variance (latency std dev): < X%
Benchmark automation success rate: 100%

Phase 7 – machine deep learning process

Phase 8 – Documentation & Onboarding
Purpose: Make the refactored system understandable and maintainable.

Entry Criteria:

Core systems stabilized (previous phases done).
Core Tasks:

 Architecture diagram (current state).
 ADRs for major decisions (data format, trainer, parallelization choices).
 Onboarding guide: “Add a new layer”, “Add a callback”, “Run benchmarks”.
 Glossary of internal terms.
 Update README with performance summary table.
 Contribution guide (coding standards, profiling workflow).
Exit Criteria:

A new contributor can implement a new layer using only docs.
All major decisions have ADR entries.
Deliverables:

docs/adr/*.md
CONTRIBUTING.md
Updated README with perf table
Success Metrics:

Time-to-first-PR for new contributor: < X days
Docs coverage satisfaction (subjective review) ≥ Y/10
Master Checklist (Condensed View)
 ✅ Phase 0: Baseline established
 ✅ Phase 1: Data loader optimized (+949% throughput)
 Phase 2: Core tensor path streamlined
 Phase 3: Modular architecture
 Phase 4: Trainer + callbacks
 Phase 5: Parallelization & AMP
 Phase 6: Bench & eval suite
 Phase 7: Machine learning
 Phase 8: Documentation complete


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
