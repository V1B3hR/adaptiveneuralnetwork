# Performance & Profiling Guide

This document consolidates runtime, profiling, coverage, and energy/phase efficiency best practices.

## 1. Profiling Entry Points

```bash
python scripts/profile.py --profile-type comprehensive
adaptive-profile --profile-type phases
```

Profile types (suggested taxonomy):
- `layers` – per-module forward/backward cost
- `phases` – time per phase transition cycle
- `energy` – energy normalization & redistribution overhead
- `sparsity` – active vs dormant node ratios
- `backend` – backend comparison harness

## 2. Metrics to Track

| Metric | Rationale | Source |
|--------|-----------|--------|
| Phase cycle duration | Detect scheduling bottlenecks | profiler output |
| Active node ratio | Gauges sparsity & adaptive pruning impact | internal stats |
| Energy convergence steps | Stability / equilibrium analysis | dynamics logs |
| Peak memory (MB) | Backend + model scaling | torch / JAX profilers |
| Time per batch | Throughput baseline | training loop |
| Robustness delta (%) | Domain shift resilience | enhanced_robustness_results.json |
| Adversarial retention (%) | Output stability post perturbation | adversarial_results.json |

## 3. Backend Comparison

Run standardized suite:
```bash
adaptive-benchmark --dataset mnist --epochs 3 --backend pytorch --report backend_pytorch.json
adaptive-benchmark --dataset mnist --epochs 3 --backend jax --report backend_jax.json
```

Aggregate:
```python
import json, statistics
pt = json.load(open("backend_pytorch.json"))
jx = json.load(open("backend_jax.json"))
print("Speedup:", pt["timing"]["epoch_mean"] / jx["timing"]["epoch_mean"])
```

## 4. Energy & Sparsity

Log distribution snapshots every N steps:
```python
energy_stats = model.get_energy_stats()   # (proposed helper)
sparsity = energy_stats["active_ratio"]
```

Plot (example):
```python
import matplotlib.pyplot as plt
plt.plot(history["active_ratio"])
plt.title("Active Node Ratio Over Training")
plt.show()
```

## 5. Coverage & Type Safety

```bash
pytest --cov adaptiveneuralnetwork --cov-report term-missing
mypy adaptiveneuralnetwork/
```

Set coverage gate in CI (example):
```
--cov-fail-under=70
```

## 6. Automation Recommendations

| Automation | Benefit |
|------------|---------|
| JSON → README table generation | Always current benchmarks |
| JAX vs PyTorch delta alert | Detect regression |
| Robustness threshold guard | Fails CI if robustness < last-release - tolerance |
| Adaptive sparsity drift detector | Alerts if network collapses to too sparse or dense |

## 7. Scaling Guidelines

| Scale Dimension | Consideration |
|-----------------|---------------|
| num_nodes | Quadratic effects if dense connectivity; encourage sparse linking |
| hidden_dim | Memory & compute heavy; benchmark JAX for large dims |
| backend | JAX JIT wins after warmup; PyTorch faster cold start |
| modalities | Ensure encoder freezing for text early epochs to stabilize dynamics |

## 8. Suggested Future Tools

- `adaptive-report` CLI to consolidate metrics into markdown.
- In-memory flamegraph integration (py-spy or torch.profiler export).

## 9. Reproducing Results

```bash
SEED=42
python scripts/run_benchmark.py --dataset mnist --epochs 5 --seed $SEED --deterministic
```

Document:
- Python version
- Backend commit SHA
- Hardware (GPU / TPU / CPU)
- Extras installed

---

Maintainer Note: Update this document whenever profiling output schema changes.
