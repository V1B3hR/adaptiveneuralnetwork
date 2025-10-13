# Integration Guide: Phase 6 Evaluation Layer

## Overview

This guide shows how to integrate the Phase 6 evaluation layer with existing evaluation scripts and workflows.

## Integration with `eval.py`

The Phase 6 evaluation layer can be used alongside the existing `eval.py` script. Here's how to integrate:

### Option 1: Direct Import

Add the Phase 6 modules to your existing evaluation workflow:

```python
# In eval.py or your custom evaluation script
from eval.metrics import compute_metrics
from eval.microbenchmark import run_microbenchmark
from eval.drift_detection import detect_drift

def evaluate_checkpoint(checkpoint_path: Path, config: WorkflowConfig):
    # ... existing code to load model and data ...
    
    # Add Phase 6 metrics computation
    from eval.metrics import compute_metrics
    import torch.nn as nn
    
    metrics = compute_metrics(
        model=model,
        data_loader=test_loader,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
        compute_detailed=True,
    )
    
    # Add to results
    results = {
        'checkpoint': str(checkpoint_path),
        'metrics': metrics.to_dict(),
        # ... other existing results ...
    }
    
    return results
```

### Option 2: Use as Standalone Tool

Keep `eval.py` as-is and use the Phase 6 evaluation layer as a complementary tool:

```bash
# Run existing evaluation
python eval.py --checkpoint checkpoints/model.pt --dataset mnist

# Run Phase 6 comprehensive evaluation
python eval/run_eval.py --model checkpoints/model.pt --dataset mnist --full
```

### Option 3: Wrapper Script

Create a wrapper that combines both:

```python
#!/usr/bin/env python3
"""Combined evaluation wrapper."""

import subprocess
import sys

def main():
    # Run legacy eval.py
    print("Running legacy evaluation...")
    subprocess.run([sys.executable, "eval.py", *sys.argv[1:]])
    
    # Run Phase 6 evaluation
    print("\nRunning Phase 6 comprehensive evaluation...")
    # Convert arguments from eval.py format to run_eval.py format
    # (e.g., --checkpoint -> --model)
    subprocess.run([sys.executable, "eval/run_eval.py", "--full", *converted_args])

if __name__ == "__main__":
    main()
```

## Integration with Benchmark Scripts

### Adding to Existing Benchmark Workflows

```python
# In benchmarks/scripts/run_benchmark.py
from eval.metrics import compute_metrics
from eval.microbenchmark import run_microbenchmark
from eval.drift_detection import detect_drift
import json
from datetime import datetime
from pathlib import Path

def run_benchmark_with_phase6(model, dataset, config):
    # ... existing benchmark code ...
    
    # Add Phase 6 metrics
    metrics = compute_metrics(
        model=model,
        data_loader=test_loader,
        device=device,
        loss_fn=loss_fn,
        compute_detailed=True,
    )
    
    # Add microbenchmarks
    microbenchmark = run_microbenchmark(
        model=model,
        data_loader=test_loader,
        device=device,
    )
    
    # Save results with versioning
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics.to_dict(),
        'microbenchmark': microbenchmark.to_dict(),
    }
    
    # Save to history
    history_dir = Path("benchmarks/history")
    history_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_file = history_dir / f"{timestamp_str}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Run drift detection
    drift_results = detect_drift(
        current_metrics=metrics.to_dict(),
        history_path=history_dir,
        lookback_n=5,
    )
    
    # Print drift warnings
    for drift in drift_results:
        if drift.drift_detected:
            print(f"⚠️  Drift detected in {drift.metric_name}: {drift.drift_direction}")
    
    return results
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark and Evaluate

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run Phase 6 Evaluation
        run: |
          python eval/run_eval.py \
            --model checkpoints/model.pt \
            --dataset mnist \
            --full \
            --output-dir benchmarks/history
      
      - name: Check for drift
        run: |
          python -c "
          from eval.drift_detection import detect_drift
          from pathlib import Path
          import json
          
          # Load latest results
          history = Path('benchmarks/history')
          latest = sorted(history.glob('*.json'))[-1]
          with open(latest) as f:
              results = json.load(f)
          
          # Check drift
          drifts = detect_drift(
              current_metrics=results['metrics'],
              history_path=history,
              lookback_n=5,
          )
          
          # Fail if critical drift detected
          for drift in drifts:
              if drift.drift_detected and abs(drift.drift_percentage) > 10:
                  print(f'Critical drift in {drift.metric_name}!')
                  exit(1)
          "
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: benchmarks/history/
```

## API Reference

### Quick Reference for Common Tasks

#### 1. Compute Metrics

```python
from eval.metrics import compute_metrics
import torch.nn as nn

metrics = compute_metrics(
    model=model,
    data_loader=test_loader,
    device=device,
    loss_fn=nn.CrossEntropyLoss(),
)

print(f"Accuracy: {metrics.accuracy:.2f}%")
```

#### 2. Run Microbenchmark

```python
from eval.microbenchmark import run_microbenchmark

results = run_microbenchmark(
    model=model,
    data_loader=test_loader,
    device=device,
)

print(f"Latency: {results.forward_latency_mean:.3f} ms")
```

#### 3. Detect Drift

```python
from eval.drift_detection import detect_drift

drift_results = detect_drift(
    current_metrics={'accuracy': 92.0},
    history_path="benchmarks/history",
    lookback_n=5,
)

for result in drift_results:
    if result.drift_detected:
        print(f"Drift in {result.metric_name}")
```

#### 4. Compare Metrics

```python
from eval.comparison import MetricsComparator

comparator = MetricsComparator("benchmarks/history")
comparisons = comparator.get_latest_runs(2)

# Generate report
report = comparator.generate_comparison_report(comparisons)
print(report)
```

## Best Practices

### 1. Versioning Results

Always save results to the history directory with timestamps:

```python
from datetime import datetime
from pathlib import Path
import json

history_dir = Path("benchmarks/history")
history_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
output_file = history_dir / f"{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
```

### 2. Reproducibility

Always set seeds for deterministic evaluation:

```python
from adaptiveneuralnetwork.utils.reproducibility import ReproducibilityHarness

harness = ReproducibilityHarness(master_seed=42, strict_mode=True)
harness.set_seed()

# Run evaluation...
```

### 3. Drift Detection

Configure appropriate thresholds based on your metrics:

```python
metric_directions = {
    "accuracy": True,      # Higher is better
    "loss": False,         # Lower is better
    "latency_ms": False,   # Lower is better
    "throughput": True,    # Higher is better
}

drift_results = detect_drift(
    current_metrics=metrics,
    history_path=history_path,
    lookback_n=5,
    threshold_std=2.0,  # 2 standard deviations
    metric_directions=metric_directions,
)
```

### 4. Error Handling

Always wrap evaluation in try-except for production:

```python
try:
    metrics = compute_metrics(model, data_loader, device)
    # Save results...
except Exception as e:
    logger.error(f"Evaluation failed: {e}")
    # Fallback handling...
```

## Migration Guide

### Migrating from Legacy Evaluation

If you have existing evaluation scripts, here's how to migrate:

**Before:**
```python
# Old evaluation
def evaluate(model, data_loader):
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total
```

**After:**
```python
# Phase 6 evaluation
from eval.metrics import compute_metrics

def evaluate(model, data_loader):
    metrics = compute_metrics(
        model=model,
        data_loader=data_loader,
        device=device,
        compute_detailed=True,
    )
    
    # Still get accuracy, but also precision, recall, F1, etc.
    return metrics
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the `eval/` directory is in your Python path
2. **Missing history**: Create `benchmarks/history/` directory if it doesn't exist
3. **Drift detection requires 2+ runs**: You need at least 2 historical runs for drift detection
4. **Memory issues with microbenchmarks**: Reduce `num_iterations` parameter

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

## Support

For questions or issues:
- See `docs/PHASE6_EVALUATION.md` for complete documentation
- Run `python demo_phase6_eval.py` to see examples
- Check `tests/test_phase6_eval.py` for usage patterns
