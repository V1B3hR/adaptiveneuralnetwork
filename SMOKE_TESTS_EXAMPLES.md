# Smoke Tests - Quick Start Examples

This document provides quick start examples for running smoke tests for quick validation.

## 🚀 Quick Start (30 seconds)

The fastest way to run smoke tests:

```bash
# Option 1: Use the Python runner
python run_smoke_tests.py

# Option 2: Use the Bash runner
./run_smoke_tests.sh

# Option 3: Direct CLI
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
```

## 📋 What You'll See

### Typical Output

```
==================================================
Training Summary (smoke mode)
==================================================
Status: ✓ SUCCESS
Runtime: 0.04 seconds
Data: 80 train, 20 val samples
Train Accuracy: 0.8750
Validation Accuracy: 0.7500

Training completed successfully!
```

### Generated Files

```
outputs/
├── smoke_test_results.json  # Detailed metrics (3.7 KB)
└── smoke_test_model.pkl     # Trained model (7.0 KB)
```

## 💡 Common Use Cases

### 1. Development Quick Check

Before committing code, verify nothing is broken:

```bash
python run_smoke_tests.py --cli-only
```

**Expected:** Completes in < 5 seconds with ✓ SUCCESS

### 2. CI/CD Pipeline

Add to your workflow:

```yaml
# .github/workflows/test.yml
- name: Run Smoke Tests
  run: |
    pip install -e '.[nlp]'
    python run_smoke_tests.py
```

### 3. Pre-Deployment Validation

Before deploying, verify the system works:

```bash
./run_smoke_tests.sh
```

**Expected:** All tests pass with performance metrics

### 4. Testing with Custom Data

Use your own CSV file:

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
    --mode smoke \
    --local-path my_data.csv \
    --output-dir results
```

**Expected:** Trains on your data and generates results in `results/`

### 5. Different Sample Sizes

Test with different amounts of data:

```bash
# Small test (50 samples)
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
    --mode smoke \
    --subset-size 50

# Medium test (200 samples)
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
    --mode smoke \
    --subset-size 200
```

## 📊 Understanding Results

### Results JSON Structure

```json
{
  "mode": "smoke",
  "success": true,
  "runtime_seconds": 0.04,
  "dataset_info": {
    "train_samples": 80,
    "val_samples": 20,
    "data_source": "synthetic"
  },
  "train_metrics": {
    "train_accuracy": 0.8750,
    "val_accuracy": 0.7500
  },
  "eval_metrics": {
    "accuracy": 0.7500,
    "precision": 0.8333,
    "recall": 0.7500,
    "f1_score": 0.7333
  }
}
```

### Key Metrics Explained

| Metric | What It Means | Good Range |
|--------|---------------|------------|
| **runtime_seconds** | How long training took | < 30s for smoke |
| **train_accuracy** | How well model fits training data | > 0.70 |
| **val_accuracy** | How well model generalizes | > 0.60 |
| **precision** | How many predictions are correct | > 0.60 |
| **recall** | How many actual cases are found | > 0.60 |
| **f1_score** | Balanced metric | > 0.60 |

## 🔧 Programmatic Usage

### Basic Usage

```python
from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test

# Run smoke test
results = run_smoke_test(
    subset_size=100,
    output_dir="outputs"
)

# Check success
if results['success']:
    print(f"✓ Accuracy: {results['eval_metrics']['accuracy']:.2%}")
else:
    print(f"✗ Error: {results['error']}")
```

### Advanced Usage

```python
from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test
from adaptiveneuralnetwork.training.models.text_baseline import TextClassificationBaseline
import json

# Run training
results = run_smoke_test(
    subset_size=100,
    output_dir="outputs"
)

# Load the trained model
model = TextClassificationBaseline()
model.load_model("outputs/smoke_test_model.pkl")

# Make predictions
texts = ["new sample text", "another example"]
predictions = model.predict(texts)
probabilities = model.predict_proba(texts)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")

# Access detailed results
with open("outputs/smoke_test_results.json") as f:
    metrics = json.load(f)
    
print(f"Feature count: {metrics['model_info']['num_features']}")
print(f"Runtime: {metrics['runtime_seconds']:.2f}s")
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
pip install -e '.[nlp]'
```

### Issue: "Smoke test failed with low accuracy"

**This is normal!** Smoke tests use synthetic data for speed. Low accuracy (even 50%) is expected and acceptable. The goal is to verify the pipeline works, not to achieve high accuracy.

### Issue: "Kaggle credentials not found"

**Solution:** Either:
1. Set environment variables:
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```
2. Or use local CSV files instead:
   ```bash
   python -m ... --local-path data.csv
   ```

### Issue: Slow execution

**Expected:** Smoke tests should complete in < 30s

If slower:
1. Check system resources
2. Reduce subset size: `--subset-size 50`
3. Use synthetic data (default, fastest)

## 📚 Additional Resources

- **Full Documentation:** See `SMOKE_TESTS.md`
- **Validation Report:** See `SMOKE_TEST_VALIDATION.md`
- **Quick Start Guide:** See `QUICKSTART.md`
- **Test Implementation:** See `tests/test_quickstart_features.py`

## ✅ Success Checklist

Before considering smoke tests successful, verify:

- [ ] Exit code is 0
- [ ] Status shows "✓ SUCCESS"
- [ ] Runtime is < 30 seconds
- [ ] Output files are created
- [ ] Accuracy is > 0.50 (50%)
- [ ] No error messages in output

## 🎯 Performance Targets

| Target | Smoke Test | Benchmark |
|--------|-----------|-----------|
| Runtime | < 30s | < 300s |
| Memory | < 500MB | < 2GB |
| Accuracy | > 50% | > 70% |
| Success Rate | 100% | 100% |

---

**Last Updated:** October 5, 2025  
**Status:** ✅ All smoke tests passing
