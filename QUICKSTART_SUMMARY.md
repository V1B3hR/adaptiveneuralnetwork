# Quick Start - Feature Summary

This document provides a quick visual summary of all Quick Start features.

## ✅ Features Implemented

All features from the problem statement are fully implemented and tested:

### 1️⃣ Run Smoke Tests for Quick Validation

**Purpose**: Fast validation for CI/CD, development, and testing

```bash
# Default smoke test (100 samples, ~5-30 seconds)
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke

# Custom smoke test
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode smoke \
  --subset-size 50 \
  --output-dir my_results
```

**Output**:
- ✓ Completes in 5-30 seconds
- ✓ Generates `smoke_test_results.json` with metrics
- ✓ Saves `smoke_test_model.pkl` for predictions

---

### 2️⃣ Run Benchmarks for Full Evaluation

**Purpose**: Comprehensive model evaluation with detailed metrics

```bash
# Basic benchmark
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --subset-size 1000

# Large-scale benchmark
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --subset-size 10000 \
  --output-dir production_models
```

**Output**:
- ✓ Detailed accuracy, precision, recall, F1 scores
- ✓ Confusion matrix
- ✓ Per-class classification report
- ✓ Feature importance analysis
- ✓ Full results in `benchmark_results.json`

---

### 3️⃣ Use Local CSV Files or Kaggle Datasets

**A. Local CSV Files**

```bash
# Prepare your CSV (columns: text, label)
# Example: data.csv with 'text' and 'label' columns

python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --local-path data/my_dataset.csv \
  --output-dir results
```

**B. Kaggle Datasets**

```bash
# Set up Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Train with Kaggle dataset
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --dataset-name username/dataset-name \
  --subset-size 5000
```

**C. Synthetic Data (Automatic Fallback)**

```bash
# No data source specified - uses synthetic data automatically
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
```

---

### 4️⃣ Configure Output Directories and Subset Sizes

**Full Configuration Examples**:

```bash
# Configure everything
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --local-path data.csv \
  --subset-size 5000 \
  --output-dir /path/to/results \
  --epochs 1 \
  --verbose

# Minimal configuration
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode smoke \
  --output-dir test_results

# Different subset sizes
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke --subset-size 50
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke --subset-size 100
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode benchmark --subset-size 1000
```

---

### 5️⃣ Access Trained Models and Detailed Metrics

**A. Load and Use Trained Models**

```python
from adaptiveneuralnetwork.training.models.text_baseline import TextClassificationBaseline

# Load model
model = TextClassificationBaseline()
model.load_model("outputs/smoke_test_model.pkl")

# Make predictions
texts = ["This is a great product", "Not happy with this"]
predictions = model.predict(texts)
probabilities = model.predict_proba(texts)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

**B. Access Detailed Metrics**

```python
import json

# Load results JSON
with open("outputs/smoke_test_results.json") as f:
    results = json.load(f)

# Access metrics
print(f"Accuracy: {results['eval_metrics']['accuracy']}")
print(f"F1 Score: {results['eval_metrics']['f1_score']}")
print(f"Runtime: {results['runtime_seconds']}s")

# Get feature importance
for class_name, features in results['feature_importance'].items():
    print(f"\nTop features for class {class_name}:")
    for feature, weight in features[:5]:
        print(f"  {feature}: {weight:.4f}")
```

**C. Analyze Confusion Matrix**

```python
# From results JSON
cm = results['eval_metrics']['confusion_matrix']
print("Confusion Matrix:")
for i, row in enumerate(cm):
    print(f"Class {i}: {row}")
```

---

## 📊 Output Files

### Smoke Mode
```
outputs/
├── smoke_test_results.json    # Metrics and configuration
└── smoke_test_model.pkl       # Trained model
```

### Benchmark Mode
```
outputs/
├── benchmark_results.json     # Detailed metrics
├── benchmark_model.pkl        # Trained model
└── confusion_matrix.png       # Visualization (if matplotlib available)
```

---

## 🧪 Testing

All features are fully tested with a comprehensive test suite:

```bash
# Run all tests
python tests/test_quickstart_features.py
```

**Test Coverage**:
- ✓ Smoke test with default settings
- ✓ Smoke test with custom settings
- ✓ Benchmark mode with full evaluation
- ✓ Local CSV file support
- ✓ Model loading and predictions
- ✓ Accessing detailed metrics
- ✓ Multiple subset size configurations

**Results**: 7/7 tests passing ✅

---

## 📖 Documentation

**Main Documentation**:
- [QUICKSTART.md](QUICKSTART.md) - Comprehensive quick start guide
- [examples/quickstart_example.py](examples/quickstart_example.py) - 5 working examples
- [docs/training/bitext_training.md](docs/training/bitext_training.md) - Detailed documentation
- [docs/training/TRAINING_GUIDE.md](docs/training/TRAINING_GUIDE.md) - Complete training guide

**Quick Links**:
- Installation: See [QUICKSTART.md](QUICKSTART.md#installation)
- Troubleshooting: See [QUICKSTART.md](QUICKSTART.md#troubleshooting)
- Command Reference: See [QUICKSTART.md](QUICKSTART.md#command-reference)

---

## 🎯 Common Use Cases

### Development Testing
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
```

### Production Training
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --local-path production_data.csv \
  --subset-size 50000 \
  --output-dir models/v1
```

### Quick Experimentation
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode smoke \
  --subset-size 50 \
  --output-dir experiments/test1
```

### CI/CD Pipeline
```yaml
# GitHub Actions example
- name: Validate Training Pipeline
  run: |
    python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
      --mode smoke \
      --subset-size 100
```

---

## ✨ What's Next?

1. **Try the Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
2. **Run Examples**: `python examples/quickstart_example.py`
3. **Read Documentation**: [docs/training/](docs/training/)
4. **Integrate into Your Workflow**: CI/CD, production, experiments

---

**Need Help?**
- Check [QUICKSTART.md](QUICKSTART.md) for detailed examples
- See [TRAINING_RUN_SUMMARY.md](TRAINING_RUN_SUMMARY.md) for verified results
- Review test examples in [tests/test_quickstart_features.py](tests/test_quickstart_features.py)
