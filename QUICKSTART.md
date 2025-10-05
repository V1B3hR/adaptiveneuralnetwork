# Quick Start Guide

Get started with Adaptive Neural Network training in 5 minutes!

## Table of Contents
- [Installation](#installation)
- [Quick Validation (Smoke Test)](#quick-validation-smoke-test)
- [Full Evaluation (Benchmark)](#full-evaluation-benchmark)
- [Using Your Own Data](#using-your-own-data)
- [Accessing Results](#accessing-results)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Basic Installation
```bash
pip install adaptiveneuralnetwork
```

### With NLP Support (Required for Text Classification)
```bash
pip install 'adaptiveneuralnetwork[nlp]'
```

This installs:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `kagglehub` - Kaggle dataset integration

### Verify Installation
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --check-deps
```

---

## Quick Validation (Smoke Test)

**Perfect for:** Quick validation, CI/CD pipelines, development testing

### 1. Basic Smoke Test (Default Settings)
Run a fast test with 100 synthetic samples:

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
```

**Output:**
- ✓ Completes in ~5-30 seconds
- ✓ Uses 80 train / 20 validation samples
- ✓ Creates `outputs/smoke_test_results.json`
- ✓ Creates `outputs/smoke_test_model.pkl`

### 2. Custom Smoke Test
Customize sample size and output location:

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode smoke \
  --subset-size 50 \
  --output-dir my_results
```

**What you get:**
```
my_results/
├── smoke_test_results.json  # Metrics and configuration
└── smoke_test_model.pkl     # Trained model
```

---

## Full Evaluation (Benchmark)

**Perfect for:** Model evaluation, performance testing, production training

### 1. Basic Benchmark
Run with synthetic data:

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --subset-size 1000
```

**Output:**
- ✓ Uses 800 train / 200 validation samples
- ✓ Creates `outputs/benchmark_results.json`
- ✓ Creates `outputs/benchmark_model.pkl`
- ✓ Detailed metrics and feature importance

### 2. Large-Scale Benchmark
For bigger datasets:

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --subset-size 10000 \
  --output-dir production_models
```

---

## Using Your Own Data

### Option 1: Local CSV Files

**CSV Format:**
```csv
text,label
"This is positive text",1
"This is negative text",0
```

**Command:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --local-path data/my_dataset.csv \
  --output-dir results
```

**Supported column names:**
- Text: `text`, `sentence`, `review`, `comment`
- Label: `label`, `target`, `sentiment`, `class`

### Option 2: Kaggle Datasets

**Setup Kaggle credentials:**
```bash
# Option A: Environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Option B: Config file (~/.kaggle/kaggle.json)
{
  "username": "your_username",
  "key": "your_api_key"
}
```

**Run with Kaggle dataset:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --dataset-name username/dataset-name \
  --subset-size 5000 \
  --output-dir kaggle_results
```

### Option 3: Automatic Fallback
Without specifying data, the system automatically generates synthetic data for testing:

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
```
✓ Perfect for testing the pipeline without real data!

---

## Accessing Results

### 1. Trained Models

**Load a trained model:**
```python
from adaptiveneuralnetwork.training.models.text_baseline import TextClassificationBaseline

# Load model
model = TextClassificationBaseline()
model.load_model("outputs/smoke_test_model.pkl")

# Make predictions
texts = ["Sample text to classify", "Another example"]
predictions = model.predict(texts)
probabilities = model.predict_proba(texts)

print(f"Predictions: {predictions}")
print(f"Probabilities: {probabilities}")
```

### 2. Detailed Metrics

**View results JSON:**
```bash
cat outputs/smoke_test_results.json
```

**JSON Structure:**
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
  "model_info": {
    "num_features": 100,
    "num_classes": 2,
    "class_names": [0, 1]
  },
  "train_metrics": {
    "train_accuracy": 0.8750,
    "val_accuracy": 0.7500
  },
  "eval_metrics": {
    "accuracy": 0.7500,
    "precision": 0.8333,
    "recall": 0.7500,
    "f1_score": 0.7333,
    "confusion_matrix": [[10, 0], [5, 5]]
  },
  "feature_importance": {
    "0": [["feature1", 0.86], ["feature2", 0.62]],
    "1": [["feature1", -0.86], ["feature2", -0.62]]
  }
}
```

### 3. Programmatic Access

**Load and analyze results:**
```python
import json

# Load results
with open("outputs/smoke_test_results.json") as f:
    results = json.load(f)

# Access metrics
print(f"Accuracy: {results['eval_metrics']['accuracy']}")
print(f"F1 Score: {results['eval_metrics']['f1_score']}")
print(f"Training time: {results['runtime_seconds']:.2f}s")

# Get feature importance
for class_name, features in results['feature_importance'].items():
    print(f"\nTop features for class {class_name}:")
    for feature, weight in features[:5]:
        print(f"  {feature}: {weight:.4f}")
```

---

## Command Reference

### All Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Training mode (`smoke` or `benchmark`) | `smoke` |
| `--dataset-name` | Kaggle dataset name (e.g., `user/dataset`) | None (synthetic) |
| `--local-path` | Path to local CSV file | None (synthetic) |
| `--subset-size` | Maximum samples to use | 100 (smoke), unlimited (benchmark) |
| `--output-dir` | Output directory for results | `outputs` |
| `--epochs` | Training epochs (future use) | 1 |
| `--verbose` | Enable verbose logging | False |
| `--check-deps` | Check dependencies and exit | False |

### Example Commands

**Minimal smoke test:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
```

**Custom smoke test:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode smoke \
  --subset-size 50 \
  --output-dir test_results
```

**Benchmark with local data:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --local-path data.csv \
  --subset-size 5000 \
  --output-dir benchmark_results
```

**Benchmark with Kaggle:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --dataset-name username/sentiment-dataset \
  --output-dir kaggle_benchmark
```

**With verbose logging:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --verbose
```

---

## Troubleshooting

### Issue: Missing Dependencies
```
ERROR: Required dependencies missing: ['pandas', 'sklearn']
```
**Solution:**
```bash
pip install 'adaptiveneuralnetwork[nlp]'
```

### Issue: Kaggle Credentials Not Found
```
WARNING: Kaggle credentials not found
```
**Solution:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```
Or create `~/.kaggle/kaggle.json`

### Issue: CSV Column Not Found
```
ERROR: Text column 'text' not found in CSV
```
**Solution:**
Ensure your CSV has columns named `text` and `label`, or rename them:
```python
import pandas as pd
df = pd.read_csv("data.csv")
df = df.rename(columns={"review": "text", "sentiment": "label"})
df.to_csv("data_fixed.csv", index=False)
```

### Issue: Memory Error
```
MemoryError: Unable to allocate array
```
**Solution:**
Reduce dataset size:
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --subset-size 1000
```

### Issue: Permission Denied (Output Directory)
```
PermissionError: [Errno 13] Permission denied: 'outputs'
```
**Solution:**
Use a different output directory:
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --output-dir ~/my_results
```

---

## Performance Expectations

### Smoke Mode
- **Data:** 50-100 samples
- **Runtime:** 5-30 seconds
- **Memory:** < 100 MB
- **Purpose:** Quick validation

### Benchmark Mode
- **Data:** 1K-100K samples
- **Runtime:** 1-10 minutes
- **Memory:** 100 MB - 2 GB
- **Purpose:** Full evaluation

### Scaling Guidelines
- **Small datasets** (< 1K): Use full data
- **Medium datasets** (1K-100K): Use `--subset-size`
- **Large datasets** (> 100K): Use smaller subset (e.g., 10K)

---

## Next Steps

1. ✅ **Run your first smoke test** (see above)
2. ✅ **Try with your own CSV data**
3. ✅ **Run a benchmark for full evaluation**
4. 📚 **Read detailed documentation:**
   - [Training Guide](docs/training/TRAINING_GUIDE.md)
   - [Bitext Training Details](docs/training/bitext_training.md)
5. 🚀 **Integrate into your workflow:**
   - Add to CI/CD pipelines
   - Automate training runs
   - Export models for production

---

## Support

- **Documentation:** See [docs/training/](docs/training/)
- **Examples:** See [TRAINING_RUN_SUMMARY.md](TRAINING_RUN_SUMMARY.md)
- **Issues:** Report at GitHub Issues

---

**Happy Training! 🚀**
