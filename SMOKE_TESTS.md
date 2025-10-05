# Smoke Tests for Quick Validation

This directory contains smoke tests for quick validation of the Adaptive Neural Network text classification features.

## What are Smoke Tests?

Smoke tests are quick, lightweight tests designed to verify that the basic functionality of the system works correctly. They are perfect for:

- ✅ Quick validation during development
- ✅ CI/CD pipeline checks
- ✅ Pre-deployment verification
- ✅ Regression testing

## Quick Start

### Option 1: Using the Python Runner (Recommended)

Run all smoke tests:
```bash
python run_smoke_tests.py
```

Run only CLI smoke test:
```bash
python run_smoke_tests.py --cli-only
```

Run only the test suite:
```bash
python run_smoke_tests.py --suite-only
```

### Option 2: Using the Shell Script

```bash
./run_smoke_tests.sh
```

### Option 3: Direct CLI Usage

Run a smoke test directly:
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
```

With custom settings:
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
    --mode smoke \
    --subset-size 50 \
    --output-dir my_results
```

### Option 4: Programmatic Usage

```python
from adaptiveneuralnetwork.training.scripts.run_bitext_training import run_smoke_test

# Run smoke test
results = run_smoke_test(
    subset_size=100,
    output_dir="outputs"
)

if results['success']:
    print(f"✓ Success! Runtime: {results['runtime_seconds']:.2f}s")
    print(f"✓ Train Accuracy: {results['train_metrics']['train_accuracy']:.4f}")
    print(f"✓ Val Accuracy: {results['eval_metrics']['accuracy']:.4f}")
```

## What Gets Tested

The smoke tests validate:

1. **Smoke Test - Default Settings**: Basic smoke test with synthetic data
2. **Smoke Test - Custom Settings**: Smoke test with custom parameters
3. **Benchmark Mode**: Full evaluation mode
4. **Local CSV Files**: Loading and training with local CSV files
5. **Trained Model Access**: Loading and using trained models
6. **Detailed Metrics**: Accessing comprehensive metrics
7. **Subset Size Configuration**: Different data subset sizes

## Expected Output

After running smoke tests, you'll see:

```
outputs/
├── smoke_test_results.json  # Detailed metrics and configuration
└── smoke_test_model.pkl     # Trained model ready for predictions
```

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
  "model_info": {
    "num_features": 100,
    "num_classes": 2
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

## Performance Expectations

- **Runtime**: ~5-30 seconds for smoke test
- **Runtime**: ~30-120 seconds for benchmark test
- **Memory**: < 500MB
- **Accuracy**: Varies with synthetic data, typically 60-90%

## Dependencies

Required:
```bash
pip install 'adaptiveneuralnetwork[nlp]'
```

This installs:
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `kagglehub` - Kaggle dataset integration (optional)

## Troubleshooting

### Missing Dependencies

If you see `ModuleNotFoundError`:
```bash
pip install -e '.[nlp]'
```

### Kaggle Dataset Issues

If using Kaggle datasets, set credentials:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

Or place `kaggle.json` in `~/.kaggle/`

### Test Failures

Check the error message and ensure:
1. All dependencies are installed
2. Python version is 3.12+
3. No conflicting package versions

## CI/CD Integration

Add to your CI pipeline:

```yaml
# GitHub Actions example
- name: Run Smoke Tests
  run: |
    pip install -e '.[nlp]'
    python run_smoke_tests.py
```

```yaml
# GitLab CI example
smoke_test:
  script:
    - pip install -e '.[nlp]'
    - python run_smoke_tests.py
```

## Advanced Usage

### Custom Dataset

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
    --mode smoke \
    --local-path my_data.csv
```

### Kaggle Dataset

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
    --mode smoke \
    --dataset-name username/dataset-name
```

## References

- See `QUICKSTART.md` for detailed usage guide
- See `tests/test_quickstart_features.py` for test implementation
- See `adaptiveneuralnetwork/training/scripts/run_bitext_training.py` for CLI implementation

## Support

For issues or questions:
- Check the [QUICKSTART.md](QUICKSTART.md) guide
- Review test output for specific error messages
- Open an issue on GitHub
