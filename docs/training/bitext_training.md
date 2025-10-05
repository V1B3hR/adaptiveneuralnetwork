# Bitext Training with Adaptive Neural Network

This document describes how to use the bitext training capabilities of the Adaptive Neural Network package for text classification tasks.

## Overview

The bitext training system provides a lightweight baseline for text classification that demonstrates state-modulated behavior without requiring heavy GPU/transformer dependencies. It uses scikit-learn's TF-IDF vectorization with logistic regression for fast, deterministic results.

## Features

- **Kaggle Integration**: Automatic dataset download via kagglehub
- **Local CSV Fallback**: Works with local CSV files when Kaggle unavailable
- **Smoke Testing**: Quick validation with minimal data
- **Benchmark Mode**: Full evaluation with detailed metrics
- **Synthetic Data**: Generates test data when no real data available
- **Graceful Degradation**: Clear warnings when dependencies missing

## Installation

### Core Dependencies (Required)

```bash
pip install adaptiveneuralnetwork
```

### NLP Dependencies (Required for bitext training)

```bash
pip install 'adaptiveneuralnetwork[nlp]'
```

This installs:
- `pandas` - Data manipulation
- `scikit-learn` - TF-IDF and classification
- `tqdm` - Progress bars
- `kagglehub` - Kaggle dataset downloads

## Quick Start

### 1. Check Dependencies

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --check-deps
```

### 2. Run Smoke Test

```bash
# With synthetic data
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke

# With local CSV file
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke --local-path data.csv
```

### 3. Run Benchmark

```bash
# Full benchmark with synthetic data
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode benchmark --subset-size 5000

# With Kaggle dataset (requires credentials)
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode benchmark --dataset-name username/dataset-name
```

## Usage

### Command Line Interface

The main entry point is `run_bitext_training.py` which provides two modes:

#### Smoke Mode
- **Purpose**: Quick validation and testing
- **Data**: Small subset (default 100 samples) 
- **Runtime**: < 2 minutes
- **Use Case**: CI/CD, development testing

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode smoke \
  --subset-size 50 \
  --output-dir outputs
```

#### Benchmark Mode
- **Purpose**: Full evaluation and model training
- **Data**: Full dataset or specified subset
- **Runtime**: Varies by dataset size
- **Use Case**: Model evaluation, performance testing

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode benchmark \
  --subset-size 10000 \
  --epochs 1 \
  --dataset-name kaggle-user/dataset-name \
  --output-dir results
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Training mode (`smoke` or `benchmark`) | `smoke` |
| `--dataset-name` | Kaggle dataset name | None |
| `--local-path` | Path to local CSV file | None |
| `--subset-size` | Maximum samples to use | 100 (smoke), unlimited (benchmark) |
| `--epochs` | Training epochs (future use) | 1 |
| `--output-dir` | Output directory | `outputs` |
| `--verbose` | Enable verbose logging | False |
| `--check-deps` | Check dependencies and exit | False |

### Data Sources

#### 1. Kaggle Datasets

**Requirements:**
- Set environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`
- Or place `kaggle.json` in `~/.kaggle/`

**Example:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --dataset-name username/sentiment-analysis-dataset
```

#### 2. Local CSV Files

**Requirements:**
- CSV file with text and label columns
- Columns can be named: `text`/`label`, `sentence`/`target`, etc.

**Example CSV format:**
```csv
text,label
"This is a positive example",positive
"This is a negative example",negative
```

**Usage:**
```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --local-path /path/to/your/data.csv
```

#### 3. Synthetic Data

When no data source is specified, the system generates synthetic text data for testing:

```python
# Synthetic data generation
from adaptiveneuralnetwork.training.bitext_dataset import create_synthetic_bitext_data

train_df, val_df = create_synthetic_bitext_data(
    num_samples=1000,
    num_classes=2,
    random_seed=42
)
```

## Programmatic Usage

### Loading Datasets

```python
from adaptiveneuralnetwork.training.bitext_dataset import BitextDatasetLoader

# Create loader
loader = BitextDatasetLoader(
    dataset_name="username/dataset-name",  # Kaggle dataset
    local_path="data.csv",                 # Local fallback
    sampling_fraction=0.1,                 # Use 10% of data
    normalize_text=True,                   # Apply text normalization
    random_seed=42                         # Reproducibility
)

# Load data
train_df, val_df = loader.load_dataset(val_split=0.2)
```

### Training Models

```python
from adaptiveneuralnetwork.training.text_baseline import TextClassificationBaseline

# Create baseline model
baseline = TextClassificationBaseline(
    max_features=10000,    # TF-IDF features
    ngram_range=(1, 2),    # Unigrams and bigrams
    C=1.0,                 # Regularization
    random_state=42        # Reproducibility
)

# Train model
metrics = baseline.fit(
    texts=train_df['text'].tolist(),
    labels=train_df['label'].tolist(),
    validation_texts=val_df['text'].tolist(),
    validation_labels=val_df['label'].tolist()
)

# Make predictions
predictions = baseline.predict(["Sample text to classify"])
probabilities = baseline.predict_proba(["Sample text to classify"])
```

## GitHub Actions Integration

### Manual Workflow Dispatch

Navigate to Actions → Bitext Training Workflow → Run workflow

**Inputs:**
- **Mode**: `smoke` or `benchmark`
- **Subset Size**: Maximum samples (e.g., `1000`)
- **Epochs**: Number of epochs (currently `1`)
- **Dataset Name**: Kaggle dataset (optional)

### Scheduled Runs

The workflow runs automatically daily at 2 AM UTC in smoke mode for continuous validation.

### Secrets Configuration

To use Kaggle datasets in GitHub Actions, configure repository secrets:

1. Go to repository Settings → Secrets and variables → Actions
2. Add secrets:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: Your Kaggle API key

**Without secrets**: Workflow runs with synthetic data and exits cleanly with warning.

### Example Workflow Usage

```yaml
# Manual trigger with custom parameters
- name: Run bitext training
  env:
    KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
    KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
  run: |
    python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
      --mode benchmark \
      --subset-size 1000 \
      --dataset-name "username/dataset" \
      --output-dir outputs
```

## Output Files

### Smoke Mode
- `smoke_test_results.json` - Test results and metrics
- `smoke_test_model.pkl` - Trained model (serialized)

### Benchmark Mode
- `benchmark_results.json` - Detailed results and metrics
- `benchmark_model.pkl` - Trained model (serialized)
- `confusion_matrix.png` - Confusion matrix plot (if matplotlib available)

### Results JSON Structure

```json
{
  "mode": "smoke",
  "success": true,
  "runtime_seconds": 15.42,
  "dataset_info": {
    "train_samples": 80,
    "val_samples": 20,
    "data_source": "synthetic"
  },
  "model_info": {
    "num_features": 234,
    "num_classes": 2,
    "class_names": [0, 1]
  },
  "train_metrics": {
    "train_accuracy": 0.9875,
    "val_accuracy": 0.85
  },
  "eval_metrics": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1_score": 0.85
  }
}
```

## Configuration Integration

The bitext training system integrates with the adaptive neural network configuration system:

```python
from adaptiveneuralnetwork.config import AdaptiveNeuralNetworkConfig

# Configure adaptive behavior
config = AdaptiveNeuralNetworkConfig()
config.trend_analysis.window = 10
config.proactive_interventions.anxiety_enabled = True

# Use with your adaptive network
from core.alive_node import AliveLoopNode

node = AliveLoopNode(
    position=[0.0, 0.0],
    velocity=[0.0, 0.0], 
    config=config
)
```

## Performance Expectations

### Smoke Mode
- **Data**: 50-100 samples
- **Runtime**: 5-30 seconds
- **Memory**: < 100 MB
- **Purpose**: Quick validation

### Benchmark Mode
- **Data**: 1K-100K samples
- **Runtime**: 1-10 minutes
- **Memory**: 100 MB - 2 GB
- **Purpose**: Full evaluation

### Scaling Guidelines
- **Small datasets** (< 1K): Use full data
- **Medium datasets** (1K-100K): Consider subset_size
- **Large datasets** (> 100K): Use sampling_fraction < 0.1

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```
   ERROR: Required dependencies missing: ['pandas', 'sklearn']
   ```
   **Solution**: Install NLP dependencies: `pip install 'adaptiveneuralnetwork[nlp]'`

2. **Kaggle Credentials Not Found**
   ```
   WARNING: Kaggle credentials not found
   ```
   **Solution**: Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables

3. **CSV Column Not Found**
   ```
   ERROR: Text column 'text' not found in CSV
   ```
   **Solution**: Specify correct column names or rename CSV columns

4. **Memory Issues with Large Datasets**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Use `--subset-size` to limit data size

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
  --mode smoke \
  --verbose
```

### Validation

The system includes built-in validation:
- JSON result validation
- Model serialization verification  
- Output file existence checks
- Dependency availability checks

## Advanced Usage

### Custom Text Preprocessing

```python
from adaptiveneuralnetwork.training.text_baseline import TextClassificationBaseline

# Custom preprocessing
def preprocess_text(text):
    # Your custom preprocessing
    return text.lower().strip()

# Apply before training
texts = [preprocess_text(text) for text in raw_texts]

baseline = TextClassificationBaseline()
baseline.fit(texts, labels)
```

### Feature Analysis

```python
# Get important features for each class
feature_importance = baseline.get_feature_importance(top_k=20)

for class_name, features in feature_importance.items():
    print(f"Top features for {class_name}:")
    for feature, weight in features[:5]:
        print(f"  {feature}: {weight:.4f}")
```

### Model Persistence

```python
# Save trained model
baseline.save_model("my_model.pkl")

# Load model later
new_baseline = TextClassificationBaseline()
new_baseline.load_model("my_model.pkl")

# Make predictions
predictions = new_baseline.predict(["New text to classify"])
```

## Integration with Adaptive Features

The bitext training system is designed to demonstrate state-modulated behavior:

1. **Configuration-driven parameters** affect model behavior
2. **Reproducible results** with deterministic seeds
3. **Lightweight baseline** suitable for continuous validation
4. **Scalable architecture** for future adaptive enhancements

This provides a foundation for more advanced adaptive text processing capabilities while maintaining simplicity and reliability for immediate use.