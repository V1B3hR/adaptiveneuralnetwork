# Human vs AI Generated Essays Classification Benchmark

This implementation provides a complete text classification benchmark for distinguishing between human-written and AI-generated essays using adaptive neural networks.

## Overview

The benchmark demonstrates the adaptive neural network's capabilities on a binary text classification task, specifically designed to identify patterns that differentiate human writing from AI-generated content.

## Features

- **50-epoch training** as specified in requirements
- **Synthetic dataset generation** for testing and development
- **Variable batch size handling** for robust training
- **Comprehensive evaluation metrics** with detailed logging
- **Full integration** with existing adaptive neural network framework

## Quick Start

### Run 50-Epoch Benchmark (Full Specification)

```bash
# Run the complete 50-epoch benchmark with 2000 samples
python run_essay_benchmark.py --synthetic --epochs 50 --samples 2000 --batch-size 32 --learning-rate 0.001
```

### Quick Testing

```bash
# Quick 5-epoch test with smaller dataset
python run_essay_benchmark.py --synthetic --epochs 5 --samples 100
```

### Run Tests

```bash
# Run comprehensive test suite
python test_essay_classification.py
```

## Command Line Options

```bash
python run_essay_benchmark.py [OPTIONS]

Dataset Options:
  --data-path PATH       Path to real Kaggle dataset (not implemented yet)
  --synthetic           Use synthetic data (default for testing)
  --samples N           Number of synthetic samples (default: 2000)

Training Options:
  --epochs N            Number of training epochs (default: 50)
  --batch-size N        Batch size (default: 32)
  --learning-rate F     Learning rate (default: 0.001)

Model Options:
  --hidden-dim N        Hidden dimension size (default: 128)
  --num-nodes N         Number of adaptive nodes (default: 100)
  --device DEVICE       Device (cpu/cuda, default: cpu)

Other Options:
  --verbose             Enable verbose logging
  --help                Show help message
```

## Implementation Details

### Architecture

The text classification benchmark uses:

1. **EssayDataset**: Handles text tokenization and preprocessing
2. **SyntheticEssayDataset**: Generates synthetic human vs AI writing patterns
3. **TextClassificationBenchmark**: Main benchmark orchestration
4. **AdaptiveModel**: The core adaptive neural network from the framework

### Data Processing

- **Tokenization**: Simple word-based tokenization with vocabulary building
- **Sequence Handling**: Fixed-length sequences with padding/truncation
- **Label Encoding**: Binary labels (0=human, 1=AI-generated)

### Model Configuration

- **Input Dimension**: Sequence length (default: 256 tokens)
- **Hidden Dimension**: Configurable (default: 128)
- **Output Dimension**: 2 (binary classification)
- **Adaptive Nodes**: Configurable (default: 100)

## Results Structure

The benchmark outputs comprehensive results including:

```json
{
  "final_test_accuracy": 0.5325,
  "best_test_accuracy": 0.5325,
  "final_train_accuracy": 0.5094,
  "training_time": 544.84,
  "epochs": 50,
  "vocab_size": 5000,
  "train_samples": 1600,
  "test_samples": 400,
  "model_parameters": 49746
}
```

## Sample 50-Epoch Results

With synthetic data (2000 samples, 50 epochs):

- **Final Test Accuracy**: 53.25%
- **Training Time**: ~9 minutes
- **Model Parameters**: 49,746 parameters
- **Vocabulary Size**: 5,000 tokens

## Extending to Real Data

To use with the actual Kaggle "Human vs AI Generated Essays" dataset:

1. Download the dataset from Kaggle
2. Implement the `load_kaggle_dataset()` function in `run_essay_benchmark.py`
3. Parse CSV files and extract text/label pairs
4. Create `EssayDataset` instance with real data

```python
def load_kaggle_dataset(data_path: str) -> EssayDataset:
    """Load real Kaggle dataset."""
    import pandas as pd
    
    # Load CSV file
    df = pd.read_csv(f"{data_path}/dataset.csv")
    
    # Extract texts and labels
    texts = df['text'].tolist()
    labels = df['generated'].astype(int).tolist()  # Assuming 'generated' column
    
    return EssayDataset(texts=texts, labels=labels)
```

## Testing

The implementation includes comprehensive tests covering:

- Dataset creation and functionality
- Model initialization and parameter counting
- Training loop correctness
- Variable batch size handling
- Accuracy improvement over epochs
- Integration testing

Run all tests with: `python test_essay_classification.py`

## Integration with Framework

This benchmark fully integrates with the existing adaptive neural network framework:

- Uses `AdaptiveConfig` for configuration management
- Leverages `AdaptiveModel` for the neural network
- Follows existing patterns for benchmarking and evaluation
- Compatible with the framework's device management and logging

## Performance Considerations

- **Memory Usage**: Scales with vocabulary size and sequence length
- **Training Time**: ~10-15 minutes for 50 epochs with 2000 samples on CPU
- **Scalability**: Supports large datasets with efficient batch processing
- **Device Support**: Works on both CPU and GPU (CUDA)

## Future Enhancements

- Real dataset integration with Kaggle API
- Advanced text preprocessing (subword tokenization, embeddings)
- Model architecture improvements (attention mechanisms)
- Hyperparameter optimization
- Cross-validation evaluation
- Additional evaluation metrics (F1-score, precision, recall)