# Part-of-Speech Tagging with Adaptive Neural Networks

This guide covers the implementation of Part-of-Speech (POS) tagging using adaptive neural networks with dynamic epoch and sampling heuristics.

## Overview

The POS tagging module provides a complete pipeline for sequence labeling tasks, including:

- **Dataset Loading**: Flexible CSV loader with column auto-detection
- **Model Architectures**: BiLSTM (default) and Transformer encoder options  
- **Adaptive Heuristics**: Dynamic epoch and batch size selection based on data characteristics
- **Comprehensive Metrics**: Token accuracy, macro/micro F1, per-tag F1 scores
- **Training Pipeline**: Full training with early stopping, checkpointing, and evaluation

## Dataset

### Supported Dataset
- **Name**: Part-of-Speech Tagging Dataset
- **URL**: https://www.kaggle.com/datasets/ruchi798/part-of-speech-tagging
- **Format**: CSV with columns for sentence ID, word/token, and POS tag
- **Task**: Token-level sequence labeling for grammatical tagging

### Expected Format
The dataset loader automatically detects columns with names like:
- **Sentence**: `sentence`, `sentence_id`, `sent_id`, `id`
- **Token**: `word`, `token`, `text`, `tokens`  
- **POS Tag**: `pos`, `tag`, `pos_tag`, `label`

Example CSV structure:
```csv
sentence,word,pos
1,The,DT
1,cat,NN
1,sat,VBD
1,on,IN
1,the,DT
1,mat,NN
2,Dogs,NNS
2,bark,VBP
2,loudly,RB
```

## Quick Start

### 1. Download Dataset
```bash
# Option 1: Manual download from Kaggle
# Download and extract to a local directory

# Option 2: Using Kaggle CLI (if available)
kaggle datasets download -d ruchi798/part-of-speech-tagging
unzip part-of-speech-tagging.zip
```

### 2. Basic Training
```bash
# Train with automatic heuristics
python train_pos_tagging.py --data-path /path/to/dataset --auto

# Train with custom parameters
python train_pos_tagging.py --data-path /path/to/dataset --epochs 20 --batch-size 32

# Small-scale testing with synthetic data
python train_pos_tagging.py --synthetic --epochs 2 --max-sentences 50
```

### 3. Model Options
```bash
# BiLSTM model (default)
python train_pos_tagging.py --data-path /path/to/dataset --model bilstm

# Transformer model
python train_pos_tagging.py --data-path /path/to/dataset --model transformer --num-heads 8
```

## Dynamic Heuristics

The system automatically selects training parameters based on dataset characteristics:

### Epoch Selection
| Dataset Size (Sentences) | Epochs | Reasoning |
|--------------------------|--------|-----------|
| ≤ 5,000 | 40 | Small datasets need more iterations |
| 5,001 - 15,000 | 30 | Medium datasets with moderate training |
| 15,001 - 40,000 | 20 | Large datasets converge faster |
| > 40,000 | 12 | Very large datasets, prevent overfitting |

### Batch Size Selection
| Token Count | Batch Size | Reasoning |
|-------------|------------|-----------|
| ≤ 800,000 | 32 | Standard batch size |
| > 800,000 | 16 | Reduced for memory efficiency |

### Gradient Accumulation
- **Threshold**: 8,000 effective tokens per batch
- **Action**: Split into accumulation steps to fit memory
- **Formula**: `steps = max(1, effective_tokens / 4000)`

## Command Line Arguments

### Data Arguments
- `--data-path`: Path to POS tagging dataset (CSV or directory)
- `--max-sentences`: Maximum sentences to load (for sampling/testing)
- `--min-token-length`: Minimum token length filter (default: 1)
- `--filter-punctuation`: Filter single-character punctuation tokens

### Training Arguments
- `--epochs`: Number of epochs (auto-determined if not specified)
- `--batch-size`: Batch size (auto-determined if not specified)
- `--learning-rate`: Learning rate (default: 0.001)
- `--gradient-accumulation-steps`: Gradient accumulation (default: 1)
- `--early-stop-patience`: Early stopping patience (default: 5)
- `--min-improve`: Minimum improvement threshold (default: 0.001)

### Model Arguments
- `--model`: Model type - `bilstm` or `transformer` (default: bilstm)
- `--embedding-dim`: Embedding dimension (default: 128)
- `--hidden-dim`: Hidden dimension (default: 256)
- `--num-layers`: Number of layers (default: 2)
- `--dropout`: Dropout rate (default: 0.3)
- `--vocab-size`: Vocabulary size (default: 10000)
- `--max-len`: Maximum sequence length (default: 512)

### Other Arguments  
- `--seed`: Random seed for reproducibility (default: 42)
- `--device`: Device selection - `auto`, `cpu`, or `cuda`
- `--output-dir`: Output directory (default: ./pos_tagging_output)
- `--verbose`: Enable verbose logging
- `--auto`: Use automatic heuristics (implied when epochs/batch-size not specified)
- `--synthetic`: Use synthetic data for testing

## Example Training Runs

### Small Dataset Example
```bash
python train_pos_tagging.py --data-path /path/to/small_dataset --max-sentences 5000
# Expected: 40 epochs, batch_size 32, ~10-15 minutes training
```

### Large Dataset Example  
```bash
python train_pos_tagging.py --data-path /path/to/large_dataset
# Expected: 12-20 epochs, batch_size 16-32, automatic memory optimization
```

### Custom Configuration
```bash
python train_pos_tagging.py \
    --data-path /path/to/dataset \
    --model transformer \
    --embedding-dim 256 \
    --hidden-dim 512 \
    --num-heads 8 \
    --epochs 25 \
    --batch-size 16 \
    --learning-rate 0.0005
```

### Development/Testing
```bash
# Quick smoke test with synthetic data
python train_pos_tagging.py --synthetic --epochs 2 --max-sentences 100

# Small real data test
python train_pos_tagging.py --data-path /path/to/dataset --max-sentences 1000 --epochs 5
```

## Expected Performance

### Metrics
The system tracks comprehensive metrics:
- **Token Accuracy**: Percentage of correctly predicted tokens
- **Macro F1**: Average F1 score across all POS tags
- **Micro F1**: Overall F1 score weighted by tag frequency
- **Per-tag F1**: Individual F1 scores for each POS tag

### Typical Results
| Dataset Size | Model | Token Accuracy | Macro F1 | Training Time |
|-------------|-------|----------------|----------|---------------|
| 5K sentences | BiLSTM | 85-92% | 0.75-0.85 | 5-10 min |
| 15K sentences | BiLSTM | 88-94% | 0.80-0.88 | 15-25 min |
| 50K sentences | BiLSTM | 90-95% | 0.85-0.92 | 45-90 min |
| 15K sentences | Transformer | 87-93% | 0.78-0.86 | 20-35 min |

*Results vary based on dataset quality, tag distribution, and hardware*

## Output Files

Training produces several output files in the specified output directory:

### Model Files
- `best_model.pt`: Best model checkpoint (highest validation F1)
- `final_model.pt`: Final model after training completion

### Metrics Files
- `metrics_pos_tagging.json`: Training history and final metrics
- `tag_report.json`: Per-tag F1 scores and tag vocabulary
- `training_config.json`: Complete configuration and heuristics used

### Example Metrics Output
```json
{
  "training_history": [
    {
      "epoch": 1,
      "train_loss": 2.1543,
      "val_loss": 1.9876,
      "token_accuracy": 0.7234,
      "macro_f1": 0.6891,
      "learning_rate": 0.001
    }
  ],
  "best_validation": {
    "token_accuracy": 0.8934,
    "macro_f1": 0.8456,
    "per_tag_f1": {
      "NN": 0.92,
      "VB": 0.87,
      "DT": 0.95
    }
  }
}
```

## Tips and Best Practices

### Data Preprocessing
- **Token Length**: Use `--min-token-length 2` to filter very short tokens
- **Punctuation**: Use `--filter-punctuation` if punctuation tags are noisy
- **Sampling**: Use `--max-sentences` for initial experiments and debugging

### Model Selection
- **BiLSTM**: Faster training, good for most POS tagging tasks
- **Transformer**: Better context modeling, slower but potentially higher accuracy
- **Memory**: Reduce `--batch-size` or increase `--gradient-accumulation-steps` for large models

### Training Optimization
- **Early Stopping**: Default patience of 5 epochs works well
- **Learning Rate**: 0.001 is a good starting point, try 0.0005 for Transformers
- **Reproducibility**: Always set `--seed` for consistent results

### Handling Large Datasets
- **Batch Size**: System automatically reduces batch size for >800K tokens
- **Gradient Accumulation**: Automatically enabled for memory efficiency
- **Sequence Length**: Truncate very long sentences with `--max-len`

### Debugging and Development
- **Synthetic Data**: Use `--synthetic` for quick pipeline testing
- **Verbose Mode**: Enable `--verbose` for detailed training logs
- **Small Samples**: Use `--max-sentences 1000` for rapid iteration

## Troubleshooting

### Common Issues

#### Memory Errors
- Reduce `--batch-size` (try 8 or 4)
- Increase `--gradient-accumulation-steps`
- Reduce `--max-len` for very long sequences
- Use `--model bilstm` instead of transformer

#### Poor Performance
- Check data quality and format
- Ensure balanced tag distribution
- Try different learning rates (0.0005, 0.002)
- Increase `--epochs` for small datasets
- Use `--model transformer` for better context

#### Training Too Slow
- Reduce `--epochs` or `--max-sentences`
- Increase `--batch-size` if memory allows
- Use `--model bilstm` for faster training
- Enable GPU with `--device cuda`

#### Data Loading Errors
- Verify CSV format and column names
- Check for missing or malformed data
- Try different encoding if characters appear corrupted
- Use `--verbose` to see column detection

### Dependencies

#### Required
- torch >= 2.0.0
- pandas >= 2.0.0  
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

#### Optional  
- seqeval >= 1.2.2 (for advanced sequence labeling metrics)
- tqdm (for progress bars)

## Integration with Existing Codebase

The POS tagging module integrates seamlessly with the existing adaptive neural network framework:

### Using with train_kaggle_datasets.py
```bash
python train_kaggle_datasets.py --dataset pos_tagging --data-path /path/to/dataset
```

### Programmatic Usage
```python
from adaptiveneuralnetwork.data.kaggle_datasets import load_pos_tagging_dataset
from adaptiveneuralnetwork.models.pos_tagger import POSTagger, POSTaggerConfig

# Load dataset
data = load_pos_tagging_dataset('/path/to/dataset')
datasets = data['datasets']

# Create model  
config = POSTaggerConfig(vocab_size=len(data['vocab']), num_tags=len(data['tag_vocab']))
model = POSTagger(config)

# Training loop (see train_pos_tagging.py for complete example)
```

## License and Contributing

This implementation follows the existing project license and contribution guidelines. Please refer to the main repository documentation for details on contributing improvements or reporting issues.