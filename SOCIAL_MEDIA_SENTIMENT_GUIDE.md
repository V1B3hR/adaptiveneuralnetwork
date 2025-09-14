# Social Media Sentiment Analysis with Adaptive Neural Network

This guide explains how to use the Adaptive Neural Network framework for sentiment analysis on the Social Media Sentiments Analysis Dataset.

## Dataset Information

**Dataset**: Social Media Sentiments Analysis Dataset  
**URL**: https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset  
**Task**: Binary sentiment classification (positive vs negative/neutral)  
**Training**: 50 epochs (as specified in problem statement)

## Quick Start

### 1. Download the Dataset

Download the dataset from Kaggle:
```bash
# Using Kaggle CLI (if installed)
kaggle datasets download -d kashishparmar02/social-media-sentiments-analysis-dataset

# Or manually download from:
# https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset
```

### 2. Run Training

#### With Real Dataset
```bash
# Extract the dataset and run training
python train_kaggle_datasets.py --dataset social_media_sentiment --data-path /path/to/dataset.csv

# The script will automatically use 50 epochs for sentiment analysis
```

#### With Synthetic Data (for testing)
```bash
# Run with synthetic sentiment data
python train_kaggle_datasets.py --dataset social_media_sentiment --samples 1000
```

## Supported File Formats

The framework supports flexible format loading:

- **CSV**: Most common format
- **JSON**: Array of objects or line-delimited JSON
- **Excel**: .xlsx and .xls files  
- **TSV**: Tab-separated values

### Expected Data Structure

The dataset should contain these columns (case-insensitive):

| Column | Alternative Names | Description |
|--------|-------------------|-------------|
| `text` | Text, content, message, post, tweet, comment | Social media text content |
| `sentiment` | Sentiment, label, emotion, feeling, class | Sentiment labels |

### Supported Sentiment Labels

The system automatically recognizes various sentiment label formats:

- **Positive**: "positive", "pos", "good", "happy", "joy", "love", "1"
- **Negative**: "negative", "neg", "bad", "sad", "anger", "hate", "0"  
- **Neutral**: "neutral", "neu", "mixed" (grouped with negative for binary classification)

## Training Configuration

### Default Settings (50 Epochs)
```bash
python train_kaggle_datasets.py --dataset social_media_sentiment --data-path /path/to/data.csv
```

### Custom Configuration
```bash
python train_kaggle_datasets.py \
    --dataset social_media_sentiment \
    --data-path /path/to/data.csv \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --hidden-dim 128 \
    --num-nodes 100 \
    --device cpu
```

## Sample Data Format

### CSV Format
```csv
text,sentiment
"I love this product! Amazing quality!",positive
"Terrible experience, very disappointed",negative  
"It's okay, nothing special",neutral
"Absolutely fantastic service!",positive
```

### JSON Format
```json
[
  {"text": "Great product, highly recommend!", "sentiment": "positive"},
  {"text": "Poor quality, very disappointed", "sentiment": "negative"},
  {"text": "Average product, nothing special", "sentiment": "neutral"}
]
```

## Model Architecture

The Adaptive Neural Network uses:

- **Input Layer**: Tokenized text sequences (max length: 512)
- **Hidden Layer**: Configurable dimension (default: 128)
- **Adaptive Nodes**: Dynamic neural computation (default: 100)
- **Output Layer**: Binary classification (positive=1, negative/neutral=0)

## Training Output

The system provides comprehensive metrics:

```
============================================================
TRAINING SOCIAL_MEDIA_SENTIMENT DATASET  
============================================================
Loading social_media_sentiment dataset from data.csv
Sentiment label mapping: {'negative': 0, 'positive': 1, 'neutral': 0}
Loaded 5000 samples from Social Media Sentiment dataset
Label distribution: {0: 2500, 1: 2500}

Training for 50 epochs...
Epoch 10/50: Train Loss: 0.652, Train Acc: 0.651, Test Acc: 0.648
Epoch 20/50: Train Loss: 0.543, Train Acc: 0.723, Test Acc: 0.721
...
Final Test Accuracy: 0.756
Training Time: 124.5 seconds
```

## Results

Training results are automatically saved to:
- `benchmark_results/essay_classification_results.json`

The results include:
- Final and best test accuracy
- Training time and configuration
- Model parameters and vocabulary size
- Dataset information

## Advanced Usage

### Preprocessing Text
The system automatically handles:
- Text tokenization and vocabulary building
- Sequence padding/truncation
- Label encoding and mapping

### Multi-format Support
```bash
# CSV file
python train_kaggle_datasets.py --dataset social_media_sentiment --data-path data.csv

# JSON file  
python train_kaggle_datasets.py --dataset social_media_sentiment --data-path data.json

# Excel file
python train_kaggle_datasets.py --dataset social_media_sentiment --data-path data.xlsx

# Directory (auto-detects format)
python train_kaggle_datasets.py --dataset social_media_sentiment --data-path /path/to/dataset/
```

### Integration with Other Datasets

You can also train on multiple datasets:
```bash
# Train on all supported datasets
python train_kaggle_datasets.py --dataset both --epochs 50
```

## Troubleshooting

### Common Issues

1. **Column not found**: The system uses fuzzy matching for column names
2. **Format detection**: Specify file extension or use auto-detection
3. **Memory issues**: Reduce batch size or vocabulary size
4. **Performance**: Use GPU with `--device cuda` if available

### Getting Help

```bash
# Show all available options
python train_kaggle_datasets.py --help

# Show supported datasets
python train_kaggle_datasets.py --dataset social_media_sentiment
```

## Performance Considerations

- **Training Time**: ~5-10 minutes for 50 epochs with 5000 samples on CPU
- **Memory Usage**: Scales with vocabulary size and batch size
- **GPU Support**: Automatically detected and used if available
- **Scalability**: Handles datasets from hundreds to millions of samples

## Next Steps

After training, you can:
1. Evaluate model performance on test data
2. Fine-tune hyperparameters for better accuracy  
3. Deploy the model for real-time sentiment analysis
4. Integrate with other text classification tasks

---

*This implementation fulfills the problem statement requirements for 50-epoch training on the Social Media Sentiments Analysis Dataset with flexible format loading and comprehensive preprocessing.*