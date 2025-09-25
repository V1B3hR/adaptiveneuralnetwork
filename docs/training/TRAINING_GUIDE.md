# Comprehensive Training Guide

This unified guide consolidates all training-related documentation for the Adaptive Neural Network system.

## Overview

The Adaptive Neural Network supports multiple training approaches across various datasets and tasks:

- **Text Classification**: Binary and multi-class classification tasks
- **Sequence Labeling**: Token-level prediction (POS tagging, NER)  
- **Sentiment Analysis**: Social media and review sentiment
- **Essay Classification**: Human vs AI-generated content detection
- **Specialized Datasets**: Domain-specific classification tasks

## Quick Start

### Installation
```bash
# Core dependencies
pip install adaptiveneuralnetwork

# For text processing tasks
pip install 'adaptiveneuralnetwork[nlp]'

# For development
pip install 'adaptiveneuralnetwork[dev]'
```

### Basic Training
```bash
# Simple text classification with synthetic data
python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke

# Full benchmark with 100 epochs
python run_essay_benchmark.py --epochs 100 --samples 2000 --synthetic
```

## Supported Datasets

### 1. Text Classification Datasets

#### ANNOMI Motivational Interviewing Dataset
- **URL**: https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing
- **Task**: Binary classification (motivational vs non-motivational responses)
- **Training**: `python train_annomi.py --epochs 100`

#### Mental Health FAQs Dataset  
- **URL**: https://www.kaggle.com/datasets/ragishehab/mental-healthfaqs
- **Task**: Binary classification (anxiety vs depression categorization)
- **Training**: `python train_mental_health.py --epochs 100`

#### Social Media Sentiments Analysis Dataset
- **URL**: https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset
- **Task**: Binary sentiment classification (positive vs negative/neutral)
- **Training**: `python train_kaggle_datasets.py --dataset social_media_sentiment`

#### Human vs AI Generated Essays
- **Task**: Binary classification distinguishing human from AI-generated essays
- **Features**: 50-epoch training, synthetic data generation, comprehensive metrics
- **Training**: `python run_essay_benchmark.py --epochs 50`

### 2. Sequence Labeling Datasets

#### Part-of-Speech Tagging Dataset
- **URL**: https://www.kaggle.com/datasets/ruchi798/part-of-speech-tagging
- **Task**: Token-level sequence labeling for grammatical tagging
- **Models**: BiLSTM (default) and Transformer encoder options
- **Training**: `python train_pos_tagging.py --epochs 20 --auto`

### 3. Specialized Datasets

#### Virtual Reality Driving Simulator Dataset
- **URL**: https://www.kaggle.com/datasets/sasanj/virtual-reality-driving-simulator-dataset
- **Task**: Regression (driving performance prediction)
- **Training**: `python train_new_datasets.py --dataset vr_driving`

#### AUTVI Dataset (Automated Vehicle Inspection)
- **URL**: https://www.kaggle.com/datasets/hassanmojab/autvi  
- **Task**: Binary classification (pass/fail inspection)
- **Training**: `python train_new_datasets.py --dataset autvi`

#### Digakust Dataset (Mensa Saarland University)
- **URL**: https://www.kaggle.com/datasets/resc28/digakust-dataset-mensa-saarland-university
- **Task**: Multi-class classification (acoustic pattern recognition)
- **Training**: `python train_new_datasets.py --dataset digakust`

## Training Scripts

### 1. Bitext Training (Lightweight Baseline)
```bash
# Smoke test with synthetic data (quick validation)
python -m adaptiveneuralnetwork.training.run_bitext_training --mode smoke

# Full benchmark
python -m adaptiveneuralnetwork.training.run_bitext_training \
  --mode benchmark \
  --subset-size 10000 \
  --dataset-name kaggle-user/dataset-name
```

### 2. Essay Classification Benchmark
```bash
# 50-epoch benchmark (specification requirement)
python run_essay_benchmark.py --synthetic --epochs 50 --samples 2000

# Quick testing
python run_essay_benchmark.py --synthetic --epochs 5 --samples 100
```

### 3. Multi-Dataset Training
```bash
# Train on all supported datasets
python train_kaggle_datasets.py --dataset all --epochs 10

# Train specific dataset
python train_kaggle_datasets.py --dataset annomi --epochs 100
```

### 4. Specialized Dataset Training
```bash
# New datasets (VR Driving, AUTVI, Digakust)
python train_new_datasets.py --dataset vr_driving --epochs 15

# POS tagging with adaptive heuristics
python train_pos_tagging.py --epochs 20 --auto
```

## Configuration Options

### Model Parameters
- `--hidden-dim`: Hidden layer dimension (default: 128)
- `--num-nodes`: Number of adaptive nodes (default: 100)
- `--device`: Training device (cpu/cuda)

### Training Parameters  
- `--epochs`: Number of training epochs (default varies by task)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)

### Dataset Parameters
- `--samples`: Number of synthetic samples (default: 2000)
- `--subset-size`: Maximum samples to use from real datasets
- `--data-path`: Path to local dataset files

## Kaggle Integration

### Setup
Set environment variables for automatic dataset download:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Or place credentials in `~/.kaggle/kaggle.json`

### Automatic Downloads
The training scripts automatically download datasets when credentials are available, falling back to synthetic data otherwise.

## Data Formats

### CSV Format (Most Common)
```csv
text,label
"This is a positive example",positive
"This is a negative example",negative
```

### POS Tagging Format
```csv
sentence,word,pos
1,The,DT
1,cat,NN
1,sat,VBD
```

### JSON Format
```json
[
  {"text": "Sample text", "label": "positive"},
  {"text": "Another example", "label": "negative"}
]
```

## Model Architectures

### Text Classification
- **Base**: TF-IDF + Logistic Regression (bitext training)
- **Advanced**: Adaptive Neural Network with embeddings
- **Features**: Configurable hidden dimensions, adaptive nodes

### Sequence Labeling
- **BiLSTM**: Bidirectional LSTM for sequence modeling (default)
- **Transformer**: Self-attention based encoder
- **Features**: Adaptive epochs, dynamic batch sizing

### Specialized Tasks
- **Regression**: Feed-forward networks for continuous outputs
- **Multi-class**: Adaptive neural networks with appropriate output dimensions

## Results and Metrics

### Text Classification Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Per-class performance analysis

### Sequence Labeling Metrics  
- Token-level accuracy
- Entity-level F1 scores
- Per-tag performance analysis

### Output Structure
```json
{
  "final_test_accuracy": 0.8725,
  "best_test_accuracy": 0.8750,
  "training_time": 245.67,
  "epochs": 50,
  "vocab_size": 5000,
  "model_parameters": 49746
}
```

## Performance Guidelines

### Training Times (Synthetic Data)
- **Text Classification**: ~30-60 seconds (1000 samples, 10 epochs)
- **POS Tagging**: ~45 seconds (1000 sentences, 10 epochs)
- **Essay Classification**: ~10-15 minutes (2000 samples, 50 epochs)

### Resource Requirements
- **Memory**: 2-4 GB per dataset depending on size
- **CPU**: Scales with epoch count and dataset size
- **GPU**: Use `--device cuda` for acceleration when available

### Scaling Guidelines
- **Small datasets** (< 1K): Use full data
- **Medium datasets** (1K-100K): Consider `--subset-size`
- **Large datasets** (> 100K): Use sampling or batch processing

## Error Handling

### Common Issues

1. **Missing Dependencies**
   ```
   ERROR: Required dependencies missing: ['pandas', 'sklearn']
   ```
   **Solution**: `pip install 'adaptiveneuralnetwork[nlp]'`

2. **Kaggle Credentials Not Found**
   ```
   WARNING: Kaggle credentials not found, using synthetic data
   ```
   **Solution**: Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Use `--subset-size` or `--batch-size` parameters

### Debug Mode
Enable verbose logging for troubleshooting:
```bash
python train_script.py --verbose
```

## Integration with Adaptive Features

The training system integrates with the adaptive neural network's core features:

- **Configuration-driven**: Uses `AdaptiveConfig` for parameter management
- **State-modulated**: Training behavior adapts based on system state
- **Reproducible**: Deterministic seeds for consistent results
- **Scalable**: Supports various deployment scenarios

## Advanced Usage

### Custom Preprocessing
```python
from adaptiveneuralnetwork.training.text_baseline import TextClassificationBaseline

def custom_preprocess(text):
    return text.lower().strip()

texts = [custom_preprocess(t) for t in raw_texts]
baseline = TextClassificationBaseline()
baseline.fit(texts, labels)
```

### Model Persistence
```python
# Save trained model
baseline.save_model("my_model.pkl")

# Load and use later
new_baseline = TextClassificationBaseline()
new_baseline.load_model("my_model.pkl")
predictions = new_baseline.predict(["New text"])
```

### Feature Analysis
```python
# Get important features
feature_importance = baseline.get_feature_importance(top_k=20)
for class_name, features in feature_importance.items():
    print(f"Top features for {class_name}:")
    for feature, weight in features[:5]:
        print(f"  {feature}: {weight:.4f}")
```

## Future Enhancements

### Planned Features
- Distributed training across multiple devices
- Hyperparameter optimization workflows
- Real-time model serving capabilities
- Advanced multimodal fusion
- Federated learning support

### Extension Points
- Custom dataset loaders in `adaptiveneuralnetwork/data/`
- New training scripts following established patterns
- Enhanced visualization and monitoring tools
- Integration with MLOps platforms

---

*This consolidated guide replaces the following individual documents:*
- `ESSAY_CLASSIFICATION_README.md`
- `KAGGLE_DATASETS_README.md` 
- `MULTI_DATASET_TRAINING_GUIDE.md`
- `docs/bitext_training.md`
- `POS_TAGGING_GUIDE.md`
- `SOCIAL_MEDIA_SENTIMENT_GUIDE.md`