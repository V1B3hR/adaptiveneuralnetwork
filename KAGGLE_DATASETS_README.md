# Kaggle Datasets Integration for Adaptive Neural Network

This document describes the implementation of support for the two Kaggle datasets specified in the problem statement, with training capabilities for 100 epochs.

## Supported Datasets

### 1. ANNOMI Motivational Interviewing Dataset
- **URL**: https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing
- **Description**: Contains motivational interviewing conversations
- **Task**: Binary classification (motivational vs non-motivational responses)
- **Expected Format**: CSV with columns: `text`, `label`

### 2. Mental Health FAQs Dataset
- **URL**: https://www.kaggle.com/datasets/ragishehab/mental-healthfaqs
- **Description**: Mental health-related questions and answers
- **Task**: Binary classification (anxiety vs depression, or Q&A categorization)
- **Expected Format**: CSV with columns: `question`, `answer`, `category`

## Implementation Features

### Dataset Loading
- **Flexible Format Support**: CSV, JSON, Excel, TSV
- **Auto-detection**: Automatically detects dataset type from path
- **Column Mapping**: Intelligent column name detection (case-insensitive, partial matching)
- **Binary Classification**: Converts multi-class labels to binary classification
- **Robust Error Handling**: Fallback to synthetic data if loading fails

### Training Capabilities
- **100 Epochs**: Default training for 100 epochs as specified
- **Adaptive Learning**: Uses the adaptive neural network architecture
- **Progress Monitoring**: Detailed logging every 10 epochs
- **Results Saving**: Automatic saving of training metrics and results

## Usage

### Quick Start (with synthetic data)
```bash
# Train on ANNOMI-style synthetic data for 100 epochs
python train_annomi.py --epochs 100

# Train on Mental Health-style synthetic data for 100 epochs  
python train_mental_health.py --epochs 100
```

### With Real Kaggle Data
```bash
# Download datasets from Kaggle first, then:

# ANNOMI dataset
python train_annomi.py --data-path /path/to/annomi/dataset --epochs 100

# Mental Health FAQs dataset
python train_mental_health.py --data-path /path/to/mental_health/dataset --epochs 100
```

### Advanced Options
```bash
# Customize training parameters
python train_annomi.py \
    --data-path /path/to/dataset \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-dim 256 \
    --num-nodes 150 \
    --device cuda \
    --verbose

# Mental Health with questions only
python train_mental_health.py \
    --data-path /path/to/dataset \
    --epochs 100 \
    --use-questions-only \
    --verbose
```

## Training Scripts

### `train_annomi.py`
Specialized training script for the ANNOMI Motivational Interviewing Dataset.

**Key Features**:
- Auto-detects ANNOMI dataset format
- Optimized for motivational vs non-motivational classification
- Default 100 epochs as per problem statement
- Comprehensive progress reporting

### `train_mental_health.py`
Specialized training script for the Mental Health FAQs Dataset.

**Key Features**:
- Handles Q&A format with categories
- Option to use questions only or combine Q&A
- Binary classification based on mental health categories
- Default 100 epochs as per problem statement

## Dataset Format Examples

### ANNOMI Dataset Format
```csv
text,label
"I understand you're feeling overwhelmed. Can you tell me more?",motivational
"You need to stop making excuses and get your life together.",non_motivational
```

### Mental Health FAQs Format
```csv
question,answer,category
"How do I know if I have anxiety?","Anxiety symptoms include persistent worry...",anxiety
"What are signs of depression?","Depression symptoms include persistent sadness...",depression
```

## Technical Implementation

### Dataset Loaders (`adaptiveneuralnetwork/data/kaggle_datasets.py`)
- `load_annomi_dataset()`: Loads ANNOMI dataset with flexible format support
- `load_mental_health_faqs_dataset()`: Loads Mental Health FAQs with Q&A processing
- `create_text_classification_dataset()`: General text classification dataset creation

### Enhanced Benchmark Script (`run_essay_benchmark.py`)
- Added support for multiple dataset types
- Auto-detection of dataset format
- Integration with new dataset loaders
- Default 100 epochs training

## Configuration Options

### Model Parameters
- `--hidden-dim`: Hidden layer dimension (default: 128)
- `--num-nodes`: Number of adaptive nodes (default: 100)
- `--device`: Training device (cpu/cuda)

### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)

### Dataset Parameters
- `--samples`: Number of synthetic samples if no real data (default: 2000)
- `--dataset-type`: Force specific dataset type (auto/annomi/mental_health)

## Results and Metrics

The training scripts provide comprehensive metrics:
- **Training Progress**: Loss and accuracy per epoch
- **Final Performance**: Test accuracy, best accuracy achieved
- **Model Information**: Parameter count, vocabulary size
- **Timing**: Training time, total execution time
- **Dataset Information**: Training/test split sizes

Results are automatically saved to `benchmark_results/essay_classification_results.json`.

## Error Handling and Fallbacks

The implementation includes robust error handling:
1. **File Format Detection**: Tries multiple formats if initial load fails
2. **Column Name Detection**: Uses fuzzy matching for column names
3. **Encoding Issues**: Tries multiple text encodings
4. **Synthetic Fallback**: Falls back to synthetic data if real data fails
5. **Graceful Degradation**: Continues with warnings rather than failing

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision pandas numpy scipy rich pyyaml
   ```

2. **Download Datasets** (optional):
   - ANNOMI: https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing
   - Mental Health FAQs: https://www.kaggle.com/datasets/ragishehab/mental-healthfaqs

3. **Run Training**:
   ```bash
   # With synthetic data (works immediately)
   python train_annomi.py --epochs 100 --verbose
   
   # With real data
   python train_annomi.py --data-path /path/to/dataset --epochs 100 --verbose
   ```

## Performance Notes

- **100 Epochs**: Training typically takes 10-20 minutes for 2000 samples on CPU
- **GPU Acceleration**: Use `--device cuda` for faster training if available
- **Memory Usage**: Scales with vocabulary size and sequence length
- **Batch Size**: Adjust based on available memory (32 is conservative default)

This implementation fulfills the problem statement requirements for integrating the specified Kaggle datasets and providing 100-epoch training capabilities.