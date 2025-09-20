# Multi-Dataset Training Guide

This guide documents the extended CI/CD workflows that support training and evaluation on multiple Kaggle datasets.

## Supported Datasets

The system now supports 7 different Kaggle datasets across various machine learning tasks:

### Text Classification Datasets
1. **ANNOMI Motivational Interviewing Dataset**
   - URL: https://www.kaggle.com/datasets/rahulmenon1758/annomi-motivational-interviewing
   - Task: Binary classification (motivational vs non-motivational)
   - Loader: `load_annomi_dataset`

2. **Mental Health FAQs Dataset**
   - URL: https://www.kaggle.com/datasets/ragishehab/mental-healthfaqs
   - Task: Binary classification (anxiety vs depression, etc.)
   - Loader: `load_mental_health_faqs_dataset`

3. **Social Media Sentiments Analysis Dataset**
   - URL: https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset
   - Task: Binary sentiment classification (positive vs negative/neutral)
   - Loader: `load_social_media_sentiment_dataset`

### Sequence Labeling Datasets
4. **Part-of-Speech Tagging Dataset**
   - URL: https://www.kaggle.com/datasets/ruchi798/part-of-speech-tagging
   - Task: Sequence labeling (token-level POS tag prediction)
   - Loader: `load_pos_tagging_dataset`

### Specialized Datasets (NEW)
5. **Virtual Reality Driving Simulator Dataset**
   - URL: https://www.kaggle.com/datasets/sasanj/virtual-reality-driving-simulator-dataset
   - Task: Regression (driving performance prediction)
   - Loader: `load_vr_driving_dataset`

6. **AUTVI Dataset (Automated Vehicle Inspection)**
   - URL: https://www.kaggle.com/datasets/hassanmojab/autvi
   - Task: Binary classification (pass/fail inspection)
   - Loader: `load_autvi_dataset`

7. **Digakust Dataset (Mensa Saarland University)**
   - URL: https://www.kaggle.com/datasets/resc28/digakust-dataset-mensa-saarland-university
   - Task: Multi-class classification (acoustic pattern recognition)
   - Loader: `load_digakust_dataset`

## Training Scripts

### 1. Legacy Datasets Training
For ANNOMI, Mental Health, and Social Media Sentiment datasets:
```bash
python train_kaggle_datasets.py --dataset <dataset_name> --epochs <num_epochs>
```

### 2. POS Tagging Training
For Part-of-Speech tagging:
```bash
python train_pos_tagging.py --epochs <num_epochs> --auto
```

### 3. New Datasets Training
For VR Driving, AUTVI, and Digakust datasets:
```bash
python train_new_datasets.py --dataset <dataset_name> --epochs <num_epochs>
```

### 4. Unified Training
Train on all datasets:
```bash
python train_kaggle_datasets.py --dataset all --epochs 10
```

## CI/CD Workflows

### 1. Main CI Workflow (`ci.yml`)
- Runs on every push and pull request
- Includes linting, type checking, testing, and training phases
- Supports both synthetic and real data training

### 2. Bitext Training Workflow (`bitext-train.yml`)
- Specialized workflow for text-based datasets
- Runs on manual trigger and schedule
- Supports matrix testing across Python versions

### 3. Multi-Dataset Training Workflow (`multi-dataset-training.yml`)
- **NEW**: Comprehensive workflow for all datasets
- Supports manual dataset selection via workflow inputs
- Automated Kaggle dataset downloading (with API credentials)
- Matrix-based training across all datasets
- Consolidated reporting and artifact management

## Workflow Parameters

### Multi-Dataset Training Workflow Inputs

#### Manual Trigger (`workflow_dispatch`)
- **dataset**: Select which dataset to train on
  - Options: `all`, `annomi`, `mental_health_faqs`, `social_media_sentiment`, `pos_tagging`, `vr_driving`, `autvi`, `digakust`
  - Default: `all`
- **epochs**: Number of training epochs
  - Default: `10`
  - Type: string
- **use_real_data**: Whether to use real Kaggle datasets (requires API key)
  - Default: `false`
  - Type: boolean

#### Automatic Triggers
- **Push/PR**: Trains all datasets with 5 epochs using synthetic data
- **Schedule**: Weekly training on Sundays at 3 AM UTC with 20 epochs

## Kaggle Integration

### API Key Setup
To use real Kaggle datasets, set up these GitHub secrets:
- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key

### Dataset Download
The workflow automatically downloads datasets when credentials are available:
```yaml
- name: Download Kaggle dataset (if configured)
  if: env.KAGGLE_CONFIGURED == 'true'
  run: |
    case "${{ matrix.dataset }}" in
      "pos_tagging")
        kaggle datasets download -d ruchi798/part-of-speech-tagging -p data/pos_tagging --unzip
        ;;
      "vr_driving")
        kaggle datasets download -d sasanj/virtual-reality-driving-simulator-dataset -p data/vr_driving --unzip
        ;;
      # ... other datasets
    esac
```

## Training Architecture

### Model Selection
Each dataset uses an appropriate model architecture:
- **Text Classification**: BERT-based adaptive neural network
- **Sequence Labeling**: LSTM/Transformer for POS tagging
- **Regression**: Feed-forward networks for VR driving
- **Multi-class Classification**: Adaptive neural networks for acoustic analysis

### Evaluation Metrics
- **Binary Classification**: Accuracy, Precision, Recall, F1-score
- **Multi-class Classification**: Accuracy, per-class metrics
- **Regression**: MSE, MAE, R²
- **Sequence Labeling**: Token-level accuracy, entity-level F1

## Outputs and Artifacts

### Training Results
Each training run generates:
- JSON results file with metrics and metadata
- Training logs and progress history
- Model checkpoints (when applicable)
- Evaluation plots and visualizations

### Artifact Structure
```
outputs/
├── <dataset_name>/
│   ├── <dataset>_training_results.json
│   ├── training_report.md
│   └── model_checkpoints/
└── all_datasets_results.json (for "all" training)
```

### Consolidated Reporting
The workflow generates a comprehensive training summary:
- Training status for each dataset
- Performance metrics comparison
- Training time and resource usage
- Links to detailed results

## Usage Examples

### Manual Training (Single Dataset)
```bash
# Train VR Driving dataset locally
python train_new_datasets.py --dataset vr_driving --epochs 20 --data-path /path/to/dataset

# Train with synthetic data
python train_new_datasets.py --dataset vr_driving --epochs 10 --num-samples 1000
```

### CI/CD Trigger (GitHub Actions)
```bash
# Trigger via GitHub CLI
gh workflow run multi-dataset-training.yml \
  -f dataset=vr_driving \
  -f epochs=15 \
  -f use_real_data=true
```

### Batch Training (All Datasets)
```bash
# Train all datasets locally
python train_kaggle_datasets.py --dataset all --epochs 5

# Train new datasets only
python train_new_datasets.py --dataset all --epochs 10
```

## Monitoring and Debugging

### Workflow Status
Monitor training progress through:
- GitHub Actions UI
- Workflow run logs
- Artifact downloads
- PR comments (for pull requests)

### Common Issues
1. **Missing Kaggle credentials**: Falls back to synthetic data
2. **Dataset download failures**: Uses cached versions or synthetic data
3. **Training timeouts**: Adjust timeout limits in workflow
4. **Memory constraints**: Reduce batch size or dataset size

### Troubleshooting
```bash
# Check dataset loading
python -c "from adaptiveneuralnetwork.data import get_dataset_info; print(get_dataset_info())"

# Test individual dataset
python train_new_datasets.py --dataset vr_driving --epochs 1 --num-samples 100 --verbose

# Validate results
ls -la outputs/
cat outputs/vr_driving_training_results.json
```

## Performance Benchmarks

### Expected Training Times (Synthetic Data)
- Text Classification: ~30 seconds (1000 samples, 10 epochs)
- POS Tagging: ~45 seconds (1000 sentences, 10 epochs) 
- VR Driving: ~25 seconds (1000 samples, 10 epochs)
- AUTVI: ~20 seconds (1000 samples, 10 epochs)
- Digakust: ~30 seconds (1000 samples, 10 epochs)

### Resource Usage
- Memory: ~2-4 GB per dataset
- CPU: Scales with epoch count and dataset size
- Disk: ~100 MB per training run (including artifacts)

## Future Enhancements

### Planned Features
- [ ] Distributed training across multiple runners
- [ ] Hyperparameter optimization workflows
- [ ] A/B testing framework for model architectures
- [ ] Real-time performance monitoring dashboard
- [ ] Integration with MLflow for experiment tracking

### Extension Points
- Custom dataset loaders in `adaptiveneuralnetwork/data/kaggle_datasets.py`
- New training scripts following the `train_new_datasets.py` pattern
- Additional workflow triggers and scheduling options
- Enhanced reporting and visualization tools