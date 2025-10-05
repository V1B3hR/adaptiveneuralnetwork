# Core Training Module

This module provides a unified entry point for training adaptive neural networks on all supported datasets.

## Quick Start

### Train on all datasets
```bash
python -m core.train --dataset all --epochs 10
```

### Train on a specific dataset
```bash
python -m core.train --dataset vr_driving --epochs 10
python -m core.train --dataset autvi --epochs 10
python -m core.train --dataset digakust --epochs 10
```

### Custom configuration
```bash
python -m core.train \
  --dataset all \
  --epochs 20 \
  --num-samples 2000 \
  --output-dir my_results \
  --verbose
```

## Supported Datasets

1. **VR Driving** - Virtual Reality Driving Simulator Dataset
2. **AUTVI** - Automated Vehicle Inspection Dataset
3. **Digakust** - Digital Acoustic Analysis Dataset

## Command-Line Options

- `--dataset`: Choose dataset to train on (`vr_driving`, `autvi`, `digakust`, or `all`)
- `--data-path`: Path to real dataset (optional, uses synthetic data if not provided)
- `--epochs`: Number of training epochs (default: 10)
- `--num-samples`: Number of synthetic samples to generate (default: 1000)
- `--output-dir`: Output directory for results (default: `outputs`)
- `--verbose`: Enable verbose logging

## Output

The training script generates:
- Individual result files for each dataset: `{dataset}_training_results.json`
- Combined results file (when using `--dataset all`): `all_datasets_results.json`

Each result file contains:
- Training metrics (accuracy, loss)
- Training history
- Final evaluation metrics
- Dataset information

## Examples

### Quick Test
```bash
python -m core.train --dataset all --epochs 2 --num-samples 100
```

### Production Training
```bash
python -m core.train \
  --dataset all \
  --epochs 100 \
  --num-samples 10000 \
  --output-dir production_results
```

### With Real Data
```bash
python -m core.train \
  --dataset vr_driving \
  --data-path /path/to/vr_driving_dataset.csv \
  --epochs 50
```
