# Comprehensive Training Guide

This guide consolidates all training-related knowledge for the Adaptive Neural Network system, including dataset-based training workflows and agent/node-level reinforcement learning.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training Workflows](#training-workflows)
  - [1. Dataset-based Training](#1-dataset-based-training)
  - [2. Agent/Node-level (AliveNode) Training](#2-agentnode-level-alivenode-training)
- [Supported Datasets](#supported-datasets)
- [Training Script Reference](#training-script-reference)
- [Configuration Options](#configuration-options)
- [Kaggle Integration](#kaggle-integration)
- [Data Formats](#data-formats)
- [Model Architectures](#model-architectures)
- [Results and Metrics](#results-and-metrics)
- [Performance Guidelines](#performance-guidelines)
- [Error Handling](#error-handling)
- [Integration with Adaptive Features](#integration-with-adaptive-features)
- [Advanced Usage](#advanced-usage)
- [Future Enhancements](#future-enhancements)

---

## Overview

The Adaptive Neural Network repository supports:
- **Text Classification**: Binary/multi-class, including essay and sentiment tasks.
- **Sequence Labeling**: POS tagging, NER.
- **Specialized Tasks**: Regression/classification on domain-specific datasets.
- **Agent/Node-level Training**: Reinforcement learning for autonomous node agents (e.g., AliveNode).

---

## Quick Start

### Installation

```bash
pip install adaptiveneuralnetwork
# For NLP/text processing
pip install 'adaptiveneuralnetwork[nlp]'
# For development
pip install 'adaptiveneuralnetwork[dev]'
```

---

## Training Workflows

### 1. Dataset-based Training

Scripts for text/sequence/specialized dataset training are in:
- `adaptiveneuralnetwork/training/scripts/`
- Example scripts: `run_bitext_training.py`, `run_essay_benchmark.py`, `train_kaggle_datasets.py`, `train_pos_tagging.py`, `train_new_datasets.py`

#### Basic Usage

- **Bitext Training (smoke test):**
  ```bash
  python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
  ```
- **Full Benchmark:**
  ```bash
  python run_essay_benchmark.py --epochs 100 --samples 2000 --synthetic
  ```

See [Training Script Reference](#training-script-reference) for details.

---

### 2. Agent/Node-level (AliveNode) Training

The repository implements agent-level reinforcement learning via the `AliveNode` class in `core/alive_node.py`.

#### Key Features
- Experience-based learning: The `train` method enables nodes to learn from interaction "experiences".
- Emotional and behavioral adaptation: Learning outcomes affect node emotions, memory, and future predictions.
- Fully tested: Unit tests for node training are provided.

#### Example: Programmatic Node Training

```python
from core.alive_node import AliveLoopNode

# Create node instance (example arguments)
node = AliveLoopNode(position=(0, 0), velocity=(1, 1), initial_energy=10.0, field_strength=1.0, node_id=1)

# Define experiences (as per RL convention)
experiences = [
    {
        'state': {'energy': 10.0, 'position': (0, 0)},
        'action': 'move_forward',
        'reward': 5.0,
        'next_state': {'energy': 9.5, 'position': (1, 0)},
        'done': False
    },
    {
        'state': {'energy': 9.5, 'position': (1, 0)},
        'action': 'interact',
        'reward': -2.0,
        'next_state': {'energy': 9.0, 'position': (1, 0)},
        'done': False
    }
]
metrics = node.train(experiences)
print("Training metrics:", metrics)
```

#### Unit Testing

See `tests/test_alive_node.py` for full examples of training tests and reproducibility.

---

## Supported Datasets

### 1. Text Classification Datasets

- **ANNOMI Motivational Interviewing Dataset**
  - Binary (motivational vs non-motivational)
  - Training: `python train_annomi.py --epochs 100`
- **Mental Health FAQs Dataset**
  - Binary (anxiety vs depression)
  - Training: `python train_mental_health.py --epochs 100`
- **Social Media Sentiments Analysis**
  - Binary sentiment (positive/negative/neutral)
  - Training: `python train_kaggle_datasets.py --dataset social_media_sentiment`
- **Human vs AI Essays**
  - Binary, synthetic support
  - Training: `python run_essay_benchmark.py --epochs 50`

### 2. Sequence Labeling

- **POS Tagging**
  - Models: BiLSTM, Transformer
  - Training: `python train_pos_tagging.py --epochs 20 --auto`

### 3. Specialized/Regression Datasets

- **VR Driving Simulator**: Regression, `python train_new_datasets.py --dataset vr_driving`
- **AUTVI**: Binary, `python train_new_datasets.py --dataset autvi`
- **Digakust**: Multi-class, `python train_new_datasets.py --dataset digakust`

---

## Training Script Reference

### Bitext Training (Lightweight Baseline)

```bash
# Smoke test
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode smoke
# Full benchmark (with Kaggle)
python -m adaptiveneuralnetwork.training.scripts.run_bitext_training --mode benchmark --subset-size 10000 --dataset-name kaggle-user/dataset-name
```

### Essay Benchmark

```bash
python run_essay_benchmark.py --synthetic --epochs 50 --samples 2000
```

### Multi-dataset Training

```bash
python train_kaggle_datasets.py --dataset all --epochs 10
python train_kaggle_datasets.py --dataset annomi --epochs 100
```

### Specialized/Sequence Training

```bash
python train_new_datasets.py --dataset vr_driving --epochs 15
python train_pos_tagging.py --epochs 20 --auto
```

### Node/Agent Training (AliveNode)

See Python code example above, and refer to `core/alive_node.py` for the `train` method.

---

## Configuration Options

### Model/Training Parameters

- `--hidden-dim`: Hidden layer dimension (default: 128)
- `--num-nodes`: Number of adaptive nodes (default: 100)
- `--device`: Training device (cpu/cuda)
- `--epochs`: Epochs (default varies)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--samples`: Number of synthetic samples (default: 2000)
- `--subset-size`: Max samples from real datasets
- `--data-path`: Path to local dataset files

#### Node/Agent Training

- `learning_rate` (parameter to `train` method, optional): Controls adaptation speed for node learning.

---

## Kaggle Integration

- Set `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables, or use `~/.kaggle/kaggle.json`.
- Training scripts will auto-download datasets when credentials are available.

---

## Data Formats

- **CSV:**  
  `text,label` or `sentence,word,pos`
- **JSON:**  
  `[{"text": ..., "label": ...}]`
- **Experience dicts (for node training):**  
  See Python example above.

---

## Model Architectures

- **Text Classification:** TF-IDF + Logistic Regression, Adaptive Neural Networks
- **Sequence Labeling:** BiLSTM, Transformer encoder
- **Agent/Node:** Custom reinforcement learning in AliveNode (see core/alive_node.py)

---

## Results and Metrics

- **Text/Sequence:** Accuracy, Precision, Recall, F1, Confusion Matrices, Token/Entity-level scores.
- **Node/Agent Training:**  
  - `total_reward`, `avg_reward`, `memories_created`, emotional state changes (see train() return value).

---

## Performance Guidelines

- **Dataset Training:**  
  - Text: ~30-60s per 1000 samples, 10 epochs (CPU)
  - POS: ~45s per 1000 sentences, 10 epochs
  - Essays: ~10-15min for 2000 samples, 50 epochs
- **Node/Agent:**  
  - Scales with number of experiences; typically fast unless running large simulations

---

## Error Handling

- **Missing Dependencies:**  
  `pip install 'adaptiveneuralnetwork[nlp]'`
- **Kaggle Credentials Not Found:**  
  Scripts fall back to synthetic data.
- **Memory Issues:**  
  Use `--subset-size` or `--batch-size`
- **Verbose Mode:**  
  `python train_script.py --verbose`

---

## Integration with Adaptive Features

- Uses `AdaptiveConfig` for parameter management.
- Training and node behaviors adapt to system state.
- Deterministic seeds for reproducibility.
- Scalable design for deployment and simulation.

---

## Advanced Usage

- **Custom Preprocessing, Model Persistence, Feature Analysis:**  
  See examples in the original guide.
- **Extension Points:**  
  - Custom dataset loaders.
  - New training scripts.
  - New node/agent behaviors.

---

## Future Enhancements

- Distributed and federated training.
- Hyperparameter optimization.
- Real-time model serving.
- Multimodal/fusion training capabilities.
- MLOps and visualization integration.

---

*This guide supersedes older, fragmented docs. For details on specific scripts, agent logic, or RL training, see code in `core/alive_node.py`, `tests/test_alive_node.py`, and all scripts in `adaptiveneuralnetwork/training/scripts/`.*
