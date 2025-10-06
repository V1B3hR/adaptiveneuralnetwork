# AliveNode Dataset Training

This directory contains scripts and tests for training AliveLoopNode instances using experiences derived from multiple datasets.

## Overview

The `train_alive_node_with_datasets.py` script enables full training of AliveNode agents using reinforcement learning principles with experiences derived from various datasets specified in `data/README.md`.

## Supported Datasets

The training script supports the following 6 datasets:

1. **IBM HR Analytics Employee Attrition Dataset**
   - Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
   - Reward signal: Based on job satisfaction and attrition

2. **Human vs AI Generated Essays**
   - Source: https://www.kaggle.com/datasets/navjotkaushal/human-vs-ai-generated-essays
   - Reward signal: Based on text quality and authenticity

3. **Disorder Dataset**
   - Source: https://www.kaggle.com/datasets/varishabatool/disorder
   - Reward signal: Based on pattern recognition confidence

4. **Emotion Prediction Dataset**
   - Source: https://www.kaggle.com/datasets/emirhanai/emotion-prediction-with-semi-supervised-learning
   - Reward signal: Based on emotion labels (positive/negative valence)

5. **Neural Networks and Deep Learning Dataset**
   - Source: https://www.kaggle.com/datasets/sanjoypator/c1-neural-networks-and-deep-learning
   - Reward signal: Based on model accuracy/performance

6. **Galas Images Dataset**
   - Source: https://www.kaggle.com/datasets/iamhungundji/galasimages
   - Reward signal: Based on image classification confidence

## Usage

### Basic Training

Train on all datasets with default parameters:

```bash
python training/scripts/train_alive_node_with_datasets.py --all
```

### Full Training with Custom Parameters

Train for multiple epochs with more samples:

```bash
python training/scripts/train_alive_node_with_datasets.py \
  --epochs 3 \
  --samples-per-dataset 200 \
  --batch-size 32 \
  --learning-rate 0.01 \
  --all \
  --verbose
```

### Training with Real Data

If you have downloaded real datasets:

```bash
python training/scripts/train_alive_node_with_datasets.py \
  --data-dir /path/to/datasets \
  --all
```

### Command Line Options

- `--epochs`: Number of training epochs (default: 1)
- `--samples-per-dataset`: Number of samples per dataset (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate (default: 0.01)
- `--data-dir`: Directory containing datasets (default: data/)
- `--all`: Train on all datasets (recommended)
- `--verbose`: Enable verbose logging
- `--output`: Output file for results (default: alive_node_training_results.json)

## How It Works

### Dataset to Experience Conversion

The script uses `DatasetToExperienceConverter` to transform dataset samples into reinforcement learning experiences with the following structure:

```python
{
    'state': {
        'energy': float,
        'position': tuple,
        'dataset_type': str
    },
    'action': str,
    'reward': float,
    'next_state': {
        'energy': float,
        'position': tuple,
        'dataset_type': str
    },
    'done': bool
}
```

### Reward Signals

Each dataset type has custom logic for determining rewards:

- **HR Analytics**: Job satisfaction and attrition influence rewards
- **Essays**: Text quality and authenticity determine rewards
- **Disorder**: Pattern recognition confidence affects rewards
- **Emotion**: Emotion valence (positive/negative) maps to rewards
- **Neural Networks**: Model performance metrics determine rewards
- **Galas Images**: Classification confidence influences rewards

### Training Process

1. **Initialize AliveNode** with starting energy and position
2. **Load datasets** (real data if available, synthetic otherwise)
3. **Convert samples** to reinforcement learning experiences
4. **Train in batches** using AliveNode.train() method
5. **Track metrics** including rewards, memories, energy, and emotional states
6. **Save results** to JSON file for analysis

## Output

The training script produces a JSON file containing:

- Configuration parameters
- Initial and final node state
- Per-dataset training results
- Aggregate training metrics

Example output structure:

```json
{
  "configuration": {
    "epochs": 2,
    "samples_per_dataset": 100,
    "batch_size": 32,
    "learning_rate": 0.01,
    "datasets": ["hr_analytics", "essays", ...]
  },
  "initial_state": {
    "energy": 50.0,
    "memory_count": 0
  },
  "final_state": {
    "energy": 50.0,
    "memory_count": 395,
    "phase": "active",
    "joy": 1.74,
    "anxiety": 0.0
  },
  "aggregate_metrics": {
    "total_experiences": 1200,
    "total_reward": 1024.57,
    "average_reward": 0.8538,
    "total_memories_created": 395
  }
}
```

## Testing

Run the test suite to validate functionality:

```bash
python -m unittest tests.test_alive_node_dataset_training -v
```

The test suite includes:

- Experience converter validation
- Synthetic dataset generation tests
- Complete training workflow tests
- Multi-dataset training tests
- Node state change validation
- Reward calculation verification

## Performance

Typical performance on CPU:

- **100 samples per dataset**: ~1-2 seconds per dataset
- **1000 samples per dataset**: ~10-15 seconds per dataset
- **Memory usage**: Scales with number of experiences and memories created

## Example Training Session

```bash
$ python training/scripts/train_alive_node_with_datasets.py --samples-per-dataset 100 --epochs 2 --all

================================================================================
ALIVE NODE TRAINING WITH ALL DATASETS
================================================================================
Training Configuration:
  Epochs: 2
  Samples per dataset: 100
  Batch size: 32
  Learning rate: 0.01
  Datasets: 6
================================================================================

Training on hr_analytics dataset...
Results:
  Total experiences: 100
  Total reward: 35.00
  Memories created: 25

[... training continues for all datasets ...]

================================================================================
TRAINING COMPLETE
================================================================================
Node Statistics:
  Initial memories: 0
  Final memories: 395
  Memories added: 395
  Current phase: active
  Joy level: 1.74
  Anxiety level: 0.00

Aggregate Training Metrics:
  Total experiences: 1200
  Total reward: 1024.57
  Average reward: 0.8538
  Total memories created: 395
================================================================================

Training completed successfully! ✅
```

## Integration with AliveNode

The training leverages AliveNode's built-in `train()` method which:

- Stores valuable experiences as memories (based on reward magnitude)
- Adjusts emotional states based on rewards (joy, anxiety, frustration, etc.)
- Updates behavioral parameters using learning rate
- Adapts energy predictions based on patterns
- Maintains memory cleanup for efficient operation

## See Also

- `core/alive_node.py` - AliveLoopNode implementation
- `tests/test_alive_node.py` - Original AliveNode tests
- `tests/test_alive_node_dataset_training.py` - Dataset training tests
- `data/README.md` - Dataset sources and information
