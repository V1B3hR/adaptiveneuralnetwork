#!/usr/bin/env python3
"""
Full training script for AliveNode using all datasets from data/README.md

This script trains an AliveLoopNode using experiences derived from multiple datasets:
1. IBM HR Analytics Employee Attrition Dataset
2. Human vs AI Generated Essays
3. Disorder Dataset
4. Emotion Prediction Dataset
5. Neural Networks and Deep Learning Dataset
6. Galas Images Dataset

The script converts dataset samples into experiences (state, action, reward, next_state)
that can be used to train the AliveNode using reinforcement learning principles.

Usage:
    python train_alive_node_with_datasets.py --epochs 10 --samples-per-dataset 100
    python train_alive_node_with_datasets.py --all --verbose
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.alive_node import AliveLoopNode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetToExperienceConverter:
    """Converts dataset samples into experiences for AliveNode training."""
    
    def __init__(self, base_energy: float = 10.0, base_position: Tuple[float, float] = (0.0, 0.0)):
        self.base_energy = base_energy
        self.base_position = np.array(base_position, dtype=float)
        self.position_counter = 0
        
    def convert_to_experience(
        self,
        sample_data: Dict[str, Any],
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Convert a dataset sample to an experience for AliveNode training.
        
        Args:
            sample_data: Dictionary containing sample data from dataset
            dataset_type: Type of dataset (for custom conversion logic)
            
        Returns:
            Experience dictionary with keys: state, action, reward, next_state, done
        """
        # Generate state based on sample characteristics
        energy = self.base_energy + np.random.randn() * 2.0
        position = self.base_position + np.array([self.position_counter % 10, self.position_counter // 10], dtype=float)
        self.position_counter += 1
        
        state = {
            'energy': max(1.0, energy),
            'position': tuple(position),
            'dataset_type': dataset_type
        }
        
        # Determine action and reward based on dataset type and sample
        action, reward = self._determine_action_reward(sample_data, dataset_type)
        
        # Generate next state
        energy_change = np.random.randn() * 1.0 + (reward * 0.1)
        next_energy = max(1.0, state['energy'] + energy_change)
        next_position = position + np.random.randn(2) * 0.5
        
        next_state = {
            'energy': next_energy,
            'position': tuple(next_position),
            'dataset_type': dataset_type
        }
        
        return {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': False
        }
    
    def _determine_action_reward(
        self,
        sample_data: Dict[str, Any],
        dataset_type: str
    ) -> Tuple[str, float]:
        """Determine action and reward based on dataset characteristics."""
        
        if dataset_type == 'hr_analytics':
            # Use attrition or satisfaction as reward signal
            attrition = sample_data.get('Attrition', 0)
            satisfaction = sample_data.get('JobSatisfaction', 3)
            reward = float(satisfaction - 2.0) * 2.0  # Range: -4 to +4
            if attrition == 1 or attrition == 'Yes':
                reward -= 3.0
            action = f"assess_employee_status"
            
        elif dataset_type == 'essays':
            # Use essay quality or authenticity as reward
            label = sample_data.get('label', 0)
            reward = float(label) * 4.0 - 2.0  # Range based on label
            action = f"analyze_text_quality"
            
        elif dataset_type == 'disorder':
            # Use disorder classification as reward (identifying patterns)
            label = sample_data.get('label', 0)
            confidence = sample_data.get('confidence', 0.5)
            reward = float(confidence) * 3.0 - 1.0
            action = f"diagnose_pattern"
            
        elif dataset_type == 'emotion':
            # Use emotion labels as reward
            emotion = sample_data.get('emotion', 'neutral')
            emotion_values = {
                'joy': 4.0, 'happiness': 4.0, 'love': 3.5,
                'surprise': 2.0, 'neutral': 0.0,
                'sadness': -2.0, 'anger': -3.0, 'fear': -3.5
            }
            reward = emotion_values.get(emotion.lower(), 0.0)
            action = f"recognize_emotion_{emotion}"
            
        elif dataset_type == 'neural_networks':
            # Use accuracy or performance metrics as reward
            accuracy = sample_data.get('accuracy', 0.5)
            reward = (float(accuracy) - 0.5) * 8.0  # Range: -4 to +4
            action = f"optimize_network"
            
        elif dataset_type == 'galas_images':
            # Use image classification confidence as reward
            confidence = sample_data.get('confidence', 0.5)
            label_correct = sample_data.get('correct', True)
            reward = float(confidence) * 3.0 if label_correct else -float(confidence) * 2.0
            action = f"classify_image"
            
        else:
            # Default case
            reward = np.random.randn() * 2.0
            action = "process_data"
        
        return action, reward


def load_synthetic_dataset(dataset_type: str, num_samples: int) -> List[Dict[str, Any]]:
    """Generate synthetic dataset samples for demonstration."""
    samples = []
    
    if dataset_type == 'hr_analytics':
        for i in range(num_samples):
            samples.append({
                'EmployeeID': i,
                'Attrition': np.random.choice([0, 1], p=[0.8, 0.2]),
                'JobSatisfaction': np.random.randint(1, 5),
                'Age': np.random.randint(22, 60),
                'MonthlyIncome': np.random.randint(2000, 15000)
            })
    
    elif dataset_type == 'essays':
        for i in range(num_samples):
            samples.append({
                'text': f"Sample essay text {i}",
                'label': np.random.choice([0, 1]),  # 0=AI, 1=Human
                'word_count': np.random.randint(100, 500)
            })
    
    elif dataset_type == 'disorder':
        disorders = ['anxiety', 'depression', 'bipolar', 'normal']
        for i in range(num_samples):
            samples.append({
                'symptoms': f"Symptom description {i}",
                'label': np.random.choice(disorders),
                'confidence': np.random.uniform(0.5, 1.0)
            })
    
    elif dataset_type == 'emotion':
        emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
        for i in range(num_samples):
            samples.append({
                'text': f"Emotion text {i}",
                'emotion': np.random.choice(emotions),
                'intensity': np.random.uniform(0.3, 1.0)
            })
    
    elif dataset_type == 'neural_networks':
        for i in range(num_samples):
            samples.append({
                'model_id': i,
                'accuracy': np.random.uniform(0.6, 0.95),
                'loss': np.random.uniform(0.1, 2.0),
                'epochs': np.random.randint(10, 100)
            })
    
    elif dataset_type == 'galas_images':
        for i in range(num_samples):
            samples.append({
                'image_id': i,
                'class': np.random.choice(['galaxy', 'star', 'artifact']),
                'confidence': np.random.uniform(0.5, 1.0),
                'correct': np.random.choice([True, False], p=[0.85, 0.15])
            })
    
    return samples


def load_real_dataset(dataset_type: str, data_path: Path) -> List[Dict[str, Any]]:
    """Load real dataset from file if available."""
    samples = []
    
    try:
        if data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            samples = df.to_dict('records')
            logger.info(f"Loaded {len(samples)} samples from {data_path}")
        elif data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict):
                    samples = [data]
            logger.info(f"Loaded {len(samples)} samples from {data_path}")
        else:
            logger.warning(f"Unsupported file format: {data_path.suffix}")
    except Exception as e:
        logger.error(f"Error loading dataset from {data_path}: {e}")
    
    return samples


def train_alive_node_on_dataset(
    node: AliveLoopNode,
    dataset_samples: List[Dict[str, Any]],
    dataset_type: str,
    batch_size: int = 32,
    learning_rate: float = 0.01
) -> Dict[str, Any]:
    """
    Train an AliveNode on a specific dataset.
    
    Args:
        node: AliveLoopNode instance to train
        dataset_samples: List of dataset samples
        dataset_type: Type of dataset
        batch_size: Number of experiences per training batch
        learning_rate: Learning rate for training
        
    Returns:
        Training metrics dictionary
    """
    converter = DatasetToExperienceConverter()
    
    # Convert all samples to experiences
    all_experiences = []
    for sample in dataset_samples:
        experience = converter.convert_to_experience(sample, dataset_type)
        all_experiences.append(experience)
    
    logger.info(f"Training on {len(all_experiences)} experiences from {dataset_type} dataset")
    
    # Train in batches
    total_metrics = {
        'total_reward': 0.0,
        'total_memories': 0,
        'batches': 0
    }
    
    for i in range(0, len(all_experiences), batch_size):
        batch = all_experiences[i:i+batch_size]
        metrics = node.train(batch, learning_rate=learning_rate)
        
        total_metrics['total_reward'] += metrics['total_reward']
        total_metrics['total_memories'] += metrics['memories_created']
        total_metrics['batches'] += 1
        
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Batch {i//batch_size + 1}: reward={metrics['total_reward']:.2f}, "
                       f"memories={metrics['memories_created']}, "
                       f"energy={metrics['current_energy']:.2f}")
    
    avg_reward = total_metrics['total_reward'] / len(all_experiences) if all_experiences else 0.0
    
    return {
        'dataset_type': dataset_type,
        'total_experiences': len(all_experiences),
        'total_reward': total_metrics['total_reward'],
        'avg_reward_per_experience': avg_reward,
        'total_memories_created': total_metrics['total_memories'],
        'batches_processed': total_metrics['batches'],
        'final_energy': node.energy,
        'final_predicted_energy': node.predicted_energy
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train AliveNode with all datasets from data/README.md"
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of training epochs (default: 1)'
    )
    parser.add_argument(
        '--samples-per-dataset',
        type=int,
        default=100,
        help='Number of samples per dataset (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/home/runner/work/adaptiveneuralnetwork/adaptiveneuralnetwork/data',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train on all datasets'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='alive_node_training_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List of all datasets from data/README.md
    datasets = [
        'hr_analytics',      # IBM HR Analytics Employee Attrition
        'essays',            # Human vs AI Generated Essays
        'disorder',          # Disorder Dataset
        'emotion',           # Emotion Prediction
        'neural_networks',   # Neural Networks and Deep Learning
        'galas_images'       # Galas Images
    ]
    
    print("=" * 80)
    print("ALIVE NODE TRAINING WITH ALL DATASETS")
    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Samples per dataset: {args.samples_per_dataset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Datasets: {len(datasets)}")
    print("=" * 80)
    print()
    
    # Initialize AliveNode
    logger.info("Initializing AliveLoopNode...")
    node = AliveLoopNode(
        position=(0, 0),
        velocity=(1, 1),
        initial_energy=50.0,  # Start with higher energy for long training
        field_strength=1.0,
        node_id=1
    )
    
    initial_energy = node.energy
    initial_memory_count = len(node.memory)
    
    # Training results
    all_results = []
    
    # Train on each dataset
    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        for dataset_type in datasets:
            print(f"\nTraining on {dataset_type} dataset...")
            print("-" * 80)
            
            # Try to load real data, fall back to synthetic
            data_dir = Path(args.data_dir)
            dataset_samples = None
            
            # Check for real data files
            possible_files = list(data_dir.glob(f"*{dataset_type}*.csv")) + \
                           list(data_dir.glob(f"*{dataset_type}*.json"))
            
            if possible_files:
                logger.info(f"Found potential data file: {possible_files[0]}")
                dataset_samples = load_real_dataset(dataset_type, possible_files[0])
            
            if not dataset_samples:
                logger.info(f"Using synthetic data for {dataset_type}")
                dataset_samples = load_synthetic_dataset(dataset_type, args.samples_per_dataset)
            
            # Limit samples if needed
            if len(dataset_samples) > args.samples_per_dataset:
                dataset_samples = dataset_samples[:args.samples_per_dataset]
            
            # Train on this dataset
            results = train_alive_node_on_dataset(
                node,
                dataset_samples,
                dataset_type,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            
            results['epoch'] = epoch + 1
            all_results.append(results)
            
            # Print results
            print(f"\nResults for {dataset_type}:")
            print(f"  Total experiences: {results['total_experiences']}")
            print(f"  Total reward: {results['total_reward']:.2f}")
            print(f"  Avg reward per experience: {results['avg_reward_per_experience']:.4f}")
            print(f"  Memories created: {results['total_memories_created']}")
            print(f"  Final energy: {results['final_energy']:.2f}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Node Statistics:")
    print(f"  Initial energy: {initial_energy:.2f}")
    print(f"  Final energy: {node.energy:.2f}")
    print(f"  Energy change: {node.energy - initial_energy:.2f}")
    print(f"  Initial memories: {initial_memory_count}")
    print(f"  Final memories: {len(node.memory)}")
    print(f"  Memories added: {len(node.memory) - initial_memory_count}")
    print(f"  Current phase: {node.phase}")
    print(f"  Joy level: {node.joy:.2f}")
    print(f"  Anxiety level: {node.anxiety:.2f}")
    print()
    
    # Calculate aggregate statistics
    total_reward = sum(r['total_reward'] for r in all_results)
    total_experiences = sum(r['total_experiences'] for r in all_results)
    total_memories = sum(r['total_memories_created'] for r in all_results)
    
    print(f"Aggregate Training Metrics:")
    print(f"  Total experiences: {total_experiences}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward: {total_reward/total_experiences:.4f}")
    print(f"  Total memories created: {total_memories}")
    print("=" * 80)
    
    # Save results
    output_data = {
        'configuration': {
            'epochs': args.epochs,
            'samples_per_dataset': args.samples_per_dataset,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'datasets': datasets
        },
        'initial_state': {
            'energy': initial_energy,
            'memory_count': initial_memory_count
        },
        'final_state': {
            'energy': float(node.energy),
            'memory_count': len(node.memory),
            'phase': node.phase,
            'joy': float(node.joy),
            'anxiety': float(node.anxiety)
        },
        'training_results': all_results,
        'aggregate_metrics': {
            'total_experiences': total_experiences,
            'total_reward': total_reward,
            'average_reward': total_reward / total_experiences if total_experiences > 0 else 0.0,
            'total_memories_created': total_memories
        }
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\nTraining completed successfully! ✅")


if __name__ == "__main__":
    main()
