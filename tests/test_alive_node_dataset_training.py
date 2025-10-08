"""
Tests for AliveNode training with datasets.

This module tests the training script that trains AliveLoopNode
with experiences derived from multiple datasets.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.alive_node import AliveLoopNode
from training.scripts.train_alive_node_with_datasets import (
    DatasetToExperienceConverter,
    load_synthetic_dataset,
    train_alive_node_on_dataset,
)


class TestAliveNodeDatasetTraining(unittest.TestCase):
    """Test AliveNode training with datasets."""

    def setUp(self):
        """Set up test fixtures."""
        self.node = AliveLoopNode(
            position=(0, 0),
            velocity=(1, 1),
            initial_energy=50.0,
            field_strength=1.0,
            node_id=1
        )
        self.converter = DatasetToExperienceConverter()

    def test_converter_creates_valid_experiences(self):
        """Test that converter creates valid experience dictionaries."""
        sample_data = {
            'Attrition': 0,
            'JobSatisfaction': 4,
            'Age': 35
        }

        experience = self.converter.convert_to_experience(sample_data, 'hr_analytics')

        # Verify experience structure
        self.assertIn('state', experience)
        self.assertIn('action', experience)
        self.assertIn('reward', experience)
        self.assertIn('next_state', experience)
        self.assertIn('done', experience)

        # Verify state structure
        self.assertIn('energy', experience['state'])
        self.assertIn('position', experience['state'])
        self.assertGreater(experience['state']['energy'], 0)

    def test_converter_different_datasets(self):
        """Test converter handles different dataset types."""
        dataset_types = [
            'hr_analytics', 'essays', 'disorder',
            'emotion', 'neural_networks', 'galas_images'
        ]

        for dataset_type in dataset_types:
            sample_data = {'dummy': 'value'}
            experience = self.converter.convert_to_experience(sample_data, dataset_type)

            self.assertIsInstance(experience, dict)
            self.assertIn('action', experience)
            self.assertIn('reward', experience)

    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset generation."""
        dataset_types = ['hr_analytics', 'essays', 'disorder']

        for dataset_type in dataset_types:
            samples = load_synthetic_dataset(dataset_type, num_samples=10)

            self.assertEqual(len(samples), 10)
            self.assertIsInstance(samples, list)
            self.assertIsInstance(samples[0], dict)

    def test_alive_node_training_with_dataset(self):
        """Test complete training workflow with a dataset."""
        # Generate synthetic dataset
        dataset_samples = load_synthetic_dataset('hr_analytics', num_samples=20)

        # Record initial state
        initial_memory_count = len(self.node.memory)
        initial_energy = self.node.energy

        # Train on dataset
        results = train_alive_node_on_dataset(
            self.node,
            dataset_samples,
            'hr_analytics',
            batch_size=10,
            learning_rate=0.01
        )

        # Verify results structure
        self.assertIn('dataset_type', results)
        self.assertIn('total_experiences', results)
        self.assertIn('total_reward', results)
        self.assertIn('total_memories_created', results)

        # Verify training occurred
        self.assertEqual(results['total_experiences'], 20)
        self.assertGreaterEqual(len(self.node.memory), initial_memory_count)

    def test_multiple_dataset_training(self):
        """Test training on multiple datasets sequentially."""
        dataset_types = ['hr_analytics', 'essays', 'emotion']
        all_results = []

        for dataset_type in dataset_types:
            dataset_samples = load_synthetic_dataset(dataset_type, num_samples=10)
            results = train_alive_node_on_dataset(
                self.node,
                dataset_samples,
                dataset_type,
                batch_size=5,
                learning_rate=0.01
            )
            all_results.append(results)

        # Verify all trainings completed
        self.assertEqual(len(all_results), 3)
        for results in all_results:
            self.assertEqual(results['total_experiences'], 10)
            self.assertGreaterEqual(results['total_memories_created'], 0)

    def test_node_state_changes_after_training(self):
        """Test that node state changes appropriately after training."""
        initial_joy = self.node.joy
        initial_memory_count = len(self.node.memory)

        # Create experiences with positive rewards
        experiences = []
        for i in range(10):
            experiences.append({
                'state': {'energy': 50.0, 'position': (i, 0)},
                'action': 'test_action',
                'reward': 5.0,  # Positive reward
                'next_state': {'energy': 51.0, 'position': (i+1, 0)},
                'done': False
            })

        # Train on experiences
        metrics = self.node.train(experiences, learning_rate=0.01)

        # Verify metrics
        self.assertEqual(metrics['total_reward'], 50.0)
        self.assertGreater(len(self.node.memory), initial_memory_count)

        # Positive rewards should increase joy
        self.assertGreater(self.node.joy, initial_joy)

    def test_reward_calculation_hr_analytics(self):
        """Test reward calculation for HR analytics dataset."""
        # High satisfaction, no attrition -> positive reward
        sample1 = {'Attrition': 0, 'JobSatisfaction': 4}
        action1, reward1 = self.converter._determine_action_reward(sample1, 'hr_analytics')
        self.assertGreater(reward1, 0)

        # Low satisfaction, attrition -> negative reward
        sample2 = {'Attrition': 1, 'JobSatisfaction': 1}
        action2, reward2 = self.converter._determine_action_reward(sample2, 'hr_analytics')
        self.assertLess(reward2, 0)

    def test_reward_calculation_emotion(self):
        """Test reward calculation for emotion dataset."""
        # Positive emotion -> positive reward
        sample1 = {'emotion': 'joy'}
        action1, reward1 = self.converter._determine_action_reward(sample1, 'emotion')
        self.assertGreater(reward1, 0)

        # Negative emotion -> negative reward
        sample2 = {'emotion': 'sadness'}
        action2, reward2 = self.converter._determine_action_reward(sample2, 'emotion')
        self.assertLess(reward2, 0)


if __name__ == '__main__':
    unittest.main()
