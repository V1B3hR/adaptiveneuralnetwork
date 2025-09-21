"""
Curriculum learning system for adaptive neural networks.

This module implements automatic task difficulty adjustment based on agent performance
as part of Phase 1: Adaptive Learning & Continual Improvement.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Performance tracking
    performance_window: int = 100  # Recent performance window size
    success_threshold: float = 0.8  # Success rate threshold for advancement
    failure_threshold: float = 0.3  # Failure rate threshold for regression

    # Difficulty adjustment
    initial_difficulty: float = 0.1  # Starting difficulty level
    max_difficulty: float = 1.0  # Maximum difficulty level
    difficulty_step: float = 0.1  # Difficulty adjustment step size

    # Adaptation parameters
    patience: int = 50  # Patience before difficulty adjustment
    min_samples: int = 20  # Minimum samples before adjustment


class DifficultyController:
    """Controls task difficulty based on performance."""

    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_difficulty = config.initial_difficulty
        self.performance_history = []
        self.steps_since_adjustment = 0

        logger.debug(f"Initialized DifficultyController at difficulty {self.current_difficulty}")

    def update_performance(self, performance: float) -> bool:
        """
        Update performance history and adjust difficulty if needed.

        Args:
            performance: Performance score (0.0 to 1.0)

        Returns:
            True if difficulty was adjusted, False otherwise
        """
        self.performance_history.append(performance)
        self.steps_since_adjustment += 1

        # Keep only recent performance
        if len(self.performance_history) > self.config.performance_window:
            self.performance_history.pop(0)

        # Check if we have enough samples for adjustment
        if (
            len(self.performance_history) < self.config.min_samples
            or self.steps_since_adjustment < self.config.patience
        ):
            return False

        return self._adjust_difficulty()

    def _adjust_difficulty(self) -> bool:
        """Adjust difficulty based on recent performance."""
        recent_performance = np.mean(self.performance_history)
        old_difficulty = self.current_difficulty

        if recent_performance >= self.config.success_threshold:
            # Increase difficulty
            self.current_difficulty = min(
                self.config.max_difficulty, self.current_difficulty + self.config.difficulty_step
            )
        elif recent_performance <= self.config.failure_threshold:
            # Decrease difficulty
            self.current_difficulty = max(
                0.0, self.current_difficulty - self.config.difficulty_step
            )
        else:
            # No adjustment needed
            return False

        # Reset adjustment counter
        self.steps_since_adjustment = 0

        # Log adjustment
        logger.info(
            f"Difficulty adjusted: {old_difficulty:.2f} -> {self.current_difficulty:.2f} "
            f"(performance: {recent_performance:.3f})"
        )

        return True

    def get_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty

    def reset(self):
        """Reset difficulty controller."""
        self.current_difficulty = self.config.initial_difficulty
        self.performance_history = []
        self.steps_since_adjustment = 0


class TaskGenerator(ABC):
    """Abstract base class for task generators."""

    @abstractmethod
    def generate_task(self, difficulty: float) -> Any:
        """Generate a task at the specified difficulty level."""
        pass

    @abstractmethod
    def evaluate_performance(self, task: Any, response: Any) -> float:
        """Evaluate performance on a task."""
        pass


class SyntheticTaskGenerator(TaskGenerator):
    """Generates synthetic tasks for curriculum learning."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def generate_task(self, difficulty: float) -> Dict[str, torch.Tensor]:
        """
        Generate a synthetic classification task.

        Args:
            difficulty: Difficulty level (0.0 to 1.0)

        Returns:
            Dictionary containing task data and labels
        """
        batch_size = 32

        # Create data with difficulty-dependent noise
        base_pattern = torch.randn(batch_size, self.input_dim)
        noise_level = difficulty * 2.0  # More difficulty = more noise
        noise = torch.randn_like(base_pattern) * noise_level

        data = base_pattern + noise

        # Create labels based on simple patterns (made harder with difficulty)
        if difficulty < 0.3:
            # Easy: based on mean
            labels = (data.mean(dim=1) > 0).long()
        elif difficulty < 0.7:
            # Medium: based on specific features
            labels = (
                data[:, : self.input_dim // 2].sum(dim=1)
                > data[:, self.input_dim // 2 :].sum(dim=1)
            ).long()
        else:
            # Hard: complex non-linear pattern
            labels = ((data[:, ::2].sum(dim=1) * data[:, 1::2].sum(dim=1)) > 0).long()

        # Ensure we have the right number of classes
        labels = labels % self.output_dim

        return {"data": data, "labels": labels, "difficulty": difficulty}

    def evaluate_performance(
        self, task: Dict[str, torch.Tensor], predictions: torch.Tensor
    ) -> float:
        """Evaluate classification performance."""
        labels = task["labels"]
        predicted_labels = predictions.argmax(dim=1)
        accuracy = (predicted_labels == labels).float().mean().item()
        return accuracy


class CurriculumLearningSystem:
    """Complete curriculum learning system."""

    def __init__(self, config: CurriculumConfig, task_generator: TaskGenerator, model: nn.Module):
        self.config = config
        self.task_generator = task_generator
        self.model = model
        self.difficulty_controller = DifficultyController(config)

        # Performance tracking
        self.episode = 0
        self.total_performance_history = []
        self.difficulty_history = []

        logger.info("Initialized CurriculumLearningSystem")

    def train_episode(self) -> Dict[str, float]:
        """
        Train one episode with current difficulty level.

        Returns:
            Dictionary with training metrics
        """
        self.episode += 1
        current_difficulty = self.difficulty_controller.get_difficulty()

        # Generate task at current difficulty
        task = self.task_generator.generate_task(current_difficulty)

        # Train model on task
        self.model.train()
        if hasattr(self.model, "train_step"):
            # Use model's train_step if available
            loss_info = self.model.train_step(task["data"], task["labels"])
        else:
            # Basic training step
            predictions = self.model(task["data"])
            loss = nn.CrossEntropyLoss()(predictions, task["labels"])

            # Backward pass (assumes optimizer is available)
            if hasattr(self.model, "optimizer"):
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            loss_info = {"loss": loss.item()}

        # Evaluate performance
        with torch.no_grad():
            self.model.eval()
            predictions = self.model(task["data"])
            performance = self.task_generator.evaluate_performance(task, predictions)

        # Update difficulty based on performance
        difficulty_adjusted = self.difficulty_controller.update_performance(performance)

        # Track metrics
        self.total_performance_history.append(performance)
        self.difficulty_history.append(current_difficulty)

        metrics = {
            "episode": self.episode,
            "difficulty": current_difficulty,
            "performance": performance,
            "difficulty_adjusted": difficulty_adjusted,
            **loss_info,
        }

        if self.episode % 100 == 0:
            avg_performance = np.mean(self.total_performance_history[-100:])
            logger.info(
                f"Episode {self.episode}: difficulty={current_difficulty:.2f}, "
                f"performance={performance:.3f}, avg_performance={avg_performance:.3f}"
            )

        return metrics

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate model across different difficulty levels.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        # Test at different difficulty levels
        difficulty_levels = np.linspace(0.1, 1.0, 10)
        results = {}

        with torch.no_grad():
            for difficulty in difficulty_levels:
                performances = []

                for _ in range(num_episodes // len(difficulty_levels)):
                    task = self.task_generator.generate_task(difficulty)
                    predictions = self.model(task["data"])
                    performance = self.task_generator.evaluate_performance(task, predictions)
                    performances.append(performance)

                results[f"difficulty_{difficulty:.1f}"] = np.mean(performances)

        # Overall metrics
        results["mean_performance"] = np.mean(list(results.values()))
        results["current_training_difficulty"] = self.difficulty_controller.get_difficulty()

        return results

    def get_learning_curve_data(self) -> Dict[str, List[float]]:
        """Get data for plotting learning curves."""
        return {
            "performance_history": self.total_performance_history.copy(),
            "difficulty_history": self.difficulty_history.copy(),
            "episodes": list(range(1, len(self.total_performance_history) + 1)),
        }

    def reset(self):
        """Reset the curriculum learning system."""
        self.difficulty_controller.reset()
        self.episode = 0
        self.total_performance_history = []
        self.difficulty_history = []


# Utility functions for curriculum learning integration


def create_curriculum_system(
    model: nn.Module, input_dim: int, output_dim: int, config: Optional[CurriculumConfig] = None
) -> CurriculumLearningSystem:
    """
    Create a curriculum learning system with default synthetic task generator.

    Args:
        model: Neural network model to train
        input_dim: Input dimension for tasks
        output_dim: Output dimension for tasks
        config: Curriculum configuration (uses defaults if None)

    Returns:
        Configured curriculum learning system
    """
    if config is None:
        config = CurriculumConfig()

    task_generator = SyntheticTaskGenerator(input_dim, output_dim)
    return CurriculumLearningSystem(config, task_generator, model)


def train_with_curriculum(
    model: nn.Module,
    input_dim: int,
    output_dim: int,
    num_episodes: int = 1000,
    config: Optional[CurriculumConfig] = None,
) -> Dict[str, Any]:
    """
    Train a model using curriculum learning.

    Args:
        model: Neural network model to train
        input_dim: Input dimension for tasks
        output_dim: Output dimension for tasks
        num_episodes: Number of training episodes
        config: Curriculum configuration

    Returns:
        Training results and metrics
    """
    curriculum_system = create_curriculum_system(model, input_dim, output_dim, config)

    training_metrics = []
    for episode in range(num_episodes):
        metrics = curriculum_system.train_episode()
        training_metrics.append(metrics)

    # Final evaluation
    final_evaluation = curriculum_system.evaluate()

    return {
        "training_metrics": training_metrics,
        "final_evaluation": final_evaluation,
        "learning_curve_data": curriculum_system.get_learning_curve_data(),
    }
