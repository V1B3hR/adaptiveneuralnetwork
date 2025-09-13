"""
Dynamic benchmark difficulty scaling system.

This module implements automatic benchmark complexity adjustment and adversarial test generation
as part of Phase 1: Adaptive Learning & Continual Improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from collections import deque
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class DynamicBenchmarkConfig:
    """Configuration for dynamic benchmarking system."""
    # Score thresholds
    plateau_threshold: float = 0.02    # Performance change threshold for plateau detection
    plateau_patience: int = 50         # Steps to wait before declaring plateau
    max_score_threshold: float = 95.0  # Score threshold for difficulty increase
    
    # Difficulty scaling
    difficulty_levels: List[float] = None  # Custom difficulty levels
    max_difficulty_multiplier: float = 3.0  # Maximum difficulty scaling
    adaptation_rate: float = 0.1       # Rate of difficulty adaptation
    
    # Adversarial parameters
    adversarial_strength: float = 0.1  # Strength of adversarial perturbations
    ood_detection_threshold: float = 0.8  # Threshold for OOD detection
    
    # Learning curve analysis
    performance_window: int = 100      # Window for performance tracking
    trend_analysis_window: int = 50    # Window for trend analysis
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]


class BenchmarkTask(ABC):
    """Abstract base class for benchmark tasks."""
    
    @abstractmethod
    def generate(self, difficulty: float, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Generate benchmark task at specified difficulty."""
        pass
    
    @abstractmethod
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Evaluate predictions against targets."""
        pass
    
    @abstractmethod
    def get_baseline_difficulty(self) -> float:
        """Get baseline difficulty level."""
        pass


class AdversarialPatternRecognitionTask(BenchmarkTask):
    """Adversarial pattern recognition benchmark task."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        self.input_dim = input_dim
        self.num_classes = num_classes
        
    def generate(self, difficulty: float, batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """Generate pattern recognition task with adversarial elements."""
        # Base patterns
        patterns = torch.randn(batch_size, self.input_dim)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        
        # Apply difficulty-based transformations
        if difficulty > 1.0:
            # Add noise proportional to difficulty
            noise_level = (difficulty - 1.0) * 0.5
            noise = torch.randn_like(patterns) * noise_level
            patterns = patterns + noise
            
        if difficulty > 1.5:
            # Add adversarial perturbations
            adversarial_strength = (difficulty - 1.5) * 0.2
            adversarial_noise = torch.sign(torch.randn_like(patterns)) * adversarial_strength
            patterns = patterns + adversarial_noise
            
        if difficulty > 2.0:
            # Introduce distributional shift
            shift_amount = (difficulty - 2.0) * 0.3
            patterns = patterns + shift_amount
            
        return {
            'data': patterns,
            'labels': labels,
            'difficulty': difficulty
        }
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Evaluate classification accuracy."""
        predicted_labels = predictions.argmax(dim=1)
        accuracy = (predicted_labels == targets).float().mean().item()
        return accuracy * 100.0  # Return as percentage
    
    def get_baseline_difficulty(self) -> float:
        return 1.0


class AdversarialTestGenerator:
    """Generates adversarial and out-of-distribution tests."""
    
    def __init__(self, config: DynamicBenchmarkConfig):
        self.config = config
        
    def generate_adversarial_examples(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        labels: torch.Tensor,
        epsilon: float = None
    ) -> torch.Tensor:
        """Generate adversarial examples using FGSM."""
        if epsilon is None:
            epsilon = self.config.adversarial_strength
            
        model.eval()
        data.requires_grad_(True)
        
        # Forward pass
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        
        # Calculate gradients
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = data.grad.data
        perturbed_data = data + epsilon * data_grad.sign()
        
        return perturbed_data.detach()
    
    def generate_ood_examples(
        self, 
        original_data: torch.Tensor, 
        ood_strength: float = 1.0
    ) -> torch.Tensor:
        """Generate out-of-distribution examples."""
        batch_size, *data_shape = original_data.shape
        
        # Method 1: Gaussian noise
        if np.random.random() < 0.4:
            ood_data = torch.randn_like(original_data) * ood_strength
            
        # Method 2: Uniform noise
        elif np.random.random() < 0.7:
            ood_data = torch.rand_like(original_data) * 2 - 1  # [-1, 1]
            ood_data = ood_data * ood_strength
            
        # Method 3: Corrupted original data
        else:
            corruption_mask = torch.rand_like(original_data) < 0.3
            ood_data = original_data.clone()
            ood_data[corruption_mask] = torch.randn_like(ood_data[corruption_mask]) * ood_strength
            
        return ood_data
    
    def generate_deceptive_patterns(
        self, 
        data: torch.Tensor, 
        labels: torch.Tensor,
        deception_rate: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate deceptive patterns with wrong labels."""
        batch_size = data.size(0)
        num_deceptive = int(batch_size * deception_rate)
        
        if num_deceptive == 0:
            return data, labels
        
        # Select random samples to make deceptive
        deceptive_indices = torch.randperm(batch_size)[:num_deceptive]
        
        deceptive_data = data.clone()
        deceptive_labels = labels.clone()
        
        # Flip labels randomly
        num_classes = labels.max().item() + 1
        for idx in deceptive_indices:
            # Choose a different label
            new_label = torch.randint(0, num_classes, (1,))
            while new_label == labels[idx]:
                new_label = torch.randint(0, num_classes, (1,))
            deceptive_labels[idx] = new_label
            
            # Slightly modify the data to be misleading
            noise = torch.randn_like(data[idx]) * 0.1
            deceptive_data[idx] = data[idx] + noise
        
        return deceptive_data, deceptive_labels


class LearningCurveAnalyzer:
    """Analyzes learning curves and performance trends."""
    
    def __init__(self, config: DynamicBenchmarkConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_window)
        self.difficulty_history = deque(maxlen=config.performance_window)
        self.timestamps = deque(maxlen=config.performance_window)
        
    def add_performance(self, score: float, difficulty: float, timestamp: float = None):
        """Add performance measurement."""
        if timestamp is None:
            import time
            timestamp = time.time()
            
        self.performance_history.append(score)
        self.difficulty_history.append(difficulty)
        self.timestamps.append(timestamp)
    
    def detect_plateau(self) -> bool:
        """Detect if performance has plateaued."""
        if len(self.performance_history) < self.config.plateau_patience:
            return False
        
        recent_scores = list(self.performance_history)[-self.config.plateau_patience:]
        
        # Check if variance is low (plateau indicator)
        score_variance = np.var(recent_scores)
        mean_score = np.mean(recent_scores)
        
        # Normalized variance check
        if mean_score > 0:
            normalized_variance = score_variance / (mean_score ** 2)
            return normalized_variance < self.config.plateau_threshold
        
        return False
    
    def analyze_trend(self) -> Dict[str, float]:
        """Analyze performance trend."""
        if len(self.performance_history) < self.config.trend_analysis_window:
            # Return default values for insufficient data
            current_scores = list(self.performance_history) if self.performance_history else [0.0]
            return {
                'trend': 0.0, 
                'confidence': 0.0, 
                'current_mean': np.mean(current_scores)
            }
        
        recent_scores = list(self.performance_history)[-self.config.trend_analysis_window:]
        x = np.arange(len(recent_scores))
        
        # Linear regression to find trend
        slope, intercept = np.polyfit(x, recent_scores, 1)
        
        # Calculate R-squared for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((recent_scores - y_pred) ** 2)
        ss_tot = np.sum((recent_scores - np.mean(recent_scores)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'trend': slope,
            'confidence': r_squared,
            'current_mean': np.mean(recent_scores[-10:]) if len(recent_scores) >= 10 else np.mean(recent_scores)
        }
    
    def should_increase_difficulty(self) -> bool:
        """Determine if difficulty should be increased."""
        if len(self.performance_history) == 0:
            return False
        
        trend_analysis = self.analyze_trend()
        recent_performance = trend_analysis['current_mean']
        
        # Increase difficulty if performance is high and stable
        return (recent_performance >= self.config.max_score_threshold and 
                (self.detect_plateau() or trend_analysis['trend'] <= 0.1))
    
    def visualize_learning_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create learning curve visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        if len(self.performance_history) == 0:
            ax1.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax2.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return fig
        
        episodes = list(range(len(self.performance_history)))
        
        # Performance curve
        ax1.plot(episodes, list(self.performance_history), 'b-', label='Performance')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Learning Curve - Performance Over Time')
        ax1.grid(True)
        ax1.legend()
        
        # Difficulty curve
        ax2.plot(episodes, list(self.difficulty_history), 'r-', label='Difficulty')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Difficulty Level')
        ax2.set_title('Difficulty Scaling Over Time')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Learning curve saved to {save_path}")
        
        return fig


class DynamicBenchmarkSystem:
    """Complete dynamic benchmarking system."""
    
    def __init__(
        self, 
        config: DynamicBenchmarkConfig,
        benchmark_task: BenchmarkTask,
        model: nn.Module
    ):
        self.config = config
        self.benchmark_task = benchmark_task
        self.model = model
        
        # Components
        self.adversarial_generator = AdversarialTestGenerator(config)
        self.curve_analyzer = LearningCurveAnalyzer(config)
        
        # State
        self.current_difficulty = benchmark_task.get_baseline_difficulty()
        self.evaluation_count = 0
        self.difficulty_adjustments = []
        
        logger.info("Initialized DynamicBenchmarkSystem")
    
    def evaluate_model(self, include_adversarial: bool = True) -> Dict[str, float]:
        """Evaluate model with current difficulty and adversarial tests."""
        self.model.eval()
        results = {}
        
        # Standard evaluation
        standard_task = self.benchmark_task.generate(self.current_difficulty)
        with torch.no_grad():
            predictions = self.model(standard_task['data'])
            standard_score = self.benchmark_task.evaluate(predictions, standard_task['labels'])
        
        results['standard_score'] = standard_score
        results['difficulty'] = self.current_difficulty
        
        if include_adversarial:
            # Adversarial evaluation
            adversarial_data = self.adversarial_generator.generate_adversarial_examples(
                self.model, standard_task['data'], standard_task['labels']
            )
            
            with torch.no_grad():
                adv_predictions = self.model(adversarial_data)
                adversarial_score = self.benchmark_task.evaluate(adv_predictions, standard_task['labels'])
            
            results['adversarial_score'] = adversarial_score
            
            # Out-of-distribution evaluation
            ood_data = self.adversarial_generator.generate_ood_examples(standard_task['data'])
            with torch.no_grad():
                ood_predictions = self.model(ood_data)
                # For OOD, we expect low confidence (high entropy)
                ood_entropy = -torch.sum(F.softmax(ood_predictions, dim=1) * F.log_softmax(ood_predictions, dim=1), dim=1)
                ood_detection_score = (ood_entropy > self.config.ood_detection_threshold).float().mean().item() * 100
            
            results['ood_detection_score'] = ood_detection_score
            
            # Deceptive pattern evaluation
            deceptive_data, deceptive_labels = self.adversarial_generator.generate_deceptive_patterns(
                standard_task['data'], standard_task['labels']
            )
            with torch.no_grad():
                deceptive_predictions = self.model(deceptive_data)
                deceptive_score = self.benchmark_task.evaluate(deceptive_predictions, deceptive_labels)
            
            results['deceptive_resistance'] = 100 - deceptive_score  # Higher is better for resistance
        
        # Update learning curve analyzer
        import time
        self.curve_analyzer.add_performance(standard_score, self.current_difficulty, time.time())
        
        # Check for difficulty adjustment
        if self.curve_analyzer.should_increase_difficulty():
            self._adjust_difficulty()
        
        self.evaluation_count += 1
        
        return results
    
    def _adjust_difficulty(self):
        """Adjust benchmark difficulty based on performance."""
        old_difficulty = self.current_difficulty
        
        # Find next difficulty level
        current_idx = 0
        for i, level in enumerate(self.config.difficulty_levels):
            if level <= self.current_difficulty:
                current_idx = i
        
        # Move to next level if possible
        if current_idx < len(self.config.difficulty_levels) - 1:
            self.current_difficulty = self.config.difficulty_levels[current_idx + 1]
        else:
            # Beyond predefined levels, multiply by adaptation rate
            self.current_difficulty = min(
                self.current_difficulty * (1 + self.config.adaptation_rate),
                self.benchmark_task.get_baseline_difficulty() * self.config.max_difficulty_multiplier
            )
        
        # Record adjustment
        adjustment_info = {
            'old_difficulty': old_difficulty,
            'new_difficulty': self.current_difficulty,
            'evaluation_count': self.evaluation_count,
            'trigger_reason': 'performance_plateau'
        }
        self.difficulty_adjustments.append(adjustment_info)
        
        logger.info(f"Difficulty adjusted: {old_difficulty:.2f} -> {self.current_difficulty:.2f}")
    
    def generate_challenge_response_data(self) -> Dict[str, Any]:
        """Generate data for challenge response visualization."""
        return {
            'learning_curve_data': {
                'performance_history': list(self.curve_analyzer.performance_history),
                'difficulty_history': list(self.curve_analyzer.difficulty_history),
                'episodes': list(range(len(self.curve_analyzer.performance_history)))
            },
            'difficulty_adjustments': self.difficulty_adjustments.copy(),
            'current_difficulty': self.current_difficulty,
            'evaluation_count': self.evaluation_count
        }
    
    def save_benchmark_results(self, filepath: str):
        """Save benchmark results to file."""
        results = self.generate_challenge_response_data()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def create_comprehensive_report(self, save_dir: str) -> Dict[str, str]:
        """Create comprehensive benchmark report with visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate learning curve
        curve_path = os.path.join(save_dir, 'learning_curve.png')
        self.curve_analyzer.visualize_learning_curve(curve_path)
        
        # Save data
        data_path = os.path.join(save_dir, 'benchmark_data.json')
        self.save_benchmark_results(data_path)
        
        # Generate summary report
        summary_path = os.path.join(save_dir, 'summary.md')
        self._generate_summary_report(summary_path)
        
        return {
            'learning_curve': curve_path,
            'data': data_path,
            'summary': summary_path
        }
    
    def _generate_summary_report(self, filepath: str):
        """Generate markdown summary report."""
        trend_analysis = self.curve_analyzer.analyze_trend()
        
        report = f"""# Dynamic Benchmark Report

## Summary
- **Current Difficulty**: {self.current_difficulty:.2f}
- **Total Evaluations**: {self.evaluation_count}
- **Difficulty Adjustments**: {len(self.difficulty_adjustments)}

## Performance Analysis
- **Current Performance**: {trend_analysis.get('current_mean', 0):.2f}
- **Performance Trend**: {trend_analysis.get('trend', 0):.4f}
- **Trend Confidence**: {trend_analysis.get('confidence', 0):.3f}

## Difficulty Progression
"""
        
        for i, adj in enumerate(self.difficulty_adjustments):
            report += f"- **Adjustment {i+1}**: {adj['old_difficulty']:.2f} → {adj['new_difficulty']:.2f} (Evaluation #{adj['evaluation_count']})\n"
        
        report += f"""
## Recommendations
"""
        
        if trend_analysis.get('trend', 0) > 0.1:
            report += "- Model is still improving, continue current training\n"
        elif self.curve_analyzer.detect_plateau():
            report += "- Performance has plateaued, consider increasing difficulty or changing strategy\n"
        else:
            report += "- Performance is stable, monitor for plateau\n"
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report generated: {filepath}")


# Utility functions for easy integration

def create_dynamic_benchmark(
    model: nn.Module,
    task_type: str = 'pattern_recognition',
    config: Optional[DynamicBenchmarkConfig] = None
) -> DynamicBenchmarkSystem:
    """Create a dynamic benchmark system with specified task type."""
    if config is None:
        config = DynamicBenchmarkConfig()
    
    if task_type == 'pattern_recognition':
        task = AdversarialPatternRecognitionTask()
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return DynamicBenchmarkSystem(config, task, model)


def run_dynamic_benchmark(
    model: nn.Module,
    num_evaluations: int = 100,
    task_type: str = 'pattern_recognition',
    config: Optional[DynamicBenchmarkConfig] = None,
    save_results: bool = True,
    results_dir: str = 'benchmark_results'
) -> Dict[str, Any]:
    """Run a complete dynamic benchmark evaluation."""
    benchmark_system = create_dynamic_benchmark(model, task_type, config)
    
    results = []
    for i in range(num_evaluations):
        evaluation_result = benchmark_system.evaluate_model()
        results.append(evaluation_result)
        
        if (i + 1) % 20 == 0:
            logger.info(f"Evaluation {i+1}/{num_evaluations}: "
                       f"Score={evaluation_result['standard_score']:.2f}, "
                       f"Difficulty={evaluation_result['difficulty']:.2f}")
    
    # Generate final report
    final_results = {
        'evaluation_results': results,
        'summary_statistics': benchmark_system.generate_challenge_response_data()
    }
    
    if save_results:
        report_paths = benchmark_system.create_comprehensive_report(results_dir)
        final_results['report_paths'] = report_paths
    
    return final_results