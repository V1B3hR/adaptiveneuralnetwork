"""
3rd Generation Neuromorphic Applications.

This module provides advanced applications leveraging 3rd generation
neuromorphic computing capabilities for real-world problems, including
Phase 1: Adaptive Learning & Continual Improvement implementations.
"""

from .continual_learning import ContinualLearningSystem, ContinualLearningConfig
from .few_shot_learning import FewShotLearningSystem, FewShotLearningConfig  
from .sensory_processing import SensoryProcessingPipeline, SensoryConfig

# Phase 1: Adaptive Learning & Continual Improvement
from .self_supervised_learning import (
    SelfSupervisedConfig,
    SelfSupervisedLearningSystem,
    SignalPredictor,
    ContrastiveRepresentationLearner
)

from .curriculum_learning import (
    CurriculumConfig,
    CurriculumLearningSystem,
    DifficultyController,
    SyntheticTaskGenerator,
    create_curriculum_system,
    train_with_curriculum
)

from .enhanced_memory_systems import (
    EnhancedMemoryConfig,
    DynamicPriorityBuffer,
    TimeSeriesAnalyzer,
    EventDrivenLearningSystem
)

from .dynamic_benchmarking import (
    DynamicBenchmarkConfig,
    DynamicBenchmarkSystem,
    AdversarialPatternRecognitionTask,
    AdversarialTestGenerator,
    LearningCurveAnalyzer,
    create_dynamic_benchmark,
    run_dynamic_benchmark
)

__all__ = [
    # Existing applications
    'ContinualLearningSystem',
    'ContinualLearningConfig', 
    'FewShotLearningSystem',
    'FewShotLearningConfig',
    'SensoryProcessingPipeline',
    'SensoryConfig',
    
    # Phase 1: Self-supervised learning
    'SelfSupervisedConfig',
    'SelfSupervisedLearningSystem',
    'SignalPredictor',
    'ContrastiveRepresentationLearner',
    
    # Phase 1: Curriculum learning
    'CurriculumConfig',
    'CurriculumLearningSystem',
    'DifficultyController',
    'SyntheticTaskGenerator',
    'create_curriculum_system',
    'train_with_curriculum',
    
    # Phase 1: Enhanced memory systems
    'EnhancedMemoryConfig',
    'DynamicPriorityBuffer',
    'TimeSeriesAnalyzer',
    'EventDrivenLearningSystem',
    
    # Phase 1: Dynamic benchmarking
    'DynamicBenchmarkConfig',
    'DynamicBenchmarkSystem',
    'AdversarialPatternRecognitionTask',
    'AdversarialTestGenerator',
    'LearningCurveAnalyzer',
    'create_dynamic_benchmark',
    'run_dynamic_benchmark'
]