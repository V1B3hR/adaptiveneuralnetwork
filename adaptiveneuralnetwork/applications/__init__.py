"""
3rd Generation Neuromorphic Applications.

This module provides advanced applications leveraging 3rd generation
neuromorphic computing capabilities for real-world problems.
"""

from .continual_learning import ContinualLearningSystem, ContinualLearningConfig
from .few_shot_learning import FewShotLearningSystem, FewShotLearningConfig  
from .sensory_processing import SensoryProcessingPipeline, SensoryConfig

__all__ = [
    'ContinualLearningSystem',
    'ContinualLearningConfig', 
    'FewShotLearningSystem',
    'FewShotLearningConfig',
    'SensoryProcessingPipeline',
    'SensoryConfig'
]