"""
Adaptive AutoML Engine for Neuromorphic Neural Networks.

This module provides comprehensive AutoML capabilities that leverage
the unique neuromorphic principles of the adaptive neural network,
including energy dynamics, phase transitions, and adaptive thresholds.
"""

from .engine import AdaptiveAutoMLEngine, create_automl_engine
from .preprocessing import (
    NeuromorphicPreprocessor,
    EnergyAwareMissingValueImputer,
    SpikeBasedOutlierDetector,
    AdaptiveFeatureScaler
)
from .feature_engineering import (
    NeuromorphicFeatureEngineer,
    PhaseBasedFeatureGenerator,
    EnergyDrivenInteractionTerms
)
from .feature_selection import (
    AdaptiveFeatureSelector,
    EnergyGuidedFeatureImportance,
    NeuromorphicMutualInformation
)
from .optimization import (
    EnergyAwareHyperparameterOptimizer,
    PhaseTransitionOptimizer,
    AdaptiveEarlyStopping
)
from .pipeline import (
    NeuromorphicPipeline,
    AutoMLWorkflow,
    DataLeakagePreventionMixin
)
from .config import AutoMLConfig

__all__ = [
    # Core engine
    'AdaptiveAutoMLEngine',
    'create_automl_engine',
    'AutoMLConfig',
    'NeuromorphicPreprocessor',
    'EnergyAwareMissingValueImputer',
    'SpikeBasedOutlierDetector',
    'AdaptiveFeatureScaler',
    
    # Feature engineering
    'NeuromorphicFeatureEngineer',
    'PhaseBasedFeatureGenerator',
    'EnergyDrivenInteractionTerms',
    
    # Feature selection
    'AdaptiveFeatureSelector',
    'EnergyGuidedFeatureImportance',
    'NeuromorphicMutualInformation',
    
    # Hyperparameter optimization
    'EnergyAwareHyperparameterOptimizer',
    'PhaseTransitionOptimizer',
    'AdaptiveEarlyStopping',
    
    # Pipeline orchestration
    'NeuromorphicPipeline',
    'AutoMLWorkflow',
    'DataLeakagePreventionMixin'
]
