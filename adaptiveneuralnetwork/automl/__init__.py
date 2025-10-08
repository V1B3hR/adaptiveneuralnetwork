"""
Adaptive AutoML Engine for Neuromorphic Neural Networks.

This module provides comprehensive AutoML capabilities that leverage
the unique neuromorphic principles of the adaptive neural network,
including energy dynamics, phase transitions, and adaptive thresholds.
"""

from .config import AutoMLConfig
from .engine import AdaptiveAutoMLEngine, create_automl_engine
from .feature_engineering import (
    EnergyDrivenInteractionTerms,
    NeuromorphicFeatureEngineer,
    PhaseBasedFeatureGenerator,
)
from .feature_selection import (
    AdaptiveFeatureSelector,
    EnergyGuidedFeatureImportance,
    NeuromorphicMutualInformation,
)
from .optimization import (
    AdaptiveEarlyStopping,
    EnergyAwareHyperparameterOptimizer,
    PhaseTransitionOptimizer,
)
from .pipeline import AutoMLWorkflow, DataLeakagePreventionMixin, NeuromorphicPipeline
from .preprocessing import (
    AdaptiveFeatureScaler,
    EnergyAwareMissingValueImputer,
    NeuromorphicPreprocessor,
    SpikeBasedOutlierDetector,
)

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
