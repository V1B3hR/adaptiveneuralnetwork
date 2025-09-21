"""
Configuration system for the Adaptive AutoML Engine.

This module provides configuration classes that integrate with the
neuromorphic principles of the adaptive neural network.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json


@dataclass
class PreprocessingConfig:
    """Configuration for automated data preprocessing."""
    
    # Missing value imputation
    imputation_strategy: str = "adaptive"  # "simple", "knn", "iterative", "adaptive"
    imputation_fill_value: Optional[Union[str, float]] = None
    imputation_n_neighbors: int = 5
    imputation_max_iter: int = 10
    
    # Outlier detection
    outlier_detection_method: str = "spike_based"  # "iqr", "isolation_forest", "z_score", "spike_based"
    outlier_contamination: float = 0.1
    outlier_threshold: float = 3.0
    spike_energy_threshold: float = 0.8
    
    # Feature scaling
    scaling_method: str = "energy_aware"  # "standard", "minmax", "robust", "energy_aware"
    scaling_feature_range: tuple = (0, 1)
    
    # Categorical encoding
    categorical_encoding: str = "auto"  # "onehot", "target", "ordinal", "auto"
    handle_unknown_categories: str = "ignore"
    
    # Data type inference
    auto_type_inference: bool = True
    datetime_inference: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Configuration for automated feature engineering."""
    
    # Polynomial features
    polynomial_degree: int = 2
    polynomial_interaction_only: bool = False
    polynomial_include_bias: bool = False
    
    # Interaction terms
    max_interaction_degree: int = 2
    energy_driven_interactions: bool = True
    interaction_threshold: float = 0.1
    
    # Statistical transformations
    enable_log_transform: bool = True
    enable_sqrt_transform: bool = True
    enable_boxcox_transform: bool = True
    transform_threshold: float = 0.05
    
    # Phase-based features
    enable_phase_features: bool = True
    phase_window_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    phase_energy_scaling: bool = True
    
    # Temporal features (for time-series)
    enable_temporal_features: bool = False
    lag_features: List[int] = field(default_factory=lambda: [1, 2, 3])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14])


@dataclass
class FeatureSelectionConfig:
    """Configuration for automated feature selection."""
    
    # Univariate selection
    univariate_method: str = "energy_guided"  # "chi2", "f_test", "mutual_info", "energy_guided"
    univariate_k_best: int = 100
    
    # Recursive feature elimination
    enable_rfe: bool = True
    rfe_n_features: int = 50
    rfe_step: int = 1
    
    # Correlation filtering
    correlation_threshold: float = 0.95
    correlation_method: str = "pearson"
    
    # Mutual information
    mi_discrete_features: str = "auto"
    mi_n_neighbors: int = 3
    neuromorphic_mi_estimation: bool = True
    
    # Energy-based selection
    energy_importance_threshold: float = 0.01
    phase_stability_weight: float = 0.3
    convergence_stability_weight: float = 0.7


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Search strategy
    search_strategy: str = "energy_aware_bayesian"  # "grid", "random", "bayesian", "energy_aware_bayesian"
    max_evaluations: int = 100
    cv_folds: int = 5
    
    # Energy-aware optimization
    energy_convergence_patience: int = 10
    energy_stability_threshold: float = 0.01
    phase_transition_monitoring: bool = True
    
    # Learning rate schedules
    adaptive_lr_scheduling: bool = True
    lr_reduction_factor: float = 0.5
    lr_patience: int = 5
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_delta: float = 0.001
    energy_based_stopping: bool = True
    
    # Bayesian optimization specific
    acquisition_function: str = "expected_improvement"
    exploration_weight: float = 0.1


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestration."""
    
    # Data splitting
    test_size: float = 0.2
    validation_size: float = 0.2
    stratify: bool = True
    random_state: int = 42
    
    # Cross-validation
    cv_strategy: str = "stratified_kfold"
    cv_folds: int = 5
    cv_shuffle: bool = True
    
    # Data leakage prevention
    prevent_data_leakage: bool = True
    fit_on_train_only: bool = True
    validate_preprocessing: bool = True
    
    # Pipeline caching
    enable_caching: bool = True
    cache_directory: Optional[str] = None
    
    # Parallel processing
    n_jobs: int = -1
    parallel_backend: str = "threading"


@dataclass
class AutoMLConfig:
    """Main configuration class for the Adaptive AutoML Engine."""
    
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Global settings
    enable_neuromorphic_integration: bool = True
    energy_awareness_level: float = 0.8
    phase_transition_sensitivity: float = 0.5
    
    # Logging and monitoring
    verbose: bool = True
    log_level: str = "INFO"
    track_energy_consumption: bool = True
    save_intermediate_results: bool = True
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        file_path = Path(file_path)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'AutoMLConfig':
        """Load configuration from a JSON file."""
        file_path = Path(file_path)
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'preprocessing': self.preprocessing.__dict__,
            'feature_engineering': self.feature_engineering.__dict__,
            'feature_selection': self.feature_selection.__dict__,
            'optimization': self.optimization.__dict__,
            'pipeline': self.pipeline.__dict__,
            'enable_neuromorphic_integration': self.enable_neuromorphic_integration,
            'energy_awareness_level': self.energy_awareness_level,
            'phase_transition_sensitivity': self.phase_transition_sensitivity,
            'verbose': self.verbose,
            'log_level': self.log_level,
            'track_energy_consumption': self.track_energy_consumption,
            'save_intermediate_results': self.save_intermediate_results
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoMLConfig':
        """Create configuration from dictionary."""
        preprocessing = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        feature_engineering = FeatureEngineeringConfig(**config_dict.get('feature_engineering', {}))
        feature_selection = FeatureSelectionConfig(**config_dict.get('feature_selection', {}))
        optimization = OptimizationConfig(**config_dict.get('optimization', {}))
        pipeline = PipelineConfig(**config_dict.get('pipeline', {}))
        
        # Extract global settings
        global_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['preprocessing', 'feature_engineering', 'feature_selection', 
                                     'optimization', 'pipeline']}
        
        return cls(
            preprocessing=preprocessing,
            feature_engineering=feature_engineering,
            feature_selection=feature_selection,
            optimization=optimization,
            pipeline=pipeline,
            **global_settings
        )


def create_default_config() -> AutoMLConfig:
    """Create a default AutoML configuration."""
    return AutoMLConfig()


def create_minimal_config() -> AutoMLConfig:
    """Create a minimal AutoML configuration for quick testing."""
    config = AutoMLConfig()
    
    # Reduce complexity for quick testing
    config.feature_engineering.polynomial_degree = 1
    config.feature_engineering.enable_phase_features = False
    config.feature_selection.univariate_k_best = 20
    config.feature_selection.enable_rfe = False
    config.optimization.max_evaluations = 10
    config.optimization.cv_folds = 3
    
    return config


def create_high_performance_config() -> AutoMLConfig:
    """Create a high-performance AutoML configuration."""
    config = AutoMLConfig()
    
    # Enable all advanced features
    config.feature_engineering.polynomial_degree = 3
    config.feature_engineering.enable_phase_features = True
    config.feature_engineering.energy_driven_interactions = True
    
    config.feature_selection.enable_rfe = True
    config.feature_selection.neuromorphic_mi_estimation = True
    
    config.optimization.max_evaluations = 500
    config.optimization.phase_transition_monitoring = True
    config.optimization.energy_based_stopping = True
    
    return config