"""
Main AutoML Engine for Adaptive Neural Networks.

This module provides the core AdaptiveAutoMLEngine that orchestrates
all AutoML components while leveraging neuromorphic principles.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import warnings

from .config import AutoMLConfig
from .preprocessing import NeuromorphicPreprocessor
from .feature_engineering import NeuromorphicFeatureEngineer
from .feature_selection import AdaptiveFeatureSelector
from .optimization import EnergyAwareHyperparameterOptimizer
from .pipeline import NeuromorphicPipeline

# Import neuromorphic components from the main package
try:
    from ..core.dynamics import AdaptiveDynamics
    from ..core.nodes import NodeState
    from ..api.model import AdaptiveModel
except ImportError:
    warnings.warn("Neuromorphic components not available. Some features will be disabled.")
    AdaptiveDynamics = None
    NodeState = None
    AdaptiveModel = None

logger = logging.getLogger(__name__)


class AdaptiveAutoMLEngine:
    """
    Main AutoML engine that leverages neuromorphic principles for intelligent
    data preprocessing, feature engineering, and hyperparameter optimization.
    
    This engine integrates energy dynamics, phase transitions, and adaptive
    thresholds from the neuromorphic network to guide AutoML decisions.
    """
    
    def __init__(
        self,
        config: Optional[AutoMLConfig] = None,
        random_state: int = 42
    ):
        """
        Initialize the Adaptive AutoML Engine.
        
        Args:
            config: AutoML configuration object
            random_state: Random seed for reproducibility
        """
        self.config = config or AutoMLConfig()
        self.random_state = random_state
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_selector = None
        self.optimizer = None
        self.pipeline = None
        
        # Neuromorphic integration
        self.neuromorphic_enabled = self.config.enable_neuromorphic_integration and AdaptiveDynamics is not None
        self.energy_monitor = None
        self.phase_tracker = None
        
        # Results storage
        self.preprocessing_results = {}
        self.feature_engineering_results = {}
        self.feature_selection_results = {}
        self.optimization_results = {}
        
        logger.info(f"Initialized AdaptiveAutoMLEngine with neuromorphic integration: {self.neuromorphic_enabled}")
    
    def _setup_logging(self):
        """Set up logging based on configuration."""
        level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(level=level)
        
        if self.config.verbose:
            logger.setLevel(logging.DEBUG)
    
    def auto_preprocess_pipeline(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray, str],
        fit_on_subset: bool = False,
        subset_size: float = 0.1
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Automatically preprocess data using energy-aware strategies.
        
        Args:
            data: Input data (DataFrame or array)
            target: Target variable (Series, array, or column name if data is DataFrame)
            fit_on_subset: Whether to fit preprocessing on a subset for efficiency
            subset_size: Size of subset to use for fitting (if fit_on_subset=True)
            
        Returns:
            Tuple of (processed_data, preprocessing_info)
        """
        logger.info("Starting automated preprocessing pipeline")
        
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
        
        # Handle target extraction
        if isinstance(target, str) and target in data.columns:
            y = data[target]
            X = data.drop(columns=[target])
        else:
            X = data.copy()
            y = target if hasattr(target, '__len__') else pd.Series(target)
        
        # Initialize preprocessor
        self.preprocessor = NeuromorphicPreprocessor(
            config=self.config.preprocessing,
            neuromorphic_enabled=self.neuromorphic_enabled,
            random_state=self.random_state
        )
        
        # Fit and transform
        if fit_on_subset and len(X) > 1000:
            subset_idx = np.random.choice(len(X), size=int(len(X) * subset_size), replace=False)
            X_subset = X.iloc[subset_idx]
            y_subset = y.iloc[subset_idx] if hasattr(y, 'iloc') else y[subset_idx]
            
            self.preprocessor.fit(X_subset, y_subset)
        else:
            self.preprocessor.fit(X, y)
        
        X_processed = self.preprocessor.transform(X)
        
        # Store results
        self.preprocessing_results = {
            'original_shape': X.shape,
            'processed_shape': X_processed.shape,
            'missing_values_handled': self.preprocessor.missing_values_count,
            'outliers_detected': self.preprocessor.outliers_count,
            'features_scaled': self.preprocessor.scaled_features,
            'categorical_encoded': self.preprocessor.categorical_features,
            'energy_consumption': getattr(self.preprocessor, 'energy_consumption', 0.0)
        }
        
        logger.info(f"Preprocessing completed: {X.shape} -> {X_processed.shape}")
        return X_processed, self.preprocessing_results
    
    def neuromorphic_feature_engineering(
        self,
        data: pd.DataFrame,
        target: Optional[Union[pd.Series, np.ndarray]] = None,
        energy_guided: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform neuromorphic-inspired feature engineering.
        
        Args:
            data: Input data
            target: Target variable (optional, for supervised feature engineering)
            energy_guided: Whether to use energy dynamics to guide feature creation
            
        Returns:
            Tuple of (engineered_data, engineering_info)
        """
        logger.info("Starting neuromorphic feature engineering")
        
        # Initialize feature engineer
        self.feature_engineer = NeuromorphicFeatureEngineer(
            config=self.config.feature_engineering,
            neuromorphic_enabled=self.neuromorphic_enabled,
            random_state=self.random_state
        )
        
        # Fit and transform
        self.feature_engineer.fit(data, target)
        data_engineered = self.feature_engineer.transform(data)
        
        # Store results
        self.feature_engineering_results = {
            'original_features': data.shape[1],
            'engineered_features': data_engineered.shape[1],
            'polynomial_features_added': getattr(self.feature_engineer, 'polynomial_features_count', 0),
            'interaction_features_added': getattr(self.feature_engineer, 'interaction_features_count', 0),
            'phase_features_added': getattr(self.feature_engineer, 'phase_features_count', 0),
            'energy_features_added': getattr(self.feature_engineer, 'energy_features_count', 0),
            'energy_consumption': getattr(self.feature_engineer, 'energy_consumption', 0.0)
        }
        
        logger.info(f"Feature engineering completed: {data.shape[1]} -> {data_engineered.shape[1]} features")
        return data_engineered, self.feature_engineering_results
    
    def adaptive_feature_selection(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        max_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform adaptive feature selection using energy dynamics.
        
        Args:
            X: Input features
            y: Target variable
            max_features: Maximum number of features to select
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        logger.info("Starting adaptive feature selection")
        
        # Initialize feature selector
        self.feature_selector = AdaptiveFeatureSelector(
            config=self.config.feature_selection,
            neuromorphic_enabled=self.neuromorphic_enabled,
            max_features=max_features,
            random_state=self.random_state
        )
        
        # Fit and transform
        self.feature_selector.fit(X, y)
        X_selected = self.feature_selector.transform(X)
        
        # Store results
        self.feature_selection_results = {
            'original_features': X.shape[1],
            'selected_features': X_selected.shape[1],
            'selection_ratio': X_selected.shape[1] / X.shape[1],
            'selected_feature_names': X_selected.columns.tolist() if hasattr(X_selected, 'columns') else None,
            'feature_scores': getattr(self.feature_selector, 'feature_scores_', {}),
            'energy_importance': getattr(self.feature_selector, 'energy_importance_', {}),
            'energy_consumption': getattr(self.feature_selector, 'energy_consumption', 0.0)
        }
        
        logger.info(f"Feature selection completed: {X.shape[1]} -> {X_selected.shape[1]} features")
        return X_selected, self.feature_selection_results
    
    def auto_hyperparameter_optimization(
        self,
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        param_space: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform energy-aware hyperparameter optimization.
        
        Args:
            model_config: Configuration for the model to optimize
            X: Input features
            y: Target variable
            param_space: Parameter space to search (if None, uses defaults)
            
        Returns:
            Tuple of (best_params, optimization_info)
        """
        logger.info("Starting energy-aware hyperparameter optimization")
        
        # Initialize optimizer
        self.optimizer = EnergyAwareHyperparameterOptimizer(
            config=self.config.optimization,
            neuromorphic_enabled=self.neuromorphic_enabled,
            random_state=self.random_state
        )
        
        # Run optimization
        best_params, optimization_info = self.optimizer.optimize(
            model_config=model_config,
            X=X,
            y=y,
            param_space=param_space
        )
        
        # Store results
        self.optimization_results = optimization_info
        
        logger.info(f"Hyperparameter optimization completed with score: {optimization_info.get('best_score', 'N/A')}")
        return best_params, optimization_info
    
    def create_full_pipeline(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray, str],
        model_config: Optional[Dict[str, Any]] = None
    ) -> NeuromorphicPipeline:
        """
        Create a complete AutoML pipeline.
        
        Args:
            data: Input data
            target: Target variable
            model_config: Model configuration (optional)
            
        Returns:
            Configured NeuromorphicPipeline
        """
        logger.info("Creating complete AutoML pipeline")
        
        # Initialize pipeline
        self.pipeline = NeuromorphicPipeline(
            config=self.config.pipeline,
            neuromorphic_enabled=self.neuromorphic_enabled,
            random_state=self.random_state
        )
        
        # Add preprocessing step
        if self.preprocessor is None:
            self.preprocessor = NeuromorphicPreprocessor(
                config=self.config.preprocessing,
                neuromorphic_enabled=self.neuromorphic_enabled,
                random_state=self.random_state
            )
        
        self.pipeline.add_step('preprocessor', self.preprocessor)
        
        # Add feature engineering step
        if self.feature_engineer is None:
            self.feature_engineer = NeuromorphicFeatureEngineer(
                config=self.config.feature_engineering,
                neuromorphic_enabled=self.neuromorphic_enabled,
                random_state=self.random_state
            )
        
        self.pipeline.add_step('feature_engineer', self.feature_engineer)
        
        # Add feature selection step
        if self.feature_selector is None:
            self.feature_selector = AdaptiveFeatureSelector(
                config=self.config.feature_selection,
                neuromorphic_enabled=self.neuromorphic_enabled,
                random_state=self.random_state
            )
        
        self.pipeline.add_step('feature_selector', self.feature_selector)
        
        logger.info("Pipeline created with all AutoML components")
        return self.pipeline
    
    def fit_transform_all(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray, str],
        optimize_hyperparameters: bool = True,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Union[pd.Series, np.ndarray], Dict[str, Any]]:
        """
        Run the complete AutoML pipeline on the data.
        
        Args:
            data: Input data
            target: Target variable
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            model_config: Model configuration for hyperparameter optimization
            
        Returns:
            Tuple of (processed_features, target, complete_results)
        """
        logger.info("Running complete AutoML pipeline")
        
        # Step 1: Preprocessing
        X_processed, preprocessing_info = self.auto_preprocess_pipeline(data, target)
        
        # Extract target if it was part of the data
        if isinstance(target, str) and target in data.columns:
            y = data[target]
        else:
            y = target
        
        # Step 2: Feature Engineering
        X_engineered, engineering_info = self.neuromorphic_feature_engineering(X_processed, y)
        
        # Step 3: Feature Selection
        X_selected, selection_info = self.adaptive_feature_selection(X_engineered, y)
        
        # Step 4: Hyperparameter Optimization (optional)
        optimization_info = {}
        if optimize_hyperparameters and model_config is not None:
            _, optimization_info = self.auto_hyperparameter_optimization(
                model_config, X_selected, y
            )
        
        # Compile complete results
        complete_results = {
            'preprocessing': preprocessing_info,
            'feature_engineering': engineering_info,
            'feature_selection': selection_info,
            'optimization': optimization_info,
            'total_energy_consumption': sum([
                preprocessing_info.get('energy_consumption', 0),
                engineering_info.get('energy_consumption', 0),
                selection_info.get('energy_consumption', 0),
                optimization_info.get('energy_consumption', 0)
            ])
        }
        
        logger.info("Complete AutoML pipeline finished successfully")
        return X_selected, y, complete_results
    
    def save_results(self, file_path: Union[str, Path]) -> None:
        """Save all AutoML results to a file."""
        results = {
            'config': self.config.to_dict(),
            'preprocessing_results': self.preprocessing_results,
            'feature_engineering_results': self.feature_engineering_results,
            'feature_selection_results': self.feature_selection_results,
            'optimization_results': self.optimization_results
        }
        
        file_path = Path(file_path)
        if file_path.suffix == '.json':
            import json
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        else:
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)
        
        logger.info(f"Results saved to {file_path}")
    
    def get_energy_report(self) -> Dict[str, Any]:
        """Get a detailed energy consumption report."""
        return {
            'preprocessing_energy': self.preprocessing_results.get('energy_consumption', 0),
            'feature_engineering_energy': self.feature_engineering_results.get('energy_consumption', 0),
            'feature_selection_energy': self.feature_selection_results.get('energy_consumption', 0),
            'optimization_energy': self.optimization_results.get('energy_consumption', 0),
            'total_energy': sum([
                self.preprocessing_results.get('energy_consumption', 0),
                self.feature_engineering_results.get('energy_consumption', 0),
                self.feature_selection_results.get('energy_consumption', 0),
                self.optimization_results.get('energy_consumption', 0)
            ]),
            'energy_efficiency_score': self._calculate_energy_efficiency()
        }
    
    def _calculate_energy_efficiency(self) -> float:
        """Calculate an energy efficiency score for the AutoML process."""
        total_energy = sum([
            self.preprocessing_results.get('energy_consumption', 0),
            self.feature_engineering_results.get('energy_consumption', 0),
            self.feature_selection_results.get('energy_consumption', 0),
            self.optimization_results.get('energy_consumption', 0)
        ])
        
        # Simple efficiency metric: more features processed per unit energy is better
        total_features_processed = (
            self.preprocessing_results.get('processed_shape', (0, 0))[1] +
            self.feature_engineering_results.get('engineered_features', 0) +
            self.feature_selection_results.get('selected_features', 0)
        )
        
        if total_energy > 0:
            return total_features_processed / total_energy
        else:
            return float('inf')  # Perfect efficiency if no energy consumed


def create_automl_engine(
    config_type: str = "default",
    **kwargs
) -> AdaptiveAutoMLEngine:
    """
    Factory function to create an AutoML engine with predefined configurations.
    
    Args:
        config_type: Type of configuration ("default", "minimal", "high_performance")
        **kwargs: Additional arguments to pass to the engine
        
    Returns:
        Configured AdaptiveAutoMLEngine instance
    """
    from .config import create_default_config, create_minimal_config, create_high_performance_config
    
    config_creators = {
        "default": create_default_config,
        "minimal": create_minimal_config,
        "high_performance": create_high_performance_config
    }
    
    if config_type not in config_creators:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    config = config_creators[config_type]()
    return AdaptiveAutoMLEngine(config=config, **kwargs)