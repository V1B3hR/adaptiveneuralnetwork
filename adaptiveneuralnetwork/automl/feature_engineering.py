"""
Neuromorphic-inspired feature engineering components.

This module provides automated feature engineering that leverages
phase transitions, energy dynamics, and spike patterns from neuromorphic networks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import warnings
from itertools import combinations
from scipy import stats
from scipy.special import boxcox

try:
    from sklearn.preprocessing import PolynomialFeatures
    HAS_SKLEARN = True
except ImportError:
    warnings.warn("scikit-learn not available. Some feature engineering will be limited.")
    HAS_SKLEARN = False

from .config import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class PhaseBasedFeatureGenerator:
    """
    Generate features based on different phases of neuromorphic dynamics.
    
    This component creates phase-specific features that capture different
    aspects of the data based on energy states and transitions.
    """
    
    def __init__(
        self,
        phase_window_sizes: List[int] = [3, 5, 7],
        energy_scaling: bool = True,
        random_state: int = 42
    ):
        self.phase_window_sizes = phase_window_sizes
        self.energy_scaling = energy_scaling
        self.random_state = random_state
        
        self.phase_features_ = {}
        self.energy_consumption = 0.0
        self.phase_statistics_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PhaseBasedFeatureGenerator':
        """Fit phase-based feature generator."""
        logger.debug("Fitting phase-based feature generator")
        
        # Analyze phase characteristics for each numeric feature
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            self.phase_statistics_[column] = self._analyze_phase_characteristics(X[column])
            self.energy_consumption += 0.02
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate phase-based features."""
        X_phase = X.copy()
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in self.phase_statistics_:
                phase_features = self._generate_phase_features(X[column], column)
                
                # Add phase features to dataframe
                for feature_name, feature_values in phase_features.items():
                    X_phase[feature_name] = feature_values
                
                self.energy_consumption += 0.01
        
        return X_phase
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _analyze_phase_characteristics(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze phase characteristics of a feature."""
        # Calculate energy levels (rolling variance)
        phase_stats = {}
        
        for window_size in self.phase_window_sizes:
            if len(series) >= window_size:
                rolling_energy = series.rolling(window=window_size, center=True).var()
                
                phase_stats[f'window_{window_size}'] = {
                    'mean_energy': rolling_energy.mean(),
                    'energy_std': rolling_energy.std(),
                    'energy_percentiles': np.percentile(rolling_energy.dropna(), [25, 50, 75])
                }
        
        return phase_stats
    
    def _generate_phase_features(self, series: pd.Series, column: str) -> Dict[str, pd.Series]:
        """Generate phase-based features for a column."""
        phase_features = {}
        
        for window_size in self.phase_window_sizes:
            if len(series) >= window_size:
                # Rolling statistics representing different phases
                rolling_mean = series.rolling(window=window_size, center=True).mean()
                rolling_std = series.rolling(window=window_size, center=True).std()
                rolling_energy = series.rolling(window=window_size, center=True).var()
                
                # Phase transition indicators
                energy_gradient = rolling_energy.diff()
                phase_transition = (energy_gradient.abs() > energy_gradient.std()).astype(int)
                
                # Energy-scaled features
                if self.energy_scaling:
                    energy_weight = rolling_energy / (rolling_energy.max() + 1e-8)
                    phase_features[f'{column}_phase_{window_size}_energy_weighted_mean'] = rolling_mean * energy_weight
                    phase_features[f'{column}_phase_{window_size}_energy_weighted_std'] = rolling_std * energy_weight
                
                # Basic phase features
                phase_features[f'{column}_phase_{window_size}_mean'] = rolling_mean
                phase_features[f'{column}_phase_{window_size}_std'] = rolling_std
                phase_features[f'{column}_phase_{window_size}_energy'] = rolling_energy
                phase_features[f'{column}_phase_{window_size}_transition'] = phase_transition
                
                # Phase stability indicator
                stability = 1.0 / (1.0 + rolling_std.fillna(0))
                phase_features[f'{column}_phase_{window_size}_stability'] = stability
        
        # Fill NaN values with feature means
        for feature_name, feature_series in phase_features.items():
            phase_features[feature_name] = feature_series.fillna(feature_series.mean())
        
        return phase_features


class EnergyDrivenInteractionTerms:
    """
    Create interaction terms guided by energy dynamics between features.
    """
    
    def __init__(
        self,
        max_interaction_degree: int = 2,
        interaction_threshold: float = 0.1,
        energy_guided: bool = True,
        random_state: int = 42
    ):
        self.max_interaction_degree = max_interaction_degree
        self.interaction_threshold = interaction_threshold
        self.energy_guided = energy_guided
        self.random_state = random_state
        
        self.interaction_pairs_ = []
        self.energy_correlations_ = {}
        self.energy_consumption = 0.0
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EnergyDrivenInteractionTerms':
        """Fit interaction term generator."""
        logger.debug("Fitting energy-driven interaction terms")
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if self.energy_guided:
            # Calculate energy correlations between features
            self._calculate_energy_correlations(X[numeric_columns])
            
            # Select interaction pairs based on energy correlations
            self._select_interaction_pairs(numeric_columns)
        else:
            # Select all possible pairs up to max degree
            self.interaction_pairs_ = list(combinations(numeric_columns, 2))
        
        self.energy_consumption += 0.05 * len(self.interaction_pairs_)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features."""
        X_interactions = X.copy()
        
        for pair in self.interaction_pairs_:
            feature1, feature2 = pair
            
            if feature1 in X.columns and feature2 in X.columns:
                # Multiplicative interaction
                interaction_name = f'{feature1}_x_{feature2}'
                X_interactions[interaction_name] = X[feature1] * X[feature2]
                
                # Additive interaction (difference)
                diff_name = f'{feature1}_diff_{feature2}'
                X_interactions[diff_name] = X[feature1] - X[feature2]
                
                # Energy-weighted interaction
                if self.energy_guided and pair in self.energy_correlations_:
                    energy_weight = abs(self.energy_correlations_[pair])
                    weighted_name = f'{feature1}_energy_weighted_{feature2}'
                    X_interactions[weighted_name] = (X[feature1] * X[feature2]) * energy_weight
                
                self.energy_consumption += 0.01
        
        return X_interactions
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _calculate_energy_correlations(self, X: pd.DataFrame):
        """Calculate energy-based correlations between features."""
        for pair in combinations(X.columns, 2):
            feature1, feature2 = pair
            
            # Calculate energy correlation using variance-based measure
            f1_energy = X[feature1].rolling(window=5, center=True).var().fillna(X[feature1].var())
            f2_energy = X[feature2].rolling(window=5, center=True).var().fillna(X[feature2].var())
            
            # Correlation between energy profiles
            energy_corr = np.corrcoef(f1_energy, f2_energy)[0, 1]
            
            if not np.isnan(energy_corr):
                self.energy_correlations_[pair] = energy_corr
    
    def _select_interaction_pairs(self, columns: pd.Index):
        """Select interaction pairs based on energy correlations."""
        # Sort pairs by absolute energy correlation
        sorted_pairs = sorted(
            self.energy_correlations_.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Select pairs above threshold
        self.interaction_pairs_ = [
            pair for pair, corr in sorted_pairs
            if abs(corr) > self.interaction_threshold
        ]
        
        # Limit to reasonable number of interactions
        max_interactions = min(len(self.interaction_pairs_), 50)
        self.interaction_pairs_ = self.interaction_pairs_[:max_interactions]


class StatisticalTransformationEngine:
    """
    Apply statistical transformations guided by energy dynamics.
    """
    
    def __init__(
        self,
        enable_log_transform: bool = True,
        enable_sqrt_transform: bool = True,
        enable_boxcox_transform: bool = True,
        transform_threshold: float = 0.05,
        random_state: int = 42
    ):
        self.enable_log_transform = enable_log_transform
        self.enable_sqrt_transform = enable_sqrt_transform
        self.enable_boxcox_transform = enable_boxcox_transform
        self.transform_threshold = transform_threshold
        self.random_state = random_state
        
        self.transformations_ = {}
        self.transform_params_ = {}
        self.energy_consumption = 0.0
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'StatisticalTransformationEngine':
        """Fit statistical transformations."""
        logger.debug("Fitting statistical transformations")
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = X[column].dropna()
            
            if len(series) > 0:
                # Analyze distribution characteristics
                skewness = abs(stats.skew(series))
                kurtosis = abs(stats.kurtosis(series))
                
                # Determine if transformation is beneficial
                if skewness > self.transform_threshold or kurtosis > 3:
                    transform_type = self._select_transformation(series, skewness, kurtosis)
                    
                    if transform_type:
                        self.transformations_[column] = transform_type
                        self.transform_params_[column] = self._fit_transformation(series, transform_type)
                        self.energy_consumption += 0.02
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply statistical transformations."""
        X_transformed = X.copy()
        
        for column, transform_type in self.transformations_.items():
            if column in X.columns:
                transformed_feature = self._apply_transformation(
                    X[column], transform_type, self.transform_params_[column]
                )
                
                # Add transformed feature with new name
                transform_name = f'{column}_{transform_type}_transformed'
                X_transformed[transform_name] = transformed_feature
                
                self.energy_consumption += 0.01
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _select_transformation(self, series: pd.Series, skewness: float, kurtosis: float) -> Optional[str]:
        """Select appropriate transformation based on distribution characteristics."""
        # Only transform positive values for log and sqrt
        has_positive_values = (series > 0).all()
        
        if skewness > 2.0 and has_positive_values:
            if self.enable_log_transform:
                return 'log'
            elif self.enable_sqrt_transform:
                return 'sqrt'
        elif skewness > 1.0 and has_positive_values:
            if self.enable_sqrt_transform:
                return 'sqrt'
        elif kurtosis > 5 and has_positive_values and self.enable_boxcox_transform:
            return 'boxcox'
        
        return None
    
    def _fit_transformation(self, series: pd.Series, transform_type: str) -> Dict[str, Any]:
        """Fit transformation parameters."""
        params = {}
        
        if transform_type == 'boxcox':
            try:
                # Find optimal lambda for Box-Cox transformation
                _, fitted_lambda = stats.boxcox(series[series > 0])
                params['lambda'] = fitted_lambda
            except:
                params['lambda'] = 0  # Equivalent to log transform
        elif transform_type == 'log':
            params['shift'] = max(0, -series.min() + 1e-8)  # Ensure positive values
        elif transform_type == 'sqrt':
            params['shift'] = max(0, -series.min())  # Ensure non-negative values
        
        return params
    
    def _apply_transformation(self, series: pd.Series, transform_type: str, params: Dict[str, Any]) -> pd.Series:
        """Apply transformation to series."""
        try:
            if transform_type == 'log':
                shift = params.get('shift', 0)
                return np.log(series + shift)
            elif transform_type == 'sqrt':
                shift = params.get('shift', 0)
                return np.sqrt(series + shift)
            elif transform_type == 'boxcox':
                lambda_param = params.get('lambda', 0)
                positive_series = series[series > 0]
                if lambda_param == 0:
                    return np.log(positive_series)
                else:
                    return (positive_series ** lambda_param - 1) / lambda_param
            else:
                return series
        except:
            # Return original series if transformation fails
            return series


class NeuromorphicFeatureEngineer:
    """
    Main feature engineering class that coordinates all neuromorphic components.
    """
    
    def __init__(
        self,
        config: Optional[FeatureEngineeringConfig] = None,
        neuromorphic_enabled: bool = True,
        random_state: int = 42
    ):
        self.config = config or FeatureEngineeringConfig()
        self.neuromorphic_enabled = neuromorphic_enabled
        self.random_state = random_state
        
        # Initialize components
        self.polynomial_generator = None
        self.phase_generator = None
        self.interaction_generator = None
        self.transform_engine = None
        
        # Track statistics
        self.polynomial_features_count = 0
        self.phase_features_count = 0
        self.interaction_features_count = 0
        self.energy_features_count = 0
        self.energy_consumption = 0.0
        self.fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NeuromorphicFeatureEngineer':
        """Fit all feature engineering components."""
        logger.info("Fitting neuromorphic feature engineer")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        original_feature_count = X.shape[1]
        
        # 1. Polynomial features
        if self.config.polynomial_degree > 1 and HAS_SKLEARN:
            self.polynomial_generator = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                interaction_only=self.config.polynomial_interaction_only,
                include_bias=self.config.polynomial_include_bias
            )
            # Fit on numeric columns only
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                self.polynomial_generator.fit(X[numeric_columns])
                self.polynomial_features_count = (
                    self.polynomial_generator.transform(X[numeric_columns]).shape[1] - len(numeric_columns)
                )
                self.energy_consumption += 0.1
        
        # 2. Phase-based features (neuromorphic)
        if self.config.enable_phase_features and self.neuromorphic_enabled:
            self.phase_generator = PhaseBasedFeatureGenerator(
                phase_window_sizes=self.config.phase_window_sizes,
                energy_scaling=self.config.phase_energy_scaling,
                random_state=self.random_state
            )
            self.phase_generator.fit(X, y)
            # Estimate phase features count
            numeric_count = len(X.select_dtypes(include=[np.number]).columns)
            self.phase_features_count = numeric_count * len(self.config.phase_window_sizes) * 5  # Approx
            self.energy_consumption += self.phase_generator.energy_consumption
        
        # 3. Energy-driven interaction terms
        if self.config.energy_driven_interactions:
            self.interaction_generator = EnergyDrivenInteractionTerms(
                max_interaction_degree=self.config.max_interaction_degree,
                interaction_threshold=self.config.interaction_threshold,
                energy_guided=self.neuromorphic_enabled,
                random_state=self.random_state
            )
            self.interaction_generator.fit(X, y)
            self.interaction_features_count = len(self.interaction_generator.interaction_pairs_) * 3  # Approx
            self.energy_consumption += self.interaction_generator.energy_consumption
        
        # 4. Statistical transformations
        if any([self.config.enable_log_transform, self.config.enable_sqrt_transform, self.config.enable_boxcox_transform]):
            self.transform_engine = StatisticalTransformationEngine(
                enable_log_transform=self.config.enable_log_transform,
                enable_sqrt_transform=self.config.enable_sqrt_transform,
                enable_boxcox_transform=self.config.enable_boxcox_transform,
                transform_threshold=self.config.transform_threshold,
                random_state=self.random_state
            )
            self.transform_engine.fit(X, y)
            self.energy_consumption += self.transform_engine.energy_consumption
        
        self.fitted_ = True
        logger.info(f"Feature engineering fitted. Estimated new features: {self.polynomial_features_count + self.phase_features_count + self.interaction_features_count}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        if not self.fitted_:
            raise ValueError("Feature engineer must be fitted before transforming")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X_engineered = X.copy()
        
        # 1. Apply polynomial features
        if self.polynomial_generator is not None:
            numeric_columns = X_engineered.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                poly_features = self.polynomial_generator.transform(X_engineered[numeric_columns])
                poly_feature_names = self.polynomial_generator.get_feature_names_out(numeric_columns)
                
                # Add new polynomial features (exclude original features)
                original_count = len(numeric_columns)
                new_poly_features = poly_features[:, original_count:]
                new_poly_names = poly_feature_names[original_count:]
                
                poly_df = pd.DataFrame(
                    new_poly_features,
                    columns=new_poly_names,
                    index=X_engineered.index
                )
                X_engineered = pd.concat([X_engineered, poly_df], axis=1)
        
        # 2. Apply phase-based features
        if self.phase_generator is not None:
            X_engineered = self.phase_generator.transform(X_engineered)
        
        # 3. Apply interaction terms
        if self.interaction_generator is not None:
            X_engineered = self.interaction_generator.transform(X_engineered)
        
        # 4. Apply statistical transformations
        if self.transform_engine is not None:
            X_engineered = self.transform_engine.transform(X_engineered)
        
        return X_engineered
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_importance_by_energy(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance based on energy contributions."""
        importance_scores = {}
        
        # Phase-based importance
        if self.phase_generator is not None:
            for column in X.select_dtypes(include=[np.number]).columns:
                if f'{column}_phase_5_energy' in X.columns:  # Use 5-window as representative
                    energy_feature = X[f'{column}_phase_5_energy']
                    importance_scores[column] = energy_feature.std() / (energy_feature.mean() + 1e-8)
        
        # Interaction importance
        if self.interaction_generator is not None:
            for pair in self.interaction_generator.interaction_pairs_:
                feature1, feature2 = pair
                interaction_name = f'{feature1}_x_{feature2}'
                if interaction_name in X.columns:
                    interaction_values = X[interaction_name]
                    importance_scores[interaction_name] = interaction_values.std() / (abs(interaction_values.mean()) + 1e-8)
        
        return importance_scores