"""
Neuromorphic-inspired data preprocessing components.

This module provides automated preprocessing capabilities that leverage
energy dynamics and spike-based processing from neuromorphic networks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

# Core scientific computing
from scipy import stats
from scipy.spatial.distance import cdist
try:
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    warnings.warn("scikit-learn not available. Some preprocessing features will be limited.")
    HAS_SKLEARN = False

from .config import PreprocessingConfig

logger = logging.getLogger(__name__)


class EnergyAwareMissingValueImputer:
    """
    Missing value imputer that uses energy dynamics to guide imputation strategy.
    """
    
    def __init__(
        self,
        strategy: str = "adaptive",
        fill_value: Optional[Union[str, float]] = None,
        n_neighbors: int = 5,
        max_iter: int = 10,
        energy_threshold: float = 0.1,
        random_state: int = 42
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.energy_threshold = energy_threshold
        self.random_state = random_state
        
        self.imputers_ = {}
        self.energy_consumption = 0.0
        self.missing_patterns_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'EnergyAwareMissingValueImputer':
        """Fit the imputer on training data."""
        logger.debug("Fitting energy-aware missing value imputer")
        
        # Analyze missing patterns
        self._analyze_missing_patterns(X)
        
        # Choose imputation strategy based on data characteristics and energy
        for column in X.columns:
            if X[column].isnull().sum() > 0:
                column_strategy = self._choose_imputation_strategy(X[column], column)
                self.imputers_[column] = self._create_imputer(column_strategy, X, column)
                
                # Simulate energy consumption for fitting
                missing_ratio = X[column].isnull().sum() / len(X)
                self.energy_consumption += missing_ratio * 0.1  # Energy cost proportional to missing data
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values."""
        X_imputed = X.copy()
        
        for column, imputer in self.imputers_.items():
            if column in X_imputed.columns and X_imputed[column].isnull().sum() > 0:
                if HAS_SKLEARN and hasattr(imputer, 'transform'):
                    # sklearn imputer
                    X_imputed[column] = imputer.transform(X_imputed[[column]]).ravel()
                else:
                    # Custom imputation
                    X_imputed[column] = self._custom_impute(X_imputed[column], imputer)
                
                # Energy consumption for transformation
                self.energy_consumption += 0.05
        
        return X_imputed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _analyze_missing_patterns(self, X: pd.DataFrame):
        """Analyze patterns in missing data."""
        for column in X.columns:
            missing_mask = X[column].isnull()
            if missing_mask.sum() > 0:
                self.missing_patterns_[column] = {
                    'missing_count': missing_mask.sum(),
                    'missing_ratio': missing_mask.sum() / len(X),
                    'dtype': X[column].dtype,
                    'unique_values': X[column].nunique(),
                    'is_numeric': pd.api.types.is_numeric_dtype(X[column])
                }
    
    def _choose_imputation_strategy(self, series: pd.Series, column: str) -> str:
        """Choose imputation strategy based on energy dynamics and data characteristics."""
        pattern = self.missing_patterns_[column]
        
        if self.strategy == "adaptive":
            # Energy-aware strategy selection
            missing_ratio = pattern['missing_ratio']
            
            if missing_ratio < 0.05:  # Low missing ratio - simple strategy
                return "simple"
            elif missing_ratio < 0.3 and pattern['is_numeric']:  # Medium missing ratio - KNN
                return "knn"
            elif pattern['is_numeric']:  # High missing ratio - iterative
                return "iterative" 
            else:  # Categorical - mode imputation
                return "simple"
        else:
            return self.strategy
    
    def _create_imputer(self, strategy: str, X: pd.DataFrame, column: str):
        """Create appropriate imputer based on strategy."""
        if not HAS_SKLEARN:
            return {'strategy': 'mean' if pd.api.types.is_numeric_dtype(X[column]) else 'mode'}
        
        if strategy == "simple":
            if pd.api.types.is_numeric_dtype(X[column]):
                return SimpleImputer(strategy='mean', missing_values=np.nan)
            else:
                return SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        elif strategy == "knn":
            return KNNImputer(n_neighbors=self.n_neighbors, missing_values=np.nan)
        elif strategy == "iterative":
            return IterativeImputer(max_iter=self.max_iter, random_state=self.random_state)
        else:
            return SimpleImputer(strategy='mean' if pd.api.types.is_numeric_dtype(X[column]) else 'most_frequent')
    
    def _custom_impute(self, series: pd.Series, imputer_info: dict) -> pd.Series:
        """Custom imputation when sklearn is not available."""
        strategy = imputer_info.get('strategy', 'mean')
        
        if strategy == 'mean':
            return series.fillna(series.mean())
        elif strategy == 'mode':
            return series.fillna(series.mode().iloc[0] if not series.mode().empty else 0)
        else:
            return series.fillna(series.mean() if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0])


class SpikeBasedOutlierDetector:
    """
    Outlier detector inspired by spike-based anomaly detection in neural networks.
    """
    
    def __init__(
        self,
        method: str = "spike_based",
        contamination: float = 0.1,
        threshold: float = 3.0,
        spike_energy_threshold: float = 0.8,
        random_state: int = 42
    ):
        self.method = method
        self.contamination = contamination
        self.threshold = threshold
        self.spike_energy_threshold = spike_energy_threshold
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.outlier_detectors_ = {}
        self.outlier_masks_ = {}
        self.energy_consumption = 0.0
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SpikeBasedOutlierDetector':
        """Fit outlier detectors on training data."""
        logger.debug("Fitting spike-based outlier detector")
        
        for column in X.select_dtypes(include=[np.number]).columns:
            if self.method == "spike_based":
                self.outlier_detectors_[column] = self._fit_spike_detector(X[column])
            elif self.method == "isolation_forest" and HAS_SKLEARN:
                detector = IsolationForest(contamination=self.contamination, random_state=self.random_state)
                detector.fit(X[[column]])
                self.outlier_detectors_[column] = detector
            elif self.method == "z_score":
                self.outlier_detectors_[column] = {
                    'mean': X[column].mean(),
                    'std': X[column].std()
                }
            elif self.method == "iqr":
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_detectors_[column] = {
                    'lower_bound': Q1 - 1.5 * IQR,
                    'upper_bound': Q3 + 1.5 * IQR
                }
            
            # Energy consumption for fitting
            self.energy_consumption += 0.05
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove or handle outliers in the data."""
        X_clean = X.copy()
        
        for column, detector in self.outlier_detectors_.items():
            if column in X_clean.columns:
                outlier_mask = self._detect_outliers(X_clean[column], detector, column)
                self.outlier_masks_[column] = outlier_mask
                
                # Handle outliers by clipping to bounds
                X_clean = self._handle_outliers(X_clean, column, outlier_mask, detector)
                
                # Energy consumption for transformation
                self.energy_consumption += 0.03
        
        return X_clean
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _fit_spike_detector(self, series: pd.Series) -> dict:
        """Fit spike-based outlier detector using energy dynamics."""
        # Calculate rolling energy (variance)
        window_size = min(10, len(series) // 10)
        rolling_energy = series.rolling(window=window_size, center=True).var().fillna(series.var())
        
        # Detect energy spikes
        energy_threshold = np.percentile(rolling_energy, 95)  # 95th percentile as threshold
        
        return {
            'rolling_energy_threshold': energy_threshold,
            'global_mean': series.mean(),
            'global_std': series.std(),
            'window_size': window_size
        }
    
    def _detect_outliers(self, series: pd.Series, detector: dict, column: str) -> np.ndarray:
        """Detect outliers using the fitted detector."""
        if self.method == "spike_based":
            # Recalculate rolling energy
            window_size = detector['window_size']
            rolling_energy = series.rolling(window=window_size, center=True).var().fillna(series.var())
            
            # Outliers are points with high energy spikes AND extreme values
            energy_outliers = rolling_energy > detector['rolling_energy_threshold']
            value_outliers = np.abs(series - detector['global_mean']) > (self.threshold * detector['global_std'])
            
            return energy_outliers & value_outliers
        
        elif self.method == "isolation_forest" and HAS_SKLEARN:
            predictions = detector.predict(series.values.reshape(-1, 1))
            return predictions == -1
        
        elif self.method == "z_score":
            z_scores = np.abs(series - detector['mean']) / detector['std']
            return z_scores > self.threshold
        
        elif self.method == "iqr":
            return (series < detector['lower_bound']) | (series > detector['upper_bound'])
        
        else:
            return np.zeros(len(series), dtype=bool)
    
    def _handle_outliers(self, X: pd.DataFrame, column: str, outlier_mask: np.ndarray, detector: dict) -> pd.DataFrame:
        """Handle detected outliers by clipping or imputing."""
        if self.method == "iqr" and 'lower_bound' in detector and 'upper_bound' in detector:
            X.loc[outlier_mask, column] = np.clip(
                X.loc[outlier_mask, column],
                detector['lower_bound'],
                detector['upper_bound']
            )
        elif 'global_mean' in detector and 'global_std' in detector:
            # Clip to reasonable bounds
            lower_bound = detector['global_mean'] - 3 * detector['global_std']
            upper_bound = detector['global_mean'] + 3 * detector['global_std']
            X.loc[outlier_mask, column] = np.clip(
                X.loc[outlier_mask, column],
                lower_bound,
                upper_bound
            )
        
        return X


class AdaptiveFeatureScaler:
    """
    Feature scaler that adapts based on node energy dynamics.
    """
    
    def __init__(
        self,
        method: str = "energy_aware",
        feature_range: Tuple[float, float] = (0, 1),
        energy_scaling_factor: float = 0.1,
        random_state: int = 42
    ):
        self.method = method
        self.feature_range = feature_range
        self.energy_scaling_factor = energy_scaling_factor
        self.random_state = random_state
        
        self.scalers_ = {}
        self.energy_consumption = 0.0
        self.energy_weights_ = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AdaptiveFeatureScaler':
        """Fit scalers on training data."""
        logger.debug("Fitting adaptive feature scaler")
        
        for column in X.select_dtypes(include=[np.number]).columns:
            # Calculate energy weight based on feature variance and distribution
            feature_energy = self._calculate_feature_energy(X[column])
            self.energy_weights_[column] = feature_energy
            
            # Choose scaling method based on energy characteristics
            if self.method == "energy_aware":
                scaler_type = self._choose_scaler_type(X[column], feature_energy)
            else:
                scaler_type = self.method
            
            # Create and fit scaler
            if HAS_SKLEARN:
                scaler = self._create_sklearn_scaler(scaler_type)
                scaler.fit(X[[column]])
                self.scalers_[column] = scaler
            else:
                self.scalers_[column] = self._create_custom_scaler(X[column], scaler_type)
            
            # Energy consumption for fitting
            self.energy_consumption += 0.02
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers."""
        X_scaled = X.copy()
        
        for column, scaler in self.scalers_.items():
            if column in X_scaled.columns:
                if HAS_SKLEARN and hasattr(scaler, 'transform'):
                    X_scaled[column] = scaler.transform(X_scaled[[column]]).ravel()
                else:
                    X_scaled[column] = self._custom_scale(X_scaled[column], scaler)
                
                # Energy consumption for transformation
                self.energy_consumption += 0.01
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _calculate_feature_energy(self, series: pd.Series) -> float:
        """Calculate energy metric for a feature based on its distribution."""
        # Energy based on variance (spread) and skewness (asymmetry)
        variance_energy = series.var() / (series.var() + 1e-8)  # Normalized variance
        skewness_energy = abs(stats.skew(series.dropna())) / 10  # Normalized skewness
        
        return variance_energy + skewness_energy
    
    def _choose_scaler_type(self, series: pd.Series, energy: float) -> str:
        """Choose scaling method based on feature energy characteristics."""
        if energy > 0.8:  # High energy (high variance/skewness) - use robust scaling
            return "robust"
        elif energy > 0.3:  # Medium energy - use standard scaling
            return "standard"
        else:  # Low energy - use min-max scaling
            return "minmax"
    
    def _create_sklearn_scaler(self, scaler_type: str):
        """Create sklearn scaler based on type."""
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        elif scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()
    
    def _create_custom_scaler(self, series: pd.Series, scaler_type: str) -> dict:
        """Create custom scaler parameters when sklearn is not available."""
        if scaler_type == "standard":
            return {'type': 'standard', 'mean': series.mean(), 'std': series.std()}
        elif scaler_type == "minmax":
            return {
                'type': 'minmax',
                'min': series.min(),
                'max': series.max(),
                'range': self.feature_range
            }
        elif scaler_type == "robust":
            return {
                'type': 'robust',
                'median': series.median(),
                'iqr': series.quantile(0.75) - series.quantile(0.25)
            }
        else:
            return {'type': 'standard', 'mean': series.mean(), 'std': series.std()}
    
    def _custom_scale(self, series: pd.Series, scaler_params: dict) -> pd.Series:
        """Apply custom scaling when sklearn is not available."""
        scaler_type = scaler_params['type']
        
        if scaler_type == 'standard':
            return (series - scaler_params['mean']) / scaler_params['std']
        elif scaler_type == 'minmax':
            min_val, max_val = scaler_params['min'], scaler_params['max']
            range_min, range_max = scaler_params['range']
            return (series - min_val) / (max_val - min_val) * (range_max - range_min) + range_min
        elif scaler_type == 'robust':
            return (series - scaler_params['median']) / scaler_params['iqr']
        else:
            return series


class NeuromorphicPreprocessor:
    """
    Main preprocessing class that coordinates all neuromorphic preprocessing components.
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        neuromorphic_enabled: bool = True,
        random_state: int = 42
    ):
        self.config = config or PreprocessingConfig()
        self.neuromorphic_enabled = neuromorphic_enabled
        self.random_state = random_state
        
        # Initialize components
        self.imputer = None
        self.outlier_detector = None
        self.scaler = None
        self.categorical_encoders = {}
        
        # Track processing statistics
        self.missing_values_count = 0
        self.outliers_count = 0
        self.scaled_features = []
        self.categorical_features = []
        self.energy_consumption = 0.0
        self.fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NeuromorphicPreprocessor':
        """Fit all preprocessing components."""
        logger.info("Fitting neuromorphic preprocessor")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X = X.copy()
        
        # 1. Handle missing values
        if X.isnull().sum().sum() > 0:
            self.imputer = EnergyAwareMissingValueImputer(
                strategy=self.config.imputation_strategy,
                fill_value=self.config.imputation_fill_value,
                n_neighbors=self.config.imputation_n_neighbors,
                max_iter=self.config.imputation_max_iter,
                random_state=self.random_state
            )
            self.imputer.fit(X, y)
            X = self.imputer.transform(X)
            self.missing_values_count = sum(X.isnull().sum() for X in [X])
            self.energy_consumption += self.imputer.energy_consumption
        
        # 2. Detect and handle outliers
        if self.config.outlier_detection_method != "none":
            self.outlier_detector = SpikeBasedOutlierDetector(
                method=self.config.outlier_detection_method,
                contamination=self.config.outlier_contamination,
                threshold=self.config.outlier_threshold,
                spike_energy_threshold=self.config.spike_energy_threshold,
                random_state=self.random_state
            )
            self.outlier_detector.fit(X, y)
            self.outliers_count = sum(len(mask) for mask in self.outlier_detector.outlier_masks_.values())
            self.energy_consumption += self.outlier_detector.energy_consumption
        
        # 3. Scale numerical features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            self.scaler = AdaptiveFeatureScaler(
                method=self.config.scaling_method,
                feature_range=self.config.scaling_feature_range,
                random_state=self.random_state
            )
            self.scaler.fit(X[numeric_columns], y)
            self.scaled_features = numeric_columns.tolist()
            self.energy_consumption += self.scaler.energy_consumption
        
        # 4. Encode categorical features
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            self._fit_categorical_encoders(X[categorical_columns])
            self.categorical_features = categorical_columns.tolist()
        
        self.fitted_ = True
        logger.info(f"Preprocessing fitted with energy consumption: {self.energy_consumption:.4f}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessors."""
        if not self.fitted_:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        X_processed = X.copy()
        
        # 1. Impute missing values
        if self.imputer is not None:
            X_processed = self.imputer.transform(X_processed)
        
        # 2. Handle outliers
        if self.outlier_detector is not None:
            X_processed = self.outlier_detector.transform(X_processed)
        
        # 3. Scale numerical features
        if self.scaler is not None:
            numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                X_processed[numeric_columns] = self.scaler.transform(X_processed[numeric_columns])
        
        # 4. Encode categorical features
        if self.categorical_encoders:
            X_processed = self._transform_categorical_features(X_processed)
        
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _fit_categorical_encoders(self, X_categorical: pd.DataFrame):
        """Fit categorical encoders."""
        for column in X_categorical.columns:
            # Choose encoding strategy
            unique_values = X_categorical[column].nunique()
            
            if self.config.categorical_encoding == "auto":
                if unique_values <= 10:  # Low cardinality - one-hot encode
                    encoding_method = "onehot"
                else:  # High cardinality - ordinal encode
                    encoding_method = "ordinal"
            else:
                encoding_method = self.config.categorical_encoding
            
            # Create encoder
            if HAS_SKLEARN:
                if encoding_method == "onehot":
                    encoder = OneHotEncoder(
                        handle_unknown=self.config.handle_unknown_categories,
                        sparse_output=False
                    )
                else:  # ordinal or target
                    encoder = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    )
                encoder.fit(X_categorical[[column]])
                self.categorical_encoders[column] = encoder
            else:
                # Custom encoding
                if encoding_method == "onehot":
                    unique_vals = X_categorical[column].dropna().unique()
                    self.categorical_encoders[column] = {
                        'type': 'onehot',
                        'categories': unique_vals
                    }
                else:
                    unique_vals = X_categorical[column].dropna().unique()
                    label_map = {val: i for i, val in enumerate(unique_vals)}
                    self.categorical_encoders[column] = {
                        'type': 'ordinal',
                        'label_map': label_map
                    }
    
    def _transform_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        X_encoded = X.copy()
        
        for column, encoder in self.categorical_encoders.items():
            if column in X_encoded.columns:
                if HAS_SKLEARN and hasattr(encoder, 'transform'):
                    # sklearn encoder
                    encoded_values = encoder.transform(X_encoded[[column]])
                    
                    if hasattr(encoder, 'categories_'):  # OneHotEncoder
                        # Create column names for one-hot encoded features
                        feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                        encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=X_encoded.index)
                        
                        # Drop original column and add encoded columns
                        X_encoded = X_encoded.drop(columns=[column])
                        X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
                    else:  # OrdinalEncoder
                        X_encoded[column] = encoded_values.ravel()
                else:
                    # Custom encoding
                    if encoder['type'] == 'onehot':
                        # Simple one-hot encoding
                        for category in encoder['categories']:
                            X_encoded[f"{column}_{category}"] = (X_encoded[column] == category).astype(int)
                        X_encoded = X_encoded.drop(columns=[column])
                    else:  # ordinal
                        X_encoded[column] = X_encoded[column].map(encoder['label_map']).fillna(-1)
        
        return X_encoded