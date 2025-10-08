"""
Neuromorphic-inspired feature selection components.

This module provides automated feature selection that uses energy dynamics
and neuromorphic principles to identify the most informative features.
"""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import (
        RFE,
        RFECV,
        SelectKBest,
        chi2,
        f_classif,
        f_regression,
        mutual_info_classif,
        mutual_info_regression,
    )
    from sklearn.metrics import mutual_info_score
    HAS_SKLEARN = True
except ImportError:
    warnings.warn("scikit-learn not available. Some feature selection will be limited.", stacklevel=2)
    HAS_SKLEARN = False

from .config import FeatureSelectionConfig

logger = logging.getLogger(__name__)


class EnergyGuidedFeatureImportance:
    """
    Calculate feature importance using energy dynamics from neuromorphic networks.
    """

    def __init__(
        self,
        energy_threshold: float = 0.01,
        phase_stability_weight: float = 0.3,
        convergence_stability_weight: float = 0.7,
        random_state: int = 42
    ):
        self.energy_threshold = energy_threshold
        self.phase_stability_weight = phase_stability_weight
        self.convergence_stability_weight = convergence_stability_weight
        self.random_state = random_state

        self.feature_energies_ = {}
        self.phase_stabilities_ = {}
        self.convergence_scores_ = {}
        self.importance_scores_ = {}
        self.energy_consumption = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> 'EnergyGuidedFeatureImportance':
        """Fit energy-guided feature importance calculator."""
        logger.debug("Fitting energy-guided feature importance")

        numeric_columns = X.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            # Calculate energy metrics
            self.feature_energies_[column] = self._calculate_feature_energy(X[column])
            self.phase_stabilities_[column] = self._calculate_phase_stability(X[column])

            if y is not None:
                self.convergence_scores_[column] = self._calculate_convergence_score(X[column], y)
            else:
                self.convergence_scores_[column] = 0.0

            # Combine metrics into importance score
            self.importance_scores_[column] = self._calculate_importance_score(column)

            self.energy_consumption += 0.02

        return self

    def transform(self, X: pd.DataFrame) -> dict[str, float]:
        """Return feature importance scores."""
        return self.importance_scores_

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, float]:
        """Fit and return importance scores."""
        return self.fit(X, y).transform(X)

    def _calculate_feature_energy(self, series: pd.Series) -> float:
        """Calculate energy metric for a feature based on variance and dynamics."""
        # Energy based on variance (information content)
        variance_energy = series.var() / (series.var() + 1e-8)

        # Dynamic energy based on changes over time
        diff_energy = series.diff().var() / (series.diff().var() + 1e-8)

        # Combine energies
        total_energy = 0.7 * variance_energy + 0.3 * diff_energy
        return min(total_energy, 1.0)  # Normalize to [0, 1]

    def _calculate_phase_stability(self, series: pd.Series) -> float:
        """Calculate phase stability of a feature."""
        # Calculate rolling variance to assess stability
        window_size = min(10, len(series) // 5)
        if window_size < 2:
            return 0.0

        rolling_var = series.rolling(window=window_size, center=True).var()
        stability = 1.0 / (1.0 + rolling_var.std())  # Higher stability = lower variance of variance

        return min(stability, 1.0)

    def _calculate_convergence_score(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate how well feature converges with target."""
        try:
            # Simple correlation-based convergence
            correlation = abs(np.corrcoef(feature.fillna(0), target.fillna(0))[0, 1])

            if np.isnan(correlation):
                return 0.0

            # Energy-weighted correlation
            feature_energy = self._calculate_feature_energy(feature)
            convergence_score = correlation * feature_energy

            return min(convergence_score, 1.0)
        except:
            return 0.0

    def _calculate_importance_score(self, column: str) -> float:
        """Calculate final importance score combining all metrics."""
        phase_stability = self.phase_stabilities_[column]
        convergence_score = self.convergence_scores_[column]

        # Weighted combination
        importance = (
            self.phase_stability_weight * phase_stability +
            self.convergence_stability_weight * convergence_score
        )

        return importance


class NeuromorphicMutualInformation:
    """
    Mutual information estimation using neuromorphic principles.
    """

    def __init__(
        self,
        n_neighbors: int = 3,
        discrete_features: str = "auto",
        neuromorphic_estimation: bool = True,
        random_state: int = 42
    ):
        self.n_neighbors = n_neighbors
        self.discrete_features = discrete_features
        self.neuromorphic_estimation = neuromorphic_estimation
        self.random_state = random_state

        self.mi_scores_ = {}
        self.energy_weighted_mi_ = {}
        self.energy_consumption = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NeuromorphicMutualInformation':
        """Fit neuromorphic mutual information estimator."""
        logger.debug("Fitting neuromorphic mutual information")

        # Determine if target is discrete or continuous
        is_classification = self._is_classification_task(y)

        numeric_columns = X.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if HAS_SKLEARN and not self.neuromorphic_estimation:
                # Use sklearn mutual information
                if is_classification:
                    mi_score = mutual_info_classif(
                        X[[column]], y,
                        discrete_features=False,
                        n_neighbors=self.n_neighbors,
                        random_state=self.random_state
                    )[0]
                else:
                    mi_score = mutual_info_regression(
                        X[[column]], y,
                        discrete_features=False,
                        n_neighbors=self.n_neighbors,
                        random_state=self.random_state
                    )[0]
            else:
                # Use neuromorphic-inspired MI estimation
                mi_score = self._neuromorphic_mutual_info(X[column], y)

            self.mi_scores_[column] = mi_score

            # Calculate energy-weighted MI
            if self.neuromorphic_estimation:
                feature_energy = self._calculate_energy_weight(X[column])
                self.energy_weighted_mi_[column] = mi_score * feature_energy
            else:
                self.energy_weighted_mi_[column] = mi_score

            self.energy_consumption += 0.03

        return self

    def transform(self, X: pd.DataFrame) -> dict[str, float]:
        """Return mutual information scores."""
        return self.energy_weighted_mi_

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Fit and return MI scores."""
        return self.fit(X, y).transform(X)

    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if task is classification based on target."""
        unique_values = y.nunique()
        total_values = len(y)

        # If unique values is small relative to total, likely classification
        return unique_values / total_values < 0.1 or unique_values <= 20

    def _neuromorphic_mutual_info(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate MI using neuromorphic-inspired approach."""
        try:
            # Discretize both feature and target for MI calculation
            feature_discrete = self._adaptive_discretize(feature)
            target_discrete = self._adaptive_discretize(target)

            # Calculate mutual information using spike-pattern similarity
            mi_score = self._spike_pattern_mutual_info(feature_discrete, target_discrete)

            return mi_score
        except:
            # Fallback to correlation-based approximation
            correlation = abs(np.corrcoef(feature.fillna(0), target.fillna(0))[0, 1])
            return correlation ** 2 if not np.isnan(correlation) else 0.0

    def _adaptive_discretize(self, series: pd.Series, n_bins: int = 10) -> np.ndarray:
        """Adaptively discretize a series based on its distribution."""
        if series.nunique() <= n_bins:
            # Already discrete enough
            return pd.Categorical(series).codes
        else:
            # Use quantile-based discretization
            return pd.qcut(series, q=n_bins, labels=False, duplicates='drop').fillna(0).astype(int)

    def _spike_pattern_mutual_info(self, x_discrete: np.ndarray, y_discrete: np.ndarray) -> float:
        """Calculate MI based on spike pattern similarity."""
        # Calculate joint and marginal distributions
        joint_counts = {}
        x_counts = {}
        y_counts = {}

        total_samples = len(x_discrete)

        for i in range(total_samples):
            x_val, y_val = x_discrete[i], y_discrete[i]

            # Joint distribution
            joint_key = (x_val, y_val)
            joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1

            # Marginal distributions
            x_counts[x_val] = x_counts.get(x_val, 0) + 1
            y_counts[y_val] = y_counts.get(y_val, 0) + 1

        # Calculate mutual information
        mi = 0.0
        for (x_val, y_val), joint_count in joint_counts.items():
            p_xy = joint_count / total_samples
            p_x = x_counts[x_val] / total_samples
            p_y = y_counts[y_val] / total_samples

            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * np.log2(p_xy / (p_x * p_y))

        return max(mi, 0.0)  # Ensure non-negative

    def _calculate_energy_weight(self, series: pd.Series) -> float:
        """Calculate energy weight for a feature."""
        # Energy based on information content (entropy approximation)
        variance_weight = series.var() / (series.var() + 1e-8)

        # Dynamic weight based on temporal patterns
        if len(series) > 1:
            autocorr = series.autocorr(lag=1)
            dynamic_weight = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 0.5
        else:
            dynamic_weight = 0.5

        return 0.6 * variance_weight + 0.4 * dynamic_weight


class AdaptiveFeatureSelector:
    """
    Main feature selection class that coordinates all neuromorphic selection methods.
    """

    def __init__(
        self,
        config: FeatureSelectionConfig | None = None,
        neuromorphic_enabled: bool = True,
        max_features: int | None = None,
        random_state: int = 42
    ):
        self.config = config or FeatureSelectionConfig()
        self.neuromorphic_enabled = neuromorphic_enabled
        self.max_features = max_features
        self.random_state = random_state

        # Initialize components
        self.energy_importance = None
        self.mutual_info_selector = None
        self.univariate_selector = None
        self.rfe_selector = None

        # Selection results
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.energy_importance_ = {}
        self.correlation_matrix_ = None
        self.energy_consumption = 0.0
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaptiveFeatureSelector':
        """Fit all feature selection components."""
        logger.info("Fitting adaptive feature selector")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        original_features = X.columns.tolist()
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # 1. Energy-guided importance (neuromorphic)
        if self.neuromorphic_enabled:
            self.energy_importance = EnergyGuidedFeatureImportance(
                energy_threshold=self.config.energy_importance_threshold,
                phase_stability_weight=self.config.phase_stability_weight,
                convergence_stability_weight=self.config.convergence_stability_weight,
                random_state=self.random_state
            )
            self.energy_importance_ = self.energy_importance.fit_transform(X[numeric_features], y)
            self.energy_consumption += self.energy_importance.energy_consumption

        # 2. Mutual information selection
        if self.config.neuromorphic_mi_estimation or not self.neuromorphic_enabled:
            self.mutual_info_selector = NeuromorphicMutualInformation(
                n_neighbors=self.config.mi_n_neighbors,
                discrete_features=self.config.mi_discrete_features,
                neuromorphic_estimation=self.config.neuromorphic_mi_estimation and self.neuromorphic_enabled,
                random_state=self.random_state
            )
            mi_scores = self.mutual_info_selector.fit_transform(X[numeric_features], y)
            self.feature_scores_.update(mi_scores)
            self.energy_consumption += self.mutual_info_selector.energy_consumption

        # 3. Univariate feature selection
        if HAS_SKLEARN and self.config.univariate_method in ['chi2', 'f_test']:
            self._fit_univariate_selector(X[numeric_features], y)

        # 4. Recursive Feature Elimination
        if self.config.enable_rfe and HAS_SKLEARN:
            self._fit_rfe_selector(X[numeric_features], y)

        # 5. Correlation-based filtering
        if self.config.correlation_threshold < 1.0:
            self.correlation_matrix_ = X[numeric_features].corr()

        # 6. Combine all selection methods
        self.selected_features_ = self._combine_selection_results(X.columns.tolist(), numeric_features)

        self.fitted_ = True
        logger.info(f"Feature selection completed: {len(original_features)} -> {len(self.selected_features_)} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features based on fitted selectors."""
        if not self.fitted_:
            raise ValueError("Feature selector must be fitted before transforming")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Select features that exist in the data
        available_features = [f for f in self.selected_features_ if f in X.columns]

        if not available_features:
            logger.warning("No selected features found in data. Returning all numeric features.")
            return X.select_dtypes(include=[np.number])

        return X[available_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _fit_univariate_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit univariate feature selector."""
        try:
            # Determine score function
            if self.config.univariate_method == 'chi2':
                # Ensure non-negative values for chi2
                X_nonneg = X - X.min() + 1e-8
                score_func = chi2
            elif self._is_classification_task(y):
                score_func = f_classif
                X_nonneg = X
            else:
                score_func = f_regression
                X_nonneg = X

            k_best = min(self.config.univariate_k_best, X.shape[1])
            self.univariate_selector = SelectKBest(score_func=score_func, k=k_best)
            self.univariate_selector.fit(X_nonneg, y)

            # Store scores
            for i, feature in enumerate(X.columns):
                self.feature_scores_[f'{feature}_univariate'] = self.univariate_selector.scores_[i]

            self.energy_consumption += 0.05
        except Exception as e:
            logger.warning(f"Univariate selection failed: {e}")
            self.univariate_selector = None

    def _fit_rfe_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit recursive feature elimination selector."""
        try:
            # Use appropriate estimator
            if self._is_classification_task(y):
                estimator = RandomForestClassifier(n_estimators=10, random_state=self.random_state)
            else:
                estimator = RandomForestRegressor(n_estimators=10, random_state=self.random_state)

            n_features = min(self.config.rfe_n_features, X.shape[1])
            self.rfe_selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=self.config.rfe_step
            )
            self.rfe_selector.fit(X, y)

            # Store rankings
            for i, feature in enumerate(X.columns):
                self.feature_scores_[f'{feature}_rfe_ranking'] = self.rfe_selector.ranking_[i]

            self.energy_consumption += 0.1
        except Exception as e:
            logger.warning(f"RFE selection failed: {e}")
            self.rfe_selector = None

    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if task is classification."""
        unique_values = y.nunique()
        total_values = len(y)
        return unique_values / total_values < 0.1 or unique_values <= 20

    def _combine_selection_results(self, all_features: list[str], numeric_features: list[str]) -> list[str]:
        """Combine results from all selection methods."""
        feature_rankings = {}

        # Initialize with all numeric features
        for feature in numeric_features:
            feature_rankings[feature] = []

        # Add energy importance rankings
        if self.energy_importance_:
            sorted_energy = sorted(self.energy_importance_.items(), key=lambda x: x[1], reverse=True)
            for rank, (feature, _) in enumerate(sorted_energy):
                if feature in feature_rankings:
                    feature_rankings[feature].append(rank)

        # Add mutual information rankings
        mi_features = [f for f in self.feature_scores_.keys() if not f.endswith('_univariate') and not f.endswith('_rfe_ranking')]
        if mi_features:
            sorted_mi = sorted(
                [(f, self.feature_scores_[f]) for f in mi_features],
                key=lambda x: x[1], reverse=True
            )
            for rank, (feature, _) in enumerate(sorted_mi):
                if feature in feature_rankings:
                    feature_rankings[feature].append(rank)

        # Add univariate rankings
        if self.univariate_selector is not None:
            univariate_features = [f for f in self.feature_scores_.keys() if f.endswith('_univariate')]
            sorted_univariate = sorted(
                [(f.replace('_univariate', ''), self.feature_scores_[f]) for f in univariate_features],
                key=lambda x: x[1], reverse=True
            )
            for rank, (feature, _) in enumerate(sorted_univariate):
                if feature in feature_rankings:
                    feature_rankings[feature].append(rank)

        # Add RFE rankings (lower rank is better for RFE)
        if self.rfe_selector is not None:
            rfe_features = [f for f in self.feature_scores_.keys() if f.endswith('_rfe_ranking')]
            sorted_rfe = sorted(
                [(f.replace('_rfe_ranking', ''), self.feature_scores_[f]) for f in rfe_features],
                key=lambda x: x[1]  # Lower is better for RFE
            )
            for rank, (feature, _) in enumerate(sorted_rfe):
                if feature in feature_rankings:
                    feature_rankings[feature].append(rank)

        # Calculate average ranking for each feature
        final_rankings = {}
        for feature, ranks in feature_rankings.items():
            if ranks:  # Only consider features that have at least one ranking
                final_rankings[feature] = np.mean(ranks)

        # Sort by average ranking and apply correlation filtering
        sorted_features = sorted(final_rankings.items(), key=lambda x: x[1])
        selected_features = []

        for feature, rank in sorted_features:
            # Check correlation with already selected features
            if self._is_feature_acceptable(feature, selected_features):
                selected_features.append(feature)

            # Stop if we have enough features
            if self.max_features and len(selected_features) >= self.max_features:
                break

        # Include non-numeric features
        non_numeric_features = [f for f in all_features if f not in numeric_features]
        selected_features.extend(non_numeric_features)

        return selected_features

    def _is_feature_acceptable(self, feature: str, selected_features: list[str]) -> bool:
        """Check if feature is acceptable based on correlation threshold."""
        if self.correlation_matrix_ is None or not selected_features:
            return True

        for selected_feature in selected_features:
            if (feature in self.correlation_matrix_.index and
                selected_feature in self.correlation_matrix_.columns):
                correlation = abs(self.correlation_matrix_.loc[feature, selected_feature])
                if correlation > self.config.correlation_threshold:
                    return False

        return True

    def get_feature_importance_report(self) -> dict[str, Any]:
        """Get detailed feature importance report."""
        return {
            'selected_features': self.selected_features_,
            'total_features_selected': len(self.selected_features_),
            'energy_importance_scores': self.energy_importance_,
            'mutual_info_scores': {k: v for k, v in self.feature_scores_.items()
                                 if not k.endswith('_univariate') and not k.endswith('_rfe_ranking')},
            'univariate_scores': {k.replace('_univariate', ''): v for k, v in self.feature_scores_.items()
                                if k.endswith('_univariate')},
            'rfe_rankings': {k.replace('_rfe_ranking', ''): v for k, v in self.feature_scores_.items()
                           if k.endswith('_rfe_ranking')},
            'energy_consumption': self.energy_consumption
        }
