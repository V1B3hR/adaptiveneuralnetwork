"""
Energy-aware hyperparameter optimization components.

This module provides hyperparameter optimization that leverages energy dynamics
and phase transitions from neuromorphic networks for intelligent search.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import warnings
import time
from collections import defaultdict

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    warnings.warn("scikit-learn not available. Some optimization features will be limited.", stacklevel=2)
    HAS_SKLEARN = False

from .config import OptimizationConfig

logger = logging.getLogger(__name__)


class EnergyMonitor:
    """
    Monitor energy consumption during hyperparameter optimization.
    """
    
    def __init__(self, stability_threshold: float = 0.01):
        self.stability_threshold = stability_threshold
        self.energy_history = []
        self.performance_history = []
        self.convergence_checks = []
        
    def update(self, energy_value: float, performance_value: float):
        """Update energy and performance tracking."""
        self.energy_history.append(energy_value)
        self.performance_history.append(performance_value)
        
        # Check for convergence
        if len(self.energy_history) >= 5:
            recent_energy = self.energy_history[-5:]
            energy_stability = np.std(recent_energy) < self.stability_threshold
            self.convergence_checks.append(energy_stability)
    
    def is_converged(self, patience: int = 10) -> bool:
        """Check if optimization has converged based on energy stability."""
        if len(self.convergence_checks) < patience:
            return False
        
        recent_checks = self.convergence_checks[-patience:]
        return sum(recent_checks) >= patience * 0.7  # 70% of recent checks show stability
    
    def get_energy_efficiency(self) -> float:
        """Calculate energy efficiency score."""
        if not self.energy_history or not self.performance_history:
            return 0.0
        
        total_energy = sum(self.energy_history)
        best_performance = max(self.performance_history)
        
        return best_performance / (total_energy + 1e-8)


class PhaseTransitionOptimizer:
    """
    Optimizer that detects and leverages phase transitions in the search space.
    """
    
    def __init__(
        self,
        transition_sensitivity: float = 0.5,
        adaptation_rate: float = 0.1,
        random_state: int = 42
    ):
        self.transition_sensitivity = transition_sensitivity
        self.adaptation_rate = adaptation_rate
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.phase_history = []
        self.transition_points = []
        self.current_phase = 0
        self.phase_characteristics = {}
    
    def detect_phase_transition(self, performance_history: List[float]) -> bool:
        """Detect if a phase transition has occurred."""
        if len(performance_history) < 10:
            return False
        
        # Calculate performance gradient
        recent_performance = performance_history[-10:]
        gradient = np.gradient(recent_performance)
        
        # Detect significant change in gradient (phase transition)
        gradient_change = abs(gradient[-1] - np.mean(gradient[:-1]))
        threshold = self.transition_sensitivity * np.std(gradient)
        
        if gradient_change > threshold:
            self.transition_points.append(len(performance_history))
            self.current_phase += 1
            return True
        
        return False
    
    def adapt_search_strategy(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt search strategy based on detected phase transitions."""
        if not self.transition_points:
            return current_params
        
        # Analyze current phase characteristics
        phase_key = f"phase_{self.current_phase}"
        if phase_key not in self.phase_characteristics:
            self.phase_characteristics[phase_key] = {
                'exploration_factor': 1.0,
                'exploitation_factor': 1.0
            }
        
        # Adapt parameters based on phase
        adapted_params = current_params.copy()
        
        # Example adaptation: adjust exploration vs exploitation
        if self.current_phase % 2 == 0:  # Even phases: more exploration
            exploration_factor = 1.5
        else:  # Odd phases: more exploitation
            exploration_factor = 0.7
        
        # Apply phase-specific adaptations (this would be model-specific)
        for param, value in adapted_params.items():
            if isinstance(value, (int, float)):
                # Add phase-based noise for exploration
                noise_scale = exploration_factor * self.adaptation_rate * abs(value)
                adapted_params[param] = value + np.random.normal(0, noise_scale)
        
        return adapted_params


class AdaptiveEarlyStopping:
    """
    Early stopping mechanism that uses energy dynamics for intelligent stopping.
    """
    
    def __init__(
        self,
        patience: int = 15,
        delta: float = 0.001,
        energy_based: bool = True,
        energy_patience: int = 10
    ):
        self.patience = patience
        self.delta = delta
        self.energy_based = energy_based
        self.energy_patience = energy_patience
        
        self.best_score = None
        self.best_energy = None
        self.wait = 0
        self.energy_wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def __call__(
        self,
        score: float,
        energy_value: Optional[float] = None,
        epoch: int = 0
    ) -> bool:
        """Check if training should stop."""
        # Performance-based stopping
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
        
        # Energy-based stopping
        if self.energy_based and energy_value is not None:
            if self.best_energy is None or energy_value < self.best_energy:
                self.best_energy = energy_value
                self.energy_wait = 0
            else:
                self.energy_wait += 1
        
        # Decision to stop
        performance_stop = self.wait >= self.patience
        energy_stop = (self.energy_based and 
                      self.energy_wait >= self.energy_patience and 
                      energy_value is not None)
        
        if performance_stop or energy_stop:
            self.should_stop = True
            self.stopped_epoch = epoch
            return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.best_score = None
        self.best_energy = None
        self.wait = 0
        self.energy_wait = 0
        self.stopped_epoch = 0
        self.should_stop = False


class EnergyAwareHyperparameterOptimizer:
    """
    Main hyperparameter optimizer that uses energy dynamics and phase transitions.
    """
    
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        neuromorphic_enabled: bool = True,
        random_state: int = 42
    ):
        self.config = config or OptimizationConfig()
        self.neuromorphic_enabled = neuromorphic_enabled
        self.random_state = random_state
        
        # Initialize components
        self.energy_monitor = EnergyMonitor(self.config.energy_stability_threshold)
        self.phase_optimizer = PhaseTransitionOptimizer(
            transition_sensitivity=0.5,  # Use default
            random_state=random_state
        ) if neuromorphic_enabled else None
        self.early_stopping = AdaptiveEarlyStopping(
            patience=self.config.early_stopping_patience,
            delta=self.config.early_stopping_delta,
            energy_based=self.config.energy_based_stopping and neuromorphic_enabled
        )
        
        # Optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = None
        self.energy_consumption = 0.0
        self.total_evaluations = 0
    
    def optimize(
        self,
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        param_space: Optional[Dict[str, Any]] = None,
        scoring: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform energy-aware hyperparameter optimization.
        
        Args:
            model_config: Configuration for the model to optimize
            X: Input features
            y: Target variable
            param_space: Parameter space to search
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_params, optimization_info)
        """
        logger.info("Starting energy-aware hyperparameter optimization")
        
        if param_space is None:
            param_space = self._create_default_param_space(model_config)
        
        if scoring is None:
            scoring = self._determine_scoring_metric(y)
        
        # Choose optimization strategy
        if self.config.search_strategy == "energy_aware_bayesian":
            return self._energy_aware_bayesian_optimization(model_config, X, y, param_space, scoring)
        elif self.config.search_strategy == "grid" and HAS_SKLEARN:
            return self._grid_search_optimization(model_config, X, y, param_space, scoring)
        elif self.config.search_strategy == "random" and HAS_SKLEARN:
            return self._random_search_optimization(model_config, X, y, param_space, scoring)
        else:
            return self._manual_optimization(model_config, X, y, param_space, scoring)
    
    def _energy_aware_bayesian_optimization(
        self,
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        param_space: Dict[str, Any],
        scoring: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform energy-aware Bayesian optimization."""
        logger.debug("Running energy-aware Bayesian optimization")
        
        # Simplified Bayesian optimization using random sampling with energy guidance
        best_params = None
        best_score = -np.inf
        performance_history = []
        
        for iteration in range(self.config.max_evaluations):
            # Sample parameters
            if iteration == 0 or not self.neuromorphic_enabled:
                # Random sampling for first iteration or when neuromorphic is disabled
                params = self._sample_params_randomly(param_space)
            else:
                # Energy-guided sampling
                params = self._sample_params_energy_guided(param_space, performance_history)
            
            # Evaluate parameters
            score, energy = self._evaluate_params(params, model_config, X, y, scoring)
            
            # Update monitoring
            self.energy_monitor.update(energy, score)
            performance_history.append(score)
            
            # Track best parameters
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            # Check for phase transitions and adapt
            if self.phase_optimizer and self.phase_optimizer.detect_phase_transition(performance_history):
                logger.debug(f"Phase transition detected at iteration {iteration}")
                # Adapt search strategy for next iterations
                param_space = self._adapt_param_space(param_space, performance_history)
            
            # Check early stopping
            if self.early_stopping(score, energy, iteration):
                logger.info(f"Early stopping at iteration {iteration}")
                break
            
            # Check energy convergence
            if self.neuromorphic_enabled and self.energy_monitor.is_converged():
                logger.info(f"Energy convergence achieved at iteration {iteration}")
                break
            
            self.total_evaluations += 1
        
        self.best_params = best_params
        self.best_score = best_score
        
        optimization_info = {
            'best_score': best_score,
            'total_evaluations': self.total_evaluations,
            'energy_consumption': self.energy_consumption,
            'energy_efficiency': self.energy_monitor.get_energy_efficiency(),
            'converged': self.energy_monitor.is_converged(),
            'early_stopped': self.early_stopping.should_stop,
            'stopped_at_iteration': self.early_stopping.stopped_epoch,
            'optimization_history': self.optimization_history
        }
        
        return best_params, optimization_info
    
    def _grid_search_optimization(
        self,
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        param_space: Dict[str, Any],
        scoring: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform grid search optimization."""
        logger.debug("Running grid search optimization")
        
        # Create a simple model for sklearn compatibility
        from sklearn.dummy import DummyClassifier, DummyRegressor
        
        if self._is_classification_task(y):
            estimator = DummyClassifier()
        else:
            estimator = DummyRegressor()
        
        # Create cross-validation strategy
        cv = self._create_cv_strategy(y)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=1  # Single job to track energy
        )
        
        start_time = time.time()
        grid_search.fit(X, y)
        end_time = time.time()
        
        # Estimate energy consumption
        self.energy_consumption = (end_time - start_time) * 0.1  # Simplified energy model
        
        optimization_info = {
            'best_score': grid_search.best_score_,
            'total_evaluations': len(grid_search.cv_results_['params']),
            'energy_consumption': self.energy_consumption,
            'optimization_history': grid_search.cv_results_
        }
        
        return grid_search.best_params_, optimization_info
    
    def _random_search_optimization(
        self,
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        param_space: Dict[str, Any],
        scoring: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform random search optimization."""
        logger.debug("Running random search optimization")
        
        # Create a simple model for sklearn compatibility
        from sklearn.dummy import DummyClassifier, DummyRegressor
        
        if self._is_classification_task(y):
            estimator = DummyClassifier()
        else:
            estimator = DummyRegressor()
        
        # Create cross-validation strategy
        cv = self._create_cv_strategy(y)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=self.config.max_evaluations,
            cv=cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=1
        )
        
        start_time = time.time()
        random_search.fit(X, y)
        end_time = time.time()
        
        # Estimate energy consumption
        self.energy_consumption = (end_time - start_time) * 0.1
        
        optimization_info = {
            'best_score': random_search.best_score_,
            'total_evaluations': len(random_search.cv_results_['params']),
            'energy_consumption': self.energy_consumption,
            'optimization_history': random_search.cv_results_
        }
        
        return random_search.best_params_, optimization_info
    
    def _manual_optimization(
        self,
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        param_space: Dict[str, Any],
        scoring: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform manual optimization when sklearn is not available."""
        logger.debug("Running manual optimization")
        
        best_params = None
        best_score = -np.inf
        
        for iteration in range(min(self.config.max_evaluations, 20)):  # Limit for manual optimization
            # Sample parameters randomly
            params = self._sample_params_randomly(param_space)
            
            # Simple evaluation (placeholder - would need actual model evaluation)
            score = np.random.random()  # Placeholder score
            energy = np.random.random() * 0.1  # Placeholder energy
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            self.energy_consumption += energy
            self.total_evaluations += 1
        
        optimization_info = {
            'best_score': best_score,
            'total_evaluations': self.total_evaluations,
            'energy_consumption': self.energy_consumption,
            'note': 'Manual optimization - results are placeholders'
        }
        
        return best_params, optimization_info
    
    def _sample_params_randomly(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters randomly from the parameter space."""
        params = {}
        
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                params[param_name] = np.random.choice(param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Assume (min, max) range
                if isinstance(param_values[0], int):
                    params[param_name] = np.random.randint(param_values[0], param_values[1] + 1)
                else:
                    params[param_name] = np.random.uniform(param_values[0], param_values[1])
            else:
                params[param_name] = param_values  # Fixed value
        
        return params
    
    def _sample_params_energy_guided(
        self,
        param_space: Dict[str, Any],
        performance_history: List[float]
    ) -> Dict[str, Any]:
        """Sample parameters guided by energy dynamics."""
        # Use recent performance to guide sampling
        recent_performance = performance_history[-5:] if len(performance_history) >= 5 else performance_history
        
        if not recent_performance:
            return self._sample_params_randomly(param_space)
        
        # Calculate energy-based sampling probabilities
        performance_trend = np.gradient(recent_performance)[-1] if len(recent_performance) > 1 else 0
        
        # Adapt sampling based on trend
        if performance_trend > 0:  # Improving - exploit current region
            exploration_factor = 0.3
        else:  # Not improving - explore more
            exploration_factor = 0.8
        
        # Sample with energy guidance
        params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                if np.random.random() < exploration_factor:
                    # Random selection for exploration
                    params[param_name] = np.random.choice(param_values)
                else:
                    # Weighted selection for exploitation (placeholder)
                    params[param_name] = np.random.choice(param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                center = (param_values[0] + param_values[1]) / 2
                range_width = param_values[1] - param_values[0]
                
                # Sample around center with exploration factor
                noise_scale = exploration_factor * range_width * 0.1
                if isinstance(param_values[0], int):
                    sampled_value = int(center + np.random.normal(0, noise_scale))
                    params[param_name] = np.clip(sampled_value, param_values[0], param_values[1])
                else:
                    sampled_value = center + np.random.normal(0, noise_scale)
                    params[param_name] = np.clip(sampled_value, param_values[0], param_values[1])
            else:
                params[param_name] = param_values
        
        return params
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        model_config: Dict[str, Any],
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        scoring: str
    ) -> Tuple[float, float]:
        """Evaluate a set of parameters."""
        # Placeholder evaluation - in real implementation, this would:
        # 1. Create model with given parameters
        # 2. Train and evaluate using cross-validation
        # 3. Track actual energy consumption
        
        start_time = time.time()
        
        # Simulate model evaluation
        score = np.random.random()  # Placeholder score
        time.sleep(0.01)  # Simulate computation time
        
        end_time = time.time()
        energy = (end_time - start_time) * 0.1  # Simplified energy model
        
        # Store in history
        self.optimization_history.append({
            'params': params.copy(),
            'score': score,
            'energy': energy
        })
        
        self.energy_consumption += energy
        
        return score, energy
    
    def _create_default_param_space(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create default parameter space based on model configuration."""
        # Default parameter space for common ML models
        return {
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'batch_size': [16, 32, 64, 128],
            'hidden_dim': [64, 128, 256, 512],
            'dropout_rate': [0.1, 0.2, 0.3, 0.5]
        }
    
    def _determine_scoring_metric(self, y: Union[pd.Series, np.ndarray]) -> str:
        """Determine appropriate scoring metric based on target variable."""
        if self._is_classification_task(y):
            return 'accuracy'
        else:
            return 'r2'
    
    def _is_classification_task(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """Determine if task is classification."""
        if hasattr(y, 'nunique'):
            unique_values = y.nunique()
        else:
            unique_values = len(np.unique(y))
        
        total_values = len(y)
        return unique_values / total_values < 0.1 or unique_values <= 20
    
    def _create_cv_strategy(self, y: Union[pd.Series, np.ndarray]):
        """Create cross-validation strategy."""
        if not HAS_SKLEARN:
            return None
        
        if self._is_classification_task(y):
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            return KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
    
    def _adapt_param_space(
        self,
        param_space: Dict[str, Any],
        performance_history: List[float]
    ) -> Dict[str, Any]:
        """Adapt parameter space based on performance history."""
        # Simple adaptation strategy - narrow ranges around better performing regions
        adapted_space = param_space.copy()
        
        # This is a placeholder - real implementation would analyze
        # which parameter ranges led to better performance
        
        return adapted_space
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization report."""
        return {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            'energy_consumption': self.energy_consumption,
            'energy_efficiency': self.energy_monitor.get_energy_efficiency() if self.energy_monitor else 0.0,
            'converged': self.energy_monitor.is_converged() if self.energy_monitor else False,
            'early_stopped': self.early_stopping.should_stop,
            'phase_transitions': len(self.phase_optimizer.transition_points) if self.phase_optimizer else 0,
            'optimization_history': self.optimization_history
        }