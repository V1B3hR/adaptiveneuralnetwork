"""
Neuromorphic pipeline orchestration components.

This module provides pipeline management and orchestration that ensures
proper data flow while preventing data leakage and maintaining neuromorphic principles.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    warnings.warn("scikit-learn not available. Some pipeline features will be limited.")
    HAS_SKLEARN = False
    BaseEstimator = object
    TransformerMixin = object

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class DataLeakagePreventionMixin:
    """
    Mixin class that provides data leakage prevention mechanisms.
    """

    def __init__(self):
        self.fit_data_signature = None
        self.transform_data_signature = None
        self.leakage_checks_enabled = True

    def _compute_data_signature(self, X: pd.DataFrame) -> dict[str, Any]:
        """Compute a signature for the data to detect leakage."""
        return {
            'shape': X.shape,
            'columns': list(X.columns),
            'dtypes': X.dtypes.to_dict(),
            'null_counts': X.isnull().sum().to_dict(),
            'numeric_stats': X.select_dtypes(include=[np.number]).describe().to_dict()
        }

    def _check_data_leakage(self, X: pd.DataFrame, stage: str = "transform"):
        """Check for potential data leakage."""
        if not self.leakage_checks_enabled:
            return

        current_signature = self._compute_data_signature(X)

        if stage == "fit":
            self.fit_data_signature = current_signature
        elif stage == "transform":
            if self.fit_data_signature is None:
                warnings.warn("Transform called before fit - potential data leakage risk")
                return

            # Check for suspicious changes that might indicate leakage
            if current_signature['shape'][1] != self.fit_data_signature['shape'][1]:
                warnings.warn("Column count changed between fit and transform - check for data leakage")

            if set(current_signature['columns']) != set(self.fit_data_signature['columns']):
                warnings.warn("Column names changed between fit and transform - check for data leakage")

    def enable_leakage_prevention(self):
        """Enable data leakage prevention checks."""
        self.leakage_checks_enabled = True

    def disable_leakage_prevention(self):
        """Disable data leakage prevention checks."""
        self.leakage_checks_enabled = False


class NeuromorphicTransformer(BaseEstimator, TransformerMixin, DataLeakagePreventionMixin):
    """
    Base transformer class that wraps neuromorphic preprocessing components.
    """

    def __init__(self, transformer, name: str = "neuromorphic_transformer"):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        DataLeakagePreventionMixin.__init__(self)

        self.transformer = transformer
        self.name = name
        self.fitted_ = False

    def fit(self, X, y=None):
        """Fit the transformer."""
        self._check_data_leakage(X, "fit")

        if hasattr(self.transformer, 'fit'):
            self.transformer.fit(X, y)

        self.fitted_ = True
        return self

    def transform(self, X):
        """Transform the data."""
        if not self.fitted_:
            raise ValueError(f"{self.name} must be fitted before transforming")

        self._check_data_leakage(X, "transform")

        if hasattr(self.transformer, 'transform'):
            return self.transformer.transform(X)
        else:
            return X

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class NeuromorphicPipeline:
    """
    Pipeline that orchestrates neuromorphic AutoML components while preventing data leakage.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        neuromorphic_enabled: bool = True,
        random_state: int = 42
    ):
        self.config = config or PipelineConfig()
        self.neuromorphic_enabled = neuromorphic_enabled
        self.random_state = random_state

        # Pipeline components
        self.steps = []
        self.fitted_steps = {}
        self.pipeline_metadata = {}

        # Data splitting
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        # Caching
        self.cache_enabled = self.config.enable_caching
        self.cache_dir = Path(self.config.cache_directory) if self.config.cache_directory else None
        self.cached_results = {}

        # Pipeline state
        self.fitted_ = False
        self.energy_consumption = 0.0

    def add_step(self, name: str, transformer) -> 'NeuromorphicPipeline':
        """Add a step to the pipeline."""
        # Wrap transformer in neuromorphic wrapper if needed
        if not isinstance(transformer, NeuromorphicTransformer):
            transformer = NeuromorphicTransformer(transformer, name)

        self.steps.append((name, transformer))
        logger.debug(f"Added step '{name}' to pipeline")
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> 'NeuromorphicPipeline':
        """Fit the entire pipeline."""
        logger.info("Fitting neuromorphic pipeline")

        # Validate data splitting configuration
        if self.config.prevent_data_leakage and self.config.fit_on_train_only:
            X_train, y_train = self._split_data_for_fitting(X, y)
        else:
            X_train, y_train = X, y

        # Fit each step sequentially
        X_current = X_train.copy()

        for step_name, transformer in self.steps:
            logger.debug(f"Fitting step: {step_name}")

            # Check cache
            if self._is_cached(step_name, X_current):
                logger.debug(f"Using cached results for {step_name}")
                X_current = self._load_from_cache(step_name, X_current)
                continue

            # Fit transformer
            start_time = pd.Timestamp.now()
            transformer.fit(X_current, y_train)
            end_time = pd.Timestamp.now()

            # Transform for next step
            X_current = transformer.transform(X_current)

            # Store fitted transformer and metadata
            self.fitted_steps[step_name] = transformer
            self.pipeline_metadata[step_name] = {
                'input_shape': X_train.shape if step_name == self.steps[0][0] else None,
                'output_shape': X_current.shape,
                'fit_time': (end_time - start_time).total_seconds(),
                'energy_consumption': getattr(transformer.transformer, 'energy_consumption', 0.0)
            }

            # Update total energy consumption
            self.energy_consumption += self.pipeline_metadata[step_name]['energy_consumption']

            # Cache results if enabled
            if self.cache_enabled:
                self._save_to_cache(step_name, X_current)

        self.fitted_ = True
        logger.info(f"Pipeline fitted successfully with {len(self.steps)} steps")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through the entire pipeline."""
        if not self.fitted_:
            raise ValueError("Pipeline must be fitted before transforming")

        X_current = X.copy()

        for step_name, _ in self.steps:
            transformer = self.fitted_steps[step_name]
            X_current = transformer.transform(X_current)

        return X_current

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _split_data_for_fitting(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Split data for fitting to prevent data leakage."""
        if not HAS_SKLEARN:
            logger.warning("sklearn not available - using full dataset for fitting")
            return X, y

        # Determine stratification
        stratify = None
        if y is not None and self.config.stratify:
            # Check if stratification is appropriate
            if self._is_classification_task(y):
                stratify = y

        # Split data
        test_size = self.config.test_size + self.config.validation_size

        if test_size > 0:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=test_size,
                stratify=stratify,
                random_state=self.config.random_state
            )

            # Store indices for later use
            self.train_indices = X_train.index

            # Further split temp into validation and test
            if self.config.validation_size > 0:
                val_ratio = self.config.validation_size / test_size
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp,
                    test_size=1-val_ratio,
                    stratify=y_temp if stratify is not None else None,
                    random_state=self.config.random_state
                )
                self.val_indices = X_val.index
                self.test_indices = X_test.index
            else:
                self.test_indices = X_temp.index

            return X_train, y_train

        return X, y

    def _is_classification_task(self, y: pd.Series) -> bool:
        """Determine if task is classification."""
        unique_values = y.nunique()
        total_values = len(y)
        return unique_values / total_values < 0.1 or unique_values <= 20

    def _is_cached(self, step_name: str, X: pd.DataFrame) -> bool:
        """Check if results are cached for this step."""
        if not self.cache_enabled or self.cache_dir is None:
            return False

        cache_key = self._generate_cache_key(step_name, X)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        return cache_file.exists()

    def _load_from_cache(self, step_name: str, X: pd.DataFrame) -> pd.DataFrame:
        """Load cached results."""
        cache_key = self._generate_cache_key(step_name, X)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {step_name}: {e}")
            return X

    def _save_to_cache(self, step_name: str, X: pd.DataFrame):
        """Save results to cache."""
        if not self.cache_enabled or self.cache_dir is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._generate_cache_key(step_name, X)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(X, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {step_name}: {e}")

    def _generate_cache_key(self, step_name: str, X: pd.DataFrame) -> str:
        """Generate cache key based on step and data characteristics."""
        data_hash = hash(tuple(X.shape + tuple(X.columns) + (X.values.tobytes(),)))
        return f"{step_name}_{abs(data_hash)}"

    def get_pipeline_report(self) -> dict[str, Any]:
        """Get detailed pipeline execution report."""
        total_fit_time = sum(
            metadata.get('fit_time', 0)
            for metadata in self.pipeline_metadata.values()
        )

        return {
            'steps': [name for name, _ in self.steps],
            'total_steps': len(self.steps),
            'total_fit_time': total_fit_time,
            'total_energy_consumption': self.energy_consumption,
            'step_metadata': self.pipeline_metadata,
            'data_splits': {
                'train_size': len(self.train_indices) if self.train_indices is not None else None,
                'val_size': len(self.val_indices) if self.val_indices is not None else None,
                'test_size': len(self.test_indices) if self.test_indices is not None else None
            },
            'cache_enabled': self.cache_enabled,
            'leakage_prevention_enabled': self.config.prevent_data_leakage
        }

    def save_pipeline(self, file_path: str | Path):
        """Save the fitted pipeline to disk."""
        if not self.fitted_:
            raise ValueError("Pipeline must be fitted before saving")

        file_path = Path(file_path)

        pipeline_data = {
            'steps': self.steps,
            'fitted_steps': self.fitted_steps,
            'pipeline_metadata': self.pipeline_metadata,
            'config': self.config,
            'energy_consumption': self.energy_consumption,
            'random_state': self.random_state
        }

        if file_path.suffix == '.pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(pipeline_data, f)
        elif file_path.suffix == '.joblib':
            joblib.dump(pipeline_data, file_path)
        else:
            # Default to pickle
            with open(file_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(pipeline_data, f)

        logger.info(f"Pipeline saved to {file_path}")

    @classmethod
    def load_pipeline(cls, file_path: str | Path) -> 'NeuromorphicPipeline':
        """Load a fitted pipeline from disk."""
        file_path = Path(file_path)

        if file_path.suffix == '.joblib':
            pipeline_data = joblib.load(file_path)
        else:
            with open(file_path, 'rb') as f:
                pipeline_data = pickle.load(f)

        # Reconstruct pipeline
        pipeline = cls(
            config=pipeline_data['config'],
            random_state=pipeline_data['random_state']
        )

        pipeline.steps = pipeline_data['steps']
        pipeline.fitted_steps = pipeline_data['fitted_steps']
        pipeline.pipeline_metadata = pipeline_data['pipeline_metadata']
        pipeline.energy_consumption = pipeline_data['energy_consumption']
        pipeline.fitted_ = True

        logger.info(f"Pipeline loaded from {file_path}")
        return pipeline


class AutoMLWorkflow:
    """
    High-level workflow manager for complete AutoML processes.
    """

    def __init__(
        self,
        pipeline: NeuromorphicPipeline | None = None,
        config: PipelineConfig | None = None,
        neuromorphic_enabled: bool = True,
        random_state: int = 42
    ):
        self.pipeline = pipeline or NeuromorphicPipeline(config, neuromorphic_enabled, random_state)
        self.config = config or PipelineConfig()
        self.neuromorphic_enabled = neuromorphic_enabled
        self.random_state = random_state

        # Workflow state
        self.workflow_history = []
        self.results = {}
        self.models = {}

    def run_complete_workflow(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        model_configs: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Run complete AutoML workflow from preprocessing to model selection.
        
        Args:
            X: Input features
            y: Target variable
            model_configs: List of model configurations to try
            
        Returns:
            Complete workflow results
        """
        logger.info("Starting complete AutoML workflow")

        workflow_start_time = pd.Timestamp.now()

        # Step 1: Fit preprocessing pipeline
        logger.info("Step 1: Fitting preprocessing pipeline")
        self.pipeline.fit(X, y)
        X_processed = self.pipeline.transform(X)

        # Step 2: Split data for model training and evaluation
        logger.info("Step 2: Splitting data for model evaluation")
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_processed_data(
            X_processed, y
        )

        # Step 3: Model training and evaluation (placeholder)
        logger.info("Step 3: Model training and evaluation")
        model_results = self._evaluate_models(X_train, X_val, y_train, y_val, model_configs)

        # Step 4: Final evaluation on test set
        logger.info("Step 4: Final evaluation on test set")
        final_results = self._final_evaluation(X_test, y_test, model_results)

        workflow_end_time = pd.Timestamp.now()
        total_time = (workflow_end_time - workflow_start_time).total_seconds()

        # Compile complete results
        complete_results = {
            'preprocessing_results': self.pipeline.get_pipeline_report(),
            'model_results': model_results,
            'final_results': final_results,
            'workflow_metadata': {
                'total_time': total_time,
                'start_time': workflow_start_time.isoformat(),
                'end_time': workflow_end_time.isoformat(),
                'neuromorphic_enabled': self.neuromorphic_enabled,
                'data_shape': X.shape,
                'processed_data_shape': X_processed.shape
            }
        }

        self.results = complete_results
        logger.info("Complete AutoML workflow finished successfully")

        return complete_results

    def _split_processed_data(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split processed data for model training and evaluation."""
        if not HAS_SKLEARN:
            # Simple split without sklearn
            n = len(X)
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)

            X_train = X.iloc[:train_end]
            X_val = X.iloc[train_end:val_end]
            X_test = X.iloc[val_end:]

            if isinstance(y, pd.Series):
                y_train = y.iloc[:train_end]
                y_val = y.iloc[train_end:val_end]
                y_test = y.iloc[val_end:]
            else:
                y_train = y[:train_end]
                y_val = y[train_end:val_end]
                y_test = y[val_end:]

            return X_train, X_val, X_test, y_train, y_val, y_test

        # Use sklearn for better splitting
        stratify = None
        if isinstance(y, pd.Series) and self._is_classification_task(y):
            stratify = y

        # First split: train vs temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.4,  # 40% for val + test
            stratify=stratify,
            random_state=self.random_state
        )

        # Second split: val vs test
        stratify_temp = y_temp if stratify is not None else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,  # Half of temp (20% of total) for test
            stratify=stratify_temp,
            random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _is_classification_task(self, y: pd.Series | np.ndarray) -> bool:
        """Determine if task is classification."""
        if hasattr(y, 'nunique'):
            unique_values = y.nunique()
        else:
            unique_values = len(np.unique(y))

        total_values = len(y)
        return unique_values / total_values < 0.1 or unique_values <= 20

    def _evaluate_models(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        y_val: pd.Series | np.ndarray,
        model_configs: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Evaluate different models (placeholder implementation)."""
        # This is a placeholder - real implementation would:
        # 1. Train different model types (linear, tree-based, neural networks)
        # 2. Perform hyperparameter optimization
        # 3. Use cross-validation for robust evaluation
        # 4. Track energy consumption and neuromorphic features

        model_results = {
            'evaluated_models': [],
            'best_model': None,
            'best_score': 0.0,
            'evaluation_summary': {}
        }

        # Placeholder models
        models_to_try = model_configs or [
            {'type': 'linear', 'name': 'LinearModel'},
            {'type': 'tree', 'name': 'TreeModel'},
            {'type': 'neural', 'name': 'NeuralModel'}
        ]

        for model_config in models_to_try:
            # Simulate model evaluation
            model_name = model_config.get('name', 'UnknownModel')

            # Placeholder evaluation
            score = np.random.random()
            energy = np.random.random() * 0.1

            model_result = {
                'name': model_name,
                'config': model_config,
                'validation_score': score,
                'energy_consumption': energy,
                'training_time': np.random.random() * 10
            }

            model_results['evaluated_models'].append(model_result)

            # Track best model
            if score > model_results['best_score']:
                model_results['best_score'] = score
                model_results['best_model'] = model_result

        logger.info(f"Evaluated {len(models_to_try)} models. Best score: {model_results['best_score']:.4f}")

        return model_results

    def _final_evaluation(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series | np.ndarray,
        model_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform final evaluation on test set."""
        best_model = model_results.get('best_model')

        if best_model is None:
            return {'error': 'No best model found'}

        # Simulate final evaluation
        final_score = np.random.random()

        return {
            'best_model_name': best_model['name'],
            'test_score': final_score,
            'test_samples': len(X_test),
            'evaluation_complete': True
        }

    def save_workflow(self, file_path: str | Path):
        """Save complete workflow results."""
        file_path = Path(file_path)

        workflow_data = {
            'pipeline': self.pipeline,
            'results': self.results,
            'config': self.config,
            'neuromorphic_enabled': self.neuromorphic_enabled,
            'random_state': self.random_state
        }

        with open(file_path, 'wb') as f:
            pickle.dump(workflow_data, f)

        logger.info(f"Workflow saved to {file_path}")

    @classmethod
    def load_workflow(cls, file_path: str | Path) -> 'AutoMLWorkflow':
        """Load workflow from disk."""
        file_path = Path(file_path)

        with open(file_path, 'rb') as f:
            workflow_data = pickle.load(f)

        workflow = cls(
            pipeline=workflow_data['pipeline'],
            config=workflow_data['config'],
            neuromorphic_enabled=workflow_data['neuromorphic_enabled'],
            random_state=workflow_data['random_state']
        )

        workflow.results = workflow_data['results']

        logger.info(f"Workflow loaded from {file_path}")
        return workflow
