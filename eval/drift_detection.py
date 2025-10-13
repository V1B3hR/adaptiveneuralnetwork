"""
Drift detection for benchmark results.

Compares recent results against historical baselines to detect
performance degradation or improvements.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    
    metric_name: str
    current_value: float
    baseline_median: float
    baseline_std: float
    drift_detected: bool
    drift_percentage: float
    drift_direction: str  # "improving", "degrading", "stable"
    
    # Statistical info
    z_score: float
    threshold: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_median": self.baseline_median,
            "baseline_std": self.baseline_std,
            "drift_detected": self.drift_detected,
            "drift_percentage": self.drift_percentage,
            "drift_direction": self.drift_direction,
            "z_score": self.z_score,
            "threshold": self.threshold,
        }


class DriftDetector:
    """Detects performance drift in benchmark results."""
    
    def __init__(
        self,
        history_path: Path | str,
        lookback_n: int = 5,
        threshold_std: float = 2.0,
    ):
        """
        Initialize drift detector.
        
        Args:
            history_path: Path to benchmark history directory
            lookback_n: Number of historical runs to compare against
            threshold_std: Number of standard deviations for drift detection
        """
        self.history_path = Path(history_path)
        self.lookback_n = lookback_n
        self.threshold_std = threshold_std
    
    def load_history(self, metric_name: str) -> list[float]:
        """
        Load historical values for a specific metric.
        
        Args:
            metric_name: Name of the metric to load
            
        Returns:
            List of historical values
        """
        history_files = sorted(self.history_path.glob("*.json"))
        
        values = []
        for file_path in history_files[-self.lookback_n:]:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Navigate nested structure to find metric
                value = self._extract_metric(data, metric_name)
                if value is not None:
                    values.append(float(value))
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return values
    
    def _extract_metric(self, data: dict[str, Any], metric_name: str) -> float | None:
        """
        Extract metric from nested dictionary structure.
        
        Args:
            data: Dictionary to search
            metric_name: Metric name to find
            
        Returns:
            Metric value or None if not found
        """
        # Try direct access
        if metric_name in data:
            return data[metric_name]
        
        # Try nested access with common patterns
        search_paths = [
            ["metrics", metric_name],
            ["final_metrics", metric_name],
            ["results", metric_name],
            ["benchmark_results", metric_name],
        ]
        
        for path in search_paths:
            current = data
            try:
                for key in path:
                    current = current[key]
                return current
            except (KeyError, TypeError):
                continue
        
        return None
    
    def detect_drift(
        self,
        current_value: float,
        metric_name: str,
        higher_is_better: bool = True,
    ) -> DriftDetectionResult:
        """
        Detect drift in a metric value.
        
        Args:
            current_value: Current metric value
            metric_name: Name of the metric
            higher_is_better: Whether higher values are better
            
        Returns:
            DriftDetectionResult with analysis
        """
        # Load historical values
        historical_values = self.load_history(metric_name)
        
        if len(historical_values) < 2:
            # Not enough history for drift detection
            return DriftDetectionResult(
                metric_name=metric_name,
                current_value=current_value,
                baseline_median=current_value,
                baseline_std=0.0,
                drift_detected=False,
                drift_percentage=0.0,
                drift_direction="stable",
                z_score=0.0,
                threshold=self.threshold_std,
            )
        
        # Compute baseline statistics
        baseline_median = float(np.median(historical_values))
        baseline_std = float(np.std(historical_values))
        
        # Compute drift
        if baseline_std > 0:
            z_score = (current_value - baseline_median) / baseline_std
        else:
            z_score = 0.0
        
        drift_detected = abs(z_score) > self.threshold_std
        
        if baseline_median != 0:
            drift_percentage = 100.0 * (current_value - baseline_median) / abs(baseline_median)
        else:
            drift_percentage = 0.0
        
        # Determine drift direction
        if not drift_detected:
            drift_direction = "stable"
        elif higher_is_better:
            drift_direction = "improving" if current_value > baseline_median else "degrading"
        else:
            drift_direction = "degrading" if current_value > baseline_median else "improving"
        
        return DriftDetectionResult(
            metric_name=metric_name,
            current_value=current_value,
            baseline_median=baseline_median,
            baseline_std=baseline_std,
            drift_detected=drift_detected,
            drift_percentage=drift_percentage,
            drift_direction=drift_direction,
            z_score=z_score,
            threshold=self.threshold_std,
        )
    
    def detect_multiple_drifts(
        self,
        current_metrics: dict[str, float],
        metric_directions: dict[str, bool] | None = None,
    ) -> list[DriftDetectionResult]:
        """
        Detect drift for multiple metrics.
        
        Args:
            current_metrics: Dictionary of metric name to current value
            metric_directions: Dictionary of metric name to higher_is_better flag
            
        Returns:
            List of DriftDetectionResult for each metric
        """
        if metric_directions is None:
            metric_directions = {}
        
        results = []
        for metric_name, current_value in current_metrics.items():
            higher_is_better = metric_directions.get(metric_name, True)
            result = self.detect_drift(current_value, metric_name, higher_is_better)
            results.append(result)
        
        return results


def detect_drift(
    current_metrics: dict[str, float],
    history_path: Path | str,
    lookback_n: int = 5,
    threshold_std: float = 2.0,
    metric_directions: dict[str, bool] | None = None,
) -> list[DriftDetectionResult]:
    """
    Convenience function to detect drift.
    
    Args:
        current_metrics: Dictionary of metric name to current value
        history_path: Path to benchmark history directory
        lookback_n: Number of historical runs to compare against
        threshold_std: Number of standard deviations for drift detection
        metric_directions: Dictionary of metric name to higher_is_better flag
        
    Returns:
        List of DriftDetectionResult for each metric
    """
    detector = DriftDetector(history_path, lookback_n, threshold_std)
    return detector.detect_multiple_drifts(current_metrics, metric_directions)
