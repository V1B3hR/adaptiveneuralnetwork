"""
Evaluation and Validation Layer for Adaptive Neural Network.

This module provides comprehensive evaluation utilities including:
- Standardized metrics computation
- Deterministic test runs
- Microbenchmarking
- Drift detection
- Metrics comparison
"""

from .metrics import StandardMetrics, compute_metrics
from .microbenchmark import MicroBenchmark, run_microbenchmark
from .drift_detection import DriftDetector, detect_drift
from .comparison import MetricsComparator, compare_metrics

__all__ = [
    "StandardMetrics",
    "compute_metrics",
    "MicroBenchmark",
    "run_microbenchmark",
    "DriftDetector",
    "detect_drift",
    "MetricsComparator",
    "compare_metrics",
]
