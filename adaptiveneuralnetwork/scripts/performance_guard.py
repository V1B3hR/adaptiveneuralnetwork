#!/usr/bin/env python3
"""
Performance Guard CLI - JAX vs PyTorch delta alert.

Compares performance between JAX and PyTorch backends and alerts if regression is detected.
Exits 0 when data is missing or within thresholds, exits 1 when violations are detected.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class PerformanceGuard:
    """Guards against performance regression between backends."""

    def __init__(self, results_dir: Path = Path(".")):
        self.results_dir = results_dir
        self.pytorch_data = None
        self.jax_data = None
        self.load_backend_data()

    def load_backend_data(self) -> None:
        """Load backend performance data."""
        # Look for PyTorch backend results
        pytorch_files = ["backend_pytorch.json", "benchmark_pytorch.json", "pytorch_results.json"]

        for filename in pytorch_files:
            filepath = self.results_dir / filename
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        self.pytorch_data = json.load(f)
                    break
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {filepath}: {e}")

        # Look for JAX backend results
        jax_files = ["backend_jax.json", "benchmark_jax.json", "jax_results.json"]

        for filename in jax_files:
            filepath = self.results_dir / filename
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        self.jax_data = json.load(f)
                    break
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {filepath}: {e}")

    def extract_performance_metric(
        self, data: Dict[str, Any], metric_paths: list
    ) -> Optional[float]:
        """Extract performance metric from data using multiple possible paths."""
        for path in metric_paths:
            keys = path.split(".")
            current = data

            try:
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        break
                else:
                    # Successfully navigated the path
                    if isinstance(current, (int, float)):
                        return float(current)
            except (KeyError, TypeError):
                continue

        return None

    def check_performance_regression(self, tolerance_percent: float = 10.0) -> tuple[bool, str]:
        """
        Check for performance regression between backends.

        Args:
            tolerance_percent: Maximum allowed performance degradation (%)

        Returns:
            Tuple of (is_violation, message)
        """
        if not self.pytorch_data or not self.jax_data:
            return False, "Backend comparison data not available - skipping performance guard"

        # Possible paths for timing metrics
        timing_paths = [
            "timing.epoch_mean",
            "timing.total_time",
            "performance.training_time",
            "duration_seconds",
            "average_test_time",
        ]

        pytorch_time = self.extract_performance_metric(self.pytorch_data, timing_paths)
        jax_time = self.extract_performance_metric(self.jax_data, timing_paths)

        if pytorch_time is None or jax_time is None:
            return False, "Timing metrics not found in backend data - skipping performance guard"

        # Calculate performance delta (positive means JAX is slower)
        if pytorch_time > 0:
            performance_delta_percent = ((jax_time - pytorch_time) / pytorch_time) * 100
        else:
            return False, "Invalid timing metrics - skipping performance guard"

        if performance_delta_percent > tolerance_percent:
            message = (
                f"Performance regression detected: JAX is {performance_delta_percent:.1f}% "
                f"slower than PyTorch (tolerance: {tolerance_percent}%)"
            )
            return True, message

        message = (
            f"Performance check passed: JAX delta is {performance_delta_percent:.1f}% "
            f"(tolerance: {tolerance_percent}%)"
        )
        return False, message


def main() -> None:
    """Main entry point for the performance guard CLI."""
    parser = argparse.ArgumentParser(
        description="Performance guard - Check for JAX vs PyTorch performance regression"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Directory containing backend performance results (default: current directory)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Maximum allowed performance degradation percentage (default: 10.0)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output except for violations"
    )

    args = parser.parse_args()

    # Create performance guard
    guard = PerformanceGuard(Path(args.results_dir))

    # Check for violations
    is_violation, message = guard.check_performance_regression(args.tolerance)

    if not args.quiet or is_violation:
        print(message)

    # Exit with appropriate code
    sys.exit(1 if is_violation else 0)


if __name__ == "__main__":
    main()
