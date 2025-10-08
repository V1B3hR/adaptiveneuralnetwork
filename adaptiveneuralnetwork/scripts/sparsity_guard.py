#!/usr/bin/env python3
"""
Sparsity Guard CLI - Adaptive sparsity drift detector.

Alerts if network collapses to too sparse or dense, detecting unhealthy sparsity patterns.
Exits 0 when data is missing or within thresholds, exits 1 when violations are detected.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class SparsityGuard:
    """Guards against unhealthy sparsity patterns."""

    def __init__(self, results_dir: Path = Path(".")):
        self.results_dir = results_dir
        self.results_data = {}
        self.load_sparsity_data()

    def load_sparsity_data(self) -> None:
        """Load all available data that might contain sparsity metrics."""
        # Files that might contain sparsity/activity data
        data_files = [
            "benchmark_results.json",
            "enhanced_robustness_results.json",
            "adversarial_results.json",
            "final_validation.json"
        ]

        for filename in data_files:
            filepath = self.results_dir / filename
            if filepath.exists():
                try:
                    with open(filepath) as f:
                        self.results_data[filename] = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {filepath}: {e}")

        # Also load from benchmark_results subdirectory
        benchmark_dir = self.results_dir / "benchmark_results"
        if benchmark_dir.exists():
            for result_file in benchmark_dir.glob("*.json"):
                try:
                    with open(result_file) as f:
                        self.results_data[f"benchmark_results_{result_file.name}"] = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load {result_file}: {e}")

    def extract_sparsity_metrics(self) -> list[dict[str, Any]]:
        """Extract all sparsity-related metrics from loaded data."""
        sparsity_metrics = []

        for source_name, data in self.results_data.items():
            if not isinstance(data, dict):
                continue

            # Look for various sparsity indicators
            sparsity_paths = [
                ("active_node_ratio", "active_node_ratio"),
                ("active_phase_ratio", "active_phase_ratio"),
                ("sparsity", "sparsity"),
                ("node_activity", "node_activity"),
                ("phase_distribution", "phase_distribution"),
                ("active_nodes_percent", "active_nodes_percent"),
                ("node_utilization", "node_utilization")
            ]

            for metric_name, path in sparsity_paths:
                value = self.extract_metric_value(data, path)
                if value is not None:
                    sparsity_metrics.append({
                        "source": source_name,
                        "metric": metric_name,
                        "value": value,
                        "path": path
                    })

        return sparsity_metrics

    def extract_metric_value(self, data: dict[str, Any], metric_path: str) -> float | None:
        """Extract metric value from nested dictionary using dot notation."""
        keys = metric_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None

            # Handle different value types
            if isinstance(current, (int, float)):
                return float(current)
            elif isinstance(current, dict):
                # For phase_distribution, calculate average activity
                if all(isinstance(v, (int, float)) for v in current.values()):
                    values = list(current.values())
                    return sum(values) / len(values) if values else None
            elif isinstance(current, list):
                # For lists, calculate average
                if all(isinstance(v, (int, float)) for v in current):
                    return sum(current) / len(current) if current else None

            return None
        except (KeyError, TypeError, ZeroDivisionError):
            return None

    def check_sparsity_health(
        self,
        min_activity: float = 0.05,  # 5% minimum activity
        max_activity: float = 0.95,  # 95% maximum activity
    ) -> tuple[bool, str]:
        """
        Check sparsity metrics for unhealthy patterns.
        
        Args:
            min_activity: Minimum healthy activity ratio (0.0-1.0)
            max_activity: Maximum healthy activity ratio (0.0-1.0)
            
        Returns:
            Tuple of (is_violation, message)
        """
        sparsity_metrics = self.extract_sparsity_metrics()

        if not sparsity_metrics:
            return False, "Sparsity data not available - skipping sparsity guard"

        violations = []
        healthy_metrics = []
        warnings = []

        for metric in sparsity_metrics:
            source = metric["source"]
            metric_name = metric["metric"]
            value = metric["value"]

            # Convert percentage to ratio if needed
            if value > 1.0:
                value = value / 100.0
                warnings.append(f"Converted {metric_name} from percentage to ratio: {value:.3f}")

            # Check for collapse to too sparse (network nearly dead)
            if value < min_activity:
                violations.append(
                    f"Network too sparse: {metric_name} ({source}) = {value:.3f} < {min_activity} (collapse risk)"
                )
            # Check for collapse to too dense (network not adaptive)
            elif value > max_activity:
                violations.append(
                    f"Network too dense: {metric_name} ({source}) = {value:.3f} > {max_activity} (over-activation)"
                )
            else:
                healthy_metrics.append(
                    f"{metric_name} ({source}) = {value:.3f} (healthy range)"
                )

        # Prepare result message
        if violations:
            message = "Sparsity violations detected:\n" + "\n".join(f"  - {v}" for v in violations)
            if healthy_metrics:
                message += "\nHealthy metrics:\n" + "\n".join(f"  - {m}" for m in healthy_metrics)
            if warnings:
                message += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings)
            return True, message
        else:
            message_parts = []
            if healthy_metrics:
                message_parts.append("Sparsity check passed:\n" + "\n".join(f"  - {m}" for m in healthy_metrics))
            if warnings:
                message_parts.append("Warnings:\n" + "\n".join(f"  - {w}" for w in warnings))

            message = "\n".join(message_parts) if message_parts else "Sparsity metrics healthy"
            return False, message


def main() -> None:
    """Main entry point for the sparsity guard CLI."""
    parser = argparse.ArgumentParser(
        description="Sparsity guard - Detect unhealthy network sparsity patterns"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Directory containing benchmark/validation results (default: current directory)"
    )
    parser.add_argument(
        "--min-activity",
        type=float,
        default=0.05,
        help="Minimum healthy activity ratio (0.0-1.0, default: 0.05)"
    )
    parser.add_argument(
        "--max-activity",
        type=float,
        default=0.95,
        help="Maximum healthy activity ratio (0.0-1.0, default: 0.95)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except for violations"
    )

    args = parser.parse_args()

    # Validate arguments
    if not (0.0 <= args.min_activity <= 1.0):
        print("Error: min-activity must be between 0.0 and 1.0")
        sys.exit(1)

    if not (0.0 <= args.max_activity <= 1.0):
        print("Error: max-activity must be between 0.0 and 1.0")
        sys.exit(1)

    if args.min_activity >= args.max_activity:
        print("Error: min-activity must be less than max-activity")
        sys.exit(1)

    # Create sparsity guard
    guard = SparsityGuard(Path(args.results_dir))

    # Check for violations
    is_violation, message = guard.check_sparsity_health(
        args.min_activity,
        args.max_activity
    )

    if not args.quiet or is_violation:
        print(message)

    # Exit with appropriate code
    sys.exit(1 if is_violation else 0)


if __name__ == "__main__":
    main()
