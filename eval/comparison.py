"""
Metrics comparison utilities for benchmark results.

Provides tools to compare metrics across different runs,
visualize trends, and generate comparison reports.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MetricComparison:
    """Result of comparing two metric values."""
    
    metric_name: str
    current_value: float
    previous_value: float
    change_absolute: float
    change_percentage: float
    is_improvement: bool
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "change_absolute": self.change_absolute,
            "change_percentage": self.change_percentage,
            "is_improvement": self.is_improvement,
        }


class MetricsComparator:
    """Compare metrics across benchmark runs."""
    
    def __init__(self, history_path: Path | str):
        """
        Initialize metrics comparator.
        
        Args:
            history_path: Path to benchmark history directory
        """
        self.history_path = Path(history_path)
    
    def load_run(self, run_file: Path | str) -> dict[str, Any]:
        """
        Load a single benchmark run.
        
        Args:
            run_file: Path to run file
            
        Returns:
            Dictionary with run data
        """
        with open(run_file, "r") as f:
            return json.load(f)
    
    def get_latest_runs(self, n: int = 2) -> list[dict[str, Any]]:
        """
        Get the latest n benchmark runs.
        
        Args:
            n: Number of runs to retrieve
            
        Returns:
            List of run dictionaries
        """
        history_files = sorted(self.history_path.glob("*.json"))
        
        runs = []
        for file_path in history_files[-n:]:
            try:
                run_data = self.load_run(file_path)
                run_data["_file_path"] = str(file_path)
                runs.append(run_data)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return runs
    
    def compare_runs(
        self,
        current_run: dict[str, Any],
        previous_run: dict[str, Any],
        metric_directions: dict[str, bool] | None = None,
    ) -> list[MetricComparison]:
        """
        Compare two benchmark runs.
        
        Args:
            current_run: Current run data
            previous_run: Previous run data
            metric_directions: Dictionary of metric name to higher_is_better flag
            
        Returns:
            List of MetricComparison objects
        """
        if metric_directions is None:
            metric_directions = {}
        
        comparisons = []
        
        # Extract metrics from both runs
        current_metrics = self._extract_all_metrics(current_run)
        previous_metrics = self._extract_all_metrics(previous_run)
        
        # Compare common metrics
        for metric_name in current_metrics:
            if metric_name in previous_metrics:
                current_value = current_metrics[metric_name]
                previous_value = previous_metrics[metric_name]
                
                change_absolute = current_value - previous_value
                
                if previous_value != 0:
                    change_percentage = 100.0 * change_absolute / abs(previous_value)
                else:
                    change_percentage = 0.0
                
                higher_is_better = metric_directions.get(metric_name, True)
                is_improvement = (
                    (change_absolute > 0 and higher_is_better)
                    or (change_absolute < 0 and not higher_is_better)
                )
                
                comparisons.append(
                    MetricComparison(
                        metric_name=metric_name,
                        current_value=current_value,
                        previous_value=previous_value,
                        change_absolute=change_absolute,
                        change_percentage=change_percentage,
                        is_improvement=is_improvement,
                    )
                )
        
        return comparisons
    
    def _extract_all_metrics(self, run_data: dict[str, Any]) -> dict[str, float]:
        """
        Extract all numeric metrics from run data.
        
        Args:
            run_data: Run data dictionary
            
        Returns:
            Dictionary of metric name to value
        """
        metrics = {}
        
        # Common metric locations
        search_keys = ["metrics", "final_metrics", "results", "benchmark_results"]
        
        for key in search_keys:
            if key in run_data and isinstance(run_data[key], dict):
                self._extract_numeric_values(run_data[key], metrics, prefix=key)
        
        return metrics
    
    def _extract_numeric_values(
        self,
        data: dict[str, Any],
        metrics: dict[str, float],
        prefix: str = "",
    ) -> None:
        """
        Recursively extract numeric values from nested dictionary.
        
        Args:
            data: Dictionary to search
            metrics: Output dictionary to populate
            prefix: Prefix for metric names
        """
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (int, float)):
                metrics[full_key] = float(value)
            elif isinstance(value, dict):
                self._extract_numeric_values(value, metrics, full_key)
    
    def generate_comparison_report(
        self,
        comparisons: list[MetricComparison],
        output_path: Path | str | None = None,
    ) -> str:
        """
        Generate a comparison report.
        
        Args:
            comparisons: List of metric comparisons
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        report_lines = [
            "=" * 80,
            "BENCHMARK COMPARISON REPORT",
            "=" * 80,
            "",
        ]
        
        # Sort by absolute change percentage
        sorted_comparisons = sorted(
            comparisons,
            key=lambda x: abs(x.change_percentage),
            reverse=True,
        )
        
        # Add summary statistics
        improvements = [c for c in comparisons if c.is_improvement]
        degradations = [c for c in comparisons if not c.is_improvement and c.change_absolute != 0]
        
        report_lines.extend([
            f"Total Metrics Compared: {len(comparisons)}",
            f"Improvements: {len(improvements)}",
            f"Degradations: {len(degradations)}",
            f"No Change: {len(comparisons) - len(improvements) - len(degradations)}",
            "",
            "-" * 80,
            "DETAILED COMPARISONS",
            "-" * 80,
            "",
        ])
        
        # Add detailed comparisons
        for comp in sorted_comparisons:
            status = "✓ IMPROVED" if comp.is_improvement else "✗ DEGRADED"
            if comp.change_absolute == 0:
                status = "- NO CHANGE"
            
            report_lines.extend([
                f"{comp.metric_name}:",
                f"  Current:  {comp.current_value:.4f}",
                f"  Previous: {comp.previous_value:.4f}",
                f"  Change:   {comp.change_absolute:+.4f} ({comp.change_percentage:+.2f}%)",
                f"  Status:   {status}",
                "",
            ])
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
        
        return report
    
    def compute_trend(
        self,
        metric_name: str,
        num_runs: int = 10,
    ) -> dict[str, Any]:
        """
        Compute trend for a specific metric across multiple runs.
        
        Args:
            metric_name: Name of the metric
            num_runs: Number of runs to analyze
            
        Returns:
            Dictionary with trend statistics
        """
        runs = self.get_latest_runs(num_runs)
        
        values = []
        timestamps = []
        
        for run in runs:
            metrics = self._extract_all_metrics(run)
            if metric_name in metrics:
                values.append(metrics[metric_name])
                
                # Try to extract timestamp
                timestamp = run.get("timestamp", run.get("_file_path", ""))
                timestamps.append(timestamp)
        
        if len(values) < 2:
            return {
                "metric_name": metric_name,
                "num_samples": len(values),
                "trend": "insufficient_data",
            }
        
        # Compute trend statistics
        values_array = np.array(values)
        
        # Simple linear regression
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values_array, 1)
        slope = coefficients[0]
        
        trend_direction = "improving" if slope > 0 else "degrading" if slope < 0 else "stable"
        
        return {
            "metric_name": metric_name,
            "num_samples": len(values),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "trend_direction": trend_direction,
            "trend_slope": float(slope),
            "values": values,
            "timestamps": timestamps,
        }


def compare_metrics(
    history_path: Path | str,
    metric_directions: dict[str, bool] | None = None,
) -> list[MetricComparison]:
    """
    Convenience function to compare latest two runs.
    
    Args:
        history_path: Path to benchmark history directory
        metric_directions: Dictionary of metric name to higher_is_better flag
        
    Returns:
        List of MetricComparison objects
    """
    comparator = MetricsComparator(history_path)
    runs = comparator.get_latest_runs(2)
    
    if len(runs) < 2:
        print("Warning: Not enough runs to compare (need at least 2)")
        return []
    
    return comparator.compare_runs(runs[-1], runs[-2], metric_directions)
