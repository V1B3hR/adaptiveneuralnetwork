"""
Tests for Phase 6 evaluation and validation layer.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import StandardMetrics, compute_metrics
from eval.microbenchmark import MicroBenchmark, run_microbenchmark
from eval.drift_detection import DriftDetector, detect_drift
from eval.comparison import MetricsComparator, compare_metrics


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    # Create synthetic data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=False)


@pytest.fixture
def temp_history_dir():
    """Create a temporary directory for history files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMetrics:
    """Test metrics computation."""
    
    def test_compute_metrics_basic(self, simple_model, simple_dataloader):
        """Test basic metrics computation."""
        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()
        
        metrics = compute_metrics(
            model=simple_model,
            data_loader=simple_dataloader,
            device=device,
            loss_fn=loss_fn,
            compute_detailed=True,
        )
        
        assert isinstance(metrics, StandardMetrics)
        assert metrics.num_samples == 100
        assert 0 <= metrics.accuracy <= 100
        assert metrics.loss >= 0
        assert metrics.evaluation_time > 0
        assert metrics.throughput > 0
    
    def test_metrics_to_dict(self, simple_model, simple_dataloader):
        """Test metrics serialization."""
        device = torch.device("cpu")
        
        metrics = compute_metrics(
            model=simple_model,
            data_loader=simple_dataloader,
            device=device,
            loss_fn=None,
            compute_detailed=False,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert "accuracy" in metrics_dict
        assert "loss" in metrics_dict
        assert "num_samples" in metrics_dict


class TestMicrobenchmark:
    """Test microbenchmarking functionality."""
    
    def test_benchmark_forward_latency(self, simple_model, simple_dataloader):
        """Test forward pass latency benchmarking."""
        device = torch.device("cpu")
        benchmark = MicroBenchmark(simple_model, device)
        
        latency = benchmark.benchmark_forward_latency(
            data_loader=simple_dataloader,
            num_iterations=10,
            warmup_iterations=2,
        )
        
        assert "mean" in latency
        assert "std" in latency
        assert "min" in latency
        assert "max" in latency
        assert latency["mean"] > 0
        assert latency["min"] <= latency["mean"] <= latency["max"]
    
    def test_benchmark_data_loader(self, simple_dataloader):
        """Test data loader throughput benchmarking."""
        device = torch.device("cpu")
        model = SimpleModel()
        benchmark = MicroBenchmark(model, device)
        
        throughput = benchmark.benchmark_data_loader(
            data_loader=simple_dataloader,
            num_batches=5,
        )
        
        assert "samples_per_second" in throughput
        assert "batches_per_second" in throughput
        assert throughput["samples_per_second"] > 0
    
    def test_run_full_benchmark(self, simple_model, simple_dataloader):
        """Test full benchmark suite."""
        device = torch.device("cpu")
        
        results = run_microbenchmark(
            model=simple_model,
            data_loader=simple_dataloader,
            device=device,
            num_iterations=10,
            warmup_iterations=2,
        )
        
        assert results.forward_latency_mean > 0
        assert results.data_loader_throughput > 0
        assert results.num_iterations == 10
        
        # Test serialization
        results_dict = results.to_dict()
        assert isinstance(results_dict, dict)
        assert "forward_latency_ms" in results_dict
        assert "throughput" in results_dict


class TestDriftDetection:
    """Test drift detection functionality."""
    
    def test_drift_detection_insufficient_history(self, temp_history_dir):
        """Test drift detection with insufficient history."""
        detector = DriftDetector(temp_history_dir, lookback_n=5, threshold_std=2.0)
        
        result = detector.detect_drift(
            current_value=0.95,
            metric_name="accuracy",
            higher_is_better=True,
        )
        
        assert not result.drift_detected
        assert result.drift_direction == "stable"
    
    def test_drift_detection_with_history(self, temp_history_dir):
        """Test drift detection with sufficient history."""
        # Create mock history files
        for i in range(5):
            history_file = temp_history_dir / f"run_{i}.json"
            with open(history_file, "w") as f:
                json.dump({
                    "metrics": {
                        "accuracy": 90.0 + i * 0.5,
                    }
                }, f)
        
        detector = DriftDetector(temp_history_dir, lookback_n=5, threshold_std=2.0)
        
        # Test with value within normal range
        result = detector.detect_drift(
            current_value=92.0,
            metric_name="accuracy",
            higher_is_better=True,
        )
        
        assert not result.drift_detected
        assert result.baseline_median > 0
        
        # Test with value far outside normal range
        result_drift = detector.detect_drift(
            current_value=80.0,  # Much lower than baseline
            metric_name="accuracy",
            higher_is_better=True,
        )
        
        assert result_drift.drift_detected
        assert result_drift.drift_direction == "degrading"
    
    def test_detect_multiple_drifts(self, temp_history_dir):
        """Test detecting drift for multiple metrics."""
        # Create mock history
        for i in range(3):
            history_file = temp_history_dir / f"run_{i}.json"
            with open(history_file, "w") as f:
                json.dump({
                    "metrics": {
                        "accuracy": 90.0,
                        "loss": 0.5,
                    }
                }, f)
        
        current_metrics = {
            "accuracy": 92.0,
            "loss": 0.48,
        }
        
        results = detect_drift(
            current_metrics=current_metrics,
            history_path=temp_history_dir,
            lookback_n=3,
            threshold_std=2.0,
        )
        
        assert len(results) == 2
        assert all(isinstance(r.drift_detected, bool) for r in results)


class TestComparison:
    """Test metrics comparison functionality."""
    
    def test_compare_runs(self, temp_history_dir):
        """Test comparing two runs."""
        # Create two mock runs
        run1 = temp_history_dir / "run_1.json"
        run2 = temp_history_dir / "run_2.json"
        
        with open(run1, "w") as f:
            json.dump({
                "metrics": {
                    "accuracy": 90.0,
                    "loss": 0.5,
                }
            }, f)
        
        with open(run2, "w") as f:
            json.dump({
                "metrics": {
                    "accuracy": 92.0,
                    "loss": 0.45,
                }
            }, f)
        
        comparator = MetricsComparator(temp_history_dir)
        runs = comparator.get_latest_runs(2)
        
        assert len(runs) == 2
        
        comparisons = comparator.compare_runs(
            current_run=runs[-1],
            previous_run=runs[-2],
            metric_directions={"accuracy": True, "loss": False},
        )
        
        assert len(comparisons) > 0
        
        # Check accuracy comparison
        acc_comp = next(c for c in comparisons if "accuracy" in c.metric_name)
        assert acc_comp.current_value == 92.0
        assert acc_comp.previous_value == 90.0
        assert acc_comp.is_improvement
    
    def test_compute_trend(self, temp_history_dir):
        """Test trend computation."""
        # Create mock history with increasing trend
        for i in range(5):
            history_file = temp_history_dir / f"run_{i}.json"
            with open(history_file, "w") as f:
                json.dump({
                    "metrics": {
                        "accuracy": 85.0 + i * 2.0,
                    }
                }, f)
        
        comparator = MetricsComparator(temp_history_dir)
        # Note: metrics are extracted with prefix, so use "metrics.accuracy"
        trend = comparator.compute_trend("metrics.accuracy", num_runs=5)
        
        assert trend["num_samples"] == 5
        assert trend["trend_direction"] == "improving"
        assert "mean" in trend
        assert "std" in trend
    
    def test_generate_comparison_report(self, temp_history_dir):
        """Test report generation."""
        comparator = MetricsComparator(temp_history_dir)
        
        # Create some mock comparisons
        from eval.comparison import MetricComparison
        comparisons = [
            MetricComparison(
                metric_name="accuracy",
                current_value=92.0,
                previous_value=90.0,
                change_absolute=2.0,
                change_percentage=2.22,
                is_improvement=True,
            ),
            MetricComparison(
                metric_name="loss",
                current_value=0.45,
                previous_value=0.50,
                change_absolute=-0.05,
                change_percentage=-10.0,
                is_improvement=True,
            ),
        ]
        
        report = comparator.generate_comparison_report(comparisons)
        
        assert isinstance(report, str)
        assert "BENCHMARK COMPARISON REPORT" in report
        assert "accuracy" in report
        assert "loss" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
