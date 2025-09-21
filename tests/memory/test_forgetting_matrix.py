"""
Memory test: Forgetting matrix construction and analysis.

This test verifies the forgetting matrix utilities work correctly
and can track catastrophic forgetting across multiple tasks.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from adaptiveneuralnetwork.utils.forgetting import compute_forgetting_metrics, forgetting_matrix


@pytest.mark.memory
def test_forgetting_matrix_construction():
    """
    Test basic construction of forgetting matrix from evaluation snapshots.
    """
    # Simulate evaluation results after learning each task
    # After task 0: only task 0 exists
    # After task 1: task 0 performance drops (forgetting), task 1 is new
    # After task 2: both previous tasks drop further, task 2 is new

    evals_by_task = [
        {0: 0.95},  # After learning task 0
        {0: 0.85, 1: 0.92},  # After learning task 1 (task 0 forgot 0.10)
        {0: 0.75, 1: 0.88, 2: 0.94},  # After learning task 2 (more forgetting)
    ]

    matrix = forgetting_matrix(evals_by_task)

    # Check shape
    assert matrix.shape == (3, 3), f"Expected shape (3, 3), got {matrix.shape}"

    # Check specific values
    assert matrix[0, 0] == 0.95, "Task 0 accuracy after learning task 0"
    assert np.isnan(matrix[0, 1]), "Task 1 should be NaN before it's learned"
    assert np.isnan(matrix[0, 2]), "Task 2 should be NaN before it's learned"

    assert matrix[1, 0] == 0.85, "Task 0 accuracy after learning task 1"
    assert matrix[1, 1] == 0.92, "Task 1 accuracy after learning task 1"
    assert np.isnan(matrix[1, 2]), "Task 2 should still be NaN"

    assert matrix[2, 0] == 0.75, "Task 0 accuracy after learning task 2"
    assert matrix[2, 1] == 0.88, "Task 1 accuracy after learning task 2"
    assert matrix[2, 2] == 0.94, "Task 2 accuracy after learning task 2"


@pytest.mark.memory
def test_forgetting_metrics_computation():
    """
    Test computation of forgetting metrics from a forgetting matrix.
    """
    # Create a forgetting matrix with known forgetting patterns
    matrix = np.array([[0.95, np.nan, np.nan], [0.85, 0.92, np.nan], [0.75, 0.88, 0.94]])

    metrics = compute_forgetting_metrics(matrix)

    # Check that all expected metrics are present
    assert "average_forgetting" in metrics
    assert "maximum_forgetting" in metrics
    assert "final_average_accuracy" in metrics

    # Task 0: best=0.95, final=0.75, forgetting=0.20
    # Task 1: best=0.92, final=0.88, forgetting=0.04
    # Task 2: only appears once, so no forgetting calculated (excluded from average)
    expected_avg_forgetting = (0.20 + 0.04) / 2  # Only tasks with multiple data points
    expected_max_forgetting = 0.20
    expected_final_avg = (0.75 + 0.88 + 0.94) / 3

    assert abs(metrics["average_forgetting"] - expected_avg_forgetting) < 1e-5
    assert abs(metrics["maximum_forgetting"] - expected_max_forgetting) < 1e-5
    assert abs(metrics["final_average_accuracy"] - expected_final_avg) < 1e-5


@pytest.mark.memory
def test_empty_forgetting_matrix():
    """
    Test handling of empty evaluation lists.
    """
    # Empty input
    matrix = forgetting_matrix([])
    assert matrix.size == 0

    # Empty evaluation dicts
    matrix = forgetting_matrix([{}, {}])
    assert matrix.size == 0

    # Test metrics on empty matrix
    metrics = compute_forgetting_metrics(np.array([]))
    assert metrics["average_forgetting"] == 0.0
    assert metrics["maximum_forgetting"] == 0.0
    assert metrics["final_average_accuracy"] == 0.0


@pytest.mark.memory
def test_single_task_no_forgetting():
    """
    Test that single task scenario shows no forgetting.
    """
    evals_by_task = [{0: 0.87}]
    matrix = forgetting_matrix(evals_by_task)

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.87

    metrics = compute_forgetting_metrics(matrix)
    assert metrics["average_forgetting"] == 0.0  # No forgetting possible with single task
    assert metrics["maximum_forgetting"] == 0.0
    assert metrics["final_average_accuracy"] == 0.87


@pytest.mark.memory
def test_forgetting_matrix_with_gaps():
    """
    Test forgetting matrix with non-consecutive task IDs.
    """
    # Task IDs 0, 2, 4 (skip 1, 3)
    evals_by_task = [{0: 0.90}, {0: 0.85, 2: 0.88}, {0: 0.80, 2: 0.83, 4: 0.91}]

    matrix = forgetting_matrix(evals_by_task)

    # Should have shape (3, 5) to accommodate task ID 4
    assert matrix.shape == (3, 5)

    # Check that unused task IDs (1, 3) are all NaN
    assert np.all(np.isnan(matrix[:, 1]))
    assert np.all(np.isnan(matrix[:, 3]))

    # Check actual values
    assert matrix[0, 0] == 0.90
    assert matrix[1, 0] == 0.85
    assert matrix[1, 2] == 0.88
    assert matrix[2, 4] == 0.91


@pytest.mark.memory
def test_continual_learning_forgetting_simulation(make_loader):
    """
    Test forgetting matrix in a realistic continual learning scenario.

    This test simulates learning multiple tasks sequentially and tracks
    the resulting forgetting pattern.
    """

    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int = 32, num_classes: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    def evaluate_model(model, loader):
        """Evaluate model accuracy on a data loader."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                output = model(batch_x)
                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total if total > 0 else 0.0

    # Create multiple task data loaders
    task_loaders = {}
    for task_id in range(3):
        task_loaders[task_id] = make_loader(n_samples=100, seed=task_id * 100)

    # Initialize model
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()

    # Track evaluations after learning each task
    evals_by_task = []

    for current_task in range(3):
        # Train on current task
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_loader = task_loaders[current_task]

        model.train()
        for epoch in range(10):  # Limited training to induce some forgetting
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate on all tasks learned so far
        current_evals = {}
        for task_id in range(current_task + 1):
            accuracy = evaluate_model(model, task_loaders[task_id])
            current_evals[task_id] = accuracy

        evals_by_task.append(current_evals)

    # Build forgetting matrix
    matrix = forgetting_matrix(evals_by_task)

    # Basic sanity checks
    assert matrix.shape[0] == 3, "Should have 3 time steps"
    assert matrix.shape[1] >= 3, "Should accommodate at least 3 tasks"

    # Check that diagonal (final performance on each task) is reasonable
    final_performance_task_0 = matrix[2, 0]  # Task 0 performance after learning all tasks
    final_performance_task_1 = matrix[2, 1]  # Task 1 performance after learning all tasks
    final_performance_task_2 = matrix[2, 2]  # Task 2 performance after learning all tasks

    # All should be above chance (0.2 for 5 classes)
    assert final_performance_task_0 > 0.25, "Task 0 final performance too low"
    assert final_performance_task_1 > 0.25, "Task 1 final performance too low"
    assert final_performance_task_2 > 0.25, "Task 2 final performance too low"

    # Compute and check forgetting metrics
    metrics = compute_forgetting_metrics(matrix)

    # Should have some measurable forgetting (this is expected in basic continual learning)
    assert metrics["average_forgetting"] >= 0.0, "Average forgetting should be non-negative"
    assert metrics["maximum_forgetting"] >= 0.0, "Maximum forgetting should be non-negative"

    # Final average accuracy should be reasonable
    assert metrics["final_average_accuracy"] > 0.25, "Final average accuracy too low"
    assert metrics["final_average_accuracy"] <= 1.0, "Final average accuracy too high"
