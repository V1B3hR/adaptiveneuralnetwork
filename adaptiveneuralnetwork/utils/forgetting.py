"""
Forgetting utility functions for continual learning analysis.

This module provides utilities for constructing and analyzing forgetting matrices
to measure catastrophic forgetting in continual learning systems.
"""

from typing import Dict, List

import numpy as np


def forgetting_matrix(evals_by_task: List[Dict[int, float]]) -> np.ndarray:
    """
    Build task-by-task forgetting matrix from accuracy snapshots.

    Args:
        evals_by_task: List where each element is a dict mapping task_id -> accuracy
                      for the state after learning each task

    Returns:
        2D numpy array M where M[i,j] is the accuracy on task j after learning task i
        NaN values indicate task j was not yet learned when task i was completed

    Example:
        # After learning task 0: {0: 0.95}
        # After learning task 1: {0: 0.85, 1: 0.92}  # task 0 forgot 0.1
        # After learning task 2: {0: 0.80, 1: 0.88, 2: 0.94}
        evals = [{0: 0.95}, {0: 0.85, 1: 0.92}, {0: 0.80, 1: 0.88, 2: 0.94}]
        matrix = forgetting_matrix(evals)
        # matrix[1, 0] = 0.85 (accuracy on task 0 after learning task 1)
    """
    if not evals_by_task:
        return np.array([])

    # Determine the number of tasks
    all_task_ids = set()
    for eval_dict in evals_by_task:
        all_task_ids.update(eval_dict.keys())

    if not all_task_ids:
        return np.array([])

    max_task_id = max(all_task_ids)
    T = len(evals_by_task)
    num_tasks = max_task_id + 1

    # Initialize matrix with NaN
    M = np.full((T, num_tasks), np.nan)

    # Fill in the matrix
    for t in range(T):
        for task_id, acc in evals_by_task[t].items():
            M[t, task_id] = acc

    return M


def compute_forgetting_metrics(matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute forgetting metrics from a forgetting matrix.

    Args:
        matrix: Forgetting matrix from forgetting_matrix()

    Returns:
        Dictionary with forgetting metrics:
        - average_forgetting: Average amount of forgetting across all tasks
        - maximum_forgetting: Maximum forgetting experienced by any task
        - final_average_accuracy: Average accuracy on all tasks at the end
    """
    if matrix.size == 0:
        return {"average_forgetting": 0.0, "maximum_forgetting": 0.0, "final_average_accuracy": 0.0}

    T, num_tasks = matrix.shape
    forgetting_values = []

    # Calculate forgetting for each task
    for task_id in range(num_tasks):
        task_accs = matrix[:, task_id]
        # Find valid (non-NaN) accuracies
        valid_accs = task_accs[~np.isnan(task_accs)]

        if len(valid_accs) > 1:
            # Forgetting is the difference between best and final accuracy
            best_acc = np.max(valid_accs)
            final_acc = valid_accs[-1]  # Last valid accuracy
            forgetting = max(0, best_acc - final_acc)
            forgetting_values.append(forgetting)

    # Final row contains final accuracies
    final_accs = matrix[-1, :]
    final_valid_accs = final_accs[~np.isnan(final_accs)]

    return {
        "average_forgetting": np.mean(forgetting_values) if forgetting_values else 0.0,
        "maximum_forgetting": np.max(forgetting_values) if forgetting_values else 0.0,
        "final_average_accuracy": np.mean(final_valid_accs) if len(final_valid_accs) > 0 else 0.0,
    }
