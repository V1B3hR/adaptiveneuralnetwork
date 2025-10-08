"""
Drift test: Gaussian drift injection and measurement.

This test verifies that drift utilities can inject measurable
distribution shifts and that they affect model performance appropriately.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adaptiveneuralnetwork.utils.drift import alternating_drift, apply_gaussian_drift, apply_shift


@pytest.mark.drift
def test_gaussian_drift_injection():
    """
    Test that Gaussian drift is properly injected into data.
    """
    # Create original data
    torch.manual_seed(42)
    original_data = torch.randn(100, 32)

    # Apply Gaussian drift
    sigma = 0.5
    drifted_data = apply_gaussian_drift(original_data, sigma=sigma)

    # Check that shapes are preserved
    assert drifted_data.shape == original_data.shape, "Drift changed data shape"

    # Check that drift was actually applied (data should be different)
    assert not torch.allclose(original_data, drifted_data), "Drift had no effect"

    # Check that the amount of drift is reasonable
    diff = drifted_data - original_data
    diff_std = diff.std().item()

    # The added noise should have approximately the specified standard deviation
    # Allow some tolerance due to finite sample size
    assert abs(diff_std - sigma) < 0.1, f"Expected std ~{sigma}, got {diff_std:.3f}"

    # Check that drift is approximately zero-mean
    diff_mean = diff.mean().item()
    assert abs(diff_mean) < 0.1, f"Drift should be zero-mean, got mean {diff_mean:.3f}"


@pytest.mark.drift
def test_constant_shift_injection():
    """
    Test that constant shift is properly applied to data.
    """
    # Create original data
    torch.manual_seed(42)
    original_data = torch.randn(50, 16)

    # Apply constant shift
    delta = 1.5
    shifted_data = apply_shift(original_data, delta=delta)

    # Check that shapes are preserved
    assert shifted_data.shape == original_data.shape, "Shift changed data shape"

    # Check that the shift is exactly as expected
    diff = shifted_data - original_data
    expected_diff = torch.full_like(original_data, delta)

    assert torch.allclose(diff, expected_diff), "Constant shift not applied correctly"

    # Check that mean shifted by exactly delta
    original_mean = original_data.mean()
    shifted_mean = shifted_data.mean()
    actual_shift = shifted_mean - original_mean

    assert abs(actual_shift - delta) < 1e-6, f"Mean shift {actual_shift:.6f} != expected {delta}"


@pytest.mark.drift
def test_alternating_drift_pattern():
    """
    Test that alternating drift follows the expected temporal pattern.
    """
    torch.manual_seed(42)
    base_data = torch.randn(20, 8)

    period = 3
    sigma = 0.4

    # Test alternating Gaussian drift
    results = []
    for step in range(10):
        result = alternating_drift(base_data, step, period=period, mode="gaussian", sigma=sigma)
        results.append(result)

    # Check pattern: steps 0,1,2 -> no drift, steps 3,4,5 -> drift, steps 6,7,8 -> no drift, step 9 -> drift
    # (step // period) % 2 == 1 means drift is applied

    # No drift periods
    for step in [0, 1, 2, 6, 7, 8]:
        assert torch.allclose(results[step], base_data), f"Step {step} should have no drift"

    # Drift periods
    for step in [3, 4, 5, 9]:
        assert not torch.allclose(results[step], base_data), f"Step {step} should have drift"
        # Check that drift magnitude is reasonable
        diff = results[step] - base_data
        diff_std = diff.std().item()
        assert 0.2 < diff_std < 0.8, f"Step {step} drift magnitude {diff_std:.3f} seems wrong"


@pytest.mark.drift
def test_alternating_shift_pattern():
    """
    Test alternating constant shift pattern.
    """
    torch.manual_seed(123)
    base_data = torch.randn(15, 12)

    period = 4
    delta = 0.8

    # Test alternating constant shift
    results = []
    for step in range(12):
        result = alternating_drift(base_data, step, period=period, mode="shift", delta=delta)
        results.append(result)

    # Check pattern with period=4:
    # steps 0,1,2,3 -> no drift, steps 4,5,6,7 -> drift, steps 8,9,10,11 -> no drift

    # No drift periods
    for step in [0, 1, 2, 3, 8, 9, 10, 11]:
        assert torch.allclose(results[step], base_data), f"Step {step} should have no drift"

    # Drift periods
    for step in [4, 5, 6, 7]:
        expected = base_data + delta
        assert torch.allclose(results[step], expected), f"Step {step} should have shift {delta}"


@pytest.mark.drift
def test_drift_affects_model_performance():
    """
    Test that injected drift measurably affects model performance.
    
    This is a critical test to ensure drift injection is effective.
    """
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int = 32, num_classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    def create_structured_data(n_samples: int, seed: int):
        """Create data with learnable structure."""
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(n_samples, 32, generator=g)

        # Create pattern: class depends on first few features
        y = torch.zeros(n_samples, dtype=torch.long)
        y[X[:, 0] + X[:, 1] > 0.5] = 1
        y[X[:, 0] - X[:, 1] > 1.0] = 2

        return X, y

    def train_and_evaluate(train_loader, test_loader):
        """Train model and return test accuracy."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Train
        model.train()
        for epoch in range(12):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total

    # Create original structured training data
    X_train, y_train = create_structured_data(200, seed=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    # Create test data (no drift)
    X_test_clean, y_test_clean = create_structured_data(80, seed=100)
    test_loader_clean = DataLoader(TensorDataset(X_test_clean, y_test_clean), batch_size=80)

    # Create drifted test data
    X_test_drifted = apply_gaussian_drift(X_test_clean, sigma=0.7)  # Strong drift
    test_loader_drifted = DataLoader(TensorDataset(X_test_drifted, y_test_clean), batch_size=80)

    # Train on clean data, test on both clean and drifted
    clean_accuracy = train_and_evaluate(train_loader, test_loader_clean)
    drifted_accuracy = train_and_evaluate(train_loader, test_loader_drifted)

    # Clean data should achieve reasonable performance
    assert clean_accuracy > 0.6, f"Clean accuracy {clean_accuracy:.3f} too low"

    # Drifted data should perform worse due to distribution shift
    performance_drop = clean_accuracy - drifted_accuracy
    assert performance_drop > 0.05, \
        f"Drift should cause performance drop >0.05, but got {performance_drop:.3f} " \
        f"(clean: {clean_accuracy:.3f}, drifted: {drifted_accuracy:.3f})"

    # Drifted performance should still be above chance but noticeably worse
    chance_level = 1.0 / 3
    assert drifted_accuracy > chance_level + 0.1, f"Drifted accuracy {drifted_accuracy:.3f} too low"


@pytest.mark.drift
def test_drift_magnitude_scaling():
    """
    Test that larger drift magnitudes cause larger performance drops.
    """
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int = 32, num_classes: int = 2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    # Create simple binary classification data
    torch.manual_seed(200)
    X_train = torch.randn(150, 32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()  # Simple decision boundary
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    X_test = torch.randn(60, 32)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).long()

    def quick_train_eval(test_data):
        """Quick training and evaluation."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()

        # Quick training
        model.train()
        for epoch in range(8):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            output = model(test_data)
            pred = output.argmax(dim=1)
            accuracy = (pred == y_test).float().mean().item()

        return accuracy

    # Test different drift magnitudes
    no_drift_acc = quick_train_eval(X_test)
    mild_drift_acc = quick_train_eval(apply_gaussian_drift(X_test, sigma=0.3))
    strong_drift_acc = quick_train_eval(apply_gaussian_drift(X_test, sigma=0.8))

    # Stronger drift should cause larger performance drops
    mild_drop = no_drift_acc - mild_drift_acc
    strong_drop = no_drift_acc - strong_drift_acc

    assert strong_drop > mild_drop, \
        f"Strong drift drop {strong_drop:.3f} should be > mild drift drop {mild_drop:.3f}"

    # All should be positive (drift should hurt performance)
    assert mild_drop > 0, f"Mild drift should hurt performance, got drop {mild_drop:.3f}"
    assert strong_drop > 0, f"Strong drift should hurt performance, got drop {strong_drop:.3f}"

    # No drift should achieve reasonable baseline
    assert no_drift_acc > 0.7, f"No drift accuracy {no_drift_acc:.3f} too low for simple pattern"
