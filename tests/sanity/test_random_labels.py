"""
Sanity test: Random labels should yield poor performance.

This test detects data leakage by verifying that a model trained on
random labels achieves near-chance performance.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@pytest.mark.sanity
def test_random_labels_poor_performance(make_loader):
    """
    Test that training on random labels yields poor performance.
    
    This is a critical sanity check - if a model achieves high accuracy
    on random labels, it indicates data leakage or overfitting.
    """
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int = 32, num_classes: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    # Create synthetic data with random labels
    train_loader = make_loader(n_samples=200, seed=42)
    val_loader = make_loader(n_samples=50, seed=123)

    # Initialize model
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Train for a few epochs (should not achieve high accuracy on random labels)
    model.train()
    for epoch in range(5):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    chance_level = 1.0 / 5  # 5 classes, so chance = 20%

    # Assert that accuracy is close to chance level (not high)
    # Allow some variance but should be well below 80%
    assert accuracy < 0.8, f"Accuracy {accuracy:.3f} too high for random labels - possible data leakage"
    assert accuracy > 0.05, f"Accuracy {accuracy:.3f} too low - model might not be learning at all"

    # Additional check: accuracy should be roughly around chance level ± reasonable margin
    margin = 0.3  # Allow ±30% around chance level
    assert abs(accuracy - chance_level) < margin, \
        f"Accuracy {accuracy:.3f} deviates too much from chance level {chance_level:.3f}"


@pytest.mark.sanity
def test_random_vs_structured_labels_difference(make_loader):
    """
    Test that structured data performs significantly better than random labels.
    
    This verifies that our test setup can distinguish between meaningful
    and random patterns.
    """
    # Create the same model architecture
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int = 32, num_classes: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    def train_and_evaluate(train_loader, val_loader):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Train
        model.train()
        for epoch in range(10):  # More epochs for structured data
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
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                pred = output.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total

    # Test with random labels (same as previous test)
    random_train = make_loader(n_samples=200, seed=42)
    random_val = make_loader(n_samples=50, seed=123)
    random_accuracy = train_and_evaluate(random_train, random_val)

    # Test with structured data (create simple pattern)
    # Generate data where class depends on feature sum sign
    torch.manual_seed(456)
    X_structured = torch.randn(200, 32)
    # Create weak but learnable pattern: class based on first few features
    y_structured = ((X_structured[:, :3].sum(dim=1) > 0).long() +
                   (X_structured[:, 3:6].sum(dim=1) > 0).long() * 2) % 5

    X_val_structured = torch.randn(50, 32)
    y_val_structured = ((X_val_structured[:, :3].sum(dim=1) > 0).long() +
                       (X_val_structured[:, 3:6].sum(dim=1) > 0).long() * 2) % 5

    from torch.utils.data import TensorDataset
    structured_train = DataLoader(TensorDataset(X_structured, y_structured), batch_size=32)
    structured_val = DataLoader(TensorDataset(X_val_structured, y_val_structured), batch_size=32)

    structured_accuracy = train_and_evaluate(structured_train, structured_val)

    # Structured data should perform notably better than random labels
    improvement = structured_accuracy - random_accuracy
    assert improvement > 0.15, \
        f"Structured accuracy {structured_accuracy:.3f} not sufficiently better than " \
        f"random accuracy {random_accuracy:.3f} (difference: {improvement:.3f})"
