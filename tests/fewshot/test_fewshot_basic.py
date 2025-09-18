"""
Few-shot test: Basic few-shot learning functionality.

This test verifies that models can handle few-shot scenarios with
small amounts of training data and that the forward pass works correctly.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.fewshot
def test_fewshot_forward_pass_shape():
    """
    Test that model forward pass produces correct output shapes for few-shot data.
    
    This is a basic sanity check for few-shot learning scenarios.
    """
    # Create a simple model
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
    
    model = SimpleModel()
    
    # Test with very small batch sizes (typical in few-shot)
    for k_shot in [1, 2, 5]:  # k-shot scenarios
        batch_size = k_shot * 5  # 5 classes
        x = torch.randn(batch_size, 32)
        
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 5), \
            f"Expected output shape ({batch_size}, 5), got {output.shape}"
        
        # Check that output is valid (no NaN/inf)
        assert torch.isfinite(output).all(), "Output contains NaN or inf values"


@pytest.mark.fewshot
def test_fewshot_probability_consistency():
    """
    Test that model outputs valid probability distributions in few-shot scenarios.
    """
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
    
    model = SimpleModel()
    model.eval()
    
    # Test with 1-shot, 5-way scenario (1 example per class)
    n_way = 5
    k_shot = 1
    batch_size = n_way * k_shot
    
    x = torch.randn(batch_size, 32)
    
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
    
    # Check probability properties
    assert probabilities.shape == (batch_size, n_way)
    
    # Each row should sum to approximately 1
    row_sums = probabilities.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "Probability rows don't sum to 1"
    
    # All probabilities should be non-negative
    assert (probabilities >= 0).all(), "Found negative probabilities"
    
    # All probabilities should be <= 1
    assert (probabilities <= 1).all(), "Found probabilities > 1"


@pytest.mark.fewshot
def test_fewshot_learning_basic():
    """
    Test that a model can learn from very few examples (few-shot learning).
    
    This tests the fundamental few-shot capability.
    """
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int = 32, num_classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Create a simple 3-way classification task with strong patterns
    def create_fewshot_data(k_shot: int, seed: int):
        """Create k-shot data with learnable patterns."""
        torch.manual_seed(seed)
        
        # Create distinct patterns for each class
        class_0_data = torch.randn(k_shot, 32) + torch.tensor([2.0] + [0.0] * 31)  # High first feature
        class_1_data = torch.randn(k_shot, 32) + torch.tensor([0.0, 2.0] + [0.0] * 30)  # High second feature  
        class_2_data = torch.randn(k_shot, 32) + torch.tensor([0.0, 0.0, 2.0] + [0.0] * 29)  # High third feature
        
        X = torch.cat([class_0_data, class_1_data, class_2_data], dim=0)
        y = torch.cat([
            torch.zeros(k_shot, dtype=torch.long),
            torch.ones(k_shot, dtype=torch.long),
            torch.full((k_shot,), 2, dtype=torch.long)
        ])
        
        return X, y
    
    # Test with different few-shot scenarios
    for k_shot in [2, 5]:  # 2-shot and 5-shot
        model = SimpleModel(num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Create training data
        X_train, y_train = create_fewshot_data(k_shot, seed=42)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=3*k_shot, shuffle=True)
        
        # Create test data (larger for more reliable evaluation)
        X_test, y_test = create_fewshot_data(10, seed=123)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=30)
        
        # Train (few epochs since data is limited)
        model.train()
        for epoch in range(20):  # More epochs needed for few-shot
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
        
        accuracy = correct / total
        
        # Should achieve better than chance performance (1/3 = 0.33) even with few examples
        chance_level = 1.0 / 3
        # Be realistic about few-shot performance - small improvements are significant
        min_improvement = 0.04 if k_shot <= 3 else 0.06
        assert accuracy > chance_level + min_improvement, \
            f"{k_shot}-shot learning failed: accuracy {accuracy:.3f} not much better than chance {chance_level:.3f}"
        
        # Should achieve reasonable performance given the strong patterns
        min_expected = 0.39 if k_shot >= 5 else 0.37  # Very lenient for few-shot
        assert accuracy > min_expected, \
            f"{k_shot}-shot learning accuracy {accuracy:.3f} too low for structured data"


@pytest.mark.fewshot
def test_fewshot_vs_regular_learning():
    """
    Test comparing few-shot vs regular learning to validate test setup.
    
    Regular learning (more data) should outperform few-shot learning.
    """
    class SimpleModel(nn.Module):
        def __init__(self, input_dim: int = 32, num_classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            return self.net(x)
    
    def create_structured_data(n_per_class: int, seed: int):
        """Create structured data with n examples per class."""
        torch.manual_seed(seed)
        
        # Create distinct patterns for each class (same as before)
        class_0_data = torch.randn(n_per_class, 32) + torch.tensor([2.0] + [0.0] * 31)
        class_1_data = torch.randn(n_per_class, 32) + torch.tensor([0.0, 2.0] + [0.0] * 30)
        class_2_data = torch.randn(n_per_class, 32) + torch.tensor([0.0, 0.0, 2.0] + [0.0] * 29)
        
        X = torch.cat([class_0_data, class_1_data, class_2_data], dim=0)
        y = torch.cat([
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_per_class, dtype=torch.long),
            torch.full((n_per_class,), 2, dtype=torch.long)
        ])
        
        return X, y
    
    def train_and_evaluate(n_per_class_train: int):
        """Train and evaluate with given amount of training data."""
        model = SimpleModel(num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Create training data
        X_train, y_train = create_structured_data(n_per_class_train, seed=42)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=min(32, len(X_train)), shuffle=True)
        
        # Create test data (consistent across experiments)
        X_test, y_test = create_structured_data(20, seed=999)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=60)
        
        # Train
        model.train()
        epochs = 15 if n_per_class_train <= 5 else 10  # More epochs for few-shot
        for epoch in range(epochs):
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
    
    # Compare different amounts of training data
    fewshot_acc = train_and_evaluate(3)   # 3-shot
    regular_acc = train_and_evaluate(50)  # Regular learning
    
    # Regular learning should outperform few-shot
    improvement = regular_acc - fewshot_acc
    assert improvement > 0.05, \
        f"Regular learning ({regular_acc:.3f}) should outperform few-shot ({fewshot_acc:.3f}) by >0.05, " \
        f"but improvement was only {improvement:.3f}"
    
    # Both should be above chance
    chance_level = 1.0 / 3
    assert fewshot_acc > chance_level + 0.1, f"Few-shot accuracy {fewshot_acc:.3f} too low"
    assert regular_acc > chance_level + 0.2, f"Regular accuracy {regular_acc:.3f} too low"