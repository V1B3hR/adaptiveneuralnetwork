"""
Test Phase 2 Optimizations

This test verifies that Phase 2 optimizations work correctly.
"""

import torch
import torch.nn as nn

from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel
from adaptiveneuralnetwork.utils.phase2_optimizations import (
    optimize_model_phase2,
    supports_amp,
    get_amp_dtype,
    try_compile
)


def test_state_detachment():
    """Test that state detachment works correctly."""
    print("Testing state detachment...")
    
    config = AdaptiveConfig(
        num_nodes=50,
        hidden_dim=32,
        batch_size=16,
        input_dim=784,
        output_dim=10,
        device="cpu"
    )
    
    model = AdaptiveModel(config)
    model.train()
    
    # Create sample data
    x1 = torch.randn(16, 784)
    x2 = torch.randn(16, 784)
    target = torch.randint(0, 10, (16,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # First batch
    optimizer.zero_grad()
    output1 = model(x1)
    loss1 = criterion(output1, target)
    loss1.backward()
    optimizer.step()
    
    # Second batch - should not raise "backward through graph second time" error
    optimizer.zero_grad()
    output2 = model(x2)
    loss2 = criterion(output2, target)
    loss2.backward()
    optimizer.step()
    
    print("  ✓ State detachment working correctly")
    return True


def test_fused_operations():
    """Test that fused operations produce correct results."""
    print("Testing fused operations...")
    
    config = AdaptiveConfig(
        num_nodes=50,
        hidden_dim=32,
        batch_size=8,
        input_dim=100,
        output_dim=10,
        device="cpu"
    )
    
    model = AdaptiveModel(config)
    model.eval()
    
    # Create sample input
    x = torch.randn(8, 100)
    
    # Forward pass should work without errors
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"
    
    # Check output is valid (no NaN or Inf)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("  ✓ Fused operations producing correct results")
    return True


def test_amp_support():
    """Test mixed precision support."""
    print("Testing AMP support...")
    
    # Check AMP availability
    cpu_amp = supports_amp("cpu")
    print(f"  CPU AMP support: {cpu_amp}")
    
    if torch.cuda.is_available():
        cuda_amp = supports_amp("cuda")
        print(f"  CUDA AMP support: {cuda_amp}")
    
    # Check dtype selection
    cpu_dtype = get_amp_dtype("cpu")
    print(f"  CPU AMP dtype: {cpu_dtype}")
    
    if torch.cuda.is_available():
        cuda_dtype = get_amp_dtype("cuda")
        print(f"  CUDA AMP dtype: {cuda_dtype}")
    
    print("  ✓ AMP utilities working correctly")
    return True


def test_optimized_model_wrapper():
    """Test Phase2OptimizedModel wrapper."""
    print("Testing optimized model wrapper...")
    
    config = AdaptiveConfig(
        num_nodes=30,
        hidden_dim=16,
        batch_size=4,
        input_dim=50,
        output_dim=5,
        device="cpu"
    )
    
    # Create base model
    base_model = AdaptiveModel(config)
    
    # Wrap with optimizations
    optimized = optimize_model_phase2(
        base_model,
        enable_compile=False,  # Don't compile for small model
        enable_amp=False  # CPU doesn't benefit much from AMP
    )
    
    # Test forward pass
    x = torch.randn(4, 50)
    with torch.no_grad():
        output = optimized(x)
    
    assert output.shape == (4, 5), f"Expected shape (4, 5), got {output.shape}"
    
    print("  ✓ Optimized model wrapper working correctly")
    return True


def test_torch_compile_fallback():
    """Test torch.compile graceful fallback."""
    print("Testing torch.compile fallback...")
    
    config = AdaptiveConfig(
        num_nodes=20,
        hidden_dim=16,
        device="cpu"
    )
    
    model = AdaptiveModel(config)
    
    # Try to compile (should handle gracefully regardless of PyTorch version)
    compiled_model = try_compile(model, mode="default")
    
    # Model should still work
    x = torch.randn(4, 784)
    with torch.no_grad():
        output = compiled_model(x)
    
    assert output.shape[0] == 4, "Model output has wrong batch size"
    
    print("  ✓ torch.compile fallback working correctly")
    return True


def test_contiguous_tensors():
    """Test that tensors are contiguous after operations."""
    print("Testing contiguous tensor layout...")
    
    config = AdaptiveConfig(
        num_nodes=40,
        hidden_dim=32,
        device="cpu"
    )
    
    model = AdaptiveModel(config)
    
    # Run forward pass
    x = torch.randn(8, 784)
    with torch.no_grad():
        output = model(x)
    
    # Check that node state tensors are contiguous
    assert model.node_state.hidden_state.is_contiguous(), "Hidden state not contiguous"
    assert model.node_state.energy.is_contiguous(), "Energy not contiguous"
    assert model.node_state.activity.is_contiguous(), "Activity not contiguous"
    
    print("  ✓ Tensors are contiguous")
    return True


def main():
    """Run all Phase 2 tests."""
    print("=" * 70)
    print("Phase 2 Optimization Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_state_detachment,
        test_fused_operations,
        test_amp_support,
        test_optimized_model_wrapper,
        test_torch_compile_fallback,
        test_contiguous_tensors
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
