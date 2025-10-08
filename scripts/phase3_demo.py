#!/usr/bin/env python
"""
Phase 3 Demonstration: Config-driven Model Assembly

This script demonstrates the new modular architecture system:
1. Layer registry with string-to-class mapping
2. Config-driven model construction from YAML/JSON
3. Multiple model variants from configuration only
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from adaptiveneuralnetwork.core import ModelBuilder, layer_registry


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demonstrate_layer_registry():
    """Demonstrate the layer registry system."""
    print_section("1. Layer Registry")

    print(f"\nTotal registered layers: {len(layer_registry.list_layers())}")
    print("\nSample of registered layers:")
    for layer_name in sorted(layer_registry.list_layers())[:10]:
        print(f"  - {layer_name}")

    print("\nCreating a layer from registry:")
    layer = layer_registry.create("linear", in_features=10, out_features=5)
    print(f"  Created: {layer}")


def demonstrate_config_driven_models():
    """Demonstrate config-driven model construction."""
    print_section("2. Config-Driven Model Construction")

    builder = ModelBuilder(seed=42)

    # Example 1: Simple MLP
    print("\nExample 1: Simple MLP")
    mlp_config = {
        "name": "demo_mlp",
        "seed": 42,
        "layers": [
            {"type": "linear", "in_features": 10, "out_features": 20},
            {"type": "relu"},
            {"type": "dropout", "p": 0.3},
            {"type": "linear", "in_features": 20, "out_features": 5}
        ]
    }

    mlp_model = builder.build_from_config(mlp_config)
    print(f"  Model: {mlp_model}")
    print(f"  Parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 10)
    output = mlp_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Example 2: CNN
    print("\nExample 2: Convolutional Network")
    cnn_config = {
        "name": "demo_cnn",
        "seed": 42,
        "layers": [
            {"type": "conv2d", "in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1},
            {"type": "batchnorm2d", "num_features": 16},
            {"type": "relu"},
            {"type": "maxpool2d", "kernel_size": 2},
            {"type": "conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "padding": 1},
            {"type": "batchnorm2d", "num_features": 32},
            {"type": "relu"},
            {"type": "adaptive_avgpool2d", "output_size": [1, 1]},
            {"type": "flatten", "start_dim": 1},
            {"type": "linear", "in_features": 32, "out_features": 10}
        ]
    }

    cnn_model = builder.build_from_config(cnn_config)
    print(f"  Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    output = cnn_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def demonstrate_config_files():
    """Demonstrate loading models from config files."""
    print_section("3. Loading Models from Config Files")

    config_dir = Path(__file__).parent.parent / "config" / "models"

    if not config_dir.exists():
        print(f"\nConfig directory not found: {config_dir}")
        return

    builder = ModelBuilder()

    # List available configs
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    print(f"\nAvailable model configs: {len(config_files)}")
    for config_file in sorted(config_files):
        print(f"  - {config_file.name}")

    # Try to load simple_mlp config
    mlp_config_path = config_dir / "simple_mlp.yaml"
    if mlp_config_path.exists():
        print(f"\nLoading model from: {mlp_config_path.name}")
        try:
            model = builder.build_from_file(mlp_config_path)
            print("  Successfully loaded model")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Test forward pass
            x = torch.randn(2, 784)
            output = model(x)
            print(f"  Test input shape: {x.shape}")
            print(f"  Test output shape: {output.shape}")
        except Exception as e:
            print(f"  Error loading model: {e}")


def demonstrate_custom_layer_registration():
    """Demonstrate registering custom layers."""
    print_section("4. Custom Layer Registration")

    # Create a custom layer
    @layer_registry.register("custom_residual_block")
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = self.relu(out)
            return out

    print("\nRegistered custom layer: 'custom_residual_block'")
    print(f"Total layers now: {len(layer_registry.list_layers())}")

    # Use it in a config
    config = {
        "name": "resnet_style",
        "layers": [
            {"type": "conv2d", "in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
            {"type": "custom_residual_block", "channels": 64},
            {"type": "custom_residual_block", "channels": 64},
        ]
    }

    builder = ModelBuilder()
    model = builder.build_from_config(config)

    print("\nBuilt model with custom layers")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Clean up
    layer_registry.unregister("custom_residual_block")


def demonstrate_reproducibility():
    """Demonstrate reproducible model initialization."""
    print_section("5. Reproducible Initialization with Localized Seeds")

    config = {
        "name": "test_model",
        "seed": 42,
        "layers": [
            {"type": "linear", "in_features": 10, "out_features": 10}
        ]
    }

    print("\nBuilding same model twice with seed=42:")

    builder1 = ModelBuilder(seed=42)
    model1 = builder1.build_from_config(config)

    builder2 = ModelBuilder(seed=42)
    model2 = builder2.build_from_config(config)

    # Check weights are identical
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())

    all_close = all(torch.allclose(p1, p2) for p1, p2 in zip(params1, params2, strict=False))

    print(f"  Weights identical: {all_close}")
    print(f"  First weight sum (model 1): {params1[0].sum().item():.6f}")
    print(f"  First weight sum (model 2): {params2[0].sum().item():.6f}")


def print_summary():
    """Print summary of Phase 3 achievements."""
    print_section("Phase 3 Summary")

    print("\n✓ Layer Registry System")
    print("  - String-to-class mapping for all layer types")
    print(f"  - {len(layer_registry.list_layers())} layers registered")
    print("  - Custom layer registration support")

    print("\n✓ Config-Driven Model Assembly")
    print("  - YAML and JSON configuration support")
    print("  - Nested layer construction")
    print("  - Multiple model variants from config only")

    print("\n✓ Localized Random Seed Management")
    print("  - Reproducible model initialization")
    print("  - No global state pollution")

    print("\n✓ Extensibility")
    print("  - New model variants without editing core code")
    print("  - Simple decorator-based layer registration")
    print("  - Factory function support")

    print("\n✓ Exit Criteria Met")
    print("  - New model variants added via config only")
    print("  - Core architecture made modular and extensible")
    print("  - Comprehensive test coverage")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  Phase 3: Model Architecture Modularization Demo")
    print("  Config-driven, modular, extensible architecture")
    print("=" * 70)

    try:
        demonstrate_layer_registry()
        demonstrate_config_driven_models()
        demonstrate_config_files()
        demonstrate_custom_layer_registration()
        demonstrate_reproducibility()
        print_summary()

        print("\n" + "=" * 70)
        print("  Demo completed successfully!")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\nError during demonstration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
