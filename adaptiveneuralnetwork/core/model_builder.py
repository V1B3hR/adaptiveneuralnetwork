"""
Config-driven model assembly system.

This module implements a builder pattern for constructing neural network
models from YAML/JSON configuration files, using the layer registry.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from .layer_registry import layer_registry


class ModelBuilder:
    """Builder for constructing models from configuration dictionaries."""

    def __init__(self, registry=None, seed: int | None = None):
        """
        Initialize model builder.
        
        Args:
            registry: Layer registry to use (defaults to global registry)
            seed: Random seed for reproducible initialization (localized)
        """
        self.registry = registry if registry is not None else layer_registry
        self.seed = seed

    def build_from_config(self, config: dict[str, Any]) -> nn.Module:
        """
        Build a model from a configuration dictionary.
        
        Args:
            config: Dictionary with model specification
                - name: Model name
                - layers: List of layer specifications
                - seed: Optional seed override
                
        Returns:
            Constructed PyTorch module
            
        Example config:
            {
                "name": "my_model",
                "seed": 42,
                "layers": [
                    {"type": "linear", "in_features": 10, "out_features": 20},
                    {"type": "relu"},
                    {"type": "linear", "in_features": 20, "out_features": 5}
                ]
            }
        """
        # Handle local seed if specified
        model_seed = config.get('seed', self.seed)
        if model_seed is not None:
            self._set_local_seed(model_seed)

        # Build layer list
        layers = []
        for layer_spec in config.get('layers', []):
            layer = self._build_layer(layer_spec)
            layers.append(layer)

        # Wrap in Sequential if multiple layers
        if len(layers) == 0:
            raise ValueError("Model must have at least one layer")
        elif len(layers) == 1:
            model = layers[0]
        else:
            model = nn.Sequential(*layers)

        return model

    def build_from_file(self, config_path: str | Path) -> nn.Module:
        """
        Build a model from a YAML or JSON configuration file.
        
        Args:
            config_path: Path to configuration file (.yaml, .yml, or .json)
            
        Returns:
            Constructed PyTorch module
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load config based on file extension
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path) as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        return self.build_from_config(config)

    def _build_layer(self, layer_spec: dict[str, Any]) -> nn.Module:
        """
        Build a single layer from specification.
        
        Args:
            layer_spec: Dictionary with layer type and parameters
            
        Returns:
            Instantiated layer module
        """
        layer_spec = layer_spec.copy()  # Don't modify original
        layer_type = layer_spec.pop('type')

        # Handle nested layers (e.g., Sequential, ModuleList)
        if 'layers' in layer_spec:
            nested_layers = [
                self._build_layer(nested_spec)
                for nested_spec in layer_spec.pop('layers')
            ]
            layer_spec['layers'] = nested_layers

        # Create layer using registry
        return self.registry.create(layer_type, **layer_spec)

    def _set_local_seed(self, seed: int) -> None:
        """
        Set random seed locally for this model construction.
        
        This ensures reproducible initialization without affecting global state.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


def register_builtin_layers():
    """Register built-in PyTorch layers with the layer registry."""

    # Linear layers
    layer_registry.register('linear', nn.Linear)
    layer_registry.register('bilinear', nn.Bilinear)
    layer_registry.register('identity', nn.Identity)

    # Convolutional layers
    layer_registry.register('conv1d', nn.Conv1d)
    layer_registry.register('conv2d', nn.Conv2d)
    layer_registry.register('conv3d', nn.Conv3d)
    layer_registry.register('conv_transpose1d', nn.ConvTranspose1d)
    layer_registry.register('conv_transpose2d', nn.ConvTranspose2d)
    layer_registry.register('conv_transpose3d', nn.ConvTranspose3d)

    # Pooling layers
    layer_registry.register('maxpool1d', nn.MaxPool1d)
    layer_registry.register('maxpool2d', nn.MaxPool2d)
    layer_registry.register('maxpool3d', nn.MaxPool3d)
    layer_registry.register('avgpool1d', nn.AvgPool1d)
    layer_registry.register('avgpool2d', nn.AvgPool2d)
    layer_registry.register('avgpool3d', nn.AvgPool3d)
    layer_registry.register('adaptive_avgpool1d', nn.AdaptiveAvgPool1d)
    layer_registry.register('adaptive_avgpool2d', nn.AdaptiveAvgPool2d)
    layer_registry.register('adaptive_avgpool3d', nn.AdaptiveAvgPool3d)

    # Activation functions
    layer_registry.register('relu', nn.ReLU)
    layer_registry.register('leaky_relu', nn.LeakyReLU)
    layer_registry.register('prelu', nn.PReLU)
    layer_registry.register('elu', nn.ELU)
    layer_registry.register('selu', nn.SELU)
    layer_registry.register('gelu', nn.GELU)
    layer_registry.register('sigmoid', nn.Sigmoid)
    layer_registry.register('tanh', nn.Tanh)
    layer_registry.register('softmax', nn.Softmax)
    layer_registry.register('log_softmax', nn.LogSoftmax)

    # Normalization layers
    layer_registry.register('batchnorm1d', nn.BatchNorm1d)
    layer_registry.register('batchnorm2d', nn.BatchNorm2d)
    layer_registry.register('batchnorm3d', nn.BatchNorm3d)
    layer_registry.register('layernorm', nn.LayerNorm)
    layer_registry.register('groupnorm', nn.GroupNorm)
    layer_registry.register('instancenorm1d', nn.InstanceNorm1d)
    layer_registry.register('instancenorm2d', nn.InstanceNorm2d)
    layer_registry.register('instancenorm3d', nn.InstanceNorm3d)

    # Dropout layers
    layer_registry.register('dropout', nn.Dropout)
    layer_registry.register('dropout1d', nn.Dropout1d)
    layer_registry.register('dropout2d', nn.Dropout2d)
    layer_registry.register('dropout3d', nn.Dropout3d)

    # Recurrent layers
    layer_registry.register('rnn', nn.RNN)
    layer_registry.register('lstm', nn.LSTM)
    layer_registry.register('gru', nn.GRU)

    # Attention layers
    layer_registry.register('multihead_attention', nn.MultiheadAttention)

    # Transformer layers
    layer_registry.register('transformer', nn.Transformer)
    layer_registry.register('transformer_encoder', nn.TransformerEncoder)
    layer_registry.register('transformer_decoder', nn.TransformerDecoder)
    layer_registry.register('transformer_encoder_layer', nn.TransformerEncoderLayer)
    layer_registry.register('transformer_decoder_layer', nn.TransformerDecoderLayer)

    # Container layers
    def create_sequential(**kwargs):
        layers = kwargs.get('layers', [])
        return nn.Sequential(*layers)

    def create_module_list(**kwargs):
        layers = kwargs.get('layers', [])
        return nn.ModuleList(layers)

    layer_registry.register_factory('sequential', create_sequential)
    layer_registry.register_factory('module_list', create_module_list)

    # Flatten
    layer_registry.register('flatten', nn.Flatten)


# Register built-in layers on import
register_builtin_layers()
