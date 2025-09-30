"""
Layer registry system for modular architecture construction.

This module provides a registry pattern for registering and retrieving
layer classes by string identifiers, enabling config-driven model assembly.
"""

from typing import Any, Callable, Dict, Optional, Type
import torch.nn as nn


class LayerRegistry:
    """Registry for layer classes that can be instantiated from configs."""
    
    def __init__(self):
        self._registry: Dict[str, Type[nn.Module]] = {}
        self._factory_functions: Dict[str, Callable] = {}
    
    def register(self, name: str, layer_class: Optional[Type[nn.Module]] = None) -> Callable:
        """
        Register a layer class with a given name.
        
        Can be used as a decorator or direct call.
        
        Args:
            name: String identifier for the layer
            layer_class: Layer class to register (optional if used as decorator)
            
        Returns:
            The layer class (for decorator pattern)
            
        Example:
            @layer_registry.register("my_layer")
            class MyLayer(nn.Module):
                pass
        """
        def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
            if name in self._registry:
                raise ValueError(f"Layer '{name}' already registered")
            self._registry[name] = cls
            return cls
        
        if layer_class is None:
            # Used as decorator
            return decorator
        else:
            # Direct registration
            return decorator(layer_class)
    
    def register_factory(self, name: str, factory_fn: Callable) -> None:
        """
        Register a factory function for creating layers.
        
        Factory functions are useful for layers that need custom initialization.
        
        Args:
            name: String identifier for the layer
            factory_fn: Function that creates and returns a layer instance
        """
        if name in self._factory_functions:
            raise ValueError(f"Factory '{name}' already registered")
        self._factory_functions[name] = factory_fn
    
    def create(self, name: str, **kwargs) -> nn.Module:
        """
        Create a layer instance by name with given parameters.
        
        Args:
            name: String identifier for the layer
            **kwargs: Parameters to pass to layer constructor
            
        Returns:
            Instantiated layer module
            
        Raises:
            ValueError: If layer name is not registered
        """
        # Check factory functions first
        if name in self._factory_functions:
            return self._factory_functions[name](**kwargs)
        
        # Check registered classes
        if name not in self._registry:
            raise ValueError(
                f"Layer '{name}' not found in registry. "
                f"Available layers: {list(self._registry.keys())}"
            )
        
        layer_class = self._registry[name]
        return layer_class(**kwargs)
    
    def list_layers(self) -> list[str]:
        """Get list of all registered layer names."""
        all_layers = set(self._registry.keys()) | set(self._factory_functions.keys())
        return sorted(list(all_layers))
    
    def has_layer(self, name: str) -> bool:
        """Check if a layer is registered."""
        return name in self._registry or name in self._factory_functions
    
    def unregister(self, name: str) -> None:
        """Remove a layer from the registry (mainly for testing)."""
        self._registry.pop(name, None)
        self._factory_functions.pop(name, None)


# Global registry instance
layer_registry = LayerRegistry()
