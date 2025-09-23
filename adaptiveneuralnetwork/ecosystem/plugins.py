"""
Plugin architecture for custom modules and community contributions.
"""

import importlib
import inspect
from typing import Dict, Any, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod
import sys

from ..api.model import AdaptiveModel
from ..api.config import AdaptiveConfig


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    author: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    license: str = "MIT"
    homepage: Optional[str] = None
    api_version: str = "1.0"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create metadata from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "license": self.license,
            "homepage": self.homepage,
            "api_version": self.api_version
        }


class PluginBase(ABC):
    """Base class for all plugins."""
    
    def __init__(self):
        self.metadata = self.get_metadata()
        self.is_enabled = False
        self.logger = logging.getLogger(f"plugin.{self.metadata.name}")
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, config: AdaptiveConfig) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def finalize(self) -> bool:
        """Cleanup plugin resources."""
        pass
    
    def pre_training(self, model: AdaptiveModel, data: Any) -> Any:
        """Called before training begins."""
        return data
    
    def post_training(self, model: AdaptiveModel, results: Any) -> Any:
        """Called after training completes."""
        return results
    
    def pre_inference(self, model: AdaptiveModel, input_data: Any) -> Any:
        """Called before inference."""
        return input_data
    
    def post_inference(self, model: AdaptiveModel, output_data: Any) -> Any:
        """Called after inference."""
        return output_data
    
    def on_model_update(self, model: AdaptiveModel, update_type: str) -> None:
        """Called when model is updated."""
        pass


class ModelEnhancementPlugin(PluginBase):
    """Base class for model enhancement plugins."""
    
    @abstractmethod
    def enhance_model(self, model: AdaptiveModel) -> AdaptiveModel:
        """Enhance the model with plugin functionality."""
        pass


class DataProcessingPlugin(PluginBase):
    """Base class for data processing plugins."""
    
    @abstractmethod
    def process_data(self, data: Any, processing_type: str) -> Any:
        """Process data with plugin functionality."""
        pass


class MetricsPlugin(PluginBase):
    """Base class for metrics collection plugins."""
    
    @abstractmethod
    def collect_metrics(self, model: AdaptiveModel) -> Dict[str, float]:
        """Collect custom metrics from the model."""
        pass


class PluginRegistry:
    """Registry for discovering and managing plugins."""
    
    def __init__(self, search_paths: Optional[List[Path]] = None):
        self.search_paths = search_paths or []
        self.plugins = {}
        self.plugin_classes = {}
        self.logger = logging.getLogger(__name__)
    
    def add_search_path(self, path: Path) -> None:
        """Add a directory to search for plugins."""
        if path.exists() and path not in self.search_paths:
            self.search_paths.append(path)
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in search paths."""
        discovered = []
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
            
            # Look for Python files
            for plugin_file in search_path.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                
                try:
                    plugin_name = plugin_file.stem
                    self._load_plugin_module(plugin_file, plugin_name)
                    discovered.append(plugin_name)
                except Exception as e:
                    self.logger.warning(f"Failed to discover plugin {plugin_file}: {e}")
            
            # Look for plugin packages
            for plugin_dir in search_path.iterdir():
                if not plugin_dir.is_dir() or plugin_dir.name.startswith("_"):
                    continue
                
                plugin_file = plugin_dir / "__init__.py"
                if plugin_file.exists():
                    try:
                        plugin_name = plugin_dir.name
                        self._load_plugin_module(plugin_dir, plugin_name)
                        discovered.append(plugin_name)
                    except Exception as e:
                        self.logger.warning(f"Failed to discover plugin {plugin_dir}: {e}")
        
        return discovered
    
    def _load_plugin_module(self, plugin_path: Path, plugin_name: str) -> None:
        """Load a plugin module and extract plugin classes."""
        # Add plugin path to sys.path temporarily
        parent_path = str(plugin_path.parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
        
        try:
            if plugin_path.is_file():
                # Load from file
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                # Load from package
                module = importlib.import_module(plugin_name)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginBase) and 
                    obj != PluginBase and 
                    not inspect.isabstract(obj)):
                    
                    plugin_key = f"{plugin_name}.{name}"
                    self.plugin_classes[plugin_key] = obj
                    self.logger.info(f"Discovered plugin class: {plugin_key}")
        
        finally:
            # Remove from sys.path
            if parent_path in sys.path:
                sys.path.remove(parent_path)
    
    def get_plugin_class(self, plugin_name: str) -> Optional[Type[PluginBase]]:
        """Get a plugin class by name."""
        return self.plugin_classes.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all discovered plugin classes."""
        return list(self.plugin_classes.keys())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a plugin."""
        plugin_class = self.get_plugin_class(plugin_name)
        if not plugin_class:
            return None
        
        try:
            # Create temporary instance to get metadata
            temp_instance = plugin_class()
            metadata = temp_instance.get_metadata()
            
            return {
                "name": plugin_name,
                "class": plugin_class.__name__,
                "module": plugin_class.__module__,
                "metadata": metadata.to_dict(),
                "base_classes": [cls.__name__ for cls in plugin_class.__bases__],
                "methods": [name for name, _ in inspect.getmembers(plugin_class, inspect.ismethod)]
            }
        except Exception as e:
            self.logger.error(f"Failed to get info for plugin {plugin_name}: {e}")
            return None


class PluginManager:
    """Manager for loading, configuring, and executing plugins."""
    
    def __init__(self, config: AdaptiveConfig, registry: Optional[PluginRegistry] = None):
        self.config = config
        self.registry = registry or PluginRegistry()
        self.loaded_plugins = {}
        self.enabled_plugins = {}
        self.plugin_hooks = {
            "pre_training": [],
            "post_training": [],
            "pre_inference": [], 
            "post_inference": [],
            "model_update": []
        }
        self.logger = logging.getLogger(__name__)
    
    def load_plugin(self, plugin_name: str, plugin_config: Optional[Dict[str, Any]] = None) -> bool:
        """Load a plugin by name."""
        if plugin_name in self.loaded_plugins:
            self.logger.warning(f"Plugin {plugin_name} already loaded")
            return True
        
        plugin_class = self.registry.get_plugin_class(plugin_name)
        if not plugin_class:
            self.logger.error(f"Plugin class {plugin_name} not found")
            return False
        
        try:
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Check dependencies
            if not self._check_dependencies(plugin_instance.metadata.dependencies):
                self.logger.error(f"Dependencies not met for plugin {plugin_name}")
                return False
            
            # Initialize plugin
            if not plugin_instance.initialize(self.config):
                self.logger.error(f"Failed to initialize plugin {plugin_name}")
                return False
            
            self.loaded_plugins[plugin_name] = plugin_instance
            self.logger.info(f"Loaded plugin: {plugin_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.loaded_plugins:
            return True
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            
            # Disable first
            self.disable_plugin(plugin_name)
            
            # Finalize
            plugin.finalize()
            
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]
            
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a loaded plugin."""
        if plugin_name not in self.loaded_plugins:
            self.logger.error(f"Plugin {plugin_name} not loaded")
            return False
        
        if plugin_name in self.enabled_plugins:
            return True
        
        plugin = self.loaded_plugins[plugin_name]
        plugin.is_enabled = True
        self.enabled_plugins[plugin_name] = plugin
        
        # Register hooks
        self._register_plugin_hooks(plugin_name, plugin)
        
        self.logger.info(f"Enabled plugin: {plugin_name}")
        return True
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable an enabled plugin."""
        if plugin_name not in self.enabled_plugins:
            return True
        
        plugin = self.enabled_plugins[plugin_name]
        plugin.is_enabled = False
        
        # Unregister hooks
        self._unregister_plugin_hooks(plugin_name)
        
        del self.enabled_plugins[plugin_name]
        
        self.logger.info(f"Disabled plugin: {plugin_name}")
        return True
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are available."""
        for dependency in dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                self.logger.error(f"Missing dependency: {dependency}")
                return False
        return True
    
    def _register_plugin_hooks(self, plugin_name: str, plugin: PluginBase) -> None:
        """Register plugin hooks."""
        # Check which methods are overridden
        base_methods = set(dir(PluginBase))
        plugin_methods = set(dir(plugin))
        
        overridden_methods = plugin_methods - base_methods
        
        # Register appropriate hooks
        if hasattr(plugin, 'pre_training') and 'pre_training' in overridden_methods:
            self.plugin_hooks["pre_training"].append((plugin_name, plugin.pre_training))
        
        if hasattr(plugin, 'post_training') and 'post_training' in overridden_methods:
            self.plugin_hooks["post_training"].append((plugin_name, plugin.post_training))
        
        if hasattr(plugin, 'pre_inference') and 'pre_inference' in overridden_methods:
            self.plugin_hooks["pre_inference"].append((plugin_name, plugin.pre_inference))
        
        if hasattr(plugin, 'post_inference') and 'post_inference' in overridden_methods:
            self.plugin_hooks["post_inference"].append((plugin_name, plugin.post_inference))
        
        if hasattr(plugin, 'on_model_update') and 'on_model_update' in overridden_methods:
            self.plugin_hooks["model_update"].append((plugin_name, plugin.on_model_update))
    
    def _unregister_plugin_hooks(self, plugin_name: str) -> None:
        """Unregister plugin hooks."""
        for hook_type, hooks in self.plugin_hooks.items():
            self.plugin_hooks[hook_type] = [
                (name, func) for name, func in hooks if name != plugin_name
            ]
    
    def execute_hook(self, hook_type: str, *args, **kwargs) -> Any:
        """Execute all plugins registered for a specific hook."""
        result = args[0] if args else None
        
        for plugin_name, hook_func in self.plugin_hooks.get(hook_type, []):
            try:
                if hook_type in ["pre_training", "post_training", "pre_inference", "post_inference"]:
                    result = hook_func(*args, **kwargs)
                    # Update args with modified result
                    if args:
                        args = (result,) + args[1:]
                else:
                    hook_func(*args, **kwargs)
                    
            except Exception as e:
                self.logger.error(f"Error executing hook {hook_type} for plugin {plugin_name}: {e}")
        
        return result
    
    def get_plugins_by_type(self, plugin_type: Type[PluginBase]) -> List[PluginBase]:
        """Get all enabled plugins of a specific type."""
        return [
            plugin for plugin in self.enabled_plugins.values()
            if isinstance(plugin, plugin_type)
        ]
    
    def list_loaded_plugins(self) -> List[str]:
        """List all loaded plugins."""
        return list(self.loaded_plugins.keys())
    
    def list_enabled_plugins(self) -> List[str]:
        """List all enabled plugins."""
        return list(self.enabled_plugins.keys())
    
    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all plugins."""
        status = {}
        
        for plugin_name, plugin in self.loaded_plugins.items():
            status[plugin_name] = {
                "loaded": True,
                "enabled": plugin_name in self.enabled_plugins,
                "metadata": plugin.metadata.to_dict(),
                "hooks": []
            }
            
            # Check which hooks are registered
            for hook_type, hooks in self.plugin_hooks.items():
                if any(name == plugin_name for name, _ in hooks):
                    status[plugin_name]["hooks"].append(hook_type)
        
        return status


# Example plugin implementations
class ExampleEnhancementPlugin(ModelEnhancementPlugin):
    """Example model enhancement plugin."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_enhancement",
            version="1.0.0",
            author="ANN Team",
            description="Example plugin for model enhancement",
            tags=["example", "enhancement"]
        )
    
    def initialize(self, config: AdaptiveConfig) -> bool:
        self.logger.info("Example enhancement plugin initialized")
        return True
    
    def finalize(self) -> bool:
        self.logger.info("Example enhancement plugin finalized")
        return True
    
    def enhance_model(self, model: AdaptiveModel) -> AdaptiveModel:
        # Example: Add some enhancement to the model
        self.logger.info("Enhancing model with example plugin")
        return model
    
    def pre_inference(self, model: AdaptiveModel, input_data: Any) -> Any:
        self.logger.debug("Pre-processing input data")
        return input_data


class ExampleMetricsPlugin(MetricsPlugin):
    """Example metrics collection plugin."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_metrics",
            version="1.0.0",
            author="ANN Team",
            description="Example plugin for custom metrics collection",
            tags=["example", "metrics"]
        )
    
    def initialize(self, config: AdaptiveConfig) -> bool:
        self.logger.info("Example metrics plugin initialized")
        return True
    
    def finalize(self) -> bool:
        self.logger.info("Example metrics plugin finalized")
        return True
    
    def collect_metrics(self, model: AdaptiveModel) -> Dict[str, float]:
        # Example: Collect some custom metrics
        return {
            "example_metric_1": 0.95,
            "example_metric_2": 0.87
        }