"""
Public API for adaptive neural networks.
"""

from .config import AdaptiveConfig
from .model import AdaptiveModel
from .backend_factory import create_adaptive_model, list_available_backends, get_backend_info

__all__ = [
    "AdaptiveConfig", 
    "AdaptiveModel",
    "create_adaptive_model",
    "list_available_backends", 
    "get_backend_info"
]
