"""
Public API for adaptive neural networks.
"""

from .backend_factory import create_adaptive_model, get_backend_info, list_available_backends
from .config import AdaptiveConfig
from .model import AdaptiveModel

__all__ = [
    "AdaptiveConfig",
    "AdaptiveModel",
    "create_adaptive_model",
    "list_available_backends",
    "get_backend_info"
]
