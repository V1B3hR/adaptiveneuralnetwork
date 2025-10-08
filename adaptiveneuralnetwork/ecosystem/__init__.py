"""
Developer ecosystem and community tools for adaptive neural networks.
"""

from .contrib import CommunityPlugin, ContributionValidator
from .plugins import PluginBase, PluginManager, PluginRegistry

# Base exports (always available)
__all__ = [
    "PluginManager",
    "PluginBase",
    "PluginRegistry",
    "CommunityPlugin",
    "ContributionValidator",
]

# Import optional components without automatic loading
_SDK_AVAILABLE = False
_INTEGRATIONS_AVAILABLE = False

def get_sdk():
    """Lazy import SDK components."""
    global _SDK_AVAILABLE
    if not _SDK_AVAILABLE:
        try:
            from .sdk import AdaptiveNeuralNetworkSDK, SDKClient
            _SDK_AVAILABLE = True
            return AdaptiveNeuralNetworkSDK, SDKClient
        except ImportError:
            return None, None
    else:
        from .sdk import AdaptiveNeuralNetworkSDK, SDKClient
        return AdaptiveNeuralNetworkSDK, SDKClient

def get_integrations():
    """Lazy import framework integrations."""
    global _INTEGRATIONS_AVAILABLE
    if not _INTEGRATIONS_AVAILABLE:
        try:
            from .integrations import JAXIntegration, PyTorchIntegration, TensorFlowIntegration
            _INTEGRATIONS_AVAILABLE = True
            return PyTorchIntegration, TensorFlowIntegration, JAXIntegration
        except ImportError:
            return None, None, None
    else:
        from .integrations import JAXIntegration, PyTorchIntegration, TensorFlowIntegration
        return PyTorchIntegration, TensorFlowIntegration, JAXIntegration
