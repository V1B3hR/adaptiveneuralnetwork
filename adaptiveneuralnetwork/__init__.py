"""
Adaptive Neural Network - Biologically inspired neural network with adaptive learning
"""

__version__ = "0.1.0"

from .api.config import AdaptiveConfig
from .api.model import AdaptiveModel
from .config import AdaptiveNeuralNetworkConfig, load_config

__all__ = ["AdaptiveModel", "AdaptiveConfig", "AdaptiveNeuralNetworkConfig", "load_config"]
