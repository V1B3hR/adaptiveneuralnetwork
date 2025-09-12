"""
Adaptive Neural Network - Biologically inspired neural network with adaptive learning
"""

__version__ = "0.1.0"

from .api.model import AdaptiveModel
from .api.config import AdaptiveConfig

__all__ = ["AdaptiveModel", "AdaptiveConfig"]