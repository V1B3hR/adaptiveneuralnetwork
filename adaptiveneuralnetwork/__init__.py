"""
Adaptive Neural Network - Biologically inspired neural network with adaptive learning
"""

__version__ = "0.1.0"

from .api.config import AdaptiveConfig
from .api.model import AdaptiveModel
from .config import AdaptiveNeuralNetworkConfig, load_config

# AutoML components
from .automl import (
    AdaptiveAutoMLEngine, 
    AutoMLConfig,
    create_automl_engine
)

__all__ = [
    "AdaptiveModel", 
    "AdaptiveConfig", 
    "AdaptiveNeuralNetworkConfig", 
    "load_config",
    # AutoML
    "AdaptiveAutoMLEngine",
    "AutoMLConfig", 
    "create_automl_engine"
]
