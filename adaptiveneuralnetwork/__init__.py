"""
Adaptive Neural Network - Biologically inspired neural network with adaptive learning
"""

__version__ = "0.1.0"

from .api.config import AdaptiveConfig
from .api.model import AdaptiveModel
from .config import AdaptiveNeuralNetworkConfig, load_config

# Optional AutoML components
try:
    from .automl import AdaptiveAutoMLEngine, AutoMLConfig, create_automl_engine
    _AUTOML_AVAILABLE = True
except ImportError:
    _AUTOML_AVAILABLE = False

# Base exports
__all__ = [
    "AdaptiveModel",
    "AdaptiveConfig",
    "AdaptiveNeuralNetworkConfig",
    "load_config",
]

# Add AutoML exports if available
if _AUTOML_AVAILABLE:
    __all__.extend([
        "AdaptiveAutoMLEngine",
        "AutoMLConfig",
        "create_automl_engine"
    ])
