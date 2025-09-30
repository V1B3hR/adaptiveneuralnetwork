"""
Training utilities for adaptive neural networks.
"""

from .datasets import create_synthetic_loaders, load_mnist, load_mnist_subset
from .loops import TrainingLoop, quick_train

# Phase 4: Training Loop Abstraction
from .callbacks import Callback, CallbackList, LoggingCallback, ProfilingCallback
from .trainer import Trainer

# Try to import new modules (with fallback for gradual implementation)
try:
    from .bitext_dataset import BitextDatasetLoader
    from .text_baseline import TextClassificationBaseline
    from .run_bitext_training import main as run_bitext_training
    
    __all__ = [
        "load_mnist",
        "load_mnist_subset", 
        "create_synthetic_loaders",
        "TrainingLoop",
        "quick_train",
        "Callback",
        "CallbackList",
        "LoggingCallback",
        "ProfilingCallback",
        "Trainer",
        "BitextDatasetLoader",
        "TextClassificationBaseline",
        "run_bitext_training",
    ]
except ImportError:
    # Fallback when new modules are not yet implemented
    __all__ = [
        "load_mnist",
        "load_mnist_subset",
        "create_synthetic_loaders",
        "TrainingLoop",
        "quick_train",
        "Callback",
        "CallbackList",
        "LoggingCallback",
        "ProfilingCallback",
        "Trainer",
    ]
