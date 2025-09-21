"""
Training utilities for adaptive neural networks.
"""

from .datasets import create_synthetic_loaders, load_mnist, load_mnist_subset
from .loops import TrainingLoop, quick_train

# Try to import new modules (with fallback for gradual implementation)
try:
    from .bitext_dataset import BitextDatasetLoader
    from .run_bitext_training import main as run_bitext_training
    from .text_baseline import TextClassificationBaseline

    __all__ = [
        "load_mnist",
        "load_mnist_subset",
        "create_synthetic_loaders",
        "TrainingLoop",
        "quick_train",
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
    ]
