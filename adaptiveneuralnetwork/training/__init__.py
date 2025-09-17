"""
Training utilities for adaptive neural networks.
"""

from .datasets import create_synthetic_loaders, load_mnist, load_mnist_subset
from .loops import TrainingLoop, quick_train

# Bitext training components (optional NLP dependencies)
try:
    from .bitext_dataset import load_bitext_dataset, create_synthetic_bitext_dataset, BitextDataset
    from .sklearn_baseline import SklearnBaseline, run_baseline_training
    __all__ = [
        "load_mnist",
        "load_mnist_subset", 
        "create_synthetic_loaders",
        "TrainingLoop",
        "quick_train",
        "load_bitext_dataset",
        "create_synthetic_bitext_dataset",
        "BitextDataset",
        "SklearnBaseline",
        "run_baseline_training",
    ]
except ImportError:
    # NLP dependencies not available
    __all__ = [
        "load_mnist",
        "load_mnist_subset",
        "create_synthetic_loaders",
        "TrainingLoop",
        "quick_train",
    ]
