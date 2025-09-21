"""
Test utilities and seed control for reproducible testing
"""

import os
import random

import numpy as np


def set_seed(seed=42):
    """Set random seed for reproducible testing"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_test_seed():
    """Get seed from environment variable or use default"""
    return int(os.environ.get("TEST_SEED", 42))


class SeedContext:
    """Context manager for temporary seed setting"""

    def __init__(self, seed):
        self.seed = seed
        self.old_random_state = None
        self.old_numpy_state = None

    def __enter__(self):
        # Save current states
        self.old_random_state = random.getstate()
        self.old_numpy_state = np.random.get_state()

        # Set new seed
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old states
        random.setstate(self.old_random_state)
        np.random.set_state(self.old_numpy_state)


def run_with_seed(seed=42):
    """Decorator to run test with specific seed"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with SeedContext(seed):
                return func(*args, **kwargs)

        return wrapper

    return decorator
