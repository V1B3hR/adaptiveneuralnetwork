"""
Core modules for adaptive neural networks.
"""

from .dynamics import AdaptiveDynamics
from .nodes import NodeConfig, NodeState
from .phases import Phase, PhaseScheduler

__all__ = [
    "NodeConfig",
    "NodeState",
    "Phase",
    "PhaseScheduler",
    "AdaptiveDynamics",
]
