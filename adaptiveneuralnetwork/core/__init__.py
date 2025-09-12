"""
Core modules for adaptive neural networks.
"""

from .nodes import NodeConfig, NodeState
from .phases import Phase, PhaseScheduler
from .dynamics import AdaptiveDynamics

__all__ = [
    "NodeConfig",
    "NodeState", 
    "Phase",
    "PhaseScheduler",
    "AdaptiveDynamics",
]