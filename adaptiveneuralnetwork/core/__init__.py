"""
Core modules for adaptive neural networks.
"""

from .dynamics import AdaptiveDynamics
from .nodes import NodeConfig, NodeState
from .phases import Phase, PhaseScheduler
from .consolidation import (
    UnifiedConsolidationManager,
    ConsolidationType,
    PhaseBasedConsolidation,
    SynapticConsolidation,
    MemoryConsolidation,
    create_default_consolidation_manager
)

# Optional video inference components
try:
    from .video_inference import (
        InferenceConfig,
        InferenceResult,
        PerformanceMonitor,
        AdaptiveProcessor,
        RealTimeInferenceEngine,
        VideoStreamInference,
        create_stream_inference
    )
    _video_inference_available = True
except ImportError:
    _video_inference_available = False

__all__ = [
    "NodeConfig",
    "NodeState",
    "Phase",
    "PhaseScheduler",
    "AdaptiveDynamics",
    "UnifiedConsolidationManager",
    "ConsolidationType",
    "PhaseBasedConsolidation", 
    "SynapticConsolidation",
    "MemoryConsolidation",
    "create_default_consolidation_manager",
]

if _video_inference_available:
    __all__.extend([
        "InferenceConfig",
        "InferenceResult",
        "PerformanceMonitor",
        "AdaptiveProcessor",
        "RealTimeInferenceEngine",
        "VideoStreamInference",
        "create_stream_inference"
    ])
