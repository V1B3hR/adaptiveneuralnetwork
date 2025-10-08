"""
Core modules for adaptive neural networks.
"""

from .consolidation import (
    ConsolidationType,
    MemoryConsolidation,
    PhaseBasedConsolidation,
    SynapticConsolidation,
    UnifiedConsolidationManager,
    create_default_consolidation_manager,
)
from .dynamics import AdaptiveDynamics
from .layer_registry import LayerRegistry, layer_registry
from .model_builder import ModelBuilder, register_builtin_layers
from .nodes import NodeConfig, NodeState
from .phases import Phase, PhaseScheduler

# Optional video inference components
try:
    from .video_inference import (
        AdaptiveProcessor,
        InferenceConfig,
        InferenceResult,
        PerformanceMonitor,
        RealTimeInferenceEngine,
        VideoStreamInference,
        create_stream_inference,
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
    "LayerRegistry",
    "layer_registry",
    "ModelBuilder",
    "register_builtin_layers",
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
