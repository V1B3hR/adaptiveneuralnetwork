"""
Utility modules for adaptive neural networks.
"""

from .profiling import PerformanceProfiler, run_torch_profiler
from .onnx_export import ModelIntrospection, ONNXExporter, export_model_with_introspection
from .reproducibility import (
    ReproducibilityHarness, 
    EnvironmentSnapshot, 
    SeedState,
    DeterminismReport,
    create_reproducible_experiment,
    set_global_seed,
    verify_reproducible_function
)
from .phase2_optimizations import (
    AMPContext,
    try_compile,
    mixed_precision_wrapper,
    Phase2OptimizedModel,
    optimize_model_phase2,
    supports_amp,
    get_amp_dtype
)

__all__ = [
    "PerformanceProfiler", 
    "run_torch_profiler",
    "ModelIntrospection",
    "ONNXExporter", 
    "export_model_with_introspection",
    "ReproducibilityHarness",
    "EnvironmentSnapshot",
    "SeedState", 
    "DeterminismReport",
    "create_reproducible_experiment",
    "set_global_seed",
    "verify_reproducible_function",
    # Phase 2 optimizations
    "AMPContext",
    "try_compile",
    "mixed_precision_wrapper",
    "Phase2OptimizedModel",
    "optimize_model_phase2",
    "supports_amp",
    "get_amp_dtype"
]
