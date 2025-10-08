"""
Mixed precision and quantization aware phase system for adaptive neural networks.

This module provides dynamic precision control that adapts to different operational
phases, optimizing computation efficiency while maintaining accuracy.
"""

from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .phases import Phase


class PrecisionLevel(Enum):
    """Available precision levels for computation."""

    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"


class QuantizationStrategy(Enum):
    """Quantization strategies for different phases."""

    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "quantization_aware_training"  # Quantization Aware Training
    PTQ = "post_training_quantization"   # Post Training Quantization


class MixedPrecisionPhaseManager:
    """Manages mixed precision and quantization based on neural network phases."""

    def __init__(
        self,
        device: str = "cpu",
        enable_amp: bool = True,
        precision_policy: dict[Phase, PrecisionLevel] = None,
        quantization_policy: dict[Phase, QuantizationStrategy] = None,
        dynamic_precision: bool = True,
        efficiency_threshold: float = 0.1
    ):
        self.device = torch.device(device)
        self.enable_amp = enable_amp and device != "cpu"  # AMP only on GPU
        self.dynamic_precision = dynamic_precision
        self.efficiency_threshold = efficiency_threshold

        # Default precision policy: higher precision for critical phases
        self.precision_policy = precision_policy or {
            Phase.ACTIVE: PrecisionLevel.FP32,      # High precision for active computation
            Phase.INTERACTIVE: PrecisionLevel.FP16,  # Medium precision for interaction
            Phase.SLEEP: PrecisionLevel.INT8,        # Low precision for dormant state
            Phase.INSPIRED: PrecisionLevel.FP32,     # High precision for creativity
        }

        # Default quantization strategy
        self.quantization_policy = quantization_policy or {
            Phase.ACTIVE: QuantizationStrategy.DYNAMIC,
            Phase.INTERACTIVE: QuantizationStrategy.STATIC,
            Phase.SLEEP: QuantizationStrategy.PTQ,
            Phase.INSPIRED: QuantizationStrategy.QAT,
        }

        # Performance tracking
        self.precision_metrics = {}
        self.quantization_metrics = {}
        self.efficiency_history = []

        # Mixed precision scaler for AMP
        self.scaler = torch.cuda.GradScaler() if self.enable_amp else None

    def get_optimal_precision(self, phase: Phase, complexity_score: float = 1.0) -> PrecisionLevel:
        """Determine optimal precision level for given phase and complexity."""
        base_precision = self.precision_policy[phase]

        if not self.dynamic_precision:
            return base_precision

        # Adjust precision based on complexity
        if complexity_score > 0.8:  # High complexity needs higher precision
            if base_precision == PrecisionLevel.INT8:
                return PrecisionLevel.FP16
            elif base_precision == PrecisionLevel.FP16:
                return PrecisionLevel.FP32
        elif complexity_score < 0.3:  # Low complexity can use lower precision
            if base_precision == PrecisionLevel.FP32:
                return PrecisionLevel.FP16
            elif base_precision == PrecisionLevel.FP16:
                return PrecisionLevel.INT8

        return base_precision

    def apply_precision_context(self, phase: Phase, complexity_score: float = 1.0):
        """Get context manager for precision-aware computation."""
        precision = self.get_optimal_precision(phase, complexity_score)

        if precision == PrecisionLevel.FP16 and self.enable_amp:
            return torch.cuda.amp.autocast()
        elif precision == PrecisionLevel.BF16 and self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            # For other precisions, return a no-op context manager
            return torch.no_grad() if precision in [PrecisionLevel.INT8, PrecisionLevel.INT4] else torch.enable_grad()

    def quantize_tensor(self, tensor: torch.Tensor, phase: Phase, strategy: QuantizationStrategy | None = None) -> torch.Tensor:
        """Apply quantization to tensor based on phase and strategy."""
        if strategy is None:
            strategy = self.quantization_policy[phase]

        if strategy == QuantizationStrategy.DYNAMIC:
            return self._dynamic_quantization(tensor)
        elif strategy == QuantizationStrategy.STATIC:
            return self._static_quantization(tensor)
        elif strategy == QuantizationStrategy.QAT:
            return self._qat_quantization(tensor)
        elif strategy == QuantizationStrategy.PTQ:
            return self._ptq_quantization(tensor)
        else:
            return tensor

    def _dynamic_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply dynamic quantization to tensor."""
        # Simple dynamic quantization to int8
        if tensor.dtype in [torch.float32, torch.float16]:
            scale = tensor.abs().max() / 127.0
            quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
            # Dequantize for computation
            return quantized.float() * scale
        return tensor

    def _static_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply static quantization with fixed scale."""
        # Use a fixed scale for static quantization
        scale = 0.1
        if tensor.dtype in [torch.float32, torch.float16]:
            quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
            return quantized.float() * scale
        return tensor

    def _qat_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply quantization-aware training style quantization."""
        # Simulate QAT with fake quantization
        if tensor.dtype in [torch.float32, torch.float16]:
            # Add noise to simulate quantization effects
            noise = torch.randn_like(tensor) * 0.01
            return tensor + noise
        return tensor

    def _ptq_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply post-training quantization."""
        # Simple PTQ implementation
        if tensor.dtype in [torch.float32, torch.float16]:
            # Calibrate scale based on tensor statistics
            scale = tensor.std() / 32.0
            quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
            return quantized.float() * scale
        return tensor

    def compute_with_phase_precision(
        self,
        computation_fn,
        inputs: torch.Tensor,
        phase: Phase,
        complexity_score: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute computation with phase-appropriate precision."""
        precision = self.get_optimal_precision(phase, complexity_score)

        # Track computation start time
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None

        if start_time:
            start_time.record()

        # Apply precision context and quantization
        with self.apply_precision_context(phase, complexity_score):
            # Quantize inputs if needed
            quantized_inputs = self.quantize_tensor(inputs, phase)

            # Perform computation
            if self.scaler and precision == PrecisionLevel.FP16:
                # Use gradient scaling for mixed precision
                outputs = computation_fn(quantized_inputs)
                if outputs.requires_grad:
                    outputs = self.scaler.scale(outputs)
            else:
                outputs = computation_fn(quantized_inputs)

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            computation_time = start_time.elapsed_time(end_time)
        else:
            computation_time = 0.0

        # Calculate efficiency metrics
        memory_usage = torch.cuda.memory_allocated() if self.device.type == "cuda" else 0

        metrics = {
            'precision_used': precision.value,
            'quantization_strategy': self.quantization_policy[phase].value,
            'computation_time_ms': computation_time,
            'memory_usage_bytes': memory_usage,
            'phase': phase.name,
            'complexity_score': complexity_score
        }

        # Update efficiency history
        efficiency_score = 1.0 / (1.0 + computation_time / 1000.0)  # Simple efficiency metric
        self.efficiency_history.append(efficiency_score)
        if len(self.efficiency_history) > 100:
            self.efficiency_history.pop(0)

        return outputs, metrics

    def adapt_precision_policy(self, performance_feedback: dict[Phase, float]):
        """Adapt precision policy based on performance feedback."""
        if not self.dynamic_precision:
            return

        for phase, performance in performance_feedback.items():
            current_precision = self.precision_policy[phase]

            # If performance is poor, increase precision
            if performance < 0.7:
                if current_precision == PrecisionLevel.INT8:
                    self.precision_policy[phase] = PrecisionLevel.FP16
                elif current_precision == PrecisionLevel.FP16:
                    self.precision_policy[phase] = PrecisionLevel.FP32
            # If performance is excellent and efficiency is low, decrease precision
            elif performance > 0.95 and np.mean(self.efficiency_history[-10:]) < self.efficiency_threshold:
                if current_precision == PrecisionLevel.FP32:
                    self.precision_policy[phase] = PrecisionLevel.FP16
                elif current_precision == PrecisionLevel.FP16:
                    self.precision_policy[phase] = PrecisionLevel.INT8

    def get_precision_metrics(self) -> dict[str, Any]:
        """Get comprehensive precision and quantization metrics."""
        return {
            'precision_policy': {phase.name: precision.value for phase, precision in self.precision_policy.items()},
            'quantization_policy': {phase.name: strategy.value for phase, strategy in self.quantization_policy.items()},
            'amp_enabled': self.enable_amp,
            'dynamic_precision': self.dynamic_precision,
            'efficiency_history_mean': np.mean(self.efficiency_history) if self.efficiency_history else 0.0,
            'efficiency_history_std': np.std(self.efficiency_history) if self.efficiency_history else 0.0,
            'device': str(self.device)
        }


class PrecisionAwareModule(nn.Module):
    """Base module that adapts its precision based on current phase."""

    def __init__(self, precision_manager: MixedPrecisionPhaseManager):
        super().__init__()
        self.precision_manager = precision_manager
        self.current_phase = Phase.ACTIVE

    def set_phase(self, phase: Phase):
        """Set the current operational phase."""
        self.current_phase = phase

    def forward_with_precision(self, x: torch.Tensor, complexity_score: float = 1.0) -> tuple[torch.Tensor, dict[str, Any]]:
        """Forward pass with phase-appropriate precision."""
        return self.precision_manager.compute_with_phase_precision(
            self.forward_impl,
            x,
            self.current_phase,
            complexity_score
        )

    def forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implement the actual forward computation."""
        raise NotImplementedError("Subclasses must implement forward_impl")
