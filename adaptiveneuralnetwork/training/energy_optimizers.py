"""
Energy-aware optimizer variants with meta-adaptation capabilities.

This module provides optimizers that adapt their behavior based on energy
dynamics and node states in the adaptive neural network.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import Optimizer

from ..core.nodes import NodeState
from ..core.phases import PhaseScheduler


class EnergyAwareOptimizer(Optimizer):
    """
    Base class for energy-aware optimizers that adapt based on node energy states.
    """

    def __init__(
        self,
        params,
        node_state: NodeState,
        phase_scheduler: Optional[PhaseScheduler] = None,
        lr: float = 1e-3,
        energy_scaling: bool = True,
        energy_threshold: float = 0.1,
        **kwargs,
    ):
        defaults = dict(
            lr=lr, energy_scaling=energy_scaling, energy_threshold=energy_threshold, **kwargs
        )
        super().__init__(params, defaults)

        self.node_state = node_state
        self.phase_scheduler = phase_scheduler
        self.step_count = 0
        self.energy_history: List[float] = []
        self.adaptation_history: List[Dict[str, float]] = []

    def get_energy_scaling_factor(self) -> torch.Tensor:
        """Compute energy-based scaling factors for learning rates."""
        if not self.defaults["energy_scaling"]:
            return torch.ones_like(self.node_state.energy)

        # Scale learning rate based on node energy levels
        energy = self.node_state.energy
        energy_threshold = self.defaults["energy_threshold"]

        # Higher energy = higher learning rate (up to 2x)
        # Lower energy = lower learning rate (down to 0.1x)
        scaling = torch.clamp(energy / energy_threshold, 0.1, 2.0)

        return scaling

    def get_phase_modulation(self) -> torch.Tensor:
        """Get phase-based modulation factors."""
        if self.phase_scheduler is None:
            return torch.ones(self.node_state.config.num_nodes, device=self.node_state.device)

        phases = self.phase_scheduler.node_phases
        modulation = torch.ones_like(phases, dtype=torch.float)

        # Different learning rate modulations for different phases
        # ACTIVE: normal learning rate (1.0)
        # SLEEP: reduced learning rate for consolidation (0.5)
        # INTERACTIVE: enhanced learning rate for adaptation (1.5)
        # INSPIRED: creative learning rate boost (2.0)

        modulation[phases == 0] = 1.0  # ACTIVE
        modulation[phases == 1] = 0.5  # SLEEP
        modulation[phases == 2] = 1.5  # INTERACTIVE
        modulation[phases == 3] = 2.0  # INSPIRED

        return modulation

    def compute_adaptive_lr(self, base_lr: float) -> torch.Tensor:
        """Compute adaptive learning rates for each node."""
        energy_scaling = self.get_energy_scaling_factor()
        phase_modulation = self.get_phase_modulation()

        # Combine energy and phase effects
        adaptive_lr = base_lr * energy_scaling * phase_modulation

        return adaptive_lr

    def update_adaptation_history(self, metrics: Dict[str, float]) -> None:
        """Update adaptation history for meta-learning."""
        current_energy = self.node_state.energy.mean().item()
        self.energy_history.append(current_energy)

        adaptation_metrics = {
            "step": self.step_count,
            "mean_energy": current_energy,
            "energy_std": self.node_state.energy.std().item(),
            "mean_activity": self.node_state.activity.mean().item(),
            **metrics,
        }

        self.adaptation_history.append(adaptation_metrics)

        # Keep only recent history
        max_history = 1000
        if len(self.energy_history) > max_history:
            self.energy_history = self.energy_history[-max_history:]
            self.adaptation_history = self.adaptation_history[-max_history:]


class EnergyAwareAdam(EnergyAwareOptimizer):
    """
    Energy-aware Adam optimizer that adapts learning rates based on node energy states.
    """

    def __init__(
        self,
        params,
        node_state: NodeState,
        phase_scheduler: Optional[PhaseScheduler] = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        energy_scaling: bool = True,
        energy_threshold: float = 0.1,
        energy_momentum: float = 0.9,
    ):
        super().__init__(
            params,
            node_state,
            phase_scheduler,
            lr=lr,
            energy_scaling=energy_scaling,
            energy_threshold=energy_threshold,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            energy_momentum=energy_momentum,
        )

    def step(self, closure=None):
        """Perform a single optimization step with energy awareness."""
        loss = None
        if closure is not None:
            loss = closure()

        # Compute adaptive learning rates
        base_lr = self.defaults["lr"]
        adaptive_lrs = self.compute_adaptive_lr(base_lr)

        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                grad = param.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.data).float()
                    state["exp_avg_sq"] = torch.zeros_like(param.data).float()
                    state["energy_momentum"] = torch.zeros_like(adaptive_lrs).float()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                energy_momentum = state["energy_momentum"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Apply weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(param.data, alpha=group["weight_decay"])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Update energy momentum
                energy_momentum.mul_(group["energy_momentum"]).add_(
                    adaptive_lrs, alpha=1 - group["energy_momentum"]
                )

                # Use energy-modulated learning rate
                if len(adaptive_lrs) == 1:
                    effective_lr = energy_momentum[0].item()
                else:
                    # For multi-node case, use mean or parameter-specific mapping
                    effective_lr = energy_momentum.mean().item()

                step_size = effective_lr * math.sqrt(bias_correction2) / bias_correction1

                # Apply update
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

        self.step_count += 1

        # Update adaptation history
        metrics = {
            "learning_rate": adaptive_lrs.mean().item(),
            "lr_std": adaptive_lrs.std().item(),
        }
        self.update_adaptation_history(metrics)

        return loss


class EnergyAwareSGD(EnergyAwareOptimizer):
    """
    Energy-aware SGD optimizer with momentum and energy-based adaptation.
    """

    def __init__(
        self,
        params,
        node_state: NodeState,
        phase_scheduler: Optional[PhaseScheduler] = None,
        lr: float = 1e-2,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        energy_scaling: bool = True,
        energy_threshold: float = 0.1,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(
            params,
            node_state,
            phase_scheduler,
            lr=lr,
            energy_scaling=energy_scaling,
            energy_threshold=energy_threshold,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    def step(self, closure=None):
        """Perform a single optimization step with energy awareness."""
        loss = None
        if closure is not None:
            loss = closure()

        # Compute adaptive learning rates
        base_lr = self.defaults["lr"]
        adaptive_lrs = self.compute_adaptive_lr(base_lr)

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for i, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                grad = param.grad.data
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                # Use energy-aware learning rate
                if len(adaptive_lrs) == 1:
                    effective_lr = adaptive_lrs[0].item()
                else:
                    effective_lr = adaptive_lrs.mean().item()

                if momentum != 0:
                    param_state = self.state[param]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(param.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                param.data.add_(grad, alpha=-effective_lr)

        self.step_count += 1

        # Update adaptation history
        metrics = {
            "learning_rate": adaptive_lrs.mean().item(),
        }
        self.update_adaptation_history(metrics)

        return loss


class MetaAdaptiveOptimizer(EnergyAwareOptimizer):
    """
    Meta-adaptive optimizer that learns to adjust its own hyperparameters
    based on energy dynamics and training progress.
    """

    def __init__(
        self,
        params,
        node_state: NodeState,
        phase_scheduler: Optional[PhaseScheduler] = None,
        lr: float = 1e-3,
        meta_lr: float = 1e-4,
        adaptation_window: int = 100,
        energy_scaling: bool = True,
        energy_threshold: float = 0.1,
    ):
        super().__init__(
            params,
            node_state,
            phase_scheduler,
            lr=lr,
            energy_scaling=energy_scaling,
            energy_threshold=energy_threshold,
            meta_lr=meta_lr,
            adaptation_window=adaptation_window,
        )

        # Meta-parameters to adapt
        self.meta_params = {
            "lr_scale": torch.tensor(1.0, requires_grad=True),
            "energy_sensitivity": torch.tensor(1.0, requires_grad=True),
            "phase_modulation_strength": torch.tensor(1.0, requires_grad=True),
        }

        # Meta-optimizer for meta-parameters
        self.meta_optimizer = torch.optim.Adam(list(self.meta_params.values()), lr=meta_lr)

        self.performance_history: List[float] = []
        self.last_meta_update = 0

    def compute_adaptive_lr(self, base_lr: float) -> torch.Tensor:
        """Compute adaptive learning rates with meta-learned parameters."""
        # Base energy and phase modulation
        energy_scaling = self.get_energy_scaling_factor()
        phase_modulation = self.get_phase_modulation()

        # Apply meta-learned modulations
        lr_scale = self.meta_params["lr_scale"]
        energy_sensitivity = self.meta_params["energy_sensitivity"]
        phase_strength = self.meta_params["phase_modulation_strength"]

        # Enhanced energy scaling with learned sensitivity
        enhanced_energy_scaling = torch.pow(energy_scaling, energy_sensitivity)

        # Enhanced phase modulation with learned strength
        enhanced_phase_modulation = 1.0 + (phase_modulation - 1.0) * phase_strength

        # Combine all factors
        adaptive_lr = base_lr * lr_scale * enhanced_energy_scaling * enhanced_phase_modulation

        return adaptive_lr

    def meta_update(self, current_performance: float) -> None:
        """Update meta-parameters based on performance."""
        self.performance_history.append(current_performance)

        if len(self.performance_history) < self.defaults["adaptation_window"]:
            return

        # Only update periodically
        if self.step_count - self.last_meta_update < self.defaults["adaptation_window"]:
            return

        # Compute performance trend
        recent_window = self.performance_history[-self.defaults["adaptation_window"] :]
        performance_trend = (recent_window[-1] - recent_window[0]) / len(recent_window)

        # Create meta-loss: we want to maximize performance improvement
        meta_loss = -performance_trend

        # Update meta-parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # Clamp meta-parameters to reasonable ranges
        with torch.no_grad():
            self.meta_params["lr_scale"].clamp_(0.1, 10.0)
            self.meta_params["energy_sensitivity"].clamp_(0.1, 3.0)
            self.meta_params["phase_modulation_strength"].clamp_(0.0, 2.0)

        self.last_meta_update = self.step_count

    def step(self, closure=None, current_performance: Optional[float] = None):
        """Perform optimization step with meta-adaptation."""
        loss = None
        if closure is not None:
            loss = closure()

        # Regular optimization step (using Adam-like update)
        base_lr = self.defaults["lr"]
        adaptive_lrs = self.compute_adaptive_lr(base_lr)

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Exponential moving averages
                exp_avg.mul_(0.9).add_(grad, alpha=0.1)
                exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)

                # Bias correction
                bias_correction1 = 1 - 0.9 ** state["step"]
                bias_correction2 = 1 - 0.999 ** state["step"]

                # Use adaptive learning rate
                effective_lr = adaptive_lrs.mean().item()
                step_size = effective_lr * math.sqrt(bias_correction2) / bias_correction1

                # Apply update
                denom = exp_avg_sq.sqrt().add_(1e-8)
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

        self.step_count += 1

        # Meta-adaptation
        if current_performance is not None:
            self.meta_update(current_performance)

        # Update adaptation history
        metrics = {
            "learning_rate": adaptive_lrs.mean().item(),
            "lr_scale": self.meta_params["lr_scale"].item(),
            "energy_sensitivity": self.meta_params["energy_sensitivity"].item(),
            "phase_modulation_strength": self.meta_params["phase_modulation_strength"].item(),
        }
        self.update_adaptation_history(metrics)

        return loss


def create_energy_aware_optimizer(
    optimizer_type: str,
    parameters,
    node_state: NodeState,
    phase_scheduler: Optional[PhaseScheduler] = None,
    **kwargs,
) -> EnergyAwareOptimizer:
    """
    Factory function to create energy-aware optimizers.

    Args:
        optimizer_type: Type of optimizer ('adam', 'sgd', 'meta')
        parameters: Model parameters
        node_state: Node state for energy information
        phase_scheduler: Optional phase scheduler
        **kwargs: Additional optimizer arguments

    Returns:
        Configured energy-aware optimizer
    """
    optimizer_classes = {
        "adam": EnergyAwareAdam,
        "sgd": EnergyAwareSGD,
        "meta": MetaAdaptiveOptimizer,
    }

    if optimizer_type not in optimizer_classes:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer_classes[optimizer_type](parameters, node_state, phase_scheduler, **kwargs)
