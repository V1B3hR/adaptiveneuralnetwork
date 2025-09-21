"""
Reproducibility harness with seed isolation and determinism reporting.

This module provides comprehensive reproducibility utilities to ensure
experiments can be reliably reproduced across different runs and environments.
"""

import hashlib
import json
import os
import platform
import random
import sys
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class EnvironmentSnapshot:
    """Captures complete environment state for reproducibility."""

    python_version: str
    torch_version: str
    numpy_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: list[int] | None
    device_count: int
    device_names: list[str]

    @classmethod
    def capture(cls) -> "EnvironmentSnapshot":
        """Capture current environment snapshot."""
        cuda_version = None
        cudnn_version = None
        device_names = []

        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if hasattr(torch.backends.cudnn, "version"):
                cudnn_version = list(torch.backends.cudnn.version())

            device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

        return cls(
            python_version=sys.version,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            cuda_available=torch.cuda.is_available(),
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            device_names=device_names,
        )


@dataclass
class SeedState:
    """Captures complete random state for reproducibility."""

    master_seed: int
    python_random_state: Any
    numpy_random_state: dict[str, Any]
    torch_random_state: Any
    torch_cuda_random_state: Any | None

    @classmethod
    def capture(cls, master_seed: int) -> "SeedState":
        """Capture current random state."""
        return cls(
            master_seed=master_seed,
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            torch_random_state=torch.get_rng_state(),
            torch_cuda_random_state=(
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        )

    def restore(self) -> None:
        """Restore random state."""
        random.setstate(self.python_random_state)
        np.random.set_state(self.numpy_random_state)
        torch.set_rng_state(self.torch_random_state)
        if self.torch_cuda_random_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_random_state)


@dataclass
class DeterminismReport:
    """Report on determinism verification."""

    test_name: str
    is_deterministic: bool
    run_count: int
    output_hashes: list[str]
    unique_outputs: int
    max_difference: float
    mean_difference: float
    std_difference: float
    environment: EnvironmentSnapshot
    seed_state: SeedState
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Handle non-serializable numpy/torch state
        result["seed_state"]["python_random_state"] = "captured"
        result["seed_state"]["numpy_random_state"] = "captured"
        result["seed_state"]["torch_random_state"] = "captured"
        result["seed_state"]["torch_cuda_random_state"] = "captured"
        return result


class ReproducibilityHarness:
    """Comprehensive reproducibility management system."""

    def __init__(self, master_seed: int = 42, strict_mode: bool = True):
        self.master_seed = master_seed
        self.strict_mode = strict_mode
        self.environment = EnvironmentSnapshot.capture()
        self.reports: list[DeterminismReport] = []
        self._original_seed_state: SeedState | None = None

    def set_seed(self, seed: int | None = None) -> int:
        """Set all random seeds for reproducibility."""
        if seed is None:
            seed = self.master_seed

        # Capture original state if not already done
        if self._original_seed_state is None:
            self._original_seed_state = SeedState.capture(seed)

        # Set Python random seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Configure PyTorch for deterministic behavior
        if self.strict_mode:
            self._enforce_deterministic_pytorch()

        return seed

    def _enforce_deterministic_pytorch(self) -> None:
        """Enforce deterministic PyTorch operations."""
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=False)

        # Set deterministic CUDA operations
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set environment variables for additional determinism
        os.environ["PYTHONHASHSEED"] = str(self.master_seed)

        # Warn about potential non-deterministic operations
        if torch.cuda.is_available():
            warnings.warn(
                "CUDA operations may still be non-deterministic. "
                "Some PyTorch operations do not have deterministic implementations.",
                UserWarning,
                stacklevel=2,
            )

    @contextmanager
    def isolated_seed_context(self, seed: int):
        """Context manager for isolated seed usage."""
        # Capture current state
        current_state = SeedState.capture(seed)

        try:
            # Set new seed
            self.set_seed(seed)
            yield
        finally:
            # Restore original state
            current_state.restore()

    def verify_determinism(
        self, test_function, test_name: str, run_count: int = 3, *args, **kwargs
    ) -> DeterminismReport:
        """
        Verify that a function produces deterministic outputs.

        Args:
            test_function: Function to test for determinism
            test_name: Name of the test
            run_count: Number of times to run the test
            *args, **kwargs: Arguments to pass to test_function

        Returns:
            DeterminismReport with verification results
        """
        outputs = []
        output_hashes = []
        warnings_list = []

        for i in range(run_count):
            with self.isolated_seed_context(self.master_seed):
                try:
                    output = test_function(*args, **kwargs)
                    outputs.append(output)

                    # Create hash of output for comparison
                    if isinstance(output, torch.Tensor):
                        output_bytes = output.detach().cpu().numpy().tobytes()
                    elif isinstance(output, np.ndarray):
                        output_bytes = output.tobytes()
                    else:
                        output_bytes = str(output).encode()

                    output_hash = hashlib.sha256(output_bytes).hexdigest()
                    output_hashes.append(output_hash)

                except Exception as e:
                    warnings_list.append(f"Run {i} failed: {str(e)}")
                    outputs.append(None)
                    output_hashes.append("error")

        # Analyze results
        unique_outputs = len(set(output_hashes))
        is_deterministic = unique_outputs == 1 and "error" not in output_hashes

        # Calculate differences if outputs are numeric
        max_diff = 0.0
        mean_diff = 0.0
        std_diff = 0.0

        if len(outputs) > 1 and all(o is not None for o in outputs):
            try:
                if isinstance(outputs[0], (torch.Tensor, np.ndarray)):
                    diffs = []
                    for i in range(1, len(outputs)):
                        if isinstance(outputs[0], torch.Tensor):
                            diff = torch.abs(outputs[0] - outputs[i]).max().item()
                        else:
                            diff = np.abs(outputs[0] - outputs[i]).max()
                        diffs.append(diff)

                    max_diff = max(diffs)
                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs)
            except Exception as e:
                warnings_list.append(f"Could not compute differences: {str(e)}")

        # Create report
        report = DeterminismReport(
            test_name=test_name,
            is_deterministic=is_deterministic,
            run_count=run_count,
            output_hashes=output_hashes,
            unique_outputs=unique_outputs,
            max_difference=max_diff,
            mean_difference=mean_diff,
            std_difference=std_diff,
            environment=self.environment,
            seed_state=SeedState.capture(self.master_seed),
            warnings=warnings_list,
        )

        self.reports.append(report)
        return report

    def generate_reproducibility_report(
        self, output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive reproducibility report."""
        report = {
            "harness_config": {
                "master_seed": self.master_seed,
                "strict_mode": self.strict_mode,
            },
            "environment": asdict(self.environment),
            "determinism_tests": [report.to_dict() for report in self.reports],
            "summary": {
                "total_tests": len(self.reports),
                "deterministic_tests": sum(1 for r in self.reports if r.is_deterministic),
                "non_deterministic_tests": sum(1 for r in self.reports if not r.is_deterministic),
                "tests_with_warnings": sum(1 for r in self.reports if r.warnings),
            },
        }

        # Add recommendations
        recommendations = []
        if report["summary"]["non_deterministic_tests"] > 0:
            recommendations.append(
                "Some tests are non-deterministic. Review implementation for random operations."
            )

        if not self.strict_mode:
            recommendations.append("Consider enabling strict_mode for maximum determinism.")

        if not torch.backends.cudnn.deterministic:
            recommendations.append(
                "Enable torch.backends.cudnn.deterministic = True for CUDA determinism."
            )

        report["recommendations"] = recommendations

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        return report

    def restore_original_state(self) -> None:
        """Restore original random state."""
        if self._original_seed_state:
            self._original_seed_state.restore()


def create_reproducible_experiment(
    name: str, seed: int = 42, strict_mode: bool = True, output_dir: str | Path | None = None
) -> ReproducibilityHarness:
    """
    Create a reproducible experiment harness.

    Args:
        name: Experiment name
        seed: Master seed for reproducibility
        strict_mode: Whether to enforce strict determinism
        output_dir: Directory to save reproducibility reports

    Returns:
        Configured ReproducibilityHarness
    """
    harness = ReproducibilityHarness(master_seed=seed, strict_mode=strict_mode)
    harness.set_seed(seed)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save initial environment snapshot
        env_file = output_dir / f"{name}_environment.json"
        with open(env_file, "w") as f:
            json.dump(asdict(harness.environment), f, indent=2, default=str)

    return harness


# Convenience functions
def set_global_seed(seed: int = 42, strict_mode: bool = True) -> None:
    """Set global random seed for reproducibility."""
    harness = ReproducibilityHarness(master_seed=seed, strict_mode=strict_mode)
    harness.set_seed(seed)


def verify_reproducible_function(func, *args, **kwargs) -> bool:
    """Quick verification that a function is reproducible."""
    harness = ReproducibilityHarness()
    report = harness.verify_determinism(func, func.__name__, *args, **kwargs)
    return report.is_deterministic
