import logging
import threading
from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Default module logger (can be overridden per-instance)
_module_logger = logging.getLogger(__name__)
_module_logger.setLevel(logging.WARNING)


class CapacitorInSpace:
    """
    A simple energy storage component with:
    - Capacity & clamped charge/discharge
    - Optional logger injection / verbosity control
    - Optional position shape & bounds validation
    - Optional immutability of position
    - Optional thread safety for multi-thread simulations
    """

    def __init__(
        self,
        position: Sequence[float],
        capacity: float = 5.0,
        initial_energy: float = 0.0,
        *,
        logger: Optional[logging.Logger] = None,
        verbosity: Optional[int] = None,
        allow_external_level_override: bool = True,
        expected_dims: Optional[int] = None,
        bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
        fixed_position: bool = False,
        thread_safe: bool = False,
    ):
        """
        Args:
            position: Iterable of floats (e.g. (x, y)[, z])
            capacity: Maximum energy storable (>=0)
            initial_energy: Starting energy (clamped into [0, capacity])
            logger: Optional externally provided logger
            verbosity: Optional logging level (e.g. logging.DEBUG)
            allow_external_level_override: If False, we do not call setLevel on injected logger
            expected_dims: If set, enforces len(position) == expected_dims
            bounds: Optional tuple per dimension: ((xmin, xmax), (ymin, ymax), ...)
            fixed_position: If True, position cannot be changed after init
            thread_safe: If True, operations are guarded by a re-entrant lock
        """
        self._logger = logger if logger is not None else _module_logger
        if verbosity is not None and (logger is None or allow_external_level_override):
            # Only adjust level if:
            #  - we created the logger (logger is None), or
            #  - caller explicitly permits override
            self._logger.setLevel(verbosity)

        self._expected_dims = expected_dims
        self._bounds = bounds
        self._fixed_position = fixed_position
        self._lock = threading.RLock() if thread_safe else None

        self.position = self._validate_and_create_position(position)

        self.capacity = max(0.0, float(capacity))
        self.energy = min(max(0.0, float(initial_energy)), self.capacity)

    # ------------- Internal Utilities -------------

    def _with_lock(self):
        if self._lock:
            return self._lock
        # Dummy context manager
        from contextlib import nullcontext

        return nullcontext()

    def _validate_and_create_position(self, position: Sequence[float]) -> np.ndarray:
        arr = np.array(position, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Position must be a 1D coordinate sequence.")
        if self._expected_dims is not None and arr.shape[0] != self._expected_dims:
            raise ValueError(
                f"Position dimension mismatch: got {arr.shape[0]}, expected {self._expected_dims}"
            )
        if self._bounds:
            if len(self._bounds) != arr.shape[0]:
                raise ValueError("Bounds dimensionality does not match position length.")
            for i, (low, high) in enumerate(self._bounds):
                if low > high:
                    raise ValueError(f"Invalid bounds for axis {i}: {low} > {high}")
                if not (low <= arr[i] <= high):
                    raise ValueError(
                        f"Position component {i}={arr[i]} outside bounds [{low}, {high}]"
                    )
        return arr

    def _validate_new_position(self, new_position: Sequence[float]) -> np.ndarray:
        arr = np.array(new_position, dtype=float)
        if arr.ndim != 1:
            raise ValueError("New position must be a 1D coordinate sequence.")
        if self._expected_dims is not None and arr.shape[0] != self._expected_dims:
            raise ValueError(
                f"New position dimension mismatch: got {arr.shape[0]}, "
                f"expected {self._expected_dims}"
            )
        if self._bounds:
            for i, (low, high) in enumerate(self._bounds):
                if not (low <= arr[i] <= high):
                    raise ValueError(
                        f"New position component {i}={arr[i]} outside bounds [{low}, {high}]"
                    )
        return arr

    # ------------- Public API -------------

    def charge(self, amount: float) -> float:
        """
        Increase stored energy by 'amount' but clamp to capacity.
        Returns actual absorbed energy.
        """
        if amount <= 0:
            return 0.0
        with self._with_lock():
            prev = self.energy
            self.energy = min(self.capacity, self.energy + amount)
            absorbed = self.energy - prev
            self._logger.debug(
                f"[charge] requested={amount:.4f} absorbed={absorbed:.4f} "
                f"energy={self.energy:.4f}/{self.capacity:.4f}"
            )
            return absorbed

    def discharge(self, amount: float) -> float:
        """
        Decrease stored energy by 'amount' but not below zero.
        Returns the actual energy released.
        """
        if amount <= 0:
            return 0.0
        with self._with_lock():
            actual = min(self.energy, amount)
            self.energy -= actual
            self._logger.debug(
                f"[discharge] requested={amount:.4f} released={actual:.4f} "
                f"energy={self.energy:.4f}/{self.capacity:.4f}"
            )
            return actual

    def update_position(self, new_position: Sequence[float]) -> np.ndarray:
        """
        Safely update the capacitor's position (validating shape & bounds).
        Raises RuntimeError if fixed_position=True.
        Returns the updated position.
        """
        if self._fixed_position:
            raise RuntimeError("Position is fixed; cannot update.")
        with self._with_lock():
            old = self.position.copy()
            arr = self._validate_new_position(new_position)
            self.position = arr
            self._logger.debug(f"[update_position] old={old.tolist()} new={arr.tolist()}")
            return self.position

    def print_status(self):
        self._logger.info(
            f"Capacitor: Position {self.position.tolist()}, "
            f"Energy {round(self.energy, 2)}/{self.capacity}"
        )

    def set_verbosity(self, level: int, override_external: bool = False):
        """
        Adjust logger level at runtime.
        If an external logger was injected and override_external=False, no change is applied.
        """
        if self._logger is _module_logger or override_external:
            self._logger.setLevel(level)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "capacity": self.capacity,
            "energy": self.energy,
            "soc": self.energy / self.capacity if self.capacity > 0 else 0.0,
            "fixed_position": self._fixed_position,
            "expected_dims": self._expected_dims,
            "bounds": self._bounds,
        }

    def __str__(self):
        return (
            f"CapacitorInSpace(pos={self.position.tolist()}, "
            f"energy={self.energy:.3f}/{self.capacity:.3f})"
        )
