import numpy as np
import random
from collections import deque
from typing import Any
import logging
from core.alive_node import Memory

# Setup logger for capacitor module
logger = logging.getLogger('capacitor')
logger.setLevel(logging.WARNING)  # Only show warnings and errors

class CapacitorInSpace:
    def __init__(self, position, capacity=5.0, initial_energy=0.0):
        self.position = np.array(position, dtype=float)
        self.capacity = max(0.0, capacity)
        self.energy = min(max(0.0, initial_energy), self.capacity)

    def charge(self, amount):
        """Increase energy, clamped to capacity."""
        logger.debug(f"Charging: Energy before: {self.energy}, Amount: {amount}")
        previous_energy = self.energy
        self.energy = min(self.capacity, self.energy + amount)
        logger.debug(f"Charging: Energy after: {self.energy}")
        return self.energy - previous_energy

    def discharge(self, amount):
        """Decrease energy, clamped to zero."""
        logger.debug(f"Discharging: Energy before: {self.energy}, Amount: {amount}")
        actual_discharge = min(self.energy, amount)
        self.energy -= actual_discharge
        logger.debug(f"Discharging: Energy after: {self.energy}")
        return actual_discharge

    def print_status(self):
        logger.info(f"Capacitor: Position {self.position}, Energy {round(self.energy,2)}/{self.capacity}")
        
    def __str__(self):
        return f"Capacitor at {self.position}, Energy {round(self.energy,2)}/{self.capacity}"
