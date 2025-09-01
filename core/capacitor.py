import numpy as np

class CapacitorInSpace:
    def __init__(self, position, capacity=5.0, initial_energy=0.0):
        self.position = np.array(position, dtype=float)
        self.capacity = max(0.0, capacity)
        self.energy = min(max(0.0, initial_energy), self.capacity)

    def charge(self, amount):
        """Increase energy, clamped to capacity."""
        self.energy = min(self.capacity, self.energy + amount)

    def discharge(self, amount):
        """Decrease energy, clamped to zero."""
        self.energy = max(0.0, self.energy - amount)

    def print_status(self):
        print(f"Capacitor: Position {self.position}, Energy {round(self.energy,2)}/{self.capacity}")
