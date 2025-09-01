import numpy as np

class CapacitorInSpace:
    def __init__(self, position, capacity=5.0, initial_energy=0.0):
        self.position = np.array(position, dtype=float)
        self.capacity = capacity
        self.energy = initial_energy
