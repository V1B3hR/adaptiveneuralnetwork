import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

@dataclass
class Memory:
    """Structured memory with importance weighting"""
    content: Any
    importance: float
    timestamp: int
    memory_type: str  # 'reward', 'shared', 'prediction', 'pattern'
    decay_rate: float = 0.95
    
    def age(self):
        # Debug: Log importance before and after aging
        print(f"[DEBUG] Memory importance before aging: {self.importance}")
        self.importance *= self.decay_rate
        print(f"[DEBUG] Memory importance after aging: {self.importance}")

class CapacitorInSpace:
    def __init__(self, position, capacity=5.0, initial_energy=0.0):
        self.position = np.array(position, dtype=float)
        self.capacity = max(0.0, capacity)
        self.energy = min(max(0.0, initial_energy), self.capacity)

    def charge(self, amount):
        """Increase energy, clamped to capacity."""
        # Debug: Log energy before charging
        print(f"[DEBUG] Charging: Energy before: {self.energy}, Amount: {amount}")
        self.energy = min(self.capacity, self.energy + amount)
        # Debug: Log energy after charging
        print(f"[DEBUG] Charging: Energy after: {self.energy}")

    def discharge(self, amount):
        """Decrease energy, clamped to zero."""
        # Debug: Log energy before discharging
        print(f"[DEBUG] Discharging: Energy before: {self.energy}, Amount: {amount}")
        self.energy = max(0.0, self.energy - amount)
        # Debug: Log energy after discharging
        print(f"[DEBUG] Discharging: Energy after: {self.energy}")

    def print_status(self):
        print(f"Capacitor: Position {self.position}, Energy {round(self.energy,2)}/{self.capacity}")

class AliveLoopNode:
    sleep_stages = ["light", "REM", "deep"]
    
    def __init__(self, position, velocity, initial_energy=0.0, field_strength=1.0, node_id=0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.energy = max(0.0, initial_energy)
        self.field_strength = field_strength
        self.radius = max(0.1, 0.1 + 0.05 * initial_energy)
        self.reward = 0.0
        self.phase = "active"
        self.sleep_stage = None
        self.node_id = node_id
        self.last_signal = None
        self.anxiety = 0.0
        self.phase_mix = set()
        self.phase_history = deque(maxlen=24)  # 24-hour rolling history
        
        # Enhanced memory system with structured storage
        self.memory = deque(maxlen=1000)  # Bounded memory for efficiency
        self.working_memory = deque(maxlen=7)  # Short-term working memory
        self.long_term_memory = {}  # Indexed by importance/patterns
        self.predicted_energy = self.energy
        
        # Debug: Initialize logging for important attributes
        print(f"[DEBUG] Node initialized: Position: {self.position}, Velocity: {self.velocity}, Energy: {self.energy}")
    
    def step_phase(self, current_time):
        """Enhanced phase management with circadian rhythms, sleep debt, and neuromodulators"""
        # Debug: Add logging for circadian phase calculations
        print(f"[DEBUG] Current time: {current_time}")
        hour = current_time % 24
        self.circadian_phase = (hour / 24) * 2 * np.pi
        print(f"[DEBUG] Circadian phase: {self.circadian_phase}")
        
        # More sophisticated phase transitions with conditions...
        if self.anxiety > 15 or self.energy < 2:
            self._enter_sleep_phase("stress")
        elif self.energy > 20 and self.anxiety < 5:
            self.phase = "inspired"
        else:
            self.phase = "active"
        
        # Debug: Log phase changes
        print(f"[DEBUG] Node phase: {self.phase}")

    def _enter_sleep_phase(self, reason):
        """Smart sleep stage selection based on needs"""
        print(f"[DEBUG] Entering sleep phase due to {reason}")
        self.phase = "sleep"
        if self.anxiety > 10 or reason == "stress":
            self.sleep_stage = "deep"
        else:
            self.sleep_stage = "REM"
        print(f"[DEBUG] Sleep stage: {self.sleep_stage}")

    def move(self):
        """Energy-efficient movement with attention-based navigation"""
        # Debug: Log position and velocity before movement
        print(f"[DEBUG] Before move: Position: {self.position}, Velocity: {self.velocity}")
        self.position += self.velocity
        # Debug: Log position after movement
        print(f"[DEBUG] After move: Position: {self.position}")

    def interact_with_capacitor(self, capacitor, threshold=0.5):
        """Enhanced interaction with learning and efficiency"""
        distance = np.linalg.norm(self.position - capacitor.position)
        effective_distance = max(0, distance - self.radius)
        
        if effective_distance < threshold:
            energy_difference = self.energy - capacitor.energy
            transfer_amount = 0.1 * energy_difference
            # Debug: Log transfer details
            print(f"[DEBUG] Interacting with capacitor: Distance: {distance}, Transfer amount: {transfer_amount}")
            if transfer_amount > 0:
                capacitor.charge(transfer_amount)
            else:
                capacitor.discharge(-transfer_amount)

    # Additional methods like absorb_external_signal, predict_energy, etc.,
    # should follow similar debugging patterns with logging at critical steps.
