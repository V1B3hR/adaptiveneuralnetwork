import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set


@dataclass
class Memory:
    """Structured memory with importance weighting"""
    content: any
    importance: float
    timestamp: int
    memory_type: str  # 'reward', 'shared', 'prediction', 'pattern'
    decay_rate: float = 0.95

    def age(self):
        self.importance *= self.decay_rate


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

        # Brain-like efficiency features
        self.attention_focus = np.array([0.0, 0.0])  # 2D attention vector
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.3
        self.energy_efficiency = 1.0  # Metabolic efficiency multiplier
        self.neural_plasticity = 0.5  # How quickly it adapts

        # Social and communication enhancements
        self.trust_network = {}  # node_id -> trust_level
        self.communication_range = 3.0
        self.signal_processing_cost = 0.1  # Energy cost per signal

        # Advanced prediction system
        self.pattern_buffer = deque(maxlen=50)
        self.prediction_accuracy = 0.5
        self.temporal_memory = deque(maxlen=100)  # For time-based patterns

        # Circadian rhythm enhancement
        self.circadian_phase = 0.0  # 0-2π for 24-hour cycle
        self.sleep_debt = 0.0
        self.optimal_sleep_duration = 6.0

        # New: Neuromodulator for sleep (e.g., acetylcholine level, 0-1)
        self.neuromodulator = 0.5  # Influences transitions, based on thalamocortical models

        # New: LTP/LTD factors for plasticity
        self.ltp_factor = 1.0
        self.ltd_factor = 1.0

    def step_phase(self, current_time):
        """Enhanced phase management with circadian rhythms, sleep debt, and neuromodulators"""
        hour = current_time % 24
        self.circadian_phase = (hour / 24) * 2 * np.pi

        # Calculate natural sleep tendency with neuromodulator influence
        circadian_sleep_drive = 0.5 * (1 + np.cos(self.circadian_phase + np.pi)) * (1 - self.neuromodulator)
        sleep_pressure = min(self.sleep_debt / 10.0, 1.0)

        self.phase_history.append(self.phase)

        # Validate circadian phase is within range
        assert 0 <= self.circadian_phase <= 2 * np.pi, f"Circadian phase out of range: {self.circadian_phase}"

        # Validate sleep pressure and sleep drive
        assert 0 <= sleep_pressure <= 1, f"Sleep pressure out of range: {sleep_pressure}"
        assert 0 <= circadian_sleep_drive <= 1, f"Circadian sleep drive out of range: {circadian_sleep_drive}"

        if self.anxiety > 15 or self.energy < 2:
            self._enter_sleep_phase("stress")
        elif (circadian_sleep_drive > 0.7 or sleep_pressure > 0.8) and hour in range(22, 6):
            self._enter_sleep_phase("natural")
        elif self.energy > 20 and self.anxiety < 5:
            self.phase = "inspired"
            self.phase_mix = {"active", "inspired"}
            self._wake_up()
        elif self.reward > 10 and len(self.trust_network) > 0:
            self.phase = "interactive"
            self.phase_mix = {"active", "interactive"}
            self._wake_up()
        else:
            self.phase = "active"
            self.phase_mix = {"active"}
            self._wake_up()

    def _enter_sleep_phase(self, reason):
        """Smart sleep stage selection based on needs"""
        self.phase = "sleep"
        self.phase_mix.clear()

        if self.anxiety > 10 or reason == "stress":
            self.sleep_stage = "deep"  # Deep sleep for stress recovery
        elif len(self.working_memory) > 5:
            self.sleep_stage = "REM"  # REM for memory consolidation
        else:
            hour = (self.circadian_phase / (2 * np.pi)) * 24
            self.sleep_stage = self.sleep_stages[int(hour / 8) % 3]

    def _wake_up(self):
        """Handle waking up and sleep debt calculation"""
        if self.phase == "sleep":
            reduction = 1.0 + self.neuromodulator
            if self.sleep_stage == "deep":
                self.sleep_debt = max(0, self.sleep_debt - 2.0 * reduction)
            elif self.sleep_stage == "REM":
                self.sleep_debt = max(0, self.sleep_debt - 1.5 * reduction)
            else:
                self.sleep_debt = max(0, self.sleep_debt - 1.0 * reduction)
        else:
            self.sleep_debt += 0.1  # Accumulate sleep debt while awake

        self.sleep_stage = None

    def move(self):
        """Energy-efficient movement with attention-based navigation"""
        base_energy_cost = 0.05

        if self.phase == "sleep":
            movement_factor = 0.1 if self.sleep_stage == "deep" else 0.3
            energy_cost = base_energy_cost * 0.2  # Very low energy during sleep
        elif "inspired" in self.phase_mix:
            movement_factor = 1.3
            energy_cost = base_energy_cost * 0.8  # More efficient when inspired
        elif "interactive" in self.phase_mix:
            movement_factor = 1.1
            energy_cost = base_energy_cost
        else:
            movement_factor = 1.0
            energy_cost = base_energy_cost

        if np.linalg.norm(self.attention_focus) > 0.1:
            self.velocity += self.attention_focus * 0.2

        self.velocity *= movement_factor
        vel_mag = np.linalg.norm(self.velocity)
        max_speed = 3.0 + 2.0 * self.energy_efficiency
        if vel_mag > max_speed:
            self.velocity = (self.velocity / vel_mag) * max_speed

        self.velocity = np.clip(self.velocity, -max_speed, max_speed)

        self.position += self.velocity
        self.energy = max(0.0, self.energy - energy_cost * vel_mag)

        # Added: Cache the position and energy for further analytics
        print(f"[DEBUG] Node {self.node_id} - Position: {self.position}, Energy: {self.energy}, Velocity: {self.velocity}")
