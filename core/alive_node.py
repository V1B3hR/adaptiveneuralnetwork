import numpy as np
import random

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
        self.memory = []
        self.predicted_energy = self.energy
        self.node_id = node_id
        self.last_signal = None
        self.anxiety = 0.0
        self.phase_mix = set()
        self.phase_history = []

    def step_phase(self, current_time):
        hour = current_time % 24
        self.phase_history.append(self.phase)
        if self.anxiety > 10 or self.energy > 25:
            self.phase = "sleep"
            self.sleep_stage = "deep"
            self.phase_mix.clear()
        elif hour in range(0, 6):
            self.phase = "sleep"
            self.sleep_stage = AliveLoopNode.sleep_stages[(hour // 2) % 3]
            self.phase_mix.clear()
        elif self.energy > 15:
            self.phase = "inspired"
            self.phase_mix = {"active", "inspired"}
            self.sleep_stage = None
        elif self.reward > 8:
            self.phase = "interactive"
            self.phase_mix = {"active", "interactive"}
            self.sleep_stage = None
        else:
            self.phase = "active"
            self.phase_mix = {"active"}
            self.sleep_stage = None

    def move(self):
        if self.phase == "sleep":
            self.velocity *= 0.7 if self.sleep_stage == "deep" else 0.9
        elif "inspired" in self.phase_mix:
            self.velocity *= 1.2
        elif "interactive" in self.phase_mix:
            self.velocity *= 1.05
        vel_mag = np.linalg.norm(self.velocity)
        if vel_mag > 5:
            self.velocity = self.velocity / vel_mag * 5
        self.position += self.velocity

    def update_size(self):
        self.radius = max(0.1, min(2.0, 0.1 + 0.05 * self.energy))

    def seek_capacitors(self, capacitors):
        if self.phase == "sleep":
            return
        best_capacitor = None
        min_dist = float("inf")
        for cap in capacitors:
            dist = np.linalg.norm(self.position - cap.position)
            if cap.energy < cap.capacity and dist < min_dist:
                min_dist = dist
                best_capacitor = cap
        if best_capacitor is not None:
            direction = best_capacitor.position - self.position
            norm = np.linalg.norm(direction)
            if norm > 0:
                self.velocity += 0.1 * direction / norm

    def interact_with_capacitor(self, capacitor, threshold=0.5):
        distance = np.linalg.norm(self.position - capacitor.position)
        effective_distance = max(0, distance - self.radius)
        if effective_distance < threshold:
            energy_difference = self.energy - capacitor.energy
            transfer_amount = 0.1 * energy_difference
            if transfer_amount > 0:
                actual_transfer = min(transfer_amount, capacitor.capacity - capacitor.energy)
            else:
                actual_transfer = max(transfer_amount, -self.energy)
            actual_transfer = max(-self.energy, min(actual_transfer, self.energy))
            self.energy -= actual_transfer
            self.energy = max(0.0, self.energy)
            capacitor.energy += actual_transfer
            capacitor.energy = min(capacitor.capacity, max(0.0, capacitor.energy))
            reward_gain = min(abs(actual_transfer), 2.0)
            self.reward += reward_gain
            self.memory.append(reward_gain)
            self.reward *= 0.98
            self.anxiety += actual_transfer * 0.2

    # --- MISSING FEATURE: API SIGNAL INTEGRATION ---
    def absorb_external_signal(self, signal_energy, signal_type="human"):
        self.last_signal = signal_type
        factor = {"human": (1.0, 0.5), "AI": (0.7, 0.3), "world": (0.4, 0.6)}
        energy_factor, reward_factor = factor.get(signal_type, (0.5, 0.5))
        self.energy += signal_energy * energy_factor
        self.reward += signal_energy * reward_factor
        self.memory.append((signal_type, signal_energy))
        self.anxiety += signal_energy * 0.3
        if self.energy > 15 or self.anxiety > 8:
            self.phase = "inspired"
            self.phase_mix = {"active", "inspired"}

    # --- MISSING FEATURE: ENERGY PREDICTION ---
    def predict_energy(self):
        recent = [e[1] for e in self.memory[-5:] if isinstance(e, tuple)]
        if recent:
            self.predicted_energy = np.mean(recent)
        else:
            self.predicted_energy = self.energy

    # --- MISSING FEATURE: MEMORY REPLAY/SHARING IN SLEEP ---
    def replay_memories(self, nodes):
        if self.phase == "sleep" and self.sleep_stage in ["REM", "deep"] and len(self.memory) > 0:
            top_memories = self.memory[-3:]
            for node in nodes:
                if node.node_id != self.node_id:
                    dist = np.linalg.norm(self.position - node.position)
                    if dist < 2.5:
                        for mem in top_memories:
                            if isinstance(mem, float):
                                node.memory.append(("shared", mem))
                                node.reward += 0.1 * mem

    # --- MISSING FEATURE: ANXIETY RELIEF IN SLEEP ---
    def clear_anxiety(self):
        if self.phase == "sleep":
            self.anxiety *= 0.6 if self.sleep_stage == "deep" else 0.8

    # --- STATUS REPORTING ---
    def print_status(self):
        print(
            f"Node {self.node_id}: Pos {self.position}, Energy {round(self.energy,2)}, "
            f"Radius {round(self.radius,2)}, Reward {round(self.reward,2)}, "
            f"Phase {self.phase}, SleepStage {self.sleep_stage}, PhaseMix {self.phase_mix}, "
            f"Anxiety {round(self.anxiety,2)}, LastSignal {self.last_signal}, PredictedEnergy {round(self.predicted_energy,2)}"
        )
