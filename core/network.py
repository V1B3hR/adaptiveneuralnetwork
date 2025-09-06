import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

class Capacitor:
    def __init__(self, capacity=5.0):
        self.capacity = capacity
        self.energy = capacity

    def release(self, amount):
        released = min(self.energy, amount)
        self.energy -= released
        return released

    def recharge(self, amount):
        self.energy = min(self.capacity, self.energy + amount)

    def __repr__(self):
        return f"Capacitor(energy={self.energy:.2f}/{self.capacity:.2f})"

class Cell:
    def __init__(self, cell_id, energy=10.0, anxiety=0.0, trust=0.5, phase="active", calm=1.0):
        self.cell_id = cell_id
        self.energy = energy
        self.anxiety = anxiety
        self.trust = trust
        self.phase = phase
        self.memory = []
        self.long_term_memory = []
        self.calm = calm
        self.successful_calm_count = 0
        self.failed_calm_count = 0

    def tick(self, external_stimulus=0.0, capacitor=None):
        prev_anxiety = self.anxiety
        self.anxiety += external_stimulus
        self.energy -= external_stimulus * 0.7

        if self.energy < 3.0 and capacitor:
            recovered = capacitor.release(2.0)
            self.energy += recovered
            logging.info(f"Cell {self.cell_id}: Drew {recovered:.2f} energy from capacitor.")

        # Calm logic (clock oil)
        if self.anxiety > 5.0:
            self.apply_calm(prev_anxiety)

        # Adaptive phase/trust logic
        if self.anxiety > 8.0:
            self.phase = "stressed"
            self.trust = max(0.0, self.trust - 0.1)
        elif self.anxiety < 2.0:
            self.phase = "resting"
            self.trust = min(1.0, self.trust + 0.05)
        else:
            self.phase = "active"

        self.memory.append({
            "energy": self.energy,
            "anxiety": self.anxiety,
            "trust": self.trust,
            "phase": self.phase,
            "calm": self.calm,
            "successful_calm": self.successful_calm_count,
            "failed_calm": self.failed_calm_count
        })
        if len(self.memory) > 10:
            self.long_term_memory.append(self.memory.pop(0))

        logging.info(f"Cell {self.cell_id}: Energy={self.energy:.2f}, Anxiety={self.anxiety:.2f}, Trust={self.trust:.2f}, Phase={self.phase}, Calm={self.calm}")

    def apply_calm(self, prev_anxiety):
        calm_effect = min(self.calm, self.anxiety * 0.4)
        self.anxiety = max(0.0, self.anxiety - calm_effect)
        logging.info(f"Cell {self.cell_id}: Calm applied! Anxiety reduced by {calm_effect:.2f}")

        # RL-style trust update: if anxiety reduced, increase trust; else decrease
        if self.anxiety < prev_anxiety:
            self.trust = min(1.0, self.trust + 0.06)
            self.successful_calm_count += 1
        else:
            self.trust = max(0.0, self.trust - 0.03)
            self.failed_calm_count += 1

    def get_status(self):
        return {
            "energy": self.energy,
            "anxiety": self.anxiety,
            "trust": self.trust,
            "phase": self.phase,
            "calm": self.calm,
            "successful_calm": self.successful_calm_count,
            "failed_calm": self.failed_calm_count
        }

    def __repr__(self):
        return f"Cell({self.cell_id}, energy={self.energy:.2f}, anxiety={self.anxiety:.2f}, trust={self.trust:.2f}, phase={self.phase}, calm={self.calm:.2f})"

class AdaptiveClockNetwork:
    def __init__(self, num_cells=3, capacitor_capacity=5.0, calm_value=5.0):
        self.cells = [Cell(cell_id=i, calm=calm_value) for i in range(num_cells)]
        self.capacitor = Capacitor(capacity=capacitor_capacity)
        self.global_calm = calm_value
        self.calm_history = []
        self.resilience_measurements = []
        self.energy_used = 0.0
        self.energy_recovered = 0.0
        self.last_avg_anxiety = None
        self.resilience_index = None

    def network_tick(self, stimuli):
        tick_energy_used = 0.0
        tick_energy_recovered = 0.0

        for cell, stim in zip(self.cells, stimuli):
            pre_energy = cell.energy
            cell.tick(external_stimulus=stim, capacitor=self.capacitor)
            tick_energy_used += max(0, pre_energy - cell.energy)
            tick_energy_recovered += max(0, cell.energy - pre_energy) if cell.energy > pre_energy else 0

        self.energy_used += tick_energy_used
        self.energy_recovered += tick_energy_recovered

        avg_anxiety = np.mean([cell.anxiety for cell in self.cells])
        logging.info(f"Network: Average Anxiety={avg_anxiety:.2f}")

        # Resilience: track return to baseline after a spike
        if self.last_avg_anxiety is not None:
            if avg_anxiety < self.last_avg_anxiety:
                self.resilience_measurements.append(self.last_avg_anxiety - avg_anxiety)
        self.last_avg_anxiety = avg_anxiety

        # Dynamic calm adjustment
        self.calm_history.append(self.global_calm)
        if avg_anxiety > 7.0:
            self.apply_global_calm()
            # If anxiety remains high, increase calm
            if avg_anxiety > 8.0:
                self.global_calm = min(10.0, self.global_calm + 0.5)
        elif avg_anxiety < 3.0 and self.global_calm > 1.0:
            self.global_calm = max(1.0, self.global_calm - 0.2)

        self.capacitor.recharge(1.0)

    def apply_global_calm(self):
        logging.info("Network: Global calm applied! Dampening anxiety for all cells.")
        for cell in self.cells:
            calm_effect = min(self.global_calm, cell.anxiety * 0.3)
            cell.anxiety = max(0.0, cell.anxiety - calm_effect)
            cell.calm = self.global_calm  # Sync cell calm to current global calm

    def calculate_metrics(self, tick_duration):
        anxieties = [cell.anxiety for cell in self.cells]
        avg_anxiety = np.mean(anxieties)
        std_anxiety = np.std(anxieties)
        stability = 1 - (std_anxiety / avg_anxiety if avg_anxiety != 0 else 0)
        trust_values = [cell.trust for cell in self.cells]
        trust_variance = np.var(trust_values)
        resilience_index = np.mean(self.resilience_measurements) if self.resilience_measurements else 0
        energy_efficiency = self.energy_recovered / (self.energy_used + 1e-6)
        return {
            "execution_time": tick_duration,
            "stability": stability,
            "avg_trust": np.mean(trust_values),
            "trust_variance": trust_variance,
            "resilience_index": resilience_index,
            "energy_efficiency": energy_efficiency,
            "stressed_cells": sum(1 for cell in self.cells if cell.phase == "stressed"),
            "global_calm": self.global_calm
        }

    def get_network_status(self):
        return [cell.get_status() for cell in self.cells] + [repr(self.capacitor)]

if __name__ == "__main__":
    best = None
    for num_cells in range(3, 6):
        for calm_value in [3.0, 5.0, 7.0]:
            network = AdaptiveClockNetwork(num_cells=num_cells, capacitor_capacity=5.0, calm_value=calm_value)
            start_time = time.perf_counter()
            for step in range(12):
                stimuli = np.random.uniform(0, 9, size=num_cells)
                network.network_tick(stimuli)
            end_time = time.perf_counter()
            tick_duration = end_time - start_time

            metrics = network.calculate_metrics(tick_duration)
            logging.info(
                f"\nTested: Nodes={num_cells}, Calm={calm_value}, Time={metrics['execution_time']:.6f}s, "
                f"Stability={metrics['stability']:.2f}, AvgTrust={metrics['avg_trust']:.2f}, TrustVar={metrics['trust_variance']:.2f}, "
                f"Resilience={metrics['resilience_index']:.2f}, EnergyEff={metrics['energy_efficiency']:.2f}, StressedCells={metrics['stressed_cells']}, GlobalCalm={metrics['global_calm']:.2f}"
            )

            if (metrics['execution_time'] < 0.0005 and metrics['stability'] > 0.8):
                best = (num_cells, calm_value, metrics)

    if best:
        print(f"\nOptimal found: Nodes={best[0]}, Calm={best[1]}, Metrics={best[2]}")
    else:
        print("\nNo optimal configuration found in sweep.")

    if best:
        network = AdaptiveClockNetwork(num_cells=best[0], capacitor_capacity=5.0, calm_value=best[1])
        for step in range(12):
            stimuli = np.random.uniform(0, 9, size=best[0])
            network.network_tick(stimuli)
        print("Final network status:", network.get_network_status())
