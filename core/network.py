"""

Advanced evolutionary extension of Network for 
- Genomes (network + per-cell genes)
- Crossover & mutation
- Diversity tracking
- Runtime self-adaptation (cells mutate in response to prolonged stress)
- Multi-objective fitness + Pareto-front (non-dominated sorting)
- CLI main entrypoint and reproducibility

Usage:
    python -m core.evolving_adaptive_clock [--seed INT] [--generations INT] [--pop_size INT]

"""

import copy
import logging
import random
import time
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")

# ------------- Core simulation classes -------------
class Capacitor:
    def __init__(self, capacity=5.0):
        self.capacity = capacity
        self.energy = capacity

    def release(self, amount) -> float:
        released = min(self.energy, amount)
        self.energy -= released
        return released

    def recharge(self, amount) -> None:
        self.energy = min(self.capacity, self.energy + amount)

    def __repr__(self) -> str:
        return f"Capacitor(energy={self.energy:.2f}/{self.capacity:.2f})"

class Cell:
    def __init__(
        self,
        cell_id: int,
        energy: float = 10.0,
        anxiety: float = 0.0,
        trust: float = 0.5,
        phase: str = "active",
        calm: float = 1.0,
        energy_capacity: float = 10.0,
        anxiety_sensitivity: float = 1.0,
        personality: str = "neutral",
        runtime_adapt: bool = True,
    ):
        self.cell_id = cell_id
        self.energy = min(energy, energy_capacity)
        self.energy_capacity = energy_capacity
        self.anxiety = anxiety
        self.trust = trust
        self.phase = phase
        self.calm = calm
        self.memory: List[Dict[str, Any]] = []
        self.long_term_memory: List[Dict[str, Any]] = []
        self.anxiety_sensitivity = anxiety_sensitivity
        self.personality = personality
        self.runtime_adapt = runtime_adapt
        self.stress_ticks = 0
        self.relief_ticks = 0

    def tick(self, external_stimulus=0.0, capacitor: Optional[Capacitor] = None, influence=0.0):
        effective_stim = external_stimulus * self.anxiety_sensitivity + influence
        self.anxiety += effective_stim
        self.energy -= effective_stim * 0.7

        if self.energy < (0.3 * self.energy_capacity) and capacitor:
            recovered = capacitor.release(min(2.0, (self.energy_capacity - self.energy)))
            self.energy += recovered
            logging.debug(f"Cell {self.cell_id}: Drew {recovered:.2f} energy from capacitor.")

        if self.anxiety > 0.0 and self.calm > 0.0:
            self.apply_calm()

        if self.anxiety > 8.0:
            self.phase = "stressed"
            self.trust = max(0.0, self.trust - 0.05)
        elif self.anxiety < 2.0:
            if self.phase == "stressed":
                self.relief_ticks += 1
            self.phase = "resting"
            self.trust = min(1.0, self.trust + 0.02)
        else:
            self.phase = "active"

        if self.runtime_adapt:
            if self.anxiety > 6.0:
                self.stress_ticks += 1
                self.relief_ticks = 0
            else:
                if self.stress_ticks > 0:
                    self.stress_ticks = max(0, self.stress_ticks - 1)
                self.relief_ticks += 1
            if self.stress_ticks >= 3:
                change = np.random.uniform(0.02, 0.08)
                self.calm += change
                self.anxiety_sensitivity = max(0.1, self.anxiety_sensitivity * (1.0 - 0.02))
                logging.debug(f"Cell {self.cell_id}: Runtime adaptation -> calm +{change:.3f}, sensitivity -> {self.anxiety_sensitivity:.3f}")
                self.stress_ticks = 0
            if self.relief_ticks >= 8 and self.calm > 0.5:
                self.calm *= 0.98
                self.relief_ticks = 0

        self.energy = min(self.energy, self.energy_capacity)
        self.anxiety = max(0.0, self.anxiety)
        self.trust = float(np.clip(self.trust, 0.0, 1.0))

        self.memory.append({
            "energy": self.energy,
            "anxiety": self.anxiety,
            "trust": self.trust,
            "phase": self.phase,
            "calm": self.calm,
        })
        if len(self.memory) > 12:
            self.long_term_memory.append(self.memory.pop(0))

    def apply_calm(self):
        calm_effect = min(self.calm, self.anxiety * 0.35)
        self.anxiety = max(0.0, self.anxiety - calm_effect)

    def get_status(self) -> Dict[str, Any]:
        return {
            "energy": self.energy,
            "anxiety": self.anxiety,
            "trust": self.trust,
            "phase": self.phase,
            "calm": self.calm,
            "sensitivity": self.anxiety_sensitivity,
        }

    def __repr__(self) -> str:
        return f"Cell({self.cell_id}, energy={self.energy:.2f}, anxiety={self.anxiety:.2f}, trust={self.trust:.2f}, phase={self.phase})"

class AdaptiveClockNetwork:
    def __init__(self, genome: Dict[str, Any], runtime_adapt: bool = True):
        self.genome = copy.deepcopy(genome)
        self.num_cells = int(genome["num_cells"])
        self.capacitor = Capacitor(capacity=genome.get("capacitor_capacity", 5.0))
        self.global_calm = genome.get("global_calm", 1.0)
        self.cells: List[Cell] = []
        per_cell_genes = genome.get("per_cell", [])
        for i in range(self.num_cells):
            g = per_cell_genes[i] if i < len(per_cell_genes) else per_cell_genes[-1]
            cell = Cell(
                cell_id=i,
                energy=g.get("energy", g.get("energy_capacity", 10.0)),
                anxiety=0.0,
                trust=g.get("trust_baseline", 0.5),
                calm=g.get("calm", 1.0),
                energy_capacity=g.get("energy_capacity", 10.0),
                anxiety_sensitivity=g.get("anxiety_sensitivity", 1.0),
                personality=g.get("personality", "neutral"),
                runtime_adapt=runtime_adapt,
            )
            self.cells.append(cell)

    def network_tick(self, stimuli: List[float]) -> None:
        anxieties = np.array([c.anxiety for c in self.cells], dtype=float)
        diffusivity = self.genome.get("diffusivity", 0.05)
        influences = diffusivity * (np.mean(anxieties) - anxieties)
        for i, (cell, stim) in enumerate(zip(self.cells, stimuli)):
            influence = float(influences[i])
            cell.tick(external_stimulus=stim, capacitor=self.capacitor, influence=influence)

        avg_anxiety = float(np.mean([cell.anxiety for cell in self.cells]))
        if avg_anxiety > self.genome.get("global_calm_trigger", 7.0):
            self.apply_global_calm()
        self.capacitor.recharge(self.genome.get("recharge_per_tick", 1.0))

    def apply_global_calm(self):
        for cell in self.cells:
            calm_effect = min(self.global_calm, cell.anxiety * 0.25)
            cell.anxiety = max(0.0, cell.anxiety - calm_effect)
            cell.calm += 0.2

    def calculate_performance_and_stability(self, tick_duration: float) -> Dict[str, float]:
        anxieties = np.array([cell.anxiety for cell in self.cells])
        energies = np.array([cell.energy for cell in self.cells])
        trusts = np.array([cell.trust for cell in self.cells])
        avg_anxiety = float(np.mean(anxieties))
        std_anxiety = float(np.std(anxieties))
        stability = 1.0 - (std_anxiety / avg_anxiety if avg_anxiety != 0 else 0.0)
        efficiency = float(np.mean(energies) / (np.mean([c.energy_capacity for c in self.cells]) + 1e-9))
        avg_trust = float(np.mean(trusts))
        stressed_cells = int(sum(1 for cell in self.cells if cell.phase == "stressed"))
        return {
            "execution_time": tick_duration,
            "stability": stability,
            "efficiency": efficiency,
            "avg_trust": avg_trust,
            "stressed_cells": stressed_cells,
            "avg_anxiety": avg_anxiety,
        }

    def get_network_status(self) -> List[Any]:
        return [cell.get_status() for cell in self.cells] + [repr(self.capacitor)]

    def serialize_genome_vector(self) -> np.ndarray:
        vec = [
            float(self.genome.get("num_cells", self.num_cells)),
            float(self.genome.get("capacitor_capacity", self.capacitor.capacity)),
            float(self.genome.get("global_calm", self.global_calm)),
            float(self.genome.get("diffusivity", 0.0)),
            float(self.genome.get("recharge_per_tick", 1.0)),
        ]
        for c in self.cells:
            vec.extend([float(c.calm), float(c.energy_capacity), float(c.anxiety_sensitivity), float(c.trust)])
        return np.array(vec, dtype=float)

# ------------- Evolutionary machinery -------------

def random_genome(min_cells=3, max_cells=6) -> Dict[str, Any]:
    num_cells = random.randint(min_cells, max_cells)
    genome = {
        "num_cells": num_cells,
        "capacitor_capacity": float(np.random.uniform(3.0, 12.0)),
        "global_calm": float(np.random.uniform(1.0, 8.0)),
        "global_calm_trigger": float(np.random.uniform(5.0, 9.0)),
        "diffusivity": float(np.random.uniform(0.0, 0.15)),
        "recharge_per_tick": float(np.random.uniform(0.5, 2.0)),
        "per_cell": [],
    }
    for _ in range(num_cells):
        per = {
            "calm": float(np.random.uniform(0.5, 6.0)),
            "energy_capacity": float(np.random.uniform(6.0, 20.0)),
            "anxiety_sensitivity": float(np.random.uniform(0.6, 1.6)),
            "trust_baseline": float(np.random.uniform(0.0, 1.0)),
            "personality": random.choice(["neutral", "resilient", "sensitive"]),
        }
        genome["per_cell"].append(per)
    return genome

def crossover(parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
    child = {}
    for k in parent_a.keys():
        if k == "per_cell":
            pa = parent_a["per_cell"]
            pb = parent_b["per_cell"]
            n = max(len(pa), len(pb))
            child_per = []
            for i in range(n):
                if i < len(pa) and i < len(pb):
                    src = random.choice([pa[i], pb[i]])
                    child_per.append(copy.deepcopy(src))
                elif i < len(pa):
                    child_per.append(copy.deepcopy(pa[i]))
                elif i < len(pb):
                    child_per.append(copy.deepcopy(pb[i]))
            child["per_cell"] = child_per
        else:
            child[k] = copy.deepcopy(random.choice([parent_a.get(k), parent_b.get(k)]))
    child["num_cells"] = int(child.get("num_cells", 3))
    while len(child["per_cell"]) < child["num_cells"]:
        child["per_cell"].append(copy.deepcopy(random.choice(child["per_cell"])))
    if len(child["per_cell"]) > child["num_cells"]:
        child["per_cell"] = child["per_cell"][: child["num_cells"]]
    return child

def mutate(genome: Dict[str, Any], mutation_rate: float = 0.15, mutation_strength: float = 0.12) -> Dict[str, Any]:
    g = copy.deepcopy(genome)
    for key in ["capacitor_capacity", "global_calm", "global_calm_trigger", "diffusivity", "recharge_per_tick"]:
        if random.random() < mutation_rate:
            factor = np.random.uniform(1.0 - mutation_strength, 1.0 + mutation_strength)
            g[key] = float(max(0.001, g.get(key, 1.0) * factor))
    if random.random() < 0.05:
        delta = random.choice([-1, 1])
        new_n = int(np.clip(g["num_cells"] + delta, 2, 8))
        if new_n != g["num_cells"]:
            g["num_cells"] = new_n
            if len(g["per_cell"]) < new_n:
                while len(g["per_cell"]) < new_n:
                    g["per_cell"].append(copy.deepcopy(random.choice(g["per_cell"])))
            else:
                g["per_cell"] = g["per_cell"][:new_n]
    for i, per in enumerate(g["per_cell"]):
        if random.random() < mutation_rate:
            per["calm"] = float(max(0.05, per.get("calm", 1.0) * np.random.uniform(1 - mutation_strength, 1 + mutation_strength)))
        if random.random() < mutation_rate:
            per["energy_capacity"] = float(max(1.0, per.get("energy_capacity", 10.0) * np.random.uniform(1 - mutation_strength, 1 + mutation_strength)))
        if random.random() < mutation_rate:
            per["anxiety_sensitivity"] = float(
                np.clip(per.get("anxiety_sensitivity", 1.0) * np.random.uniform(1 - mutation_strength, 1 + mutation_strength), 0.05, 3.0)
            )
        if random.random() < mutation_rate * 0.5:
            per["trust_baseline"] = float(np.clip(per.get("trust_baseline", 0.5) + np.random.uniform(-0.05, 0.05), 0.0, 1.0))
        if random.random() < (mutation_rate * 0.08):
            per["personality"] = random.choice(["neutral", "resilient", "sensitive"])
    return g

# ---------- Pareto + selection utilities ----------
def dominates(a: Dict[str, float], b: Dict[str, float], metrics: List[str]) -> bool:
    better_or_equal = True
    strictly_better = False
    for m in metrics:
        if a[m] < b[m] - 1e-12:
            better_or_equal = False
            break
        if a[m] > b[m] + 1e-12:
            strictly_better = True
    return better_or_equal and strictly_better

def non_dominated_sort(population_objs: List[Dict[str, float]], metrics: List[str]) -> List[int]:
    n = len(population_objs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(population_objs[j], population_objs[i], metrics):
                dominated[i] = True
                break
    front = [i for i, d in enumerate(dominated) if not d]
    return front

def compute_diversity(networks: List[AdaptiveClockNetwork]) -> float:
    vecs = np.vstack([net.serialize_genome_vector() for net in networks])
    return float(np.mean(np.std(vecs, axis=0)))

# ---------- Evolutionary run ----------

class EvolutionEngine:
    def __init__(
        self,
        pop_size: int = 24,
        generations: int = 40,
        survivor_count: int = 8,
        mutation_rate: float = 0.12,
        mutation_strength: float = 0.12,
        min_cells: int = 3,
        max_cells: int = 6,
        diversity_threshold: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.survivor_count = survivor_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.diversity_threshold = diversity_threshold

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            logging.info(f"Random seed set to {random_seed}")

        self.population_genomes: List[Dict[str, Any]] = [random_genome(min_cells, max_cells) for _ in range(pop_size)]
        self.population_networks: List[AdaptiveClockNetwork] = [AdaptiveClockNetwork(g) for g in self.population_genomes]

    def evaluate(self, network: AdaptiveClockNetwork, ticks: int = 16, verbose=False) -> Dict[str, float]:
        start = time.perf_counter()
        for t in range(ticks):
            stimuli = list(np.random.uniform(0.0, 9.0, size=network.num_cells))
            network.network_tick(stimuli)
        end = time.perf_counter()
        metrics = network.calculate_performance_and_stability(end - start)
        speed = 1.0 / (metrics["execution_time"] + 1e-9)
        stressed_score = 1.0 / (1 + metrics["stressed_cells"])
        obj = {
            "stability": float(np.clip(metrics["stability"], -2.0, 2.0)),
            "efficiency": float(np.clip(metrics["efficiency"], 0.0, 2.0)),
            "avg_trust": float(np.clip(metrics["avg_trust"], 0.0, 2.0)),
            "speed": float(speed),
            "stress_resilience": float(stressed_score),
        }
        if verbose:
            logging.info(f"Evaluated network: {obj}")
        return obj

    def run(self):
        history = []
        for gen in range(self.generations):
            logging.info(f"\n=== Generation {gen+1}/{self.generations} ===")
            evaluated_objs = []
            nets_for_eval = []
            for g in self.population_genomes:
                net = AdaptiveClockNetwork(g, runtime_adapt=True)
                nets_for_eval.append(net)
            for net in nets_for_eval:
                obj = self.evaluate(net)
                evaluated_objs.append(obj)
            pareto_metrics = ["stability", "efficiency", "avg_trust", "speed", "stress_resilience"]
            front_indices = non_dominated_sort(evaluated_objs, pareto_metrics)
            logging.info(f"Pareto front size: {len(front_indices)}")
            survivors = []
            if len(front_indices) <= self.survivor_count:
                survivors.extend(front_indices)
                scores = []
                for idx, obj in enumerate(evaluated_objs):
                    s = sum(obj[m] for m in pareto_metrics)
                    scores.append((s, idx))
                scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)
                for s, idx in scores_sorted:
                    if idx not in survivors:
                        survivors.append(idx)
                    if len(survivors) >= self.survivor_count:
                        break
            else:
                candidate_nets = [nets_for_eval[i] for i in front_indices]
                chosen = []
                remaining = set(range(len(candidate_nets)))
                while len(chosen) < self.survivor_count and remaining:
                    best_choice = None
                    best_div = -1
                    for r in list(remaining):
                        trial = chosen + [r]
                        vecs = np.vstack([candidate_nets[i].serialize_genome_vector() for i in trial])
                        div = float(np.mean(np.std(vecs, axis=0)))
                        if div > best_div:
                            best_div = div
                            best_choice = r
                    chosen.append(best_choice)
                    remaining.remove(best_choice)
                for c in chosen:
                    survivors.append(front_indices[c])
            new_population = []
            elites = [self.population_genomes[i] for i in survivors[: self.survivor_count]]
            for e in elites:
                new_population.append(mutate(e, mutation_rate=self.mutation_rate * 0.5, mutation_strength=self.mutation_strength * 0.5))
            while len(new_population) < self.pop_size:
                a = self.tournament_selection(evaluated_objs)
                b = self.tournament_selection(evaluated_objs)
                parent_a = self.population_genomes[a]
                parent_b = self.population_genomes[b]
                child = crossover(parent_a, parent_b)
                child = mutate(child, mutation_rate=self.mutation_rate, mutation_strength=self.mutation_strength)
                new_population.append(child)
            candidate_networks = [AdaptiveClockNetwork(g) for g in new_population]
            div = compute_diversity(candidate_networks)
            logging.info(f"Generation {gen+1} diversity: {div:.4f}")
            if div < self.diversity_threshold:
                injections = int(np.ceil((self.diversity_threshold - div) / (self.diversity_threshold + 1e-9) * self.pop_size))
                injections = max(1, injections)
                logging.info(f"Low diversity ({div:.4f}) -> injecting {injections} random genomes.")
                for i in range(injections):
                    new_population[random.randrange(len(new_population))] = random_genome(self.min_cells, self.max_cells)
            self.population_genomes = new_population
            self.population_networks = [AdaptiveClockNetwork(g) for g in self.population_genomes]
            best_idx = max(range(len(evaluated_objs)), key=lambda i: sum(evaluated_objs[i][m] for m in pareto_metrics))
            best_obj = evaluated_objs[best_idx]
            history.append(best_obj)
            logging.info(f"Best-of-gen {gen+1} (index {best_idx}): {best_obj}")
        return history

    def tournament_selection(self, evaluated_objs: List[Dict[str, float]], k: int = 3) -> int:
        candidates = random.sample(range(len(evaluated_objs)), k)
        pareto_metrics = ["stability", "efficiency", "avg_trust", "speed", "stress_resilience"]
        scores = [(sum(evaluated_objs[i][m] for m in pareto_metrics), i) for i in candidates]
        scores.sort(reverse=True)
        return scores[0][1]

# ---------- CLI & Demo Entry Point ----------

def main():
    parser = argparse.ArgumentParser(description="Evolutionary AdaptiveClockNetwork for V1B3hR/adaptiveneuralnetwork")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--generations", type=int, default=12, help="Number of generations")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument("--survivor_count", type=int, default=6, help="Number of survivors per generation")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    engine = EvolutionEngine(
        pop_size=args.pop_size,
        generations=args.generations,
        survivor_count=args.survivor_count,
        mutation_rate=0.14,
        mutation_strength=0.15,
        min_cells=3,
        max_cells=6,
        diversity_threshold=0.03,
        random_seed=args.seed,
    )

    history = engine.run()
    final_scores = []
    final_networks = [AdaptiveClockNetwork(g, runtime_adapt=False) for g in engine.population_genomes]
    engine_eval_results = []
    for i, net in enumerate(final_networks):
        obj = engine.evaluate(net, ticks=20)
        engine_eval_results.append((sum(obj[m] for m in ["stability", "efficiency", "avg_trust", "speed", "stress_resilience"]), i, obj))
    engine_eval_results.sort(reverse=True, key=lambda x: x[0])

    logging.info("\n=== Final top solutions ===")
    for s, idx, obj in engine_eval_results[:5]:
        logging.info(f"Rank score={s:.3f}, idx={idx}, metrics={obj}")
        logging.info(f" Genome sample (first cell): {engine.population_genomes[idx]['per_cell'][0]}")

    best_idx = engine_eval_results[0][1]
    best_genome = engine.population_genomes[best_idx]
    best_network = AdaptiveClockNetwork(best_genome, runtime_adapt=True)
    logging.info("\n--- Simulating best evolved network for 30 ticks ---")
    for t in range(30):
        stimuli = list(np.random.uniform(0.0, 9.0, size=best_network.num_cells))
        best_network.network_tick(stimuli)
        if t % 5 == 0:
            logging.info(f"Tick {t}: {best_network.get_network_status()}")
    logging.info("\nDone.")

if __name__ == "__main__":
    main()
