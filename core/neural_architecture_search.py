"""
Neural Architecture Search (NAS) System - Phase 4.2

This module provides:
- Neural architecture search for network topology optimization
- Multi-objective hyperparameter optimization
- Automated feature engineering
- Self-debugging and error correction mechanisms
"""

import time
import json
import logging
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable, Union
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NetworkArchitecture:
    """Represents a neural network architecture"""
    architecture_id: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]  # (from_layer, to_layer)
    activation_functions: List[str]
    optimizer_config: Dict[str, Any]
    regularization: Dict[str, Any]
    performance_metrics: Dict[str, float]
    complexity_metrics: Dict[str, float]
    energy_metrics: Dict[str, float]
    timestamp: float


@dataclass
class SearchSpace:
    """Defines the search space for architecture optimization"""
    layer_types: List[str]
    layer_size_range: Tuple[int, int]
    max_layers: int
    activation_functions: List[str]
    optimizer_options: List[str]
    regularization_options: Dict[str, List[float]]
    connection_patterns: List[str]  # "sequential", "skip", "dense", "residual"


@dataclass
class MultiObjective:
    """Represents a multi-objective optimization goal"""
    objective_id: str
    name: str
    weight: float
    maximize: bool
    target_range: Optional[Tuple[float, float]]
    evaluator: Callable[[Dict[str, Any]], float]


@dataclass
class OptimizationResult:
    """Results from architecture optimization"""
    best_architecture: NetworkArchitecture
    pareto_front: List[NetworkArchitecture]
    search_history: List[Dict[str, Any]]
    convergence_metrics: Dict[str, Any]
    total_evaluations: int
    optimization_time: float
    improvement_over_baseline: float


class ArchitectureGenerator(ABC):
    """Abstract base class for architecture generation strategies"""
    
    @abstractmethod
    def generate_architecture(self, search_space: SearchSpace, 
                            previous_architectures: List[NetworkArchitecture] = None) -> NetworkArchitecture:
        """Generate a new architecture based on search space and history"""
        pass
    
    @abstractmethod
    def mutate_architecture(self, architecture: NetworkArchitecture, 
                          mutation_rate: float = 0.1) -> NetworkArchitecture:
        """Mutate an existing architecture"""
        pass


class EvolutionaryGenerator(ArchitectureGenerator):
    """Evolutionary algorithm for architecture generation"""
    
    def __init__(self, population_size: int = 50, elite_ratio: float = 0.2):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.generation_history: List[List[NetworkArchitecture]] = []
        
    def generate_architecture(self, search_space: SearchSpace, 
                            previous_architectures: List[NetworkArchitecture] = None) -> NetworkArchitecture:
        """Generate architecture using evolutionary approach"""
        
        current_time = time.time()
        architecture_id = f"evo_arch_{len(self.generation_history)}_{random.randint(1000, 9999)}"
        
        if previous_architectures is None or len(previous_architectures) == 0:
            # Generate random initial architecture
            return self._generate_random_architecture(search_space, architecture_id, current_time)
        
        # Select parents from previous architectures
        parents = self._select_parents(previous_architectures)
        
        # Create offspring through crossover and mutation
        offspring = self._crossover(parents[0], parents[1], search_space)
        offspring.architecture_id = architecture_id
        offspring.timestamp = current_time
        
        return offspring
    
    def mutate_architecture(self, architecture: NetworkArchitecture, 
                          mutation_rate: float = 0.1) -> NetworkArchitecture:
        """Mutate architecture with given probability"""
        
        mutated = NetworkArchitecture(
            architecture_id=f"mut_{architecture.architecture_id}_{random.randint(100, 999)}",
            layers=architecture.layers.copy(),
            connections=architecture.connections.copy(),
            activation_functions=architecture.activation_functions.copy(),
            optimizer_config=architecture.optimizer_config.copy(),
            regularization=architecture.regularization.copy(),
            performance_metrics={},
            complexity_metrics={},
            energy_metrics={},
            timestamp=time.time()
        )
        
        # Mutate layers
        if random.random() < mutation_rate:
            self._mutate_layers(mutated)
        
        # Mutate activation functions
        if random.random() < mutation_rate:
            self._mutate_activations(mutated)
        
        # Mutate connections
        if random.random() < mutation_rate:
            self._mutate_connections(mutated)
        
        # Mutate optimizer
        if random.random() < mutation_rate:
            self._mutate_optimizer(mutated)
        
        return mutated
    
    def _generate_random_architecture(self, search_space: SearchSpace, 
                                    architecture_id: str, timestamp: float) -> NetworkArchitecture:
        """Generate a random architecture within search space"""
        
        # Random number of layers
        num_layers = random.randint(2, search_space.max_layers)
        
        # Generate layers
        layers = []
        for i in range(num_layers):
            layer_type = random.choice(search_space.layer_types)
            layer_size = random.randint(search_space.layer_size_range[0], search_space.layer_size_range[1])
            
            layer = {
                "layer_id": i,
                "type": layer_type,
                "size": layer_size,
                "parameters": self._generate_layer_parameters(layer_type)
            }
            layers.append(layer)
        
        # Generate connections (start with sequential)
        connections = [(i, i+1) for i in range(num_layers-1)]
        
        # Add some skip connections randomly
        if random.random() < 0.3:  # 30% chance for skip connections
            for i in range(num_layers-2):
                if random.random() < 0.5:
                    target = random.randint(i+2, num_layers-1)
                    connections.append((i, target))
        
        # Generate activation functions
        activation_functions = [
            random.choice(search_space.activation_functions) 
            for _ in range(num_layers)
        ]
        
        # Generate optimizer config
        optimizer_name = random.choice(search_space.optimizer_options)
        optimizer_config = {
            "name": optimizer_name,
            "learning_rate": random.uniform(0.0001, 0.1),
            "parameters": self._generate_optimizer_parameters(optimizer_name)
        }
        
        # Generate regularization
        regularization = {}
        for reg_type, values in search_space.regularization_options.items():
            if random.random() < 0.5:  # 50% chance to include each regularization
                regularization[reg_type] = random.choice(values)
        
        return NetworkArchitecture(
            architecture_id=architecture_id,
            layers=layers,
            connections=connections,
            activation_functions=activation_functions,
            optimizer_config=optimizer_config,
            regularization=regularization,
            performance_metrics={},
            complexity_metrics={},
            energy_metrics={},
            timestamp=timestamp
        )
    
    def _select_parents(self, population: List[NetworkArchitecture]) -> Tuple[NetworkArchitecture, NetworkArchitecture]:
        """Select two parents using tournament selection"""
        
        def tournament_select(pop: List[NetworkArchitecture], tournament_size: int = 3) -> NetworkArchitecture:
            tournament = random.sample(pop, min(tournament_size, len(pop)))
            # Select based on performance (assuming higher is better)
            best = max(tournament, key=lambda arch: arch.performance_metrics.get("overall_score", 0))
            return best
        
        parent1 = tournament_select(population)
        parent2 = tournament_select(population)
        
        # Ensure different parents
        attempts = 0
        while parent1.architecture_id == parent2.architecture_id and attempts < 10:
            parent2 = tournament_select(population)
            attempts += 1
        
        return parent1, parent2
    
    def _crossover(self, parent1: NetworkArchitecture, parent2: NetworkArchitecture, 
                  search_space: SearchSpace) -> NetworkArchitecture:
        """Create offspring through crossover of two parents"""
        
        # Choose the better performing parent as primary template
        if parent1.performance_metrics.get("overall_score", 0) > parent2.performance_metrics.get("overall_score", 0):
            primary, secondary = parent1, parent2
        else:
            primary, secondary = parent2, parent1
        
        # Start with primary parent's structure
        offspring_layers = primary.layers.copy()
        offspring_connections = primary.connections.copy()
        
        # Crossover layers (50% chance to inherit from secondary parent)
        for i in range(len(offspring_layers)):
            if i < len(secondary.layers) and random.random() < 0.5:
                offspring_layers[i] = secondary.layers[i].copy()
        
        # Crossover activation functions
        offspring_activations = primary.activation_functions.copy()
        for i in range(len(offspring_activations)):
            if i < len(secondary.activation_functions) and random.random() < 0.5:
                offspring_activations[i] = secondary.activation_functions[i]
        
        # Crossover optimizer (blend parameters)
        offspring_optimizer = primary.optimizer_config.copy()
        if random.random() < 0.5:
            offspring_optimizer["name"] = secondary.optimizer_config["name"]
        
        # Blend learning rate
        lr1 = primary.optimizer_config.get("learning_rate", 0.01)
        lr2 = secondary.optimizer_config.get("learning_rate", 0.01)
        offspring_optimizer["learning_rate"] = (lr1 + lr2) / 2
        
        # Crossover regularization
        offspring_regularization = primary.regularization.copy()
        for reg_type, value in secondary.regularization.items():
            if random.random() < 0.5:
                offspring_regularization[reg_type] = value
        
        return NetworkArchitecture(
            architecture_id=f"cross_{primary.architecture_id[:8]}_{secondary.architecture_id[:8]}",
            layers=offspring_layers,
            connections=offspring_connections,
            activation_functions=offspring_activations,
            optimizer_config=offspring_optimizer,
            regularization=offspring_regularization,
            performance_metrics={},
            complexity_metrics={},
            energy_metrics={},
            timestamp=time.time()
        )
    
    def _mutate_layers(self, architecture: NetworkArchitecture) -> None:
        """Mutate layer configuration"""
        if not architecture.layers:
            return
        
        mutation_type = random.choice(["size", "add", "remove", "type"])
        
        if mutation_type == "size" and architecture.layers:
            layer_idx = random.randint(0, len(architecture.layers) - 1)
            current_size = architecture.layers[layer_idx]["size"]
            # Modify size by ±20%
            size_change = int(current_size * random.uniform(-0.2, 0.2))
            new_size = max(1, current_size + size_change)
            architecture.layers[layer_idx]["size"] = new_size
            
        elif mutation_type == "add" and len(architecture.layers) < 20:  # Max 20 layers
            insert_idx = random.randint(0, len(architecture.layers))
            new_layer = {
                "layer_id": len(architecture.layers),
                "type": random.choice(["dense", "conv1d", "lstm"]),
                "size": random.randint(10, 256),
                "parameters": {}
            }
            architecture.layers.insert(insert_idx, new_layer)
            
            # Update connections
            self._update_connections_after_layer_insertion(architecture, insert_idx)
            
        elif mutation_type == "remove" and len(architecture.layers) > 2:  # Keep at least 2 layers
            remove_idx = random.randint(1, len(architecture.layers) - 2)  # Don't remove first or last
            del architecture.layers[remove_idx]
            
            # Update connections
            self._update_connections_after_layer_removal(architecture, remove_idx)
    
    def _mutate_activations(self, architecture: NetworkArchitecture) -> None:
        """Mutate activation functions"""
        if not architecture.activation_functions:
            return
        
        activations = ["relu", "tanh", "sigmoid", "leaky_relu", "swish", "gelu"]
        mutation_idx = random.randint(0, len(architecture.activation_functions) - 1)
        architecture.activation_functions[mutation_idx] = random.choice(activations)
    
    def _mutate_connections(self, architecture: NetworkArchitecture) -> None:
        """Mutate network connections"""
        if random.random() < 0.5 and len(architecture.layers) > 2:
            # Add a skip connection
            source = random.randint(0, len(architecture.layers) - 3)
            target = random.randint(source + 2, len(architecture.layers) - 1)
            new_connection = (source, target)
            
            if new_connection not in architecture.connections:
                architecture.connections.append(new_connection)
        
        elif architecture.connections and random.random() < 0.3:
            # Remove a connection (but keep basic connectivity)
            skip_connections = [conn for conn in architecture.connections 
                             if conn[1] - conn[0] > 1]  # Non-sequential connections
            if skip_connections:
                remove_connection = random.choice(skip_connections)
                architecture.connections.remove(remove_connection)
    
    def _mutate_optimizer(self, architecture: NetworkArchitecture) -> None:
        """Mutate optimizer configuration"""
        # Mutate learning rate
        current_lr = architecture.optimizer_config.get("learning_rate", 0.01)
        lr_multiplier = random.uniform(0.5, 2.0)
        new_lr = min(0.1, max(0.0001, current_lr * lr_multiplier))
        architecture.optimizer_config["learning_rate"] = new_lr
        
        # Occasionally change optimizer type
        if random.random() < 0.2:
            optimizers = ["adam", "sgd", "rmsprop", "adagrad"]
            architecture.optimizer_config["name"] = random.choice(optimizers)
    
    def _generate_layer_parameters(self, layer_type: str) -> Dict[str, Any]:
        """Generate parameters specific to layer type"""
        if layer_type == "conv1d":
            return {
                "kernel_size": random.choice([3, 5, 7]),
                "stride": random.choice([1, 2]),
                "padding": random.choice(["same", "valid"])
            }
        elif layer_type == "lstm":
            return {
                "return_sequences": random.choice([True, False]),
                "dropout": random.uniform(0.0, 0.3)
            }
        elif layer_type == "dense":
            return {
                "use_bias": random.choice([True, False])
            }
        else:
            return {}
    
    def _generate_optimizer_parameters(self, optimizer_name: str) -> Dict[str, Any]:
        """Generate optimizer-specific parameters"""
        params = {}
        
        if optimizer_name == "adam":
            params.update({
                "beta1": random.uniform(0.8, 0.95),
                "beta2": random.uniform(0.9, 0.999),
                "epsilon": random.uniform(1e-8, 1e-6)
            })
        elif optimizer_name == "sgd":
            params.update({
                "momentum": random.uniform(0.0, 0.9),
                "nesterov": random.choice([True, False])
            })
        elif optimizer_name == "rmsprop":
            params.update({
                "rho": random.uniform(0.8, 0.95),
                "epsilon": random.uniform(1e-8, 1e-6)
            })
        
        return params
    
    def _update_connections_after_layer_insertion(self, architecture: NetworkArchitecture, insert_idx: int) -> None:
        """Update connections after inserting a layer"""
        # Update layer IDs
        for layer in architecture.layers[insert_idx+1:]:
            layer["layer_id"] += 1
        
        # Update connections
        updated_connections = []
        for source, target in architecture.connections:
            new_source = source if source < insert_idx else source + 1
            new_target = target if target < insert_idx else target + 1
            updated_connections.append((new_source, new_target))
        
        # Add connections for the new layer
        if insert_idx > 0:
            updated_connections.append((insert_idx - 1, insert_idx))
        if insert_idx < len(architecture.layers) - 1:
            updated_connections.append((insert_idx, insert_idx + 1))
        
        architecture.connections = updated_connections
    
    def _update_connections_after_layer_removal(self, architecture: NetworkArchitecture, remove_idx: int) -> None:
        """Update connections after removing a layer"""
        # Update layer IDs
        for layer in architecture.layers[remove_idx:]:
            layer["layer_id"] -= 1
        
        # Update connections
        updated_connections = []
        for source, target in architecture.connections:
            if source == remove_idx or target == remove_idx:
                continue  # Remove connections involving deleted layer
            
            new_source = source if source < remove_idx else source - 1
            new_target = target if target < remove_idx else target - 1
            updated_connections.append((new_source, new_target))
        
        architecture.connections = updated_connections


class MultiObjectiveOptimizer:
    """Multi-objective optimization for neural architectures"""
    
    def __init__(self, objectives: List[MultiObjective]):
        self.objectives = objectives
        self.pareto_front: List[NetworkArchitecture] = []
        self.dominated_solutions: List[NetworkArchitecture] = []
        
    def evaluate_architecture(self, architecture: NetworkArchitecture, 
                            evaluation_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate architecture against all objectives"""
        
        objective_scores = {}
        
        for objective in self.objectives:
            try:
                score = objective.evaluator(evaluation_data)
                objective_scores[objective.objective_id] = score
            except Exception as e:
                logger.warning(f"Failed to evaluate objective {objective.objective_id}: {e}")
                objective_scores[objective.objective_id] = 0.0
        
        return objective_scores
    
    def update_pareto_front(self, new_architecture: NetworkArchitecture) -> bool:
        """Update Pareto front with new architecture"""
        
        new_scores = self._get_objective_scores(new_architecture)
        
        # Check if new architecture is dominated by any in Pareto front
        dominated_by_existing = False
        for existing_arch in self.pareto_front:
            existing_scores = self._get_objective_scores(existing_arch)
            if self._dominates(existing_scores, new_scores):
                dominated_by_existing = True
                break
        
        if dominated_by_existing:
            self.dominated_solutions.append(new_architecture)
            return False
        
        # Remove architectures from Pareto front that are dominated by new architecture
        non_dominated = []
        for existing_arch in self.pareto_front:
            existing_scores = self._get_objective_scores(existing_arch)
            if not self._dominates(new_scores, existing_scores):
                non_dominated.append(existing_arch)
            else:
                self.dominated_solutions.append(existing_arch)
        
        # Add new architecture to Pareto front
        self.pareto_front = non_dominated
        self.pareto_front.append(new_architecture)
        
        return True
    
    def get_best_compromise_solution(self) -> Optional[NetworkArchitecture]:
        """Get best compromise solution from Pareto front"""
        
        if not self.pareto_front:
            return None
        
        # Calculate weighted sum for each solution
        best_arch = None
        best_score = float('-inf')
        
        for arch in self.pareto_front:
            weighted_score = 0.0
            arch_scores = self._get_objective_scores(arch)
            
            for objective in self.objectives:
                score = arch_scores.get(objective.objective_id, 0.0)
                if not objective.maximize:
                    score = 1.0 - score  # Convert minimization to maximization
                weighted_score += objective.weight * score
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_arch = arch
        
        return best_arch
    
    def _get_objective_scores(self, architecture: NetworkArchitecture) -> Dict[str, float]:
        """Extract objective scores from architecture"""
        scores = {}
        
        for objective in self.objectives:
            if objective.objective_id in architecture.performance_metrics:
                scores[objective.objective_id] = architecture.performance_metrics[objective.objective_id]
            else:
                scores[objective.objective_id] = 0.0
        
        return scores
    
    def _dominates(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
        """Check if scores1 dominates scores2 (Pareto dominance)"""
        
        at_least_one_better = False
        
        for objective in self.objectives:
            score1 = scores1.get(objective.objective_id, 0.0)
            score2 = scores2.get(objective.objective_id, 0.0)
            
            if objective.maximize:
                if score1 < score2:
                    return False  # scores1 is worse in at least one objective
                elif score1 > score2:
                    at_least_one_better = True
            else:  # minimize
                if score1 > score2:
                    return False  # scores1 is worse in at least one objective
                elif score1 < score2:
                    at_least_one_better = True
        
        return at_least_one_better


class AutomatedFeatureEngineer:
    """Automated feature engineering system"""
    
    def __init__(self):
        self.feature_transformations = {
            "polynomial": self._polynomial_features,
            "interaction": self._interaction_features,
            "statistical": self._statistical_features,
            "temporal": self._temporal_features,
            "frequency": self._frequency_features
        }
        
        self.feature_history: List[Dict[str, Any]] = []
        
    def engineer_features(self, data: Dict[str, Any], 
                         target_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """Automatically engineer features from input data"""
        
        current_time = time.time()
        
        engineered_features = {}
        transformation_log = {
            "timestamp": current_time,
            "input_features": list(data.keys()),
            "transformations_applied": [],
            "output_features": [],
            "performance_impact": {}
        }
        
        # Apply different feature engineering techniques
        for transform_name, transform_func in self.feature_transformations.items():
            try:
                new_features = transform_func(data)
                
                if new_features:
                    engineered_features.update(new_features)
                    transformation_log["transformations_applied"].append(transform_name)
                    transformation_log["output_features"].extend(list(new_features.keys()))
                    
            except Exception as e:
                logger.warning(f"Feature transformation {transform_name} failed: {e}")
        
        # Evaluate feature quality if target metrics provided
        if target_metrics:
            feature_quality = self._evaluate_feature_quality(engineered_features, target_metrics)
            transformation_log["performance_impact"] = feature_quality
        
        # Store history
        self.feature_history.append(transformation_log)
        
        # Combine original and engineered features
        result = {**data, **engineered_features}
        
        return result
    
    def _polynomial_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate polynomial features"""
        poly_features = {}
        
        numerical_features = {k: v for k, v in data.items() 
                            if isinstance(v, (int, float)) and abs(v) < 1000}
        
        # Generate squared features
        for name, value in numerical_features.items():
            poly_features[f"{name}_squared"] = value ** 2
        
        # Generate cube root features for large values
        for name, value in numerical_features.items():
            if abs(value) > 1:
                poly_features[f"{name}_cuberoot"] = np.sign(value) * (abs(value) ** (1/3))
        
        return poly_features
    
    def _interaction_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interaction features between variables"""
        interaction_features = {}
        
        numerical_features = {k: v for k, v in data.items() 
                            if isinstance(v, (int, float))}
        
        feature_names = list(numerical_features.keys())
        
        # Generate pairwise interactions (limited to avoid explosion)
        for i, name1 in enumerate(feature_names[:10]):  # Limit to first 10 features
            for name2 in feature_names[i+1:10]:
                if name1 != name2:
                    value1 = numerical_features[name1]
                    value2 = numerical_features[name2]
                    
                    # Multiplicative interaction
                    interaction_features[f"{name1}_x_{name2}"] = value1 * value2
                    
                    # Ratio interaction (avoid division by zero)
                    if abs(value2) > 1e-10:
                        interaction_features[f"{name1}_div_{name2}"] = value1 / value2
        
        return interaction_features
    
    def _statistical_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical features"""
        stat_features = {}
        
        numerical_values = [v for v in data.values() if isinstance(v, (int, float))]
        
        if len(numerical_values) > 1:
            stat_features["mean_all_features"] = np.mean(numerical_values)
            stat_features["std_all_features"] = np.std(numerical_values)
            stat_features["max_all_features"] = np.max(numerical_values)
            stat_features["min_all_features"] = np.min(numerical_values)
            stat_features["range_all_features"] = np.max(numerical_values) - np.min(numerical_values)
        
        # Generate z-scores for each numerical feature
        if len(numerical_values) > 1:
            mean_val = np.mean(numerical_values)
            std_val = np.std(numerical_values)
            
            if std_val > 1e-10:
                for name, value in data.items():
                    if isinstance(value, (int, float)):
                        stat_features[f"{name}_zscore"] = (value - mean_val) / std_val
        
        return stat_features
    
    def _temporal_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate temporal features if time-related data is present"""
        temporal_features = {}
        
        # Look for time-related features
        time_related_keys = [k for k in data.keys() 
                           if any(time_word in k.lower() 
                                for time_word in ["time", "timestamp", "duration", "delay"])]
        
        for key in time_related_keys:
            value = data[key]
            if isinstance(value, (int, float)):
                # Generate time-based features
                temporal_features[f"{key}_sin"] = np.sin(2 * np.pi * value / 24)  # Daily cycle
                temporal_features[f"{key}_cos"] = np.cos(2 * np.pi * value / 24)
                temporal_features[f"{key}_log"] = np.log(max(1e-10, abs(value)))
        
        return temporal_features
    
    def _frequency_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate frequency-domain features for array-like data"""
        freq_features = {}
        
        # Look for array-like features
        for name, value in data.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 4:
                try:
                    # Convert to numpy array
                    arr = np.array(value, dtype=float)
                    
                    if len(arr) > 0:
                        # Basic frequency domain features
                        fft = np.fft.fft(arr)
                        fft_magnitude = np.abs(fft)
                        
                        freq_features[f"{name}_fft_mean"] = np.mean(fft_magnitude)
                        freq_features[f"{name}_fft_std"] = np.std(fft_magnitude)
                        freq_features[f"{name}_fft_max"] = np.max(fft_magnitude)
                        
                        # Dominant frequency
                        dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
                        freq_features[f"{name}_dominant_freq"] = dominant_freq_idx
                        
                except Exception as e:
                    logger.debug(f"Failed to compute frequency features for {name}: {e}")
        
        return freq_features
    
    def _evaluate_feature_quality(self, features: Dict[str, Any], 
                                 target_metrics: Dict[str, float]) -> Dict[str, float]:
        """Evaluate quality of engineered features"""
        quality_metrics = {}
        
        try:
            # Calculate feature variance (higher is potentially better)
            numerical_features = {k: v for k, v in features.items() 
                                if isinstance(v, (int, float))}
            
            if numerical_features:
                values = list(numerical_features.values())
                quality_metrics["variance"] = np.var(values)
                quality_metrics["mean_abs_value"] = np.mean([abs(v) for v in values])
                quality_metrics["non_zero_ratio"] = len([v for v in values if abs(v) > 1e-10]) / len(values)
        
        except Exception as e:
            logger.warning(f"Failed to evaluate feature quality: {e}")
            quality_metrics["error"] = str(e)
        
        return quality_metrics


class SelfDebuggingSystem:
    """Self-debugging and error correction system"""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.correction_strategies: Dict[str, Callable] = {
            "memory_error": self._handle_memory_error,
            "convergence_failure": self._handle_convergence_failure,
            "nan_values": self._handle_nan_values,
            "poor_performance": self._handle_poor_performance,
            "architecture_error": self._handle_architecture_error
        }
        
        self.debug_history: List[Dict[str, Any]] = []
        
    def detect_and_correct_errors(self, architecture: NetworkArchitecture, 
                                 training_metrics: Dict[str, Any],
                                 error_context: Dict[str, Any] = None) -> Tuple[NetworkArchitecture, List[str]]:
        """Detect errors and apply corrections"""
        
        current_time = time.time()
        detected_errors = []
        applied_corrections = []
        
        # Detect different types of errors
        error_checks = [
            ("memory_error", self._check_memory_error),
            ("convergence_failure", self._check_convergence_failure),
            ("nan_values", self._check_nan_values),
            ("poor_performance", self._check_poor_performance),
            ("architecture_error", self._check_architecture_error)
        ]
        
        corrected_architecture = architecture
        
        for error_type, check_func in error_checks:
            try:
                if check_func(architecture, training_metrics, error_context):
                    detected_errors.append(error_type)
                    
                    # Apply correction
                    correction_func = self.correction_strategies.get(error_type)
                    if correction_func:
                        corrected_architecture, correction_msg = correction_func(
                            corrected_architecture, training_metrics, error_context
                        )
                        applied_corrections.append(correction_msg)
                        
            except Exception as e:
                logger.warning(f"Error during {error_type} check/correction: {e}")
        
        # Log debug session
        debug_session = {
            "timestamp": current_time,
            "original_architecture_id": architecture.architecture_id,
            "corrected_architecture_id": corrected_architecture.architecture_id,
            "detected_errors": detected_errors,
            "applied_corrections": applied_corrections,
            "training_metrics": training_metrics,
            "context": error_context or {}
        }
        
        self.debug_history.append(debug_session)
        
        # Store error patterns for learning
        for error_type in detected_errors:
            self.error_patterns[error_type].append({
                "architecture": asdict(architecture),
                "metrics": training_metrics,
                "context": error_context,
                "timestamp": current_time
            })
        
        return corrected_architecture, applied_corrections
    
    def _check_memory_error(self, architecture: NetworkArchitecture, 
                           training_metrics: Dict[str, Any], 
                           context: Dict[str, Any] = None) -> bool:
        """Check for memory-related errors"""
        # Check if architecture is too large
        total_params = sum(layer.get("size", 0) for layer in architecture.layers)
        
        if total_params > 1000000:  # More than 1M parameters
            return True
        
        # Check memory metrics if available
        if context and "memory_usage" in context:
            if context["memory_usage"] > 0.9:  # More than 90% memory usage
                return True
        
        return False
    
    def _check_convergence_failure(self, architecture: NetworkArchitecture, 
                                  training_metrics: Dict[str, Any], 
                                  context: Dict[str, Any] = None) -> bool:
        """Check for convergence failures"""
        # Check if loss is not decreasing
        if "loss_history" in training_metrics:
            loss_history = training_metrics["loss_history"]
            if len(loss_history) > 10:
                recent_losses = loss_history[-10:]
                if all(loss >= recent_losses[0] for loss in recent_losses):
                    return True  # Loss not decreasing
        
        # Check learning rate
        lr = architecture.optimizer_config.get("learning_rate", 0.01)
        if lr > 0.1 or lr < 1e-6:
            return True  # Learning rate too high or too low
        
        return False
    
    def _check_nan_values(self, architecture: NetworkArchitecture, 
                         training_metrics: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> bool:
        """Check for NaN values in metrics"""
        # Check performance metrics
        for metric_name, value in architecture.performance_metrics.items():
            if np.isnan(value) or np.isinf(value):
                return True
        
        # Check training metrics
        for metric_name, value in training_metrics.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                return True
        
        return False
    
    def _check_poor_performance(self, architecture: NetworkArchitecture, 
                               training_metrics: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> bool:
        """Check for poor performance"""
        # Check if accuracy is too low
        if "accuracy" in architecture.performance_metrics:
            if architecture.performance_metrics["accuracy"] < 0.1:
                return True
        
        # Check if loss is too high
        if "loss" in training_metrics:
            if training_metrics["loss"] > 10.0:
                return True
        
        return False
    
    def _check_architecture_error(self, architecture: NetworkArchitecture, 
                                 training_metrics: Dict[str, Any], 
                                 context: Dict[str, Any] = None) -> bool:
        """Check for architecture-related errors"""
        # Check for disconnected layers
        max_layer_id = max(layer["layer_id"] for layer in architecture.layers) if architecture.layers else -1
        
        for source, target in architecture.connections:
            if source > max_layer_id or target > max_layer_id:
                return True  # Invalid connection
        
        # Check for very small or very large layers
        for layer in architecture.layers:
            size = layer.get("size", 0)
            if size <= 0 or size > 10000:
                return True
        
        return False
    
    def _handle_memory_error(self, architecture: NetworkArchitecture, 
                           training_metrics: Dict[str, Any], 
                           context: Dict[str, Any] = None) -> Tuple[NetworkArchitecture, str]:
        """Handle memory-related errors"""
        corrected = NetworkArchitecture(
            architecture_id=f"mem_corrected_{architecture.architecture_id}",
            layers=architecture.layers.copy(),
            connections=architecture.connections.copy(),
            activation_functions=architecture.activation_functions.copy(),
            optimizer_config=architecture.optimizer_config.copy(),
            regularization=architecture.regularization.copy(),
            performance_metrics=architecture.performance_metrics.copy(),
            complexity_metrics=architecture.complexity_metrics.copy(),
            energy_metrics=architecture.energy_metrics.copy(),
            timestamp=time.time()
        )
        
        # Reduce layer sizes
        for layer in corrected.layers:
            current_size = layer.get("size", 0)
            if current_size > 512:
                layer["size"] = min(512, current_size // 2)
        
        return corrected, "Reduced layer sizes to address memory constraints"
    
    def _handle_convergence_failure(self, architecture: NetworkArchitecture, 
                                   training_metrics: Dict[str, Any], 
                                   context: Dict[str, Any] = None) -> Tuple[NetworkArchitecture, str]:
        """Handle convergence failures"""
        corrected = NetworkArchitecture(
            architecture_id=f"conv_corrected_{architecture.architecture_id}",
            layers=architecture.layers.copy(),
            connections=architecture.connections.copy(),
            activation_functions=architecture.activation_functions.copy(),
            optimizer_config=architecture.optimizer_config.copy(),
            regularization=architecture.regularization.copy(),
            performance_metrics=architecture.performance_metrics.copy(),
            complexity_metrics=architecture.complexity_metrics.copy(),
            energy_metrics=architecture.energy_metrics.copy(),
            timestamp=time.time()
        )
        
        # Adjust learning rate
        current_lr = corrected.optimizer_config.get("learning_rate", 0.01)
        if current_lr > 0.01:
            corrected.optimizer_config["learning_rate"] = 0.01
        elif current_lr < 0.001:
            corrected.optimizer_config["learning_rate"] = 0.001
        
        # Add regularization
        corrected.regularization["dropout"] = 0.2
        
        return corrected, "Adjusted learning rate and added regularization for better convergence"
    
    def _handle_nan_values(self, architecture: NetworkArchitecture, 
                          training_metrics: Dict[str, Any], 
                          context: Dict[str, Any] = None) -> Tuple[NetworkArchitecture, str]:
        """Handle NaN values"""
        corrected = NetworkArchitecture(
            architecture_id=f"nan_corrected_{architecture.architecture_id}",
            layers=architecture.layers.copy(),
            connections=architecture.connections.copy(),
            activation_functions=architecture.activation_functions.copy(),
            optimizer_config=architecture.optimizer_config.copy(),
            regularization=architecture.regularization.copy(),
            performance_metrics=architecture.performance_metrics.copy(),
            complexity_metrics=architecture.complexity_metrics.copy(),
            energy_metrics=architecture.energy_metrics.copy(),
            timestamp=time.time()
        )
        
        # Reset NaN metrics
        for key in corrected.performance_metrics:
            if np.isnan(corrected.performance_metrics[key]) or np.isinf(corrected.performance_metrics[key]):
                corrected.performance_metrics[key] = 0.0
        
        # Reduce learning rate to prevent numerical instability
        corrected.optimizer_config["learning_rate"] = min(0.001, 
                                                         corrected.optimizer_config.get("learning_rate", 0.01))
        
        # Change activation functions to more stable ones
        for i, activation in enumerate(corrected.activation_functions):
            if activation in ["sigmoid", "tanh"]:  # Potentially unstable
                corrected.activation_functions[i] = "relu"
        
        return corrected, "Reset NaN values, reduced learning rate, and switched to stable activations"
    
    def _handle_poor_performance(self, architecture: NetworkArchitecture, 
                                training_metrics: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> Tuple[NetworkArchitecture, str]:
        """Handle poor performance"""
        corrected = NetworkArchitecture(
            architecture_id=f"perf_corrected_{architecture.architecture_id}",
            layers=architecture.layers.copy(),
            connections=architecture.connections.copy(),
            activation_functions=architecture.activation_functions.copy(),
            optimizer_config=architecture.optimizer_config.copy(),
            regularization=architecture.regularization.copy(),
            performance_metrics=architecture.performance_metrics.copy(),
            complexity_metrics=architecture.complexity_metrics.copy(),
            energy_metrics=architecture.energy_metrics.copy(),
            timestamp=time.time()
        )
        
        # Increase model complexity
        for layer in corrected.layers:
            if layer.get("type") == "dense":
                current_size = layer.get("size", 32)
                layer["size"] = min(512, int(current_size * 1.5))
        
        # Change optimizer to Adam if not already
        if corrected.optimizer_config.get("name") != "adam":
            corrected.optimizer_config["name"] = "adam"
            corrected.optimizer_config["learning_rate"] = 0.001
        
        return corrected, "Increased model complexity and switched to Adam optimizer"
    
    def _handle_architecture_error(self, architecture: NetworkArchitecture, 
                                  training_metrics: Dict[str, Any], 
                                  context: Dict[str, Any] = None) -> Tuple[NetworkArchitecture, str]:
        """Handle architecture errors"""
        corrected = NetworkArchitecture(
            architecture_id=f"arch_corrected_{architecture.architecture_id}",
            layers=architecture.layers.copy(),
            connections=[],  # Reset connections
            activation_functions=architecture.activation_functions.copy(),
            optimizer_config=architecture.optimizer_config.copy(),
            regularization=architecture.regularization.copy(),
            performance_metrics=architecture.performance_metrics.copy(),
            complexity_metrics=architecture.complexity_metrics.copy(),
            energy_metrics=architecture.energy_metrics.copy(),
            timestamp=time.time()
        )
        
        # Fix layer IDs
        for i, layer in enumerate(corrected.layers):
            layer["layer_id"] = i
            
            # Fix layer sizes
            size = layer.get("size", 0)
            if size <= 0:
                layer["size"] = 32
            elif size > 10000:
                layer["size"] = 512
        
        # Rebuild sequential connections
        for i in range(len(corrected.layers) - 1):
            corrected.connections.append((i, i + 1))
        
        # Ensure we have activation functions for all layers
        while len(corrected.activation_functions) < len(corrected.layers):
            corrected.activation_functions.append("relu")
        
        return corrected, "Fixed architecture errors: reset connections, fixed layer IDs and sizes"


class NeuralArchitectureSearchSystem:
    """Main system for neural architecture search with multi-objective optimization"""
    
    def __init__(self, search_space: SearchSpace, objectives: List[MultiObjective]):
        self.search_space = search_space
        self.objectives = objectives
        
        self.architecture_generator = EvolutionaryGenerator()
        self.multi_objective_optimizer = MultiObjectiveOptimizer(objectives)
        self.feature_engineer = AutomatedFeatureEngineer()
        self.debug_system = SelfDebuggingSystem()
        
        self.search_history: List[NetworkArchitecture] = []
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Neural Architecture Search System initialized")
    
    def search_optimal_architecture(self, evaluation_function: Callable[[NetworkArchitecture], Dict[str, Any]],
                                   max_evaluations: int = 100, 
                                   population_size: int = 20) -> OptimizationResult:
        """Search for optimal architecture using evolutionary approach"""
        
        start_time = time.time()
        current_population: List[NetworkArchitecture] = []
        
        logger.info(f"Starting architecture search with {max_evaluations} evaluations")
        
        # Generate initial population
        for i in range(population_size):
            arch = self.architecture_generator.generate_architecture(
                self.search_space, current_population
            )
            current_population.append(arch)
        
        evaluations_performed = 0
        generation = 0
        
        while evaluations_performed < max_evaluations:
            logger.info(f"Generation {generation}, Evaluations: {evaluations_performed}/{max_evaluations}")
            
            # Evaluate current population
            for arch in current_population:
                if arch.architecture_id not in self.evaluation_results:
                    try:
                        # Evaluate architecture
                        evaluation_data = evaluation_function(arch)
                        
                        # Engineer features for better evaluation
                        enhanced_evaluation = self.feature_engineer.engineer_features(
                            evaluation_data, arch.performance_metrics
                        )
                        
                        # Calculate objective scores
                        objective_scores = self.multi_objective_optimizer.evaluate_architecture(
                            arch, enhanced_evaluation
                        )
                        
                        # Update architecture with scores
                        arch.performance_metrics.update(objective_scores)
                        
                        # Check for errors and apply corrections
                        corrected_arch, corrections = self.debug_system.detect_and_correct_errors(
                            arch, enhanced_evaluation, {"generation": generation}
                        )
                        
                        if corrections:
                            logger.info(f"Applied corrections to {arch.architecture_id}: {corrections}")
                            # Re-evaluate corrected architecture
                            corrected_evaluation = evaluation_function(corrected_arch)
                            corrected_scores = self.multi_objective_optimizer.evaluate_architecture(
                                corrected_arch, corrected_evaluation
                            )
                            corrected_arch.performance_metrics.update(corrected_scores)
                            
                            # Use corrected architecture if it's better
                            if self._is_better_architecture(corrected_arch, arch):
                                arch = corrected_arch
                        
                        # Update Pareto front
                        self.multi_objective_optimizer.update_pareto_front(arch)
                        
                        # Store results
                        self.evaluation_results[arch.architecture_id] = enhanced_evaluation
                        self.search_history.append(arch)
                        
                        evaluations_performed += 1
                        
                        if evaluations_performed >= max_evaluations:
                            break
                            
                    except Exception as e:
                        logger.error(f"Failed to evaluate architecture {arch.architecture_id}: {e}")
                        # Remove problematic architecture
                        if arch in current_population:
                            current_population.remove(arch)
            
            # Generate next generation
            if evaluations_performed < max_evaluations:
                next_population = self._generate_next_population(
                    current_population, population_size
                )
                current_population = next_population
                generation += 1
        
        # Prepare results
        end_time = time.time()
        optimization_time = end_time - start_time
        
        best_architecture = self.multi_objective_optimizer.get_best_compromise_solution()
        pareto_front = self.multi_objective_optimizer.pareto_front.copy()
        
        # Calculate improvement over baseline (first architecture)
        baseline_score = 0.0
        best_score = 0.0
        
        if self.search_history:
            baseline_arch = self.search_history[0]
            baseline_score = sum(baseline_arch.performance_metrics.values()) / len(baseline_arch.performance_metrics)
            
        if best_architecture:
            best_score = sum(best_architecture.performance_metrics.values()) / len(best_architecture.performance_metrics)
        
        improvement = ((best_score - baseline_score) / max(baseline_score, 1e-10)) * 100
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics()
        
        result = OptimizationResult(
            best_architecture=best_architecture,
            pareto_front=pareto_front,
            search_history=self.search_history.copy(),
            convergence_metrics=convergence_metrics,
            total_evaluations=evaluations_performed,
            optimization_time=optimization_time,
            improvement_over_baseline=improvement
        )
        
        logger.info(f"Architecture search completed. Best improvement: {improvement:.2f}%")
        
        return result
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics of the search process"""
        
        if not self.search_history:
            return {"message": "No search history available"}
        
        # Performance trends
        performance_trend = []
        for arch in self.search_history:
            overall_score = sum(arch.performance_metrics.values()) / max(len(arch.performance_metrics), 1)
            performance_trend.append({
                "architecture_id": arch.architecture_id,
                "timestamp": arch.timestamp,
                "overall_score": overall_score,
                "individual_scores": arch.performance_metrics
            })
        
        # Architecture diversity analysis
        diversity_metrics = self._analyze_architecture_diversity()
        
        # Feature engineering impact
        feature_impact = self._analyze_feature_engineering_impact()
        
        # Debug system effectiveness
        debug_effectiveness = self._analyze_debug_effectiveness()
        
        analytics = {
            "search_summary": {
                "total_architectures_evaluated": len(self.search_history),
                "pareto_front_size": len(self.multi_objective_optimizer.pareto_front),
                "dominated_solutions": len(self.multi_objective_optimizer.dominated_solutions),
                "unique_architectures": len(set(arch.architecture_id for arch in self.search_history)),
            },
            "performance_trends": performance_trend,
            "diversity_metrics": diversity_metrics,
            "feature_engineering_impact": feature_impact,
            "debug_system_effectiveness": debug_effectiveness,
            "objective_analysis": self._analyze_objective_performance(),
            "convergence_analysis": self._calculate_convergence_metrics()
        }
        
        return analytics
    
    def _generate_next_population(self, current_population: List[NetworkArchitecture], 
                                 population_size: int) -> List[NetworkArchitecture]:
        """Generate next population using evolutionary operators"""
        
        # Sort population by overall performance
        sorted_population = sorted(
            current_population, 
            key=lambda arch: sum(arch.performance_metrics.values()) / max(len(arch.performance_metrics), 1),
            reverse=True
        )
        
        next_population = []
        
        # Keep elite individuals (top 20%)
        elite_count = max(1, int(population_size * 0.2))
        next_population.extend(sorted_population[:elite_count])
        
        # Generate offspring
        while len(next_population) < population_size:
            # Generate new architecture
            new_arch = self.architecture_generator.generate_architecture(
                self.search_space, sorted_population
            )
            
            # Mutate with some probability
            if random.random() < 0.3:  # 30% mutation rate
                new_arch = self.architecture_generator.mutate_architecture(new_arch, 0.1)
            
            next_population.append(new_arch)
        
        return next_population[:population_size]
    
    def _is_better_architecture(self, arch1: NetworkArchitecture, arch2: NetworkArchitecture) -> bool:
        """Compare two architectures to determine which is better"""
        
        scores1 = arch1.performance_metrics
        scores2 = arch2.performance_metrics
        
        if not scores1 and not scores2:
            return False
        if not scores1:
            return False
        if not scores2:
            return True
        
        # Calculate weighted sum based on objectives
        weighted_sum1 = 0.0
        weighted_sum2 = 0.0
        
        for objective in self.objectives:
            score1 = scores1.get(objective.objective_id, 0.0)
            score2 = scores2.get(objective.objective_id, 0.0)
            
            if not objective.maximize:
                score1 = 1.0 - score1
                score2 = 1.0 - score2
            
            weighted_sum1 += objective.weight * score1
            weighted_sum2 += objective.weight * score2
        
        return weighted_sum1 > weighted_sum2
    
    def _calculate_convergence_metrics(self) -> Dict[str, Any]:
        """Calculate convergence metrics"""
        
        if len(self.search_history) < 10:
            return {"message": "Insufficient data for convergence analysis"}
        
        # Calculate performance improvement over time
        performance_values = []
        for arch in self.search_history:
            overall_score = sum(arch.performance_metrics.values()) / max(len(arch.performance_metrics), 1)
            performance_values.append(overall_score)
        
        # Calculate moving average
        window_size = min(10, len(performance_values) // 4)
        moving_averages = []
        
        for i in range(window_size, len(performance_values)):
            avg = np.mean(performance_values[i-window_size:i])
            moving_averages.append(avg)
        
        # Detect convergence
        convergence_detected = False
        if len(moving_averages) > 5:
            recent_variance = np.var(moving_averages[-5:])
            if recent_variance < 0.001:  # Low variance indicates convergence
                convergence_detected = True
        
        return {
            "convergence_detected": convergence_detected,
            "final_performance": performance_values[-1] if performance_values else 0,
            "best_performance": max(performance_values) if performance_values else 0,
            "performance_variance": np.var(performance_values) if performance_values else 0,
            "improvement_rate": self._calculate_improvement_rate(performance_values)
        }
    
    def _calculate_improvement_rate(self, performance_values: List[float]) -> float:
        """Calculate rate of improvement"""
        if len(performance_values) < 2:
            return 0.0
        
        # Calculate linear regression slope
        n = len(performance_values)
        x = list(range(n))
        y = performance_values
        
        if n > 1:
            slope = (n * sum(i * v for i, v in enumerate(y)) - sum(x) * sum(y)) / (n * sum(i*i for i in x) - sum(x)**2)
            return slope
        
        return 0.0
    
    def _analyze_architecture_diversity(self) -> Dict[str, Any]:
        """Analyze diversity of explored architectures"""
        
        if not self.search_history:
            return {"message": "No architectures to analyze"}
        
        # Analyze layer count diversity
        layer_counts = [len(arch.layers) for arch in self.search_history]
        layer_count_diversity = len(set(layer_counts)) / len(layer_counts)
        
        # Analyze activation function diversity
        all_activations = []
        for arch in self.search_history:
            all_activations.extend(arch.activation_functions)
        
        unique_activations = len(set(all_activations))
        
        # Analyze optimizer diversity
        optimizers = [arch.optimizer_config.get("name", "unknown") for arch in self.search_history]
        optimizer_diversity = len(set(optimizers)) / len(optimizers)
        
        return {
            "layer_count_diversity": layer_count_diversity,
            "unique_activation_functions": unique_activations,
            "optimizer_diversity": optimizer_diversity,
            "architecture_count": len(self.search_history),
            "unique_architecture_patterns": len(set(
                tuple((layer["type"], layer["size"]) for layer in arch.layers)
                for arch in self.search_history
            ))
        }
    
    def _analyze_feature_engineering_impact(self) -> Dict[str, Any]:
        """Analyze impact of feature engineering"""
        
        if not self.feature_engineer.feature_history:
            return {"message": "No feature engineering data available"}
        
        total_transformations = sum(
            len(record["transformations_applied"]) 
            for record in self.feature_engineer.feature_history
        )
        
        avg_features_added = np.mean([
            len(record["output_features"]) - len(record["input_features"])
            for record in self.feature_engineer.feature_history
        ])
        
        return {
            "total_feature_transformations": total_transformations,
            "average_features_added": avg_features_added,
            "transformation_success_rate": len([
                record for record in self.feature_engineer.feature_history
                if record["transformations_applied"]
            ]) / len(self.feature_engineer.feature_history),
            "most_used_transformations": self._get_most_used_transformations()
        }
    
    def _analyze_debug_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of debug system"""
        
        if not self.debug_system.debug_history:
            return {"message": "No debug data available"}
        
        total_corrections = sum(
            len(session["applied_corrections"])
            for session in self.debug_system.debug_history
        )
        
        error_types = defaultdict(int)
        for session in self.debug_system.debug_history:
            for error_type in session["detected_errors"]:
                error_types[error_type] += 1
        
        return {
            "total_debug_sessions": len(self.debug_system.debug_history),
            "total_corrections_applied": total_corrections,
            "most_common_errors": dict(error_types),
            "debug_success_rate": len([
                session for session in self.debug_system.debug_history
                if session["applied_corrections"]
            ]) / len(self.debug_system.debug_history) if self.debug_system.debug_history else 0
        }
    
    def _analyze_objective_performance(self) -> Dict[str, Any]:
        """Analyze performance across different objectives"""
        
        if not self.search_history:
            return {"message": "No search history available"}
        
        objective_analysis = {}
        
        for objective in self.objectives:
            objective_id = objective.objective_id
            values = [
                arch.performance_metrics.get(objective_id, 0.0)
                for arch in self.search_history
                if objective_id in arch.performance_metrics
            ]
            
            if values:
                objective_analysis[objective_id] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": "improving" if len(values) > 1 and values[-1] > values[0] else "stable"
                }
        
        return objective_analysis
    
    def _get_most_used_transformations(self) -> Dict[str, int]:
        """Get most frequently used feature transformations"""
        
        transformation_counts = defaultdict(int)
        
        for record in self.feature_engineer.feature_history:
            for transformation in record["transformations_applied"]:
                transformation_counts[transformation] += 1
        
        return dict(transformation_counts)