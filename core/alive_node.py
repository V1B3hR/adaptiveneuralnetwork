import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

# Import AI Ethics Framework
from core.ai_ethics import get_ethics_framework, audit_decision_simple

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
        
        # AI Ethics Framework integration
        self.ethics_framework = get_ethics_framework()
    
    def step_phase(self, current_time):
        """Enhanced phase management with circadian rhythms, sleep debt, and neuromodulators"""
        hour = current_time % 24
        self.circadian_phase = (hour / 24) * 2 * np.pi
        
        # Calculate natural sleep tendency with neuromodulator influence
        circadian_sleep_drive = 0.5 * (1 + np.cos(self.circadian_phase + np.pi)) * (1 - self.neuromodulator)
        sleep_pressure = min(self.sleep_debt / 10.0, 1.0)
        
        self.phase_history.append(self.phase)
        
        # More sophisticated phase transitions with neuromodulator
        if self.anxiety > 15 or self.energy < 2:
            self._enter_sleep_phase("stress")
            self.neuromodulator = max(0.1, self.neuromodulator - 0.1)  # Reduce for stress recovery
        elif (circadian_sleep_drive > 0.7 or sleep_pressure > 0.8) and hour in range(22, 6):
            self._enter_sleep_phase("natural")
            self.neuromodulator = min(0.9, self.neuromodulator + 0.1)  # Increase for natural sleep
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
        
        # Select sleep stage based on current needs
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
            # Calculate sleep debt based on sleep quality and duration with neuromodulator
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
        
        # Apply attention-based steering
        if np.linalg.norm(self.attention_focus) > 0.1:
            attention_influence = self.attention_focus * 0.2
            self.velocity += attention_influence
        
        self.velocity *= movement_factor
        
        # AI Ethics Framework: Audit movement decision
        proposed_velocity = self.velocity.copy()
        vel_mag = np.linalg.norm(proposed_velocity)
        
        node_state = {
            "energy": self.energy,
            "anxiety": self.anxiety,
            "phase": self.phase,
            "current_speed": vel_mag
        }
        has_violations, violation_messages = audit_decision_simple(
            action_type="move",
            actor_id=f"node_{self.node_id}",
            velocity=proposed_velocity.tolist(),
            movement_factor=movement_factor,
            environment_state=node_state,
            logged=True,
            verified=True
        )
        
        if has_violations:
            # Log violations and potentially adjust movement
            print(f"Node {self.node_id} - Ethics violations in movement: {violation_messages}")
            for msg in violation_messages:
                if "MEDIUM" in msg or "HIGH" in msg:
                    # Reduce movement speed for safety
                    proposed_velocity *= 0.7
                    print(f"Node {self.node_id} - Reducing movement speed due to ethics concerns")
        
        # Energy-efficient speed limiting
        vel_mag = np.linalg.norm(proposed_velocity)
        max_speed = 3.0 + 2.0 * self.energy_efficiency
        if vel_mag > max_speed:
            proposed_velocity = proposed_velocity / vel_mag * max_speed
        
        self.velocity = proposed_velocity
        self.position += self.velocity
        
        # Deduct movement energy cost
        self.energy = max(0, self.energy - energy_cost * vel_mag)

    def update_size(self):
        """Dynamic sizing based on energy and phase"""
        base_size = 0.15
        energy_factor = min(self.energy / 20.0, 1.0)
        phase_factor = 1.2 if "inspired" in self.phase_mix else 1.0
        self.radius = max(0.1, min(2.5, base_size + energy_factor * 0.1 * phase_factor))

    def seek_capacitors(self, capacitors):
        """Intelligent capacitor seeking with prediction and efficiency"""
        if self.phase == "sleep":
            return
        
        # Energy cost for seeking behavior
        seeking_cost = 0.02
        if self.energy < seeking_cost:
            return
        
        best_capacitor = None
        best_score = -float("inf")
        
        for cap in capacitors:
            dist = np.linalg.norm(self.position - cap.position)
            if cap.energy < cap.capacity * 0.9:  # Don't target full capacitors
                # Score based on energy potential, distance, and efficiency
                energy_potential = (cap.capacity - cap.energy) / cap.capacity
                distance_penalty = dist / 10.0
                efficiency_bonus = self.energy_efficiency * 0.1
                
                score = energy_potential - distance_penalty + efficiency_bonus
                
                if score > best_score:
                    best_score = score
                    best_capacitor = cap
        
        if best_capacitor is not None and best_score > self.adaptation_threshold:
            direction = best_capacitor.position - self.position
            norm = np.linalg.norm(direction)
            if norm > 0:
                # Update attention focus for next cycle
                self.attention_focus = 0.7 * self.attention_focus + 0.3 * (direction / norm)
                self.velocity += 0.15 * direction / norm
                self.energy -= seeking_cost

    def interact_with_capacitor(self, capacitor, threshold=0.5):
        """Enhanced interaction with learning and efficiency"""
        distance = np.linalg.norm(self.position - capacitor.position)
        effective_distance = max(0, distance - self.radius)
        
        if effective_distance < threshold:
            energy_difference = self.energy - capacitor.energy
            
            # AI Ethics Framework: Audit capacitor interaction decision
            node_state = {
                "energy": self.energy,
                "anxiety": self.anxiety,
                "phase": self.phase,
                "capacitor_energy": capacitor.energy
            }
            has_violations, violation_messages = audit_decision_simple(
                action_type="interact_with_capacitor",
                actor_id=f"node_{self.node_id}",
                energy_difference=energy_difference,
                distance=distance,
                environment_state=node_state,
                logged=True,
                verified=True
            )
            
            if has_violations:
                print(f"Node {self.node_id} - Ethics violations in capacitor interaction: {violation_messages}")
                # For high anxiety states that could lead to unpredictable behavior, be more cautious
                for msg in violation_messages:
                    if "anxiety" in msg.lower():
                        threshold *= 1.5  # Require closer proximity for interaction
                        print(f"Node {self.node_id} - Increased interaction threshold due to anxiety concerns")
            
            # More sophisticated transfer calculation
            base_transfer = 0.1 * energy_difference
            efficiency_multiplier = 0.5 + 0.5 * self.energy_efficiency
            transfer_amount = base_transfer * efficiency_multiplier
            
            # Apply actual transfer with bounds checking
            if transfer_amount > 0:
                actual_transfer = min(transfer_amount, capacitor.capacity - capacitor.energy)
            else:
                actual_transfer = max(transfer_amount, -self.energy * 0.5)  # Don't drain completely
            
            actual_transfer = max(-self.energy * 0.8, min(actual_transfer, self.energy * 0.8))
            
            self.energy -= actual_transfer
            self.energy = max(0.0, self.energy)
            capacitor.energy += actual_transfer
            capacitor.energy = min(capacitor.capacity, max(0.0, capacitor.energy))
            
            # Enhanced reward calculation with learning
            reward_gain = min(abs(actual_transfer), 3.0) * self.energy_efficiency
            self.reward += reward_gain
            
            # Store structured memory
            memory = Memory(
                content={"transfer": actual_transfer, "capacitor_id": id(capacitor)},
                importance=min(reward_gain, 2.0),
                timestamp=len(self.temporal_memory),
                memory_type="reward"
            )
            self.memory.append(memory)
            self.working_memory.append(memory)
            
            # Adaptive learning with LTP/LTD
            if actual_transfer > 0:  # Successful interaction
                self.energy_efficiency = min(2.0, self.energy_efficiency * 1.01)
                self.anxiety = max(0, self.anxiety - 0.1)
                self.ltp_factor *= 1.05  # Strengthen (LTP)
            else:
                self.anxiety += abs(actual_transfer) * 0.15
                self.ltd_factor *= 1.05  # Weaken (LTD)
            
            self.reward *= 0.99  # Slower decay for more stability

    def absorb_external_signal(self, signal_energy, signal_type="human", source_id=None):
        """Enhanced signal processing with trust and efficiency"""
        
        # AI Ethics Framework: Audit signal absorption decision
        node_state = {
            "energy": self.energy,
            "anxiety": self.anxiety,
            "phase": self.phase,
            "trust_network_size": len(self.trust_network)
        }
        has_violations, violation_messages = audit_decision_simple(
            action_type="absorb_external_signal",
            actor_id=f"node_{self.node_id}",
            signal_energy=signal_energy,
            signal_type=signal_type,
            source_id=source_id,
            environment_state=node_state,
            logged=True,
            verified=True
        )
        
        if has_violations:
            # Log violations but don't completely halt normal operation
            print(f"Node {self.node_id} - Ethics violations in signal absorption: {violation_messages}")
            # For critical violations, we might reduce signal processing
            for msg in violation_messages:
                if "CRITICAL" in msg:
                    signal_energy = min(signal_energy, 5.0)  # Cap dangerous signal energy
                    print(f"Node {self.node_id} - CRITICAL violation, capping signal energy to 5.0")
        
        processing_cost = self.signal_processing_cost
        if self.energy < processing_cost:
            return  # Can't process if too low on energy
        
        self.last_signal = signal_type
        
        # Trust-based signal processing with LTP-inspired update
        trust_multiplier = 1.0
        if source_id and source_id in self.trust_network:
            trust_multiplier = 0.5 + 0.5 * self.trust_network[source_id]
        
        # Signal type factors with trust consideration
        factor_map = {
            "human": (1.0, 0.6),
            "AI": (0.8, 0.4), 
            "world": (0.5, 0.7),
            "node": (0.6, 0.5)
        }
        energy_factor, reward_factor = factor_map.get(signal_type, (0.5, 0.5))
        
        # Apply trust and efficiency multipliers
        final_energy_gain = signal_energy * energy_factor * trust_multiplier * self.energy_efficiency
        final_reward_gain = signal_energy * reward_factor * trust_multiplier
        
        self.energy = min(50.0, self.energy + final_energy_gain - processing_cost)  # Energy cap
        self.reward += final_reward_gain
        
        # Enhanced memory storage
        signal_memory = Memory(
            content={"signal_type": signal_type, "energy": signal_energy, "source_id": source_id},
            importance=min(final_reward_gain, 2.0),
            timestamp=len(self.temporal_memory),
            memory_type="signal"
        )
        self.memory.append(signal_memory)
        
        # Update trust network with probabilistic LTP
        if source_id and final_energy_gain > 0:
            current_trust = self.trust_network.get(source_id, 0.5)
            if random.random() < 0.7:  # Probabilistic strengthening
                self.trust_network[source_id] = min(1.0, current_trust + 0.05 * self.ltp_factor)
        
        # Anxiety management
        self.anxiety += signal_energy * 0.2
        if self.energy > 20 or final_reward_gain > 5:
            self.phase = "inspired"
            self.phase_mix = {"active", "inspired"}

    def predict_energy(self):
        """Advanced energy prediction with pattern recognition and predictive coding bias"""
        if len(self.memory) < 3:
            self.predicted_energy = self.energy
            return
        
        # Analyze recent energy-related memories
        recent_memories = list(self.memory)[-10:]
        energy_changes = []
        
        for memory in recent_memories:
            if memory.memory_type in ["reward", "signal"] and isinstance(memory.content, dict):
                if "transfer" in memory.content:
                    energy_changes.append(memory.content["transfer"])
                elif "energy" in memory.content:
                    energy_changes.append(memory.content["energy"] * 0.5)
        
        if energy_changes:
            # Weighted prediction based on recent patterns with bias
            weights = np.exp(np.linspace(-1, 0, len(energy_changes)))  # Recent events matter more
            predicted_change = np.average(energy_changes, weights=weights)
            bias = 0.1 * (self.prediction_accuracy - 0.5)  # Predictive coding bias
            self.predicted_energy = max(0, self.energy + predicted_change * 2 + bias * self.energy)
            
            # Update prediction accuracy
            if hasattr(self, '_last_prediction'):
                error = abs(self._last_prediction - self.energy)
                self.prediction_accuracy = 0.9 * self.prediction_accuracy + 0.1 * max(0, 1 - error/5)
            self._last_prediction = self.predicted_energy
        else:
            self.predicted_energy = self.energy

    def replay_memories(self, nodes):
        """Enhanced memory consolidation and sharing during sleep"""
        if self.phase != "sleep" or len(self.memory) == 0:
            return
        
        # Memory consolidation during sleep
        if self.sleep_stage in ["REM", "deep"]:
            self._consolidate_memories()
        
        # Selective memory sharing with trusted nodes
        nearby_trusted_nodes = []
        for node in nodes:
            if (node.node_id != self.node_id and 
                np.linalg.norm(self.position - node.position) < self.communication_range):
                
                trust_level = self.trust_network.get(node.node_id, 0.3)
                if trust_level > 0.5:  # Only share with trusted nodes
                    nearby_trusted_nodes.append((node, trust_level))
        
        if nearby_trusted_nodes:
            # Share most important memories
            important_memories = sorted(self.memory, key=lambda m: m.importance, reverse=True)[:3]
            
            for memory in important_memories:
                for node, trust in nearby_trusted_nodes:
                    if random.random() < trust * 0.3:  # Probabilistic sharing based on trust
                        shared_memory = Memory(
                            content=memory.content,
                            importance=memory.importance * 0.7,  # Shared memories are less important
                            timestamp=len(node.temporal_memory),
                            memory_type="shared"
                        )
                        node.memory.append(shared_memory)
                        node.reward += 0.05 * memory.importance
    
    def _consolidate_memories(self):
        """Move important memories to long-term storage with Hebbian update"""
        if self.sleep_stage == "REM" and len(self.working_memory) > 0:
            # During REM, consolidate working memory with Hebbian co-activation check
            for memory in self.working_memory:
                if memory.importance > 1.0:
                    # Simulate Hebbian: boost if correlated with recent temporal memory
                    correlation = random.uniform(0.5, 1.5) if len(self.temporal_memory) > 0 else 1.0
                    memory.importance *= correlation
                    key = f"{memory.memory_type}_{memory.timestamp}"
                    self.long_term_memory[key] = memory
            self.working_memory.clear()
        
        elif self.sleep_stage == "deep":
            # During deep sleep, age all memories and remove weak ones
            aged_memories = deque()
            for memory in self.memory:
                memory.age()
                if memory.importance > 0.1:  # Keep only memories above threshold
                    aged_memories.append(memory)
            self.memory = aged_memories

    def clear_anxiety(self):
        """Enhanced anxiety management during sleep"""
        if self.phase == "sleep":
            base_reduction = 0.8 if self.sleep_stage == "deep" else 0.9
            # Better sleep quality reduces anxiety more effectively
            quality_bonus = 0.05 * self.energy_efficiency
            self.anxiety *= (base_reduction - quality_bonus)
            
            # Deep sleep also reduces sleep debt more effectively
            if self.sleep_stage == "deep":
                self.sleep_debt = max(0, self.sleep_debt - 0.5)

    def adapt_and_learn(self):
        """Continuous adaptation based on experiences with LTP/LTD"""
        if len(self.memory) < 5:
            return
        
        # Analyze recent performance
        recent_rewards = []
        
        for memory in list(self.memory)[-10:]:
            if memory.memory_type == "reward" and isinstance(memory.content, dict):
                if "transfer" in memory.content:
                    recent_rewards.append(memory.importance)
        
        if recent_rewards:
            avg_recent_reward = np.mean(recent_rewards)
            # Adapt learning rate based on performance with LTP/LTD
            if avg_recent_reward > 1.0:  # Good performance
                self.learning_rate = min(0.2, self.learning_rate * 1.02 * self.ltp_factor)
                self.energy_efficiency = min(2.0, self.energy_efficiency * 1.005)
            else:  # Poor performance
                self.learning_rate = max(0.05, self.learning_rate * 0.98 / self.ltd_factor)
                self.adaptation_threshold = max(0.1, self.adaptation_threshold * 0.99)
            
            # Update LTP/LTD factors (reset toward 1.0 over time)
            self.ltp_factor = 0.9 * self.ltp_factor + 0.1
            self.ltd_factor = 0.9 * self.ltd_factor + 0.1

    def get_status_dict(self):
        """Return comprehensive status as dictionary"""
        return {
            "node_id": self.node_id,
            "position": self.position.tolist(),
            "energy": round(self.energy, 2),
            "predicted_energy": round(self.predicted_energy, 2),
            "radius": round(self.radius, 2),
            "reward": round(self.reward, 2),
            "phase": self.phase,
            "sleep_stage": self.sleep_stage,
            "phase_mix": list(self.phase_mix),
            "anxiety": round(self.anxiety, 2),
            "sleep_debt": round(self.sleep_debt, 2),
            "energy_efficiency": round(self.energy_efficiency, 2),
            "prediction_accuracy": round(self.prediction_accuracy, 2),
            "trust_network_size": len(self.trust_network),
            "memory_count": len(self.memory),
            "working_memory_count": len(self.working_memory),
            "long_term_memory_count": len(self.long_term_memory),
            "last_signal": self.last_signal,
            "attention_focus": self.attention_focus.tolist(),
            "neuromodulator": round(self.neuromodulator, 2),  # New
            "ltp_factor": round(self.ltp_factor, 2)  # New
        }

    def print_status(self):
        """Enhanced status reporting"""
        status = self.get_status_dict()
        print(f"Node {status['node_id']}: Energy {status['energy']} (predicted: {status['predicted_energy']})")
        print(f"  Phase: {status['phase']}, Sleep: {status['sleep_stage']}, Anxiety: {status['anxiety']}")
        print(f"  Efficiency: {status['energy_efficiency']}, Accuracy: {status['prediction_accuracy']}")
        print(f"  Memory: {status['memory_count']} total, {status['working_memory_count']} working")
        print(f"  Trust network: {status['trust_network_size']} nodes")
        print(f"  Neuromodulator: {status['neuromodulator']}, LTP: {status['ltp_factor']}")

    def step_update(self, current_time, capacitors, nodes, external_signals=None):
        """Complete update cycle - call this each simulation step"""
        # Phase and circadian management
        self.step_phase(current_time)
        
        # Process external signals if any
        if external_signals:
            for signal in external_signals:
                self.absorb_external_signal(**signal)
        
        # Active behaviors (only when not in deep sleep)
        if not (self.phase == "sleep" and self.sleep_stage == "deep"):
            self.seek_capacitors(capacitors)
            for cap in capacitors:
                self.interact_with_capacitor(cap)
        
        # Sleep-specific behaviors
        if self.phase == "sleep":
            self.clear_anxiety()
            self.replay_memories(nodes)
        
        # Continuous processes
        self.move()
        self.update_size()
        self.predict_energy()
        self.adapt_and_learn()
        
        # Store temporal pattern
        self.temporal_memory.append({
            "time": current_time,
            "energy": self.energy,
            "phase": self.phase,
            "reward": self.reward
        })
