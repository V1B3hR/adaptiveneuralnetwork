import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import uuid
import logging
from core.ai_ethics import audit_decision

# Setup logger for alive_node module
logger = logging.getLogger('alive_node')
logger.setLevel(logging.WARNING)  # Only show warnings and errors


@dataclass
class Memory:
    """Structured memory with importance weighting and privacy controls"""
    content: Any
    importance: float
    timestamp: int
    memory_type: str  # 'reward', 'shared', 'prediction', 'pattern'
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    decay_rate: float = 0.95
    access_count: int = 0
    source_node: Optional[int] = None  # Who shared this memory (if applicable)
    validation_count: int = 0  # How many nodes have validated this memory
    
    # Privacy controls
    private: bool = False  # Whether this memory should not be shared
    classification: str = "public"  # "public", "protected", "private", "confidential"
    retention_limit: Optional[int] = None  # Time steps before auto-deletion
    audit_log: List[str] = None  # Track access without storing content
    
    def __post_init__(self):
        if self.audit_log is None:
            self.audit_log = []

    def age(self):
        self.importance *= self.decay_rate
        if abs(self.emotional_valence) > 0.7:
            self.decay_rate = min(0.997, 0.97 + (abs(self.emotional_valence) - 0.7) * 0.1)
        
        # Check retention limit
        if self.retention_limit is not None and self.timestamp + self.retention_limit < self.timestamp:
            self.importance = 0  # Mark for deletion
    
    def access(self, accessor_id: int) -> Any:
        """Access memory content with audit logging"""
        self.access_count += 1
        self.audit_log.append(f"accessed_by_{accessor_id}_at_{self.timestamp}")
        
        # Redaction for shared memories based on classification
        if self.classification == "private" and accessor_id != self.source_node:
            return "[REDACTED]"
        elif self.classification == "protected":
            # Return summary instead of full content
            return f"[SUMMARY: {str(self.content)[:50]}...]"
        
        return self.content


class SocialSignal:
    """Structured signal for node-to-node communication"""
    def __init__(self, content: Any, signal_type: str, urgency: float, 
                 source_id: int, requires_response: bool = False):
        self.id = str(uuid.uuid4())
        self.content = content
        self.signal_type = signal_type  # 'memory', 'query', 'warning', 'resource'
        self.urgency = urgency  # 0.0 to 1.0
        self.source_id = source_id
        self.timestamp = 0
        self.requires_response = requires_response
        self.response = None


class AliveLoopNode:
    sleep_stages = ["light", "REM", "deep"]

    def __init__(self, position, velocity, initial_energy=10.0, field_strength=1.0, node_id=0):
        # Basic node attributes
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.energy = float(initial_energy)
        self.field_strength = float(field_strength)
        self.node_id = int(node_id)
        
        # Phase and time management
        self.phase = "active"  # active, sleep, interactive, inspired
        self._time = 0
        self.circadian_cycle = 0
        
        # Memory systems
        self.memory = []  # List of Memory objects
        self.long_term_memory = {}
        self.working_memory = deque(maxlen=50)
        self.predicted_energy = self.energy
        
        # Behavioral attributes
        self.anxiety = 0.0
        self.sleep_stage = "light"
        self.attention_focus = np.array([0.0, 0.0])
        self.radius = 0.5
        
        # Trust and social networks
        self.trust_network = {}  # node_id -> trust_score
        
        # Enhanced social learning attributes
        self.communication_queue = deque(maxlen=20)  # Incoming signals to process
        self.signal_history = deque(maxlen=100)  # History of sent/received signals
        self.collaborative_memories = {}  # Memories shared by other nodes
        self.social_learning_rate = 0.7  # How much to learn from others
        self.influence_network = {}  # node_id -> influence_score (how much this node influences us)
        
        # New: Social emotion contagion
        self.emotional_contagion_sensitivity = 0.5  # How susceptible to others' emotions
        self.shared_experience_buffer = deque(maxlen=10)  # Recent shared experiences
        
        # New: Collective intelligence metrics
        self.collective_contribution = 0.0  # How much this node contributes to collective
        self.knowledge_diversity = 1.0  # Diversity of knowledge sources
        
        # New: Communication styles based on personality
        self.communication_style = {
            "directness": 0.7,  # 0.0 (indirect) to 1.0 (direct)
            "formality": 0.3,   # 0.0 (casual) to 1.0 (formal)
            "expressiveness": 0.6  # 0.0 (reserved) to 1.0 (expressive)
        }
        
        # Missing attributes initialization
        self.communication_range = 2.0  # Default communication range
        self.emotional_state = {
            "valence": 0.0,  # -1 (negative) to +1 (positive)
            "arousal": 0.0   # 0 (calm) to 1 (excited)
        }
        
        # Memory and communication bounds
        self.max_memory_size = 1000  # Maximum memories to keep
        self.max_communications_per_step = 5  # Rate limiting
        self.communications_this_step = 0
        self.last_step_time = 0
        
        # Anxiety overwhelm safety protocol attributes
        self.anxiety_threshold = 8.0  # Threshold for activating help protocol
        self.calm = 1.0  # Calm level (0.0 to 5.0)
        self.help_signals_sent = 0  # Count of help signals sent this time period
        self.max_help_signals_per_period = 3  # Maximum help requests per time period
        self.help_signal_cooldown = 10  # Cooldown period between help requests
        self.last_help_signal_time = 0  # Time of last help signal
        self.anxiety_unload_capacity = 2.0  # How much anxiety can be unloaded per interaction
        self.received_help_this_period = False  # Track if received help recently
        self.anxiety_history = deque(maxlen=20)  # Track anxiety over time

    def send_signal(self, target_nodes: List['AliveLoopNode'], signal_type: str, 
                   content: Any, urgency: float = 0.5, requires_response: bool = False):
        """Send a signal to other nodes with rate limiting"""
        # Rate limiting check
        if self.communications_this_step >= self.max_communications_per_step:
            return []  # Hit rate limit
            
        if self.energy < 1.0 or self.phase == "sleep":
            return []  # Not enough energy or asleep to communicate
            
        signal = SocialSignal(content, signal_type, urgency, self.node_id, requires_response)
        signal.timestamp = self._time
        
        # Energy cost based on signal complexity and urgency
        energy_cost = 0.1 + (0.2 * urgency) + (0.1 * len(str(content)))
        self.energy = max(0, self.energy - energy_cost)
        
        responses = []
        for target in target_nodes:
            if self.communications_this_step >= self.max_communications_per_step:
                break  # Hit rate limit
                
            if self._can_communicate_with(target):
                # Adjust signal based on relationship with target
                adjusted_signal = self._adjust_signal_for_target(signal, target)
                response = target.receive_signal(adjusted_signal)
                if response:
                    responses.append(response)
                    
                # Update trust based on communication
                self._update_trust_after_communication(target, signal_type)
                self.communications_this_step += 1
        
        self.signal_history.append(signal)
        return responses

    def receive_signal(self, signal: SocialSignal) -> Optional[SocialSignal]:
        """Process an incoming signal from another node"""
        if self.energy < 0.5:
            return None  # Not enough energy to process
            
        # Energy cost to process signal
        processing_cost = 0.05 + (0.1 * signal.urgency)
        self.energy = max(0, self.energy - processing_cost)
        
        # Add to communication queue
        self.communication_queue.append(signal)
        
        # Process based on signal type
        response = None
        if signal.signal_type == "memory":
            self._process_memory_signal(signal)
        elif signal.signal_type == "query":
            response = self._process_query_signal(signal)
        elif signal.signal_type == "warning":
            self._process_warning_signal(signal)
        elif signal.signal_type == "resource":
            self._process_resource_signal(signal)
        elif signal.signal_type == "anxiety_help":
            response = self._process_anxiety_help_signal(signal)
        elif signal.signal_type == "anxiety_help_response":
            self._process_anxiety_help_response(signal)
            
        # Emotional contagion
        if hasattr(signal.content, 'emotional_valence'):
            self._apply_emotional_contagion(signal.content.emotional_valence, signal.source_id)
            
        return response

    def _process_memory_signal(self, signal: SocialSignal):
        """Process a memory shared by another node"""
        memory = signal.content
        
        # Determine trustworthiness of the source
        trust_level = self.trust_network.get(signal.source_id, 0.5)
        influence_level = self.influence_network.get(signal.source_id, 0.5)
        
        # Adjust memory importance based on trust and influence
        adjusted_importance = memory.importance * trust_level * (0.5 + 0.5 * influence_level)
        
        # Record as shared memory
        shared_memory = Memory(
            content=memory.content,
            importance=adjusted_importance,
            timestamp=self._time,
            memory_type="shared",
            emotional_valence=memory.emotional_valence,
            source_node=signal.source_id
        )
        
        self.memory.append(shared_memory)
        # Use memory timestamp and source as key instead of content
        memory_key = f"{signal.source_id}_{shared_memory.timestamp}_{shared_memory.memory_type}"
        self.collaborative_memories[memory_key] = shared_memory
        
        # Update knowledge diversity
        unique_sources = len(set(m.source_node for m in self.memory if m.source_node is not None))
        self.knowledge_diversity = min(1.0, unique_sources / 10.0)
        
        # Social learning - adjust own memories based on shared information
        self._integrate_shared_knowledge(shared_memory)

    def _process_query_signal(self, signal: SocialSignal) -> Optional[SocialSignal]:
        """Process a query from another node and return response"""
        if signal.requires_response:
            # Find relevant memories to share
            relevant_memories = []
            for memory in self.memory:
                if (self._is_memory_relevant(memory, signal.content) and 
                    memory.importance > 0.3 and
                    memory.memory_type != "shared"):  # Don't reshare shared memories
                    relevant_memories.append(memory)
            
            # Select most important memory to share
            if relevant_memories:
                relevant_memories.sort(key=lambda m: m.importance, reverse=True)
                memory_to_share = relevant_memories[0]
                
                # Create response signal
                return SocialSignal(
                    content=memory_to_share,
                    signal_type="memory",
                    urgency=0.5,
                    source_id=self.node_id
                )
        return None

    def _process_warning_signal(self, signal: SocialSignal):
        """Process a warning from another node"""
        # Increase anxiety based on urgency and trust in source
        trust_level = self.trust_network.get(signal.source_id, 0.5)
        self.anxiety += signal.urgency * trust_level * 2.0
        
        # Direct attention to potential threat
        if "danger" in str(signal.content).lower():
            self.attention_focus = np.array([1.0, 0.0])  # Focus on threat direction

    def _process_resource_signal(self, signal: SocialSignal):
        """Process a resource sharing signal"""
        # For now, just record the resource information
        resource_memory = Memory(
            content=signal.content,
            importance=0.7,
            timestamp=self._time,
            memory_type="resource",
            source_node=signal.source_id
        )
        self.memory.append(resource_memory)

    def _integrate_shared_knowledge(self, shared_memory: Memory):
        """Integrate knowledge from other nodes into own memories"""
        # Check if we have similar memories
        similar_memories = []
        for memory in self.memory:
            if (memory.content == shared_memory.content and 
                memory.memory_type != "shared"):
                similar_memories.append(memory)
                
        # Update existing memories based on shared information
        for memory in similar_memories:
            # Consensus increases importance, conflict decreases it
            if memory.emotional_valence * shared_memory.emotional_valence >= 0:
                # Generally agreeing memories
                memory.importance = min(1.0, memory.importance + 0.1 * shared_memory.importance)
                memory.validation_count += 1
            else:
                # Conflicting memories - reduce importance
                memory.importance *= 0.8
                
            # Emotional alignment
            memory.emotional_valence = (memory.emotional_valence + 
                                       shared_memory.emotional_valence * 0.2) / 1.2

    def _apply_emotional_contagion(self, emotional_valence: float, source_id: int):
        """Adjust emotions based on others' emotions"""
        trust_level = self.trust_network.get(source_id, 0.5)
        influence_level = self.influence_network.get(source_id, 0.5)
        
        # More influenced by trusted and influential nodes
        contagion_factor = self.emotional_contagion_sensitivity * trust_level * influence_level
        
        # Adjust emotional state
        self.emotional_state["valence"] = (
            (1 - contagion_factor) * self.emotional_state["valence"] + 
            contagion_factor * emotional_valence
        )
        
        # Record shared experience
        self.shared_experience_buffer.append({
            "source": source_id,
            "valence": emotional_valence,
            "timestamp": self._time
        })

    def _can_communicate_with(self, other_node: 'AliveLoopNode') -> bool:
        """Check if communication is possible with another node"""
        # Check distance
        distance = np.linalg.norm(self.position - other_node.position)
        if distance > self.communication_range:
            return False
            
        # Check if other node is awake and has energy
        if other_node.phase == "sleep" or other_node.energy < 0.5:
            return False
            
        # Check trust level (won't communicate with untrusted nodes)
        trust_level = self.trust_network.get(other_node.node_id, 0.5)
        return trust_level > 0.3

    def _adjust_signal_for_target(self, signal: SocialSignal, target: 'AliveLoopNode') -> SocialSignal:
        """Adjust signal based on relationship with target"""
        trust_level = self.trust_network.get(target.node_id, 0.5)
        
        # Adjust urgency based on trust
        adjusted_urgency = signal.urgency * trust_level
        
        # Adjust content based on communication style differences
        style_diff = abs(self.communication_style["directness"] - 
                        target.communication_style["directness"])
        
        # More direct if styles are similar
        if style_diff < 0.3:
            adjusted_urgency *= 1.2
            
        return SocialSignal(
            content=signal.content,
            signal_type=signal.signal_type,
            urgency=adjusted_urgency,
            source_id=self.node_id,
            requires_response=signal.requires_response
        )

    def _update_trust_after_communication(self, target: 'AliveLoopNode', signal_type: str):
        """Update trust based on communication outcome"""
        current_trust = self.trust_network.get(target.node_id, 0.5)
        
        # Different trust updates based on signal type
        if signal_type == "memory":
            # Trust increases when sharing valuable memories
            self.trust_network[target.node_id] = min(1.0, current_trust + 0.05)
        elif signal_type == "warning":
            # Trust updates happen after warning is validated
            pass  # Will be updated later when warning is proven true/false
        elif signal_type == "resource":
            # Trust increases when sharing resources
            self.trust_network[target.node_id] = min(1.0, current_trust + 0.1)

    def _is_memory_relevant(self, memory, query: Any) -> bool:
        """Check if a memory is relevant to a query"""
        # Handle both Memory objects and dict objects for backward compatibility
        if hasattr(memory, 'content'):
            content = memory.content
        else:
            content = memory.get('content', '')
            
        # Simple relevance check - could be enhanced with semantic similarity
        if isinstance(query, str) and isinstance(content, str):
            return query.lower() in content.lower()
        elif isinstance(query, str) and isinstance(content, dict):
            return any(query.lower() in str(v).lower() for v in content.values())
        return False

    def process_social_interactions(self):
        """Process all queued social signals"""
        processed_signals = []
        while self.communication_queue:
            signal = self.communication_queue.popleft()
            self.receive_signal(signal)
            processed_signals.append(signal)
            
        return processed_signals

    def share_valuable_memory(self, nodes: List['AliveLoopNode']) -> List[SocialSignal]:
        """Share the most valuable memory with other nodes"""
        if self.phase != "interactive" or self.energy < 2.0:
            return []
            
        # Find most valuable memory to share
        valuable_memories = [m for m in self.memory if m.importance > 0.7 and m.memory_type != "shared"]
        if not valuable_memories:
            return []
            
        valuable_memories.sort(key=lambda m: m.importance, reverse=True)
        memory_to_share = valuable_memories[0]
        
        # Ethics check before sharing memory
        ethics_result = audit_decision({
            "action": "memory_sharing",
            "preserve_life": True,  # Memory sharing doesn't harm life
            "absolute_honesty": True,  # Sharing truthful memories
            "privacy": not hasattr(memory_to_share, 'private') or not memory_to_share.private,
            "human_authority": True,  # Respecting human oversight
            "proportionality": True  # Sharing valuable information is proportional
        })
        
        if not ethics_result.get("compliant", True):
            return []  # Don't share if ethics violation
        
        # Share with nodes that would find it most valuable
        recipients = []
        for node in nodes:
            if (self._can_communicate_with(node) and 
                self._would_node_value_memory(node, memory_to_share)):
                recipients.append(node)
                
        if recipients:
            return self.send_signal(recipients, "memory", memory_to_share, urgency=0.7)
        return []

    def _would_node_value_memory(self, node: 'AliveLoopNode', memory: Memory) -> bool:
        """Check if another node would value this memory"""
        # Simple heuristic - nodes with low energy value energy-related memories
        if node.energy < 5 and "energy" in str(memory.content).lower():
            return True
            
        # Nodes with high anxiety value safety-related memories
        if node.anxiety > 10 and "safe" in str(memory.content).lower():
            return True
            
        # Generally, share with nodes we have high trust with
        return self.trust_network.get(node.node_id, 0.5) > 0.7

    def _cleanup_memory(self):
        """Clean up memory to prevent unbounded growth"""
        # Remove memories that have decayed too much or hit retention limits
        self.memory = [m for m in self.memory if m.importance > 0.1]
        
        # If still too many memories, remove oldest with lowest importance
        if len(self.memory) > self.max_memory_size:
            self.memory.sort(key=lambda m: (m.importance, -m.timestamp))
            self.memory = self.memory[-self.max_memory_size:]
        
        # Age all memories
        for memory in self.memory:
            memory.age()
    
    def _reset_rate_limits(self, current_time: int):
        """Reset communication rate limits if new time step"""
        if current_time != self.last_step_time:
            self.communications_this_step = 0
            self.last_step_time = current_time

    # [Previous methods remain the same]

    def move(self):
        # [Previous move code]
        
        # Social beings might move toward trusted nodes
        if self.phase == "interactive" and len(self.trust_network) > 0:
            # Potential extension: move toward most trusted nodes
            pass

    def step_phase(self, current_time: int):
        """Enhanced phase management with circadian rhythms"""
        self._time = current_time
        self.circadian_cycle = current_time % 24
        
        # Reset communication rate limits for new time step
        self._reset_rate_limits(current_time)
        
        # Clean up memory periodically
        if current_time % 10 == 0:  # Every 10 steps
            self._cleanup_memory()
        
        # Phase transitions based on energy, anxiety, and time
        if self.energy < 3.0 or self.circadian_cycle > 20:
            self.phase = "sleep"
            # Sleep stage based on anxiety level
            if self.anxiety > 10:
                self.sleep_stage = "deep"  # Stress-induced deep sleep
            elif self.anxiety > 5:
                self.sleep_stage = "REM"
            else:
                self.sleep_stage = "light"
        elif self.energy > 20.0 and self.anxiety < 5.0:
            self.phase = "inspired"  # High energy, low anxiety
        elif self.energy > 8.0 and 6 <= self.circadian_cycle <= 18:
            self.phase = "active"
        else:
            self.phase = "interactive"
            
        # Social phases might affect communication style
        if self.phase == "interactive":
            self.communication_style["directness"] = min(1.0, self.communication_style["directness"] + 0.1)
        elif self.phase == "sleep":
            self.communication_style["directness"] = max(0.0, self.communication_style["directness"] - 0.1)
            
        # Anxiety management and safety protocol
        self.anxiety_history.append(self.anxiety)
        
        # Apply natural calm effect
        self.apply_calm_effect()
        
        # Reset help signal limits periodically
        if current_time % 50 == 0:  # Every 50 time steps
            self.reset_help_signal_limits()
            
        # Automatic anxiety help signal for overwhelm
        # Note: This will be called by the network when it has access to nearby nodes

    def move(self):
        """Energy-efficient movement with basic navigation"""
        if self.phase in ["active", "interactive"] and self.energy > 1.0:
            # Update position based on velocity
            self.position += self.velocity
            
            # Energy cost for movement
            movement_cost = 0.1 * np.linalg.norm(self.velocity)
            self.energy = max(0, self.energy - movement_cost)

    def interact_with_capacitor(self, capacitor, threshold=0.5):
        """Enhanced interaction with capacitors"""
        distance = np.linalg.norm(self.position - capacitor.position)
        
        if distance < threshold:
            energy_difference = self.energy - capacitor.energy
            transfer_amount = 0.1 * energy_difference
            
            if transfer_amount > 0:
                # Transfer energy to capacitor
                actual_transfer = min(transfer_amount, self.energy - 1.0)  # Keep some energy
                if actual_transfer > 0:
                    capacitor.energy = min(capacitor.capacity, capacitor.energy + actual_transfer)
                    self.energy -= actual_transfer
            else:
                # Receive energy from capacitor
                actual_receive = min(-transfer_amount, capacitor.energy)
                if actual_receive > 0:
                    self.energy += actual_receive
                    capacitor.energy -= actual_receive

    def predict_energy(self):
        """Predict future energy based on memory patterns"""
        # Simple prediction based on recent memory
        energy_memories = []
        for m in self.memory:
            # Handle both Memory objects and dict objects for backward compatibility
            if hasattr(m, 'content'):
                content = m.content
            else:
                content = m.get('content', {})
            
            if isinstance(content, dict) and 'energy' in content:
                energy_memories.append(content['energy'])
            elif 'energy' in str(content):
                energy_memories.append(1.0)  # Default energy value
                
        if energy_memories:
            avg_energy_change = sum(energy_memories[-5:]) / len(energy_memories[-5:])
            self.predicted_energy = max(0, self.energy + avg_energy_change)
        else:
            self.predicted_energy = self.energy

    def clear_anxiety(self):
        """Reduce anxiety, especially during deep sleep"""
        if self.phase == "sleep" and self.sleep_stage == "deep":
            self.anxiety = max(0, self.anxiety - 2.0)
        else:
            self.anxiety = max(0, self.anxiety - 0.5)
            
    # Anxiety Overwhelm Safety Protocol Methods
    
    def check_anxiety_overwhelm(self) -> bool:
        """Check if node is experiencing anxiety overwhelm"""
        return self.anxiety >= self.anxiety_threshold
        
    def can_send_help_signal(self) -> bool:
        """Check if node can send a help signal (respects cooldown and limits)"""
        current_time = self._time
        
        # Check cooldown period
        if current_time - self.last_help_signal_time < self.help_signal_cooldown:
            return False
            
        # Check help signal limits
        if self.help_signals_sent >= self.max_help_signals_per_period:
            return False
            
        # Must have minimum energy to send help signals
        if self.energy < 2.0:
            return False
            
        return True
        
    def send_help_signal(self, nearby_nodes: List['AliveLoopNode']) -> List['AliveLoopNode']:
        """
        Send help signal to nearby trusted nodes when experiencing anxiety overwhelm.
        
        Args:
            nearby_nodes: List of nodes within communication range
            
        Returns:
            List of nodes that responded to help signal
        """
        if not self.check_anxiety_overwhelm() or not self.can_send_help_signal():
            return []
            
        # Create help signal content
        help_content = {
            "type": "anxiety_help_request",
            "anxiety_level": self.anxiety,
            "energy_level": self.energy,
            "urgency": min(1.0, self.anxiety / 10.0),
            "requesting_node": self.node_id,
            "timestamp": self._time,
            "unload_capacity_needed": min(self.anxiety - self.anxiety_threshold, self.anxiety_unload_capacity)
        }
        
        # Filter nodes by trust level and proximity
        trusted_helpers = []
        for node in nearby_nodes:
            if (node.node_id != self.node_id and 
                self.trust_network.get(node.node_id, 0.5) >= 0.4 and
                node.energy >= 3.0 and  # Helper needs sufficient energy
                node.anxiety < 6.0):    # Helper shouldn't be overwhelmed themselves
                trusted_helpers.append(node)
                
        if not trusted_helpers:
            logger.debug(f"Node {self.node_id}: No trusted helpers available for anxiety help")
            return []
            
        # Send help signal to trusted nodes
        responses = self.send_signal(
            target_nodes=trusted_helpers,
            signal_type="anxiety_help",
            content=help_content,
            urgency=help_content["urgency"],
            requires_response=True
        )
        
        # Update help signal tracking
        self.help_signals_sent += 1
        self.last_help_signal_time = self._time
        
        # Record the help request in memory
        help_memory = Memory(
            content={"action": "sent_help_request", "anxiety_level": self.anxiety},
            importance=0.8,
            timestamp=self._time,
            memory_type="help_signal",
            emotional_valence=-0.3  # Slightly negative as it indicates distress
        )
        self.memory.append(help_memory)
        
        logger.info(f"Node {self.node_id}: Sent anxiety help signal to {len(trusted_helpers)} trusted nodes")
        
        return [node for node in trusted_helpers if any(r.source_id == node.node_id for r in responses)]
        
    def _process_anxiety_help_signal(self, signal: SocialSignal) -> Optional[SocialSignal]:
        """Process an anxiety help request from another node"""
        help_content = signal.content
        
        if not isinstance(help_content, dict) or help_content.get("type") != "anxiety_help_request":
            return None
            
        requesting_node_id = help_content.get("requesting_node")
        if requesting_node_id is None:
            return None
            
        # Check if we can provide help
        trust_level = self.trust_network.get(requesting_node_id, 0.5)
        if (trust_level < 0.3 or  # Not trusted enough
            self.energy < 3.0 or  # Not enough energy
            self.anxiety > 7.0):  # Too anxious ourselves
            return None
            
        # Provide anxiety support
        anxiety_reduction = min(
            help_content.get("unload_capacity_needed", 1.0),
            self.anxiety_unload_capacity,
            self.energy - 2.0  # Keep some energy for ourselves
        )
        
        # Create help response
        help_response = {
            "type": "anxiety_help_response",
            "helper_node": self.node_id,
            "anxiety_reduction_offered": anxiety_reduction,
            "support_message": "You're not alone. This will pass.",
            "timestamp": self._time
        }
        
        # Record helping action in memory
        help_memory = Memory(
            content={"action": "provided_help", "to_node": requesting_node_id, "reduction": anxiety_reduction},
            importance=0.7,
            timestamp=self._time,
            memory_type="help_given",
            emotional_valence=0.4  # Positive for helping others
        )
        self.memory.append(help_memory)
        
        # Energy cost for providing help
        energy_cost = anxiety_reduction * 0.5
        self.energy = max(1.0, self.energy - energy_cost)
        
        # Increase trust with the helped node
        current_trust = self.trust_network.get(requesting_node_id, 0.5)
        self.trust_network[requesting_node_id] = min(1.0, current_trust + 0.1)
        
        logger.debug(f"Node {self.node_id}: Provided anxiety help to node {requesting_node_id}")
        
        return SocialSignal(
            content=help_response,
            signal_type="anxiety_help_response",
            urgency=0.7,
            source_id=self.node_id
        )
        
    def _process_anxiety_help_response(self, signal: SocialSignal):
        """Process a help response from another node"""
        help_response = signal.content
        
        if not isinstance(help_response, dict) or help_response.get("type") != "anxiety_help_response":
            return
            
        helper_node_id = help_response.get("helper_node")
        anxiety_reduction = help_response.get("anxiety_reduction_offered", 0.0)
        
        if helper_node_id is None or anxiety_reduction <= 0:
            return
            
        # Apply anxiety reduction
        old_anxiety = self.anxiety
        self.anxiety = max(0, self.anxiety - anxiety_reduction)
        
        # Increase calm level
        self.calm = min(5.0, self.calm + anxiety_reduction * 0.3)
        
        # Increase trust with helper
        current_trust = self.trust_network.get(helper_node_id, 0.5)
        self.trust_network[helper_node_id] = min(1.0, current_trust + 0.15)
        
        # Record received help in memory
        help_memory = Memory(
            content={
                "action": "received_help", 
                "from_node": helper_node_id, 
                "anxiety_before": old_anxiety,
                "anxiety_after": self.anxiety,
                "message": help_response.get("support_message", "")
            },
            importance=0.9,
            timestamp=self._time,
            memory_type="help_received",
            emotional_valence=0.6  # Positive for receiving help
        )
        self.memory.append(help_memory)
        
        self.received_help_this_period = True
        
        logger.info(f"Node {self.node_id}: Received anxiety help from node {helper_node_id}, "
                   f"anxiety reduced from {old_anxiety:.2f} to {self.anxiety:.2f}")
                   
    def apply_calm_effect(self):
        """Apply calm effect to reduce anxiety naturally"""
        if self.calm > 0.0 and self.anxiety > 0.0:
            calm_effect = min(self.calm * 0.2, self.anxiety * 0.3)
            self.anxiety = max(0, self.anxiety - calm_effect)
            # Calm depletes slightly when used
            self.calm = max(0, self.calm - calm_effect * 0.1)
            
    def reset_help_signal_limits(self):
        """Reset help signal limits (called periodically)"""
        self.help_signals_sent = 0
        self.received_help_this_period = False
        
    def get_anxiety_status(self) -> Dict[str, Any]:
        """Get comprehensive anxiety and help status"""
        return {
            "anxiety_level": self.anxiety,
            "calm_level": self.calm,
            "is_overwhelmed": self.check_anxiety_overwhelm(),
            "can_send_help": self.can_send_help_signal(),
            "help_signals_sent": self.help_signals_sent,
            "last_help_signal_time": self.last_help_signal_time,
            "received_help_recently": self.received_help_this_period,
            "anxiety_threshold": self.anxiety_threshold,
            "trust_network_size": len(self.trust_network),
            "avg_trust_level": np.mean(list(self.trust_network.values())) if self.trust_network else 0.0
        }


# Example usage in a multi-node simulation
def run_social_simulation():
    # Create nodes
    node1 = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1)
    node2 = AliveLoopNode(position=(2, 0), velocity=(0, 0), initial_energy=12.0, node_id=2)
    node3 = AliveLoopNode(position=(0, 2), velocity=(0, 0), initial_energy=10.0, node_id=3)
    
    nodes = [node1, node2, node3]
    
    # Initialize trust
    for node in nodes:
        for other in nodes:
            if other.node_id != node.node_id:
                node.trust_network[other.node_id] = 0.5
                node.influence_network[other.node_id] = 0.5
    
    # Run simulation
    for t in range(50):
        logger.info(f"Time step {t}")
        
        for node in nodes:
            node.step_phase(t)
            
            # Process social interactions
            node.process_social_interactions()
            
            # Occasionally share valuable memories
            if t % 5 == 0:
                other_nodes = [n for n in nodes if n.node_id != node.node_id]
                node.share_valuable_memory(other_nodes)
            
            # Move and process signals
            node.move()
            
            # Display social status
            logger.debug(f"Node {node.node_id}: Trust network: {node.trust_network}")

if __name__ == "__main__":
    run_social_simulation()
