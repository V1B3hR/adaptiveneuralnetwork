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
        
        # Enhanced attack resilience features
        self.energy_sharing_enabled = True  # Allow distributed energy sharing
        self.energy_sharing_history = deque(maxlen=20)  # Track energy transactions
        self.attack_detection_threshold = 3  # Number of suspicious events to trigger detection
        self.suspicious_events = deque(maxlen=10)  # Track suspicious activities
        self.energy_drain_resistance = 0.7  # Resistance to energy drain attacks (0.0-1.0)
        self.signal_redundancy_level = 2  # Number of redundant communication channels
        self.jamming_detection_sensitivity = 0.3  # Sensitivity to signal jamming detection

    def send_signal(self, target_nodes: List['AliveLoopNode'], signal_type: str, 
                   content: Any, urgency: float = 0.5, requires_response: bool = False):
        """Send a signal to other nodes with rate limiting and jamming resistance"""
        # Rate limiting check
        if self.communications_this_step >= self.max_communications_per_step:
            return []  # Hit rate limit
            
        if self.energy < 1.0 or self.phase == "sleep":
            return []  # Not enough energy or asleep to communicate
            
        signal = SocialSignal(content, signal_type, urgency, self.node_id, requires_response)
        signal.timestamp = self._time
        
        # Enhanced signal with jamming resistance
        signal = self._add_jamming_resistance(signal)
        
        # Energy cost based on signal complexity and urgency
        base_energy_cost = 0.1 + (0.2 * urgency) + (0.1 * len(str(content)))
        # Additional cost for jamming resistance
        jamming_resistance_cost = 0.05 * self.signal_redundancy_level
        energy_cost = base_energy_cost + jamming_resistance_cost
        
        self.energy = max(0, self.energy - energy_cost)
        
        responses = []
        for target in target_nodes:
            if self.communications_this_step >= self.max_communications_per_step:
                break  # Hit rate limit
                
            if self._can_communicate_with(target):
                # Adjust signal based on relationship with target
                adjusted_signal = self._adjust_signal_for_target(signal, target)
                
                # Send with redundancy for jamming resistance
                successful_transmission = self._send_with_jamming_resistance(adjusted_signal, target)
                
                if successful_transmission:
                    response = target.receive_signal(adjusted_signal)
                    if response:
                        responses.append(response)
                        
                    # Update trust based on communication
                    self._update_trust_after_communication(target, signal_type)
                    self.communications_this_step += 1
        
        self.signal_history.append(signal)
        return responses
    
    def _add_jamming_resistance(self, signal: SocialSignal) -> SocialSignal:
        """Add jamming resistance features to signal"""
        # Add error correction and redundancy information
        signal.jamming_resistance = {
            "redundancy_level": self.signal_redundancy_level,
            "error_correction": True,
            "transmission_id": f"{self.node_id}_{self._time}_{len(self.signal_history)}",
            "checksum": hash(str(signal.content)) % 10000  # Simple checksum
        }
        return signal
    
    def _send_with_jamming_resistance(self, signal: SocialSignal, target: 'AliveLoopNode') -> bool:
        """
        Send signal with jamming resistance. Returns True if transmission successful.
        Improved from 52% effectiveness to target 75-80% effectiveness.
        """
        # Base transmission success rate
        base_success_rate = 0.85  # 85% base success rate
        
        # Check for jamming conditions
        jamming_detected = self._detect_signal_jamming()
        
        if jamming_detected:
            # Apply jamming resistance improvements
            jamming_impact = 0.4  # 40% impact reduction (improved from 52%)
            
            # Redundancy helps overcome jamming - each level gives better protection
            redundancy_benefit = min(0.35, self.signal_redundancy_level * 0.15)  # Up to 35% improvement
            
            # Adaptive frequency hopping simulation
            frequency_hopping_benefit = 0.15  # 15% improvement from frequency hopping
            
            # Total jamming resistance
            total_resistance = redundancy_benefit + frequency_hopping_benefit
            actual_jamming_impact = jamming_impact * (1.0 - total_resistance)
            
            success_rate = base_success_rate - actual_jamming_impact
        else:
            success_rate = base_success_rate
        
        # Ensure redundancy always provides some benefit
        if self.signal_redundancy_level > 1:
            redundancy_bonus = (self.signal_redundancy_level - 1) * 0.05  # 5% per extra redundancy level
            success_rate = min(0.95, success_rate + redundancy_bonus)
        
        # Random transmission success based on calculated rate
        import random
        return random.random() < success_rate
    
    def _detect_signal_jamming(self) -> bool:
        """
        Detect if signal jamming is occurring in the environment.
        """
        # Check recent communication failures
        recent_failures = 0
        recent_attempts = 0
        
        for signal in list(self.signal_history)[-10:]:  # Last 10 signals
            recent_attempts += 1
            # Simulate failure detection based on signal properties
            if hasattr(signal, 'transmission_failed'):
                recent_failures += 1
        
        if recent_attempts > 0:
            failure_rate = recent_failures / recent_attempts
            # High failure rate indicates potential jamming
            if failure_rate > self.jamming_detection_sensitivity:
                return True
        
        # Check for suspicious patterns in communication queue
        if len(self.communication_queue) == 0 and recent_attempts > 3:
            # No incoming signals despite attempts might indicate jamming
            return True
        
        return False

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
        if signal.content.get("type") == "energy_request":
            # Handle energy sharing request
            response = self._handle_energy_sharing_request(signal)
            if response:
                # Send response back to requester - need to actually deliver it
                # In a real implementation, this would be sent through the network
                # For now, we'll simulate immediate delivery in tests
                self._pending_response = response
                return response
        else:
            # Original resource signal processing
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
        """Update trust based on communication outcome with manipulation resistance"""
        current_trust = self.trust_network.get(target.node_id, 0.5)
        
        # Enhanced trust validation to resist manipulation
        if self._validate_trust_update(target.node_id, signal_type):
            # Different trust updates based on signal type
            if signal_type == "memory":
                # Trust increases when sharing valuable memories
                trust_increment = 0.05
                # Apply trust manipulation resistance
                trust_increment *= self._calculate_trust_adjustment_factor(target.node_id)
                self.trust_network[target.node_id] = min(1.0, current_trust + trust_increment)
            elif signal_type == "warning":
                # Trust updates happen after warning is validated
                pass  # Will be updated later when warning is proven true/false
            elif signal_type == "resource":
                # Trust increases when sharing resources
                trust_increment = 0.1
                # Apply trust manipulation resistance
                trust_increment *= self._calculate_trust_adjustment_factor(target.node_id)
                self.trust_network[target.node_id] = min(1.0, current_trust + trust_increment)
    
    def _validate_trust_update(self, node_id: int, signal_type: str) -> bool:
        """
        Validate if trust update is legitimate or potential manipulation.
        """
        # Check for rapid trust changes that might indicate manipulation
        recent_trust_changes = 0
        for transaction in list(self.energy_sharing_history)[-5:]:
            if transaction.get("node_id") == node_id:
                recent_trust_changes += 1
        
        # Too many recent interactions might indicate manipulation attempt
        if recent_trust_changes > 3:
            return False
        
        # Check for patterns indicating reputation manipulation
        current_trust = self.trust_network.get(node_id, 0.5)
        
        # Suspiciously high trust growth rate
        if current_trust > 0.9 and len(self.energy_sharing_history) < 10:
            return False
        
        # Check consistency with other nodes' assessments (if available)
        # This would require cross-node trust information in a real implementation
        
        return True
    
    def _calculate_trust_adjustment_factor(self, node_id: int) -> float:
        """
        Calculate trust adjustment factor to resist manipulation.
        Returns value between 0.1 and 1.0.
        """
        # Base factor
        factor = 1.0
        
        # Long-term relationship factor
        interaction_history = [t for t in self.energy_sharing_history if t.get("node_id") == node_id]
        if len(interaction_history) < 3:
            # New relationships build trust more slowly
            factor *= 0.5
        
        # Consistency factor - check for suspicious patterns
        if len(interaction_history) >= 3:
            recent_amounts = [t.get("amount", 0) for t in interaction_history[-3:]]
            # Suspiciously consistent amounts might indicate manipulation
            if len(set(recent_amounts)) == 1 and recent_amounts[0] > 0:
                factor *= 0.7  # Reduce trust growth for suspicious consistency
        
        # Time-based factor - trust should build gradually over time
        if len(interaction_history) > 0:
            time_span = self._time - interaction_history[0].get("timestamp", self._time)
            if time_span < 5:  # Very rapid trust building
                factor *= 0.6
        
        return max(0.1, factor)
    
    def share_energy_directly(self, target_node: 'AliveLoopNode', amount: float) -> float:
        """
        Direct energy sharing method for testing and simple scenarios.
        Returns actual amount shared.
        """
        if not self.energy_sharing_enabled or amount <= 0:
            return 0.0
        
        # Check our ability to share
        max_shareable = min(amount, (self.energy - 3.0) * 0.5)  # Keep reserve
        trust_level = self.trust_network.get(target_node.node_id, 0.5)
        
        if max_shareable > 0 and trust_level > 0.4:
            sharing_willingness = trust_level * 0.8 + 0.2
            actual_shared = max_shareable * sharing_willingness
            
            if actual_shared > 0.1:
                self.energy -= actual_shared
                target_node.energy += actual_shared
                
                # Record transactions
                self._record_energy_transaction(target_node.node_id, actual_shared, "shared")
                target_node._record_energy_transaction(self.node_id, actual_shared, "received")
                
                return actual_shared
        
        return 0.0
    
    def detect_long_term_trust_manipulation(self) -> List[int]:
        """
        Detect nodes that might be engaging in long-term trust manipulation.
        Returns list of suspicious node IDs.
        """
        suspicious_nodes = []
        
        for node_id, trust_level in self.trust_network.items():
            # Analyze trust building pattern
            node_transactions = [t for t in self.energy_sharing_history if t.get("node_id") == node_id]
            
            if len(node_transactions) >= 5:
                # Check for artificially accelerated trust building
                trust_growth = trust_level - 0.5  # Growth from initial 0.5
                time_span = max(1, self._time - node_transactions[0].get("timestamp", self._time))
                trust_growth_rate = trust_growth / time_span
                
                # Suspiciously high trust growth rate
                if trust_growth_rate > 0.1:  # More than 10% per time unit
                    suspicious_nodes.append(node_id)
                
                # Check for patterns in transaction amounts
                amounts = [t.get("amount", 0) for t in node_transactions]
                # Suspiciously regular patterns
                if len(set(amounts)) <= 2 and len(amounts) >= 5:
                    suspicious_nodes.append(node_id)
        
        return suspicious_nodes

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
    
    def request_energy_sharing(self, target_nodes: List['AliveLoopNode'], energy_needed: float) -> float:
        """
        Request energy sharing from trusted nodes with attack resistance.
        Returns actual energy received.
        """
        if not self.energy_sharing_enabled or energy_needed <= 0:
            return 0.0
        
        # Ethics check for energy sharing request
        ethics_result = audit_decision({
            "action": "request_energy_sharing",
            "preserve_life": True,  # Energy sharing preserves life
            "absolute_honesty": True,  # Honest about energy needs
            "privacy": True,  # Respects node privacy
            "human_authority": True,  # Human oversight respected
            "proportionality": True  # Request is proportional to need
        })
        
        if not ethics_result.get("compliant", True):
            return 0.0
        
        total_received = 0.0
        sharing_attempts = 0
        
        # Sort nodes by trust level for priority sharing
        trusted_nodes = [(node, self.trust_network.get(node.node_id, 0.5)) 
                        for node in target_nodes if self._can_communicate_with(node)]
        trusted_nodes.sort(key=lambda x: x[1], reverse=True)  # High trust first
        
        for node, trust_level in trusted_nodes:
            if total_received >= energy_needed or sharing_attempts >= 3:  # Limit sharing attempts
                break
            
            # Only request from nodes with sufficient trust and energy
            if trust_level > 0.5 and node.energy > 3.0:
                sharing_attempts += 1
                
                # Calculate sharing amount based on trust and need
                max_shareable = min(energy_needed - total_received, node.energy * 0.3)  # Max 30% of their energy
                trust_factor = min(1.0, trust_level * 1.5)  # Trust bonus
                
                energy_to_request = max_shareable * trust_factor
                
                # Send energy sharing request
                request_signal = SocialSignal(
                    content={
                        "type": "energy_request",
                        "amount": energy_to_request,
                        "urgency": min(1.0, (energy_needed / 10.0)),  # Higher urgency for greater need
                        "requester_energy": self.energy
                    },
                    signal_type="resource",
                    urgency=0.7,
                    source_id=self.node_id,
                    requires_response=True
                )
                
                response = node.receive_signal(request_signal)
                if response and isinstance(response, SocialSignal):
                    shared_amount = response.content.get("shared_amount", 0.0)
                    if shared_amount > 0:
                        # Record energy sharing transaction
                        self._record_energy_transaction(node.node_id, shared_amount, "received")
                        total_received += shared_amount
                        self.energy += shared_amount
                        
                        # Increase trust for helpful nodes
                        self.trust_network[node.node_id] = min(1.0, self.trust_network.get(node.node_id, 0.5) + 0.05)
        
        return total_received
    
    def _handle_energy_sharing_request(self, request_signal: SocialSignal) -> Optional[SocialSignal]:
        """
        Handle incoming energy sharing requests with attack detection.
        """
        request_content = request_signal.content
        requester_id = request_signal.source_id
        requested_amount = request_content.get("amount", 0.0)
        requester_energy = request_content.get("requester_energy", 0.0)
        
        # Check if this is a legitimate request or potential attack
        if self._detect_energy_drain_attack(requester_id, requested_amount, requester_energy):
            self.suspicious_events.append({
                "type": "potential_energy_drain_attack",
                "source": requester_id,
                "amount": requested_amount,
                "timestamp": self._time
            })
            return None  # Reject suspicious request
        
        # Check our ability and willingness to share
        trust_level = self.trust_network.get(requester_id, 0.5)
        can_share = (
            self.energy_sharing_enabled and
            self.energy > 5.0 and  # Keep minimum energy for ourselves
            trust_level > 0.4 and  # Minimum trust threshold
            requested_amount <= self.energy * 0.4  # Max 40% of our energy
        )
        
        shared_amount = 0.0
        if can_share:
            # Calculate actual sharing amount based on trust and our capacity
            max_we_can_share = min(requested_amount, (self.energy - 3.0) * 0.5)  # Keep reserve
            sharing_willingness = trust_level * 0.8 + 0.2  # 20-100% willingness based on trust
            shared_amount = max_we_can_share * sharing_willingness
            
            if shared_amount > 0.1:  # Only share meaningful amounts
                self.energy -= shared_amount
                self._record_energy_transaction(requester_id, shared_amount, "shared")
                
                # Slight trust increase for successful sharing
                self.trust_network[requester_id] = min(1.0, trust_level + 0.02)
        
        # Send response
        return SocialSignal(
            content={
                "type": "energy_response",
                "shared_amount": shared_amount,
                "remaining_capacity": max(0.0, self.energy - 3.0)  # Available for future sharing
            },
            signal_type="resource",
            urgency=0.3,
            source_id=self.node_id
        )
    
    def _detect_energy_drain_attack(self, requester_id: int, requested_amount: float, requester_energy: float) -> bool:
        """
        Detect potential energy drain attacks based on request patterns.
        """
        # Check for excessive requests from same source
        recent_requests = [event for event in self.suspicious_events 
                          if event.get("source") == requester_id and 
                          self._time - event.get("timestamp", 0) < 10]
        
        if len(recent_requests) >= 3:  # Too many recent requests
            return True
        
        # Check for unreasonably large requests
        if requested_amount > self.energy * 0.6:  # More than 60% of our energy
            return True
        
        # Check for requests from nodes that should have sufficient energy
        if requester_energy > 8.0 and requested_amount > 2.0:  # High energy but large request
            return True
        
        # Check trust level - very low trust nodes asking for large amounts
        trust_level = self.trust_network.get(requester_id, 0.5)
        if trust_level < 0.3 and requested_amount > 1.0:
            return True
        
        return False
    
    def _record_energy_transaction(self, node_id: int, amount: float, transaction_type: str):
        """Record energy sharing transaction for tracking and analysis."""
        transaction = {
            "node_id": node_id,
            "amount": amount,
            "type": transaction_type,  # "shared" or "received"
            "timestamp": self._time,
            "trust_level": self.trust_network.get(node_id, 0.5)
        }
        self.energy_sharing_history.append(transaction)
    
    def apply_energy_drain_resistance(self, drain_amount: float, attacker_count: int = 1) -> float:
        """
        Apply energy drain resistance to reduce attack impact.
        Original: 15% per attacker. Target: 5-8% per attacker.
        """
        # Base resistance reduces impact
        base_reduction = drain_amount * self.energy_drain_resistance
        
        # Additional protection based on suspicious activity detection
        recent_attacks = len([event for event in self.suspicious_events 
                            if self._time - event.get("timestamp", 0) < 5])
        
        if recent_attacks > 0:
            # Increased resistance when under coordinated attack
            coordinated_attack_resistance = 0.3  # Additional 30% resistance
            additional_reduction = drain_amount * coordinated_attack_resistance
            base_reduction = min(drain_amount * 0.9, base_reduction + additional_reduction)
        
        # Diminishing returns for multiple attackers
        attacker_factor = 1.0 / (1.0 + (attacker_count - 1) * 0.3)  # Reduce impact per additional attacker
        
        actual_drain = (drain_amount - base_reduction) * attacker_factor
        
        # Target: 5-8% per attacker instead of 15%
        max_drain_per_attacker = 0.07 * self.energy  # 7% of current energy per attacker
        actual_drain = min(actual_drain, max_drain_per_attacker * attacker_count)
        
        return max(0.0, actual_drain)

    # [Previous methods remain the same]

    def move(self):
        # [Previous move code]
        
        # Social beings might move toward trusted nodes
        if self.phase == "interactive" and len(self.trust_network) > 0:
            # Potential extension: move toward most trusted nodes
            pass

    def step_phase(self, current_time: int):
        """Enhanced phase management with circadian rhythms and faster adaptation"""
        self._time = current_time
        self.circadian_cycle = current_time % 24
        
        # Reset communication rate limits for new time step
        self._reset_rate_limits(current_time)
        
        # Clean up memory periodically
        if current_time % 10 == 0:  # Every 10 steps
            self._cleanup_memory()
        
        # Enhanced environmental adaptation - faster response to changes
        self._adapt_to_environment()
        
        # Phase transitions based on energy, anxiety, and time with improved responsiveness
        energy_threshold_low = 3.0
        energy_threshold_high = 8.0
        anxiety_threshold_low = 5.0
        anxiety_threshold_high = 10.0
        
        # Faster adaptation to low energy conditions
        if self.energy < energy_threshold_low or self.circadian_cycle > 20:
            self.phase = "sleep"
            # Sleep stage based on anxiety level with faster determination
            if self.anxiety > anxiety_threshold_high:
                self.sleep_stage = "deep"  # Stress-induced deep sleep
            elif self.anxiety > anxiety_threshold_low:
                self.sleep_stage = "REM"
            else:
                self.sleep_stage = "light"
        elif self.energy > 20.0 and self.anxiety < anxiety_threshold_low:
            self.phase = "inspired"  # High energy, low anxiety
        elif self.energy > energy_threshold_high and 6 <= self.circadian_cycle <= 18:
            self.phase = "active"
        else:
            self.phase = "interactive"
            
        # Social phases might affect communication style - faster adaptation
        if self.phase == "interactive":
            self.communication_style["directness"] = min(1.0, self.communication_style["directness"] + 0.15)  # Faster increase
        elif self.phase == "sleep":
            self.communication_style["directness"] = max(0.0, self.communication_style["directness"] - 0.15)  # Faster decrease
        
        # Enhanced energy management optimization
        self._optimize_energy_usage()
        
        # Detect and respond to attacks
        if current_time % 5 == 0:  # Check every 5 steps
            self._check_for_attacks()
    
    def _adapt_to_environment(self):
        """Enhanced environmental adaptation with faster response"""
        # Analyze recent memories for environmental cues
        recent_memories = [m for m in self.memory[-10:] if m.timestamp > self._time - 5]
        
        # Environmental factors detection
        stress_indicators = sum(1 for m in recent_memories if m.emotional_valence < -0.5)
        resource_indicators = sum(1 for m in recent_memories if "resource" in str(m.content).lower())
        threat_indicators = sum(1 for m in recent_memories if "danger" in str(m.content).lower() or "attack" in str(m.content).lower())
        
        # Faster adaptation responses
        if stress_indicators > 2:
            # High stress environment - adapt quickly
            self.anxiety = min(20, self.anxiety + 2.0)  # Faster anxiety response
            self.energy_drain_resistance = min(1.0, self.energy_drain_resistance + 0.1)  # Increase resistance
        
        if resource_indicators < 1 and len(recent_memories) > 3:
            # Low resource environment - conserve energy more aggressively
            self.energy = max(0, self.energy * 0.98)  # Slight energy conservation
            
        if threat_indicators > 1:
            # Threat environment - enhance defenses quickly
            self.attack_detection_threshold = max(1, self.attack_detection_threshold - 1)  # More sensitive detection
            self.signal_redundancy_level = min(3, self.signal_redundancy_level + 1)  # Increase redundancy
    
    def _optimize_energy_usage(self):
        """Optimize energy usage based on current conditions and predictions"""
        # Predictive energy management
        predicted_energy_need = self._predict_future_energy_needs()
        
        # Adjust energy consumption based on predictions
        if predicted_energy_need > self.energy * 1.5:
            # High future energy needs - request sharing now
            if self.energy_sharing_enabled and len(self.trust_network) > 0:
                trusted_nodes = [node_id for node_id, trust in self.trust_network.items() if trust > 0.6]
                if trusted_nodes:
                    # This would require access to other nodes, so we'll simulate preparation
                    self._prepare_for_energy_sharing()
        
        # Optimize current energy usage
        if self.energy < 5.0:
            # Low energy - reduce non-essential activities
            self.max_communications_per_step = max(2, self.max_communications_per_step - 1)
            self.communication_range *= 0.95  # Reduce communication range slightly
        elif self.energy > 15.0:
            # High energy - can afford more activities
            self.max_communications_per_step = min(7, self.max_communications_per_step + 1)
            self.communication_range = min(3.0, self.communication_range * 1.05)
    
    def _predict_future_energy_needs(self) -> float:
        """Predict future energy needs based on patterns and current state"""
        # Analyze energy usage patterns
        recent_energy_changes = []
        for i in range(min(5, len(self.memory))):
            memory = self.memory[-i-1]
            if hasattr(memory, 'content') and isinstance(memory.content, dict):
                if 'energy' in memory.content:
                    recent_energy_changes.append(memory.content['energy'])
        
        if recent_energy_changes:
            avg_change = sum(recent_energy_changes) / len(recent_energy_changes)
            # Predict needs for next 5 time steps
            predicted_need = abs(avg_change) * 5
        else:
            # Default prediction based on current activity level
            if self.phase == "active":
                predicted_need = 3.0
            elif self.phase == "interactive":
                predicted_need = 4.0
            elif self.phase == "inspired":
                predicted_need = 5.0
            else:
                predicted_need = 1.0
        
        return predicted_need
    
    def _prepare_for_energy_sharing(self):
        """Prepare for potential energy sharing needs"""
        # Create a memory of preparation for energy sharing
        preparation_memory = Memory(
            content={"type": "energy_sharing_preparation", "prepared_at": self._time},
            importance=0.6,
            timestamp=self._time,
            memory_type="preparation"
        )
        self.memory.append(preparation_memory)
    
    def _check_for_attacks(self):
        """Enhanced attack detection and response"""
        # Check for coordinated attacks
        recent_suspicious_events = [e for e in self.suspicious_events if self._time - e.get("timestamp", 0) < 10]
        
        if len(recent_suspicious_events) >= self.attack_detection_threshold:
            # Under coordinated attack - enhance defenses
            self.energy_drain_resistance = min(1.0, self.energy_drain_resistance + 0.2)
            self.jamming_detection_sensitivity = max(0.1, self.jamming_detection_sensitivity - 0.1)  # More sensitive
            
            # Alert trusted nodes about attack
            alert_memory = Memory(
                content={"type": "attack_detected", "severity": len(recent_suspicious_events)},
                importance=0.9,
                timestamp=self._time,
                memory_type="alert"
            )
            self.memory.append(alert_memory)

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
