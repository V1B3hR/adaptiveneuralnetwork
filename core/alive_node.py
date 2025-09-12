import numpy as np
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import uuid
import logging
from core.ai_ethics import audit_decision
from core.time_manager import get_time_manager, get_timestamp
from core.trust_network import TrustNetwork

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
    """Structured signal for node-to-node communication with production features"""
    def __init__(self, content: Any, signal_type: str, urgency: float, 
                 source_id: int, requires_response: bool = False,
                 idempotency_key: Optional[str] = None, partition_key: Optional[str] = None,
                 correlation_id: Optional[str] = None, schema_version: str = "1.0"):
        self.id = str(uuid.uuid4())
        self.content = content
        self.signal_type = signal_type  # 'memory', 'query', 'warning', 'resource'
        self.urgency = urgency  # 0.0 to 1.0
        self.source_id = source_id
        self.timestamp = get_timestamp()
        self.requires_response = requires_response
        self.response = None
        
        # Production features
        self.idempotency_key = idempotency_key or f"{source_id}_{signal_type}_{uuid.uuid4().hex[:8]}"
        self.partition_key = partition_key or f"{source_id}_{signal_type}"  # For ordering guarantees
        self.correlation_id = correlation_id or str(uuid.uuid4())  # For distributed tracing
        self.schema_version = schema_version  # For schema evolution
        self.retry_count = 0  # Track retry attempts
        self.created_at = get_timestamp()  # Creation timestamp for age calculation
        self.processing_attempts = []  # Track processing history


class AliveLoopNode:
    sleep_stages = ["light", "REM", "deep"]

    def __init__(self, position, velocity, initial_energy=10.0, field_strength=1.0, node_id=0, spatial_dims=None):
        # Import spatial utilities
        from core.spatial_utils import validate_spatial_dimensions, zero_vector
        
        # Basic node attributes
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        
        # Determine spatial dimensions
        if spatial_dims is not None:
            self.spatial_dims = int(spatial_dims)
        else:
            # Infer from position length
            self.spatial_dims = len(self.position)
        
        # Validate position and velocity dimensions
        try:
            validate_spatial_dimensions([self.position, self.velocity], self.spatial_dims)
        except ValueError as e:
            raise ValueError(f"Node {node_id} dimension validation failed: {e}")
        
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
        self.attention_focus = zero_vector(self.spatial_dims)  # Dimension-aware attention focus
        self.radius = 0.5
        
        # Trust and social networks - Enhanced trust system
        self.trust_network_system = TrustNetwork(node_id)
        self.trust_network = self.trust_network_system.trust_network  # Backward compatibility
        
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
        # These attributes support robustness against adversarial or resource-exhaustion behaviors.
        self.energy_sharing_enabled = True              # Allow distributed energy sharing
        self.energy_sharing_history = deque(maxlen=20)  # Track energy transactions (tuples or dicts)
        self.attack_detection_threshold = 3             # Suspicious event count to trigger detection
        self.suspicious_events = deque(maxlen=10)       # Recent suspicious activity records
        self.energy_drain_resistance = 0.7              # Resistance factor (0.0 - 1.0)
        self.signal_redundancy_level = 2                # Number of redundant communication channels
        self.jamming_detection_sensitivity = 0.3        # Lower = less sensitive, higher = more false positives

        # Anxiety overwhelm safety protocol attributes
        # Supports internal emotional / load regulation and cooperative help signaling.
        self.anxiety_threshold = 8.0           # Threshold for activating help protocol
        self.calm = 1.0                        # Calm level (0.0 to 5.0); higher = more regulated
        self.help_signals_sent = 0             # Count of help signals sent in current period
        self.max_help_signals_per_period = 3   # Cap to prevent spam
        self.help_signal_cooldown = 10         # Seconds between allowed help requests
        self.last_help_signal_time = 0         # Timestamp of last help signal
        self.anxiety_unload_capacity = 2.0     # How much anxiety can be reduced per assistance interaction
        self.received_help_this_period = False # Flag to avoid redundant requests
        self.anxiety_history = deque(maxlen=20)# Rolling history for trend analysis
        
        # Extended emotional states and their histories
        self.joy = 0.0                         # Joy level (0.0 to 5.0)
        self.grief = 0.0                       # Grief level (0.0 to 5.0) 
        self.sadness = 0.0                     # Sadness level (0.0 to 5.0)
        self.joy_history = deque(maxlen=20)    # Rolling history for joy trend analysis
        self.grief_history = deque(maxlen=20)  # Rolling history for grief trend analysis
        self.sadness_history = deque(maxlen=20)# Rolling history for sadness trend analysis
        self.calm_history = deque(maxlen=20)   # Rolling history for calm trend analysis
        self.energy_history = deque(maxlen=20) # Rolling history for energy trend analysis

        # Initialize period boundaries if needed
        self._help_period_start = get_timestamp()
        self.help_period_duration = 60  # seconds (adjust as needed)
        
        # Production signal processing features
        self.deduplication_store = {}  # idempotency_key -> {timestamp, ttl}
        self.dedupe_ttl = 300  # 5 minutes TTL for deduplication
        self.dead_letter_queue = deque(maxlen=100)  # DLQ for poison messages
        self.signal_metrics = {
            'processed_count': 0,
            'error_count': 0,
            'dlq_count': 0,
            'duplicate_count': 0,
            'processing_times': deque(maxlen=100)
        }
        self.circuit_breaker = {
            'state': 'closed',  # closed, open, half-open
            'failure_count': 0,
            'failure_threshold': 5,
            'timeout': 30,  # seconds
            'last_failure_time': 0
        }
        self.persisted_signals = deque(maxlen=1000)  # For replay capabilities
        self.partition_queues = {}  # partition_key -> deque for ordering guarantees

    def send_signal(self, target_nodes: List['AliveLoopNode'], signal_type: str, 
                   content: Any, urgency: float = 0.5, requires_response: bool = False,
                   idempotency_key: Optional[str] = None, partition_key: Optional[str] = None):
        """Send a signal to other nodes with rate limiting and flow control"""
        # Rate limiting check
        if self.communications_this_step >= self.max_communications_per_step:
            return []  # Hit rate limit
            
        if self.energy < 1.0 or self.phase == "sleep":
            return []  # Not enough energy or asleep to communicate
        
        # Producer-side flow control: Check target node queue capacity
        available_targets = []
        for target in target_nodes:
            if hasattr(target, 'communication_queue'):
                queue_depth = len(target.communication_queue)
                queue_capacity = target.communication_queue.maxlen
                
                # Implement backpressure if queue is getting full
                if queue_depth >= queue_capacity * 0.8:  # 80% capacity threshold
                    logger.warning(f"Node {self.node_id}: Target node {target.node_id} queue near capacity ({queue_depth}/{queue_capacity})")
                    # Skip this target or apply circuit breaker logic
                    if queue_depth >= queue_capacity * 0.95:  # 95% capacity - circuit break
                        continue
                
                available_targets.append(target)
        
        if not available_targets:
            logger.warning(f"Node {self.node_id}: No available targets due to backpressure")
            return []
            
        signal = SocialSignal(
            content=content, 
            signal_type=signal_type, 
            urgency=urgency, 
            source_id=self.node_id, 
            requires_response=requires_response,
            idempotency_key=idempotency_key,
            partition_key=partition_key or f"{self.node_id}_{signal_type}"
        )
        signal.timestamp = self._time
        
        # Energy cost based on signal complexity and urgency
        energy_cost = 0.1 + (0.2 * urgency) + (0.1 * len(str(content)))
        self.energy = max(0, self.energy - energy_cost)
        
        responses = []
        for target in available_targets:
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

    def _is_duplicate_signal(self, signal: SocialSignal) -> bool:
        """Check if signal is a duplicate based on idempotency key"""
        current_time = get_timestamp()
        
        if signal.idempotency_key in self.deduplication_store:
            entry = self.deduplication_store[signal.idempotency_key]
            if current_time - entry['timestamp'] < self.dedupe_ttl:
                return True
            else:
                # TTL expired, remove entry
                del self.deduplication_store[signal.idempotency_key]
        
        return False
    
    def _record_signal_processed(self, signal: SocialSignal):
        """Record signal as processed for deduplication"""
        self.deduplication_store[signal.idempotency_key] = {
            'timestamp': get_timestamp(),
            'ttl': self.dedupe_ttl
        }
    
    def _validate_signal_schema(self, signal: SocialSignal) -> bool:
        """Validate signal schema version compatibility"""
        supported_versions = ["1.0", "1.1"]  # Add new versions as needed
        return signal.schema_version in supported_versions
    
    def _should_circuit_break(self) -> bool:
        """Check if circuit breaker should open"""
        if self.circuit_breaker['state'] == 'open':
            current_time = get_timestamp()
            if current_time - self.circuit_breaker['last_failure_time'] > self.circuit_breaker['timeout']:
                self.circuit_breaker['state'] = 'half-open'
                return False
            return True
        return False
    
    def _record_processing_error(self, signal: SocialSignal, error: str):
        """Record processing error and update circuit breaker"""
        self.signal_metrics['error_count'] += 1
        self.circuit_breaker['failure_count'] += 1
        self.circuit_breaker['last_failure_time'] = get_timestamp()
        
        if self.circuit_breaker['failure_count'] >= self.circuit_breaker['failure_threshold']:
            self.circuit_breaker['state'] = 'open'
        
        # Add to DLQ
        dlq_entry = {
            'signal': signal,
            'error': error,
            'timestamp': get_timestamp(),
            'node_id': self.node_id
        }
        self.dead_letter_queue.append(dlq_entry)
        self.signal_metrics['dlq_count'] += 1
        
        logger.warning(f"Node {self.node_id}: Signal {signal.id} moved to DLQ due to error: {error}")
    
    def _add_to_partition_queue(self, signal: SocialSignal):
        """Add signal to appropriate partition queue for ordering"""
        if signal.partition_key not in self.partition_queues:
            self.partition_queues[signal.partition_key] = deque(maxlen=50)
        
        self.partition_queues[signal.partition_key].append(signal)
    
    def _get_queue_metrics(self) -> Dict[str, Any]:
        """Get current queue metrics for observability"""
        current_time = get_timestamp()
        
        # Calculate queue depths
        main_queue_depth = len(self.communication_queue)
        partition_queue_depths = {k: len(v) for k, v in self.partition_queues.items()}
        
        # Calculate message ages
        message_ages = []
        for signal in self.communication_queue:
            if hasattr(signal, 'created_at'):
                age = current_time - signal.created_at
                message_ages.append(age)
        
        # Calculate throughput (messages per second over last minute)
        recent_times = [t for t in self.signal_metrics['processing_times'] 
                       if current_time - t < 60]
        throughput = len(recent_times) / 60.0 if recent_times else 0.0
        
        return {
            'queue_depth': main_queue_depth,
            'partition_queue_depths': partition_queue_depths,
            'max_message_age': max(message_ages) if message_ages else 0,
            'avg_message_age': sum(message_ages) / len(message_ages) if message_ages else 0,
            'throughput_per_second': throughput,
            'error_rate': self.signal_metrics['error_count'] / max(1, self.signal_metrics['processed_count']),
            'dlq_count': self.signal_metrics['dlq_count'],
            'duplicate_count': self.signal_metrics['duplicate_count'],
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def receive_signal(self, signal: SocialSignal) -> Optional[SocialSignal]:
        """Process an incoming signal from another node with production features"""
        processing_start_time = get_timestamp()
        
        try:
            # Circuit breaker check
            if self._should_circuit_break():
                logger.warning(f"Node {self.node_id}: Circuit breaker open, rejecting signal {signal.id}")
                return None
            
            # Schema validation
            if not self._validate_signal_schema(signal):
                error_msg = f"Unsupported schema version: {signal.schema_version}"
                self._record_processing_error(signal, error_msg)
                return None
            
            # Deduplication check
            if self._is_duplicate_signal(signal):
                self.signal_metrics['duplicate_count'] += 1
                logger.debug(f"Node {self.node_id}: Duplicate signal {signal.id} ignored")
                return None
            
            # Energy check
            if self.energy < 0.5:
                return None  # Not enough energy to process
                
            # Energy cost to process signal
            processing_cost = 0.05 + (0.1 * signal.urgency)
            self.energy = max(0, self.energy - processing_cost)
            
            # Add signal processing attempt
            signal.processing_attempts.append({
                'node_id': self.node_id,
                'timestamp': processing_start_time,
                'correlation_id': signal.correlation_id
            })
            
            # Add to partition queue for ordering guarantees
            self._add_to_partition_queue(signal)
            
            # Add to communication queue (existing behavior)
            self.communication_queue.append(signal)
            
            # Persist signal for replay capabilities
            self.persisted_signals.append({
                'signal': signal,
                'timestamp': processing_start_time,
                'node_id': self.node_id
            })
            
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
            elif signal.signal_type == "joy_share":
                self._process_joy_share_signal(signal)
            elif signal.signal_type == "grief_support_request":
                response = self._process_grief_support_request_signal(signal)
            elif signal.signal_type == "grief_support_response":
                self._process_grief_support_response_signal(signal)
            elif signal.signal_type == "celebration_invite":
                response = self._process_celebration_invite_signal(signal)
            elif signal.signal_type == "comfort_request":
                response = self._process_comfort_request_signal(signal)
            elif signal.signal_type == "comfort_response":
                self._process_comfort_response_signal(signal)
            else:
                error_msg = f"Unknown signal type: {signal.signal_type}"
                self._record_processing_error(signal, error_msg)
                return None
                
            # Emotional contagion
            if hasattr(signal.content, 'emotional_valence'):
                self._apply_emotional_contagion(signal.content.emotional_valence, signal.source_id)
            
            # Record successful processing
            self._record_signal_processed(signal)
            self.signal_metrics['processed_count'] += 1
            self.signal_metrics['processing_times'].append(processing_start_time)
            
            # Reset circuit breaker on success
            if self.circuit_breaker['state'] == 'half-open':
                self.circuit_breaker['state'] = 'closed'
                self.circuit_breaker['failure_count'] = 0
            
            return response
            
        except Exception as e:
            error_msg = f"Exception during signal processing: {str(e)}"
            self._record_processing_error(signal, error_msg)
            logger.error(f"Node {self.node_id}: {error_msg}")
            return None

    def _process_memory_signal(self, signal: SocialSignal):
        """Process a memory shared by another node"""
        memory = signal.content
        
        # Handle case where content is not a Memory object (create one)
        if not isinstance(memory, Memory):
            memory = Memory(
                content=memory,
                importance=0.5,  # Default importance
                timestamp=self._time,
                memory_type="shared",
                emotional_valence=0.0,
                source_node=signal.source_id
            )
        
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
        from core.spatial_utils import zero_vector
        
        # Increase anxiety based on urgency and trust in source
        trust_level = self.trust_network.get(signal.source_id, 0.5)
        self.anxiety += signal.urgency * trust_level * 2.0
        
        # Direct attention to potential threat (dimension-aware)
        if "danger" in str(signal.content).lower():
            # Create attention focus vector with first component set to 1.0
            self.attention_focus = zero_vector(self.spatial_dims)
            if self.spatial_dims > 0:
                self.attention_focus[0] = 1.0  # Focus on first axis direction

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
        """Update trust based on communication outcome using enhanced trust system"""
        # Create context for trust update
        context = {
            'timestamp': self._time,
            'source_energy': self.energy,
            'target_energy': getattr(target, 'energy', None),
            'communication_history': len(self.signal_history)
        }
        
        # Use enhanced trust network system
        new_trust = self.trust_network_system.update_trust(target, signal_type, context)
        
        # Update the backward compatibility dict
        self.trust_network[target.node_id] = new_trust
    
    def get_trust_summary(self):
        """Get overview of trust network health"""
        return self.trust_network_system.get_trust_summary()
    
    def process_trust_verification_request(self, verification_request):
        """Process a community trust verification request"""
        subject_id = verification_request.get('subject')
        if subject_id is None:
            return None
            
        # Provide our opinion on the subject
        our_trust = self.trust_network_system.get_trust(subject_id)
        return {
            'trust_level': our_trust,
            'responder_id': self.node_id,
            'confidence': 0.8 if subject_id in self.trust_network else 0.3
        }
    
    def handle_community_trust_feedback(self, subject_id, feedback_list):
        """Handle community feedback about a suspicious node"""
        self.trust_network_system.process_community_feedback(subject_id, feedback_list)
        # Update backward compatibility dict
        self.trust_network[subject_id] = self.trust_network_system.get_trust(subject_id)

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

    def step_phase(self, current_time: Optional[int] = None):
        """Enhanced phase management with circadian rhythms using centralized time management"""
        # Use centralized time manager for simulation time, but allow override for backward compatibility
        time_manager = get_time_manager()
        if current_time is not None:
            # Backward compatibility: set time manager to match provided time
            # Reset and advance to the desired time
            if current_time != time_manager.simulation_step:
                time_manager.reset()
                time_manager.advance_simulation(current_time)
        
        self._time = time_manager.simulation_step
        self.circadian_cycle = time_manager.circadian_time
        
        # Reset communication rate limits for new time step
        self._reset_rate_limits(self._time)
        
        # Clean up memory periodically
        if self._time % 10 == 0:  # Every 10 steps
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
            
        # Emotional state management and safety protocol
        self.update_emotional_states()  # Record all emotional states in history
        
        # Apply natural calm effect
        self.apply_calm_effect()
        
        # Reset help signal limits periodically
        if self._time % 50 == 0:  # Every 50 time steps
            self.reset_help_signal_limits()
            
        # Proactive intervention assessment using all emotional states
        if self._time % 10 == 0:  # Check every 10 steps to avoid overload
            intervention_assessment = self.assess_intervention_need()
            if intervention_assessment['intervention_needed']:
                # Store assessment for network to act upon
                # This allows the network layer to coordinate interventions
                self._last_intervention_assessment = intervention_assessment
            
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
        # Validate spatial dimensions match
        if hasattr(capacitor, 'position'):
            if self.position.shape != capacitor.position.shape:
                raise ValueError(f"Node {self.node_id} spatial dimensions {self.position.shape} "
                               f"don't match capacitor dimensions {capacitor.position.shape}")
        
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
                   
    # Joy and Positive Emotion Sharing Methods
    
    def send_joy_share_signal(self, nearby_nodes: List['AliveLoopNode'], joy_content: Dict[str, Any]) -> List['AliveLoopNode']:
        """
        Share positive emotions and joy with nearby nodes.
        
        Args:
            nearby_nodes: List of nodes within communication range
            joy_content: Dictionary containing joy details (achievement, celebration, etc.)
            
        Returns:
            List of nodes that received the joy signal
        """
        if self.energy < 1.0 or self.phase == "sleep":
            return []
            
        # Create joy sharing content
        joy_signal_content = {
            "type": "joy_share",
            "source_node": self.node_id,
            "emotional_valence": 0.8,  # High positive valence
            "celebration_type": joy_content.get("type", "general"),
            "description": joy_content.get("description", "Sharing positive energy"),
            "intensity": joy_content.get("intensity", 0.7),
            "timestamp": self._time,
            "invite_celebration": joy_content.get("invite_celebration", False)
        }
        
        # Filter nodes by proximity and trust
        joy_recipients = []
        for node in nearby_nodes:
            if (node.node_id != self.node_id and 
                self.trust_network.get(node.node_id, 0.5) >= 0.3 and
                node.energy >= 1.0):  # Node needs some energy to appreciate joy
                joy_recipients.append(node)
                
        if not joy_recipients:
            return []
            
        # Send joy signal to recipients
        self.send_signal(
            target_nodes=joy_recipients,
            signal_type="joy_share",
            content=joy_signal_content,
            urgency=0.3,  # Low urgency for positive sharing
            requires_response=False
        )
        
        # Record the joy sharing in memory
        joy_memory = Memory(
            content={"action": "shared_joy", "type": joy_content.get("type", "general")},
            importance=0.6,
            timestamp=self._time,
            memory_type="joy_shared",
            emotional_valence=0.8  # Positive for sharing joy
        )
        self.memory.append(joy_memory)
        
        # Boost own emotional state from sharing
        self.emotional_state["valence"] = min(1.0, self.emotional_state["valence"] + 0.1)
        
        logger.info(f"Node {self.node_id}: Shared joy with {len(joy_recipients)} nodes")
        return joy_recipients
        
    def _process_joy_share_signal(self, signal: SocialSignal):
        """Process a joy sharing signal from another node"""
        joy_content = signal.content
        
        if not isinstance(joy_content, dict) or joy_content.get("type") != "joy_share":
            return
            
        source_node_id = joy_content.get("source_node")
        if source_node_id is None:
            return
            
        # Apply positive emotional contagion - enhanced with new emotional states
        trust_level = self.trust_network.get(source_node_id, 0.5)
        joy_intensity = joy_content.get("intensity", 0.7)
        
        # Base emotional boost
        emotional_boost = joy_intensity * trust_level * 0.5
        
        # Amplify boost if we're in need (high grief/sadness, low joy)
        emotional_need = (self.grief * 0.2) + (self.sadness * 0.2) + max(0, (2.0 - self.joy) * 0.1)
        amplified_boost = emotional_boost * (1.0 + emotional_need)
        
        # Update all relevant emotional states
        # Boost emotional valence
        self.emotional_state["valence"] = min(1.0, self.emotional_state["valence"] + amplified_boost)
        
        # Increase joy directly
        joy_increase = min(amplified_boost * 1.5, 2.0)  # Cap the increase
        self.update_joy(joy_increase)
        
        # Reduce negative emotions through positive influence
        if self.anxiety > 0:
            anxiety_reduction = amplified_boost * 0.8
            self.anxiety = max(0, self.anxiety - anxiety_reduction)
            
        if self.sadness > 0:
            sadness_reduction = amplified_boost * 0.6
            self.update_sadness(-sadness_reduction)
            
        if self.grief > 0:
            grief_reduction = amplified_boost * 0.4  # Grief is harder to reduce
            self.update_grief(-grief_reduction)
            
        # Increase calm through shared joy
        calm_increase = amplified_boost * 0.3
        self.update_calm(calm_increase)
        
        # Increase trust with the joy sharer
        current_trust = self.trust_network.get(source_node_id, 0.5)
        self.trust_network[source_node_id] = min(1.0, current_trust + 0.05)
        
        # Record received joy in memory
        joy_memory = Memory(
            content={
                "action": "received_joy",
                "from_node": source_node_id,
                "celebration_type": joy_content.get("celebration_type", "general"),
                "description": joy_content.get("description", "")
            },
            importance=0.6,
            timestamp=self._time,
            memory_type="joy_received",
            emotional_valence=0.7  # Positive for receiving joy
        )
        self.memory.append(joy_memory)
        
        logger.debug(f"Node {self.node_id}: Received joy from node {source_node_id}, "
                    f"emotional boost: {emotional_boost:.2f}")
                    
    # Grief and Emotional Support Methods
    
    def send_grief_support_request(self, nearby_nodes: List['AliveLoopNode'], grief_details: Dict[str, Any]) -> List['AliveLoopNode']:
        """
        Request emotional support during times of grief or sadness.
        
        Args:
            nearby_nodes: List of nodes within communication range
            grief_details: Details about the grief or emotional support needed
            
        Returns:
            List of nodes that responded with support
        """
        if self.energy < 2.0:  # Need some energy to reach out
            return []
            
        # Create grief support request content
        support_request = {
            "type": "grief_support_request",
            "requesting_node": self.node_id,
            "emotional_valence": grief_details.get("emotional_valence", -0.6),
            "support_type": grief_details.get("support_type", "emotional"),
            "grief_intensity": grief_details.get("intensity", 0.7),
            "description": grief_details.get("description", "Requesting emotional support"),
            "urgency": min(1.0, grief_details.get("intensity", 0.7)),
            "timestamp": self._time
        }
        
        # Filter nodes by trust and availability
        trusted_supporters = []
        for node in nearby_nodes:
            if (node.node_id != self.node_id and 
                self.trust_network.get(node.node_id, 0.5) >= 0.4 and
                node.energy >= 3.0 and  # Supporter needs energy to help
                node.anxiety < 8.0 and  # Supporter shouldn't be overwhelmed
                node.emotional_state.get("valence", 0) > -0.5):  # Supporter not too sad themselves
                trusted_supporters.append(node)
                
        if not trusted_supporters:
            logger.debug(f"Node {self.node_id}: No available supporters for grief support")
            return []
            
        # Send grief support request
        responses = self.send_signal(
            target_nodes=trusted_supporters,
            signal_type="grief_support_request", 
            content=support_request,
            urgency=support_request["urgency"],
            requires_response=True
        )
        
        # Record the support request in memory
        support_memory = Memory(
            content={"action": "requested_grief_support", "grief_type": grief_details.get("support_type", "emotional")},
            importance=0.8,
            timestamp=self._time,
            memory_type="support_requested",
            emotional_valence=-0.4  # Slightly negative as it indicates distress
        )
        self.memory.append(support_memory)
        
        logger.info(f"Node {self.node_id}: Requested grief support from {len(trusted_supporters)} trusted nodes")
        
        return [node for node in trusted_supporters if any(r.source_id == node.node_id for r in responses)]
        
    def _process_grief_support_request_signal(self, signal: SocialSignal) -> Optional[SocialSignal]:
        """Process a grief support request from another node"""
        support_request = signal.content
        
        if not isinstance(support_request, dict) or support_request.get("type") != "grief_support_request":
            return None
            
        requesting_node_id = support_request.get("requesting_node")
        if requesting_node_id is None:
            return None
            
        # Check if we can provide support - enhanced with new emotional factors
        trust_level = self.trust_network.get(requesting_node_id, 0.5)
        
        # Enhanced decision logic using new emotional states and predictions
        predicted_anxiety = self.predict_emotional_state('anxiety', 2)
        predicted_energy = self.predict_emotional_state('energy', 2)
        
        # Can't help if we're too compromised emotionally or will be soon
        if (trust_level < 0.3 or  # Not trusted enough
            self.energy < 3.0 or  # Not enough energy
            predicted_energy < 2.0 or  # Energy will be too low
            self.anxiety > 8.0 or  # Too anxious ourselves
            predicted_anxiety > 7.0 or  # Will be too anxious
            self.grief > 4.0 or  # Too much grief ourselves
            self.sadness > 4.0 or  # Too sad ourselves  
            (self.sadness + self.grief) > 6.0 or  # Combined negative emotions too high
            self.emotional_state.get("valence", 0) < -0.6):  # General negative state
            return None
            
        # Provide emotional support - enhanced calculation using new emotional states
        grief_intensity = support_request.get("grief_intensity", 0.7)
        
        # Base support from calm level
        base_support = self.calm * 0.5
        
        # Boost support if we have high joy (sharing positive emotions)
        joy_boost = min(self.joy * 0.2, 1.0) if self.joy > 2.0 else 0.0
        
        # Reduce support if we have our own grief/sadness
        emotional_burden = (self.grief * 0.3) + (self.sadness * 0.25)
        
        support_amount = min(grief_intensity * 0.8, base_support + joy_boost - emotional_burden, 2.5)
        support_amount = max(0.1, support_amount)  # Minimum support level
        
        # Create support response
        support_response = {
            "type": "grief_support_response",
            "supporter_node": self.node_id,
            "emotional_support_offered": support_amount,
            "comfort_message": "You're not alone in this. I'm here to support you.",
            "support_type": support_request.get("support_type", "emotional"),
            "timestamp": self._time
        }
        
        # Record providing support in memory
        support_memory = Memory(
            content={"action": "provided_grief_support", "to_node": requesting_node_id, "support": support_amount},
            importance=0.7,
            timestamp=self._time,
            memory_type="support_given",
            emotional_valence=0.3  # Positive for helping others despite grief context
        )
        self.memory.append(support_memory)
        
        # Energy cost for providing emotional support
        energy_cost = support_amount * 0.3
        self.energy = max(1.0, self.energy - energy_cost)
        
        # Increase trust with the supported node
        current_trust = self.trust_network.get(requesting_node_id, 0.5)
        self.trust_network[requesting_node_id] = min(1.0, current_trust + 0.1)
        
        logger.debug(f"Node {self.node_id}: Provided grief support to node {requesting_node_id}")
        
        return SocialSignal(
            content=support_response,
            signal_type="grief_support_response",
            urgency=0.6,
            source_id=self.node_id
        )
        
    def _process_grief_support_response_signal(self, signal: SocialSignal):
        """Process a grief support response from another node"""
        support_response = signal.content
        
        if not isinstance(support_response, dict) or support_response.get("type") != "grief_support_response":
            return
            
        supporter_node_id = support_response.get("supporter_node")
        emotional_support = support_response.get("emotional_support_offered", 0.0)
        
        if supporter_node_id is None or emotional_support <= 0:
            return
            
        # Apply emotional support - gradual improvement in emotional valence
        old_valence = self.emotional_state["valence"]
        valence_improvement = emotional_support * 0.3
        self.emotional_state["valence"] = min(1.0, self.emotional_state["valence"] + valence_improvement)
        
        # Increase calm level 
        self.calm = min(5.0, self.calm + emotional_support * 0.2)
        
        # Reduce anxiety through emotional support
        if self.anxiety > 0:
            anxiety_reduction = emotional_support * 0.4
            self.anxiety = max(0, self.anxiety - anxiety_reduction)
        
        # Increase trust with supporter
        current_trust = self.trust_network.get(supporter_node_id, 0.5)
        self.trust_network[supporter_node_id] = min(1.0, current_trust + 0.15)
        
        # Record received support in memory
        support_memory = Memory(
            content={
                "action": "received_grief_support",
                "from_node": supporter_node_id,
                "valence_before": old_valence,
                "valence_after": self.emotional_state["valence"],
                "message": support_response.get("comfort_message", "")
            },
            importance=0.9,
            timestamp=self._time,
            memory_type="support_received",
            emotional_valence=0.4  # Positive for receiving support
        )
        self.memory.append(support_memory)
        
        logger.info(f"Node {self.node_id}: Received grief support from node {supporter_node_id}, "
                   f"emotional valence improved from {old_valence:.2f} to {self.emotional_state['valence']:.2f}")
                   
    # Celebration and Comfort Methods
    
    def _process_celebration_invite_signal(self, signal: SocialSignal) -> Optional[SocialSignal]:
        """Process an invitation to celebrate something"""
        celebration_invite = signal.content
        
        if not isinstance(celebration_invite, dict) or celebration_invite.get("type") != "celebration_invite":
            return None
            
        inviter_node_id = celebration_invite.get("inviter_node")
        if inviter_node_id is None:
            return
            
        # Accept invitation based on energy, trust, and current emotional state
        trust_level = self.trust_network.get(inviter_node_id, 0.5)
        if (trust_level >= 0.4 and 
            self.energy >= 2.0 and
            self.anxiety < 6.0 and
            self.emotional_state.get("valence", 0) > -0.3):
            
            # Participate in celebration - boost emotions
            celebration_boost = 0.4 * trust_level
            self.emotional_state["valence"] = min(1.0, self.emotional_state["valence"] + celebration_boost)
            self.calm = min(5.0, self.calm + celebration_boost * 0.2)
            
            # Energy cost for celebrating
            self.energy = max(0, self.energy - 0.5)
            
            # Strengthen trust
            self.trust_network[inviter_node_id] = min(1.0, trust_level + 0.08)
            
            # Record celebration participation
            celebration_memory = Memory(
                content={"action": "participated_in_celebration", "with_node": inviter_node_id},
                importance=0.6,
                timestamp=self._time,
                memory_type="celebration",
                emotional_valence=0.6
            )
            self.memory.append(celebration_memory)
            
            logger.debug(f"Node {self.node_id}: Participated in celebration with node {inviter_node_id}")
            
            return SocialSignal(
                content={"type": "celebration_acceptance", "participant_node": self.node_id},
                signal_type="celebration_response", 
                urgency=0.3,
                source_id=self.node_id
            )
        
        return None
        
    def _process_comfort_request_signal(self, signal: SocialSignal) -> Optional[SocialSignal]:
        """Process a request for general comfort/support"""
        comfort_request = signal.content
        
        if not isinstance(comfort_request, dict) or comfort_request.get("type") != "comfort_request":
            return None
            
        requesting_node_id = comfort_request.get("requesting_node")
        if requesting_node_id is None:
            return None
            
        # Provide comfort if able
        trust_level = self.trust_network.get(requesting_node_id, 0.5)
        if (trust_level >= 0.3 and 
            self.energy >= 2.0 and
            self.anxiety < 7.0):
            
            comfort_amount = min(1.5, self.calm * 0.4)
            
            # Create comfort response
            comfort_response = {
                "type": "comfort_response",
                "comforter_node": self.node_id,
                "comfort_offered": comfort_amount,
                "message": "Sending you comfort and support.",
                "timestamp": self._time
            }
            
            # Cost of providing comfort
            self.energy = max(1.0, self.energy - comfort_amount * 0.2)
            
            # Strengthen relationship
            self.trust_network[requesting_node_id] = min(1.0, trust_level + 0.05)
            
            logger.debug(f"Node {self.node_id}: Provided comfort to node {requesting_node_id}")
            
            return SocialSignal(
                content=comfort_response,
                signal_type="comfort_response",
                urgency=0.4, 
                source_id=self.node_id
            )
        
        return None
        
    def _process_comfort_response_signal(self, signal: SocialSignal):
        """Process a comfort response from another node"""
        comfort_response = signal.content
        
        if not isinstance(comfort_response, dict) or comfort_response.get("type") != "comfort_response":
            return
            
        comforter_node_id = comfort_response.get("comforter_node")
        comfort_amount = comfort_response.get("comfort_offered", 0.0)
        
        if comforter_node_id is None or comfort_amount <= 0:
            return
            
        # Apply comfort
        self.calm = min(5.0, self.calm + comfort_amount)
        if self.anxiety > 0:
            self.anxiety = max(0, self.anxiety - comfort_amount * 0.3)
            
        # Improve emotional state slightly
        self.emotional_state["valence"] = min(1.0, self.emotional_state["valence"] + comfort_amount * 0.2)
        
        # Record received comfort
        comfort_memory = Memory(
            content={"action": "received_comfort", "from_node": comforter_node_id},
            importance=0.6,
            timestamp=self._time,
            memory_type="comfort_received",
            emotional_valence=0.3
        )
        self.memory.append(comfort_memory)
        
        logger.debug(f"Node {self.node_id}: Received comfort from node {comforter_node_id}")
                   
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

    # Example helper methods (add or adapt depending on existing architecture)

    def record_suspicious_event(self, event):
        """Record a suspicious event and evaluate whether mitigation should trigger."""
        self.suspicious_events.append((get_timestamp(), event))
        if len(self.suspicious_events) >= self.attack_detection_threshold:
            self.handle_attack_detection()

    def handle_attack_detection(self):
        """Trigger defensive adaptations when suspicious activity surpasses threshold."""
        # Example placeholder logic:
        # - Increase redundancy
        # - Throttle external interactions
        # - Flag node state
        if self.signal_redundancy_level < 5:
            self.signal_redundancy_level += 1
        # Optional: escalate logging, broadcast alert, etc.

    def share_energy(self, amount, recipient_id):
        """Log and authorize an energy-sharing interaction."""
        if not self.energy_sharing_enabled or amount <= 0:
            return False
        # Apply drain resistance as a safeguard
        effective_amount = amount * (1.0 - (1.0 - self.energy_drain_resistance))
        self.energy_sharing_history.append({
            "t": get_timestamp(),
            "recipient": recipient_id,
            "requested": amount,
            "transferred": effective_amount
        })
        # Implement actual energy transfer logic elsewhere
        return True

    def update_anxiety(self, delta):
        """Adjust anxiety (inverse of calm) and record history."""
        # Convert 'calm' into an implicit anxiety measure if needed
        # For clarity, treat 'calm' as a stabilizer: higher calm reduces net anxiety accumulation.
        adjusted = delta - (0.1 * self.calm)
        timestamp = get_timestamp()
        self.anxiety_history.append((timestamp, adjusted))
        # Optionally derive a rolling anxiety score
        rolling_anxiety = sum(v for _, v in self.anxiety_history)
        if rolling_anxiety >= self.anxiety_threshold:
            self.try_send_help_signal()

    def update_joy(self, delta):
        """Adjust joy level and record history."""
        self.joy = max(0.0, min(5.0, self.joy + delta))
        timestamp = get_timestamp()
        self.joy_history.append((timestamp, self.joy))
        
    def update_grief(self, delta):
        """Adjust grief level and record history."""
        self.grief = max(0.0, min(5.0, self.grief + delta))
        timestamp = get_timestamp()
        self.grief_history.append((timestamp, self.grief))
        
    def update_sadness(self, delta):
        """Adjust sadness level and record history."""
        self.sadness = max(0.0, min(5.0, self.sadness + delta))
        timestamp = get_timestamp()
        self.sadness_history.append((timestamp, self.sadness))
        
    def update_calm(self, delta):
        """Adjust calm level and record history."""
        self.calm = max(0.0, min(5.0, self.calm + delta))
        timestamp = get_timestamp()
        self.calm_history.append((timestamp, self.calm))
        
    def update_emotional_states(self):
        """Record current emotional states in history at each simulation step."""
        timestamp = get_timestamp()
        self.energy_history.append((timestamp, self.energy))
        # Record current states if not already recorded this timestamp
        if not self.joy_history or self.joy_history[-1][0] != timestamp:
            self.joy_history.append((timestamp, self.joy))
        if not self.grief_history or self.grief_history[-1][0] != timestamp:
            self.grief_history.append((timestamp, self.grief))
        if not self.sadness_history or self.sadness_history[-1][0] != timestamp:
            self.sadness_history.append((timestamp, self.sadness))
        if not self.calm_history or self.calm_history[-1][0] != timestamp:
            self.calm_history.append((timestamp, self.calm))
        if not self.anxiety_history or (len(self.anxiety_history) > 0 and 
                                       isinstance(self.anxiety_history[-1], tuple) and 
                                       self.anxiety_history[-1][0] != timestamp):
            self.anxiety_history.append((timestamp, self.anxiety))
    
    def predict_emotional_state(self, state_name: str, steps_ahead: int = 5) -> float:
        """Predict future emotional state based on historical data using simple trend analysis."""
        history_mapping = {
            'joy': self.joy_history,
            'grief': self.grief_history, 
            'sadness': self.sadness_history,
            'calm': self.calm_history,
            'anxiety': self.anxiety_history,
            'energy': self.energy_history
        }
        
        history = history_mapping.get(state_name)
        if not history or len(history) < 2:
            # Return current value if insufficient history
            current_mapping = {
                'joy': self.joy,
                'grief': self.grief,
                'sadness': self.sadness,
                'calm': self.calm,
                'anxiety': self.anxiety,
                'energy': self.energy
            }
            return current_mapping.get(state_name, 0.0)
        
        # Simple linear trend calculation using last few data points
        recent_points = list(history)[-min(5, len(history)):]
        if len(recent_points) < 2:
            return recent_points[-1][1]
            
        # Calculate average rate of change
        total_change = 0.0
        count = 0
        for i in range(1, len(recent_points)):
            time_diff = recent_points[i][0] - recent_points[i-1][0]
            if time_diff > 0:
                rate = (recent_points[i][1] - recent_points[i-1][1]) / time_diff
                total_change += rate
                count += 1
        
        if count == 0:
            return recent_points[-1][1]
            
        avg_rate = total_change / count
        current_value = recent_points[-1][1]
        predicted_value = current_value + (avg_rate * steps_ahead)
        
        # Apply bounds based on emotional state type
        if state_name in ['joy', 'grief', 'sadness', 'calm']:
            return max(0.0, min(5.0, predicted_value))
        elif state_name == 'anxiety':
            return max(0.0, predicted_value)
        elif state_name == 'energy':
            return max(0.0, predicted_value)
        else:
            return predicted_value
    
    def get_emotional_trends(self) -> Dict[str, str]:
        """Analyze trends in all emotional states and return trend directions."""
        trends = {}
        states = ['joy', 'grief', 'sadness', 'calm', 'anxiety', 'energy']
        
        for state in states:
            current = getattr(self, state)
            predicted = self.predict_emotional_state(state, 3)
            
            # Use absolute difference for states that have bounds, percentage for unbounded states
            if state in ['joy', 'grief', 'sadness', 'calm']:
                # For bounded states (0-5), use absolute threshold
                if predicted > current + 0.2:  # Increasing by more than 0.2
                    trends[state] = 'increasing'
                elif predicted < current - 0.2:  # Decreasing by more than 0.2
                    trends[state] = 'decreasing'
                else:
                    trends[state] = 'stable'
            else:
                # For unbounded states, use percentage threshold
                if predicted > current * 1.1:  # 10% threshold for trend detection
                    trends[state] = 'increasing'
                elif predicted < current * 0.9:
                    trends[state] = 'decreasing'
                else:
                    trends[state] = 'stable'
                
        return trends
    
    def assess_intervention_need(self) -> Dict[str, Any]:
        """Assess need for proactive intervention based on all emotional trends and predictions."""
        trends = self.get_emotional_trends()
        
        # Predict emotional states 3 steps ahead
        predictions = {}
        for state in ['joy', 'grief', 'sadness', 'calm', 'anxiety', 'energy']:
            predictions[state] = self.predict_emotional_state(state, 3)
        
        intervention_needed = False
        intervention_type = None
        urgency = 0.0
        reasons = []
        
        # Check for anxiety escalation
        if (trends['anxiety'] == 'increasing' and predictions['anxiety'] > self.anxiety_threshold) or \
           (self.anxiety > self.anxiety_threshold * 0.8):
            intervention_needed = True
            intervention_type = 'anxiety_help'
            urgency = max(urgency, 0.8)
            reasons.append('anxiety_escalation_predicted')
        
        # Check for overwhelming grief
        if (trends['grief'] == 'increasing' and predictions['grief'] > 4.0) or self.grief > 3.5:
            intervention_needed = True
            intervention_type = 'grief_support'
            urgency = max(urgency, 0.7)
            reasons.append('grief_overwhelming')
        
        # Check for deep sadness trends
        if (trends['sadness'] == 'increasing' and predictions['sadness'] > 4.0) or \
           (self.sadness > 3.0 and trends['joy'] == 'decreasing'):
            intervention_needed = True
            intervention_type = 'comfort_request'
            urgency = max(urgency, 0.6)
            reasons.append('sadness_trend_concerning')
        
        # Check for energy depletion with negative emotional states
        if (trends['energy'] == 'decreasing' and predictions['energy'] < 2.0) and \
           (self.grief > 2.0 or self.sadness > 2.0 or self.anxiety > 5.0):
            intervention_needed = True
            intervention_type = 'energy_support'
            urgency = max(urgency, 0.5)
            reasons.append('energy_emotional_crisis')
        
        # Check for positive intervention opportunities (share joy)
        if trends['joy'] == 'increasing' and self.joy > 3.0 and self.calm > 2.5:
            intervention_needed = True
            intervention_type = 'joy_share'
            urgency = max(urgency, 0.3)
            reasons.append('joy_sharing_opportunity')
        
        return {
            'intervention_needed': intervention_needed,
            'intervention_type': intervention_type,
            'urgency': urgency,
            'reasons': reasons,
            'trends': trends,
            'predictions': predictions,
            'emotional_summary': {
                'joy': self.joy,
                'grief': self.grief,
                'sadness': self.sadness,
                'calm': self.calm,
                'anxiety': self.anxiety,
                'energy': self.energy
            }
        }

    def try_send_help_signal(self):
        """Attempt to send a help signal respecting cooldown and rate limits."""
        now = get_timestamp()
        # Reset period if expired
        if now - self._help_period_start >= self.help_period_duration:
            self._help_period_start = now
            self.help_signals_sent = 0
            self.received_help_this_period = False

        if self.received_help_this_period:
            return False
        if self.help_signals_sent >= self.max_help_signals_per_period:
            return False
        if now - self.last_help_signal_time < self.help_signal_cooldown:
            return False

        # Perform help signal action here (broadcast, queue event, etc.)
        self.help_signals_sent += 1
        self.last_help_signal_time = now
        return True

    def receive_help(self, assistance_value=1.0):
        """Reduce accumulated anxiety through cooperative interaction."""
        unload = min(self.anxiety_unload_capacity, assistance_value)
        # Adjust calm upward (bounded)
        self.calm = min(5.0, self.calm + unload * 0.2)
        self.received_help_this_period = True
        # Optionally prune anxiety history to simulate relief
        if self.anxiety_history:
            trimmed = []
            for ts, val in self.anxiety_history:
                trimmed.append((ts, val * 0.7))  # decay past anxiety
            self.anxiety_history.clear()
            self.anxiety_history.extend(trimmed)
        return unload

    def graceful_shutdown(self, timeout: int = 30) -> bool:
        """Gracefully shutdown node by draining queues and finishing work"""
        logger.info(f"Node {self.node_id}: Starting graceful shutdown with {timeout}s timeout")
        start_time = get_timestamp()
        
        # Stop accepting new signals
        self.circuit_breaker['state'] = 'open'
        
        # Process remaining signals in queues
        while get_timestamp() - start_time < timeout:
            # Drain main communication queue
            if self.communication_queue:
                signal = self.communication_queue.popleft()
                try:
                    self.receive_signal(signal)
                except Exception as e:
                    logger.error(f"Node {self.node_id}: Error processing signal during shutdown: {e}")
                continue
            
            # Drain partition queues
            processed_any = False
            for partition_key, queue in self.partition_queues.items():
                if queue:
                    signal = queue.popleft()
                    try:
                        self.receive_signal(signal)
                        processed_any = True
                    except Exception as e:
                        logger.error(f"Node {self.node_id}: Error processing partition signal during shutdown: {e}")
            
            if not processed_any:
                break  # All queues empty
                
        # Log final metrics
        metrics = self._get_queue_metrics()
        logger.info(f"Node {self.node_id}: Shutdown complete. Final metrics: {metrics}")
        
        return len(self.communication_queue) == 0 and all(len(q) == 0 for q in self.partition_queues.values())
    
    def replay_signals(self, from_timestamp: Optional[int] = None, to_timestamp: Optional[int] = None) -> List[Dict]:
        """Replay persisted signals for recovery or audit purposes"""
        if from_timestamp is None:
            from_timestamp = 0
        if to_timestamp is None:
            to_timestamp = get_timestamp()
            
        replayed_signals = []
        for entry in self.persisted_signals:
            signal_time = entry['timestamp']
            if from_timestamp <= signal_time <= to_timestamp:
                signal = entry['signal']
                
                # Create replay entry
                replay_entry = {
                    'signal_id': signal.id,
                    'signal_type': signal.signal_type,
                    'source_id': signal.source_id,
                    'timestamp': signal_time,
                    'correlation_id': signal.correlation_id,
                    'content_summary': str(signal.content)[:100] + "..." if len(str(signal.content)) > 100 else str(signal.content)
                }
                replayed_signals.append(replay_entry)
        
        logger.info(f"Node {self.node_id}: Replayed {len(replayed_signals)} signals from {from_timestamp} to {to_timestamp}")
        return replayed_signals
    
    def process_dlq_messages(self) -> List[Dict]:
        """Process and return dead letter queue messages for manual review"""
        dlq_messages = []
        
        for entry in self.dead_letter_queue:
            signal = entry['signal']
            dlq_message = {
                'signal_id': signal.id,
                'signal_type': signal.signal_type,
                'source_id': signal.source_id,
                'error': entry['error'],
                'timestamp': entry['timestamp'],
                'correlation_id': signal.correlation_id,
                'retry_count': signal.retry_count,
                'content': signal.content
            }
            dlq_messages.append(dlq_message)
        
        logger.info(f"Node {self.node_id}: Found {len(dlq_messages)} messages in DLQ")
        return dlq_messages
    
    def reprocess_dlq_message(self, signal_id: str) -> bool:
        """Attempt to reprocess a message from the DLQ"""
        for i, entry in enumerate(self.dead_letter_queue):
            if entry['signal'].id == signal_id:
                signal = entry['signal']
                signal.retry_count += 1
                
                # Temporarily enable circuit breaker for reprocessing
                original_state = self.circuit_breaker['state']
                self.circuit_breaker['state'] = 'closed'
                
                # Attempt reprocessing
                try:
                    response = self.receive_signal(signal)
                    # Only remove from DLQ if processing was actually successful
                    # Check if signal was processed (not None response or no error thrown)
                    if signal.id not in [entry['signal'].id for entry in self.dead_letter_queue]:
                        logger.info(f"Node {self.node_id}: Successfully reprocessed DLQ message {signal_id}")
                        return True
                    else:
                        logger.error(f"Node {self.node_id}: Failed to reprocess DLQ message {signal_id}: still in DLQ")
                        return False
                except Exception as e:
                    logger.error(f"Node {self.node_id}: Failed to reprocess DLQ message {signal_id}: {e}")
                    return False
                finally:
                    # Restore original circuit breaker state
                    self.circuit_breaker['state'] = original_state
        
        logger.warning(f"Node {self.node_id}: DLQ message {signal_id} not found")
        return False
    
    def cleanup_expired_deduplication_entries(self):
        """Clean up expired entries from deduplication store"""
        current_time = get_timestamp()
        expired_keys = []
        
        for key, entry in self.deduplication_store.items():
            if current_time - entry['timestamp'] >= entry['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.deduplication_store[key]
        
        if expired_keys:
            logger.debug(f"Node {self.node_id}: Cleaned up {len(expired_keys)} expired deduplication entries")


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
