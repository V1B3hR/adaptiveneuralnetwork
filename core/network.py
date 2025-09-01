import numpy as np
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

# Import your API modules
from api_integration.human_api import fetch_human_signal
from api_integration.ai import fetch_ai_signal  
from api_integration.world_api import fetch_world_signal

# Import AI Ethics Framework
from core.ai_ethics import get_ethics_framework, audit_decision_simple

class NetworkMetrics:
    """Tracks network-wide performance and health metrics"""
    def __init__(self):
        self.total_energy = deque(maxlen=1000)
        self.network_efficiency = deque(maxlen=1000)
        self.interaction_count = deque(maxlen=1000)
        self.signal_processing_rate = deque(maxlen=1000)
        self.trust_network_density = deque(maxlen=1000)
        self.collective_anxiety = deque(maxlen=1000)
        self.phase_distribution = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        
    def update(self, nodes, capacitors, interactions_this_step, signals_processed):
        """Update all metrics"""
        total_energy = sum(node.energy for node in nodes) + sum(cap.energy for cap in capacitors)
        avg_efficiency = np.mean([node.energy_efficiency for node in nodes])
        avg_anxiety = np.mean([node.anxiety for node in nodes])
        
        # Phase distribution
        phase_counts = defaultdict(int)
        for node in nodes:
            phase_counts[node.phase] += 1
        
        # Trust network density
        total_connections = sum(len(node.trust_network) for node in nodes)
        max_possible = len(nodes) * (len(nodes) - 1)
        trust_density = total_connections / max_possible if max_possible > 0 else 0
        
        # Memory usage
        total_memories = sum(len(node.memory) + len(node.long_term_memory) for node in nodes)
        
        self.total_energy.append(total_energy)
        self.network_efficiency.append(avg_efficiency)
        self.interaction_count.append(interactions_this_step)
        self.signal_processing_rate.append(signals_processed)
        self.trust_network_density.append(trust_density)
        self.collective_anxiety.append(avg_anxiety)
        self.phase_distribution.append(dict(phase_counts))
        self.memory_usage.append(total_memories)
    
    def get_health_score(self):
        """Calculate overall network health (0-1)"""
        if len(self.total_energy) < 10:
            return 0.5  # Not enough data
        
        # Recent averages
        recent_efficiency = np.mean(list(self.network_efficiency)[-10:])
        recent_anxiety = np.mean(list(self.collective_anxiety)[-10:])
        recent_trust = np.mean(list(self.trust_network_density)[-10:])
        recent_interactions = np.mean(list(self.interaction_count)[-10:])
        
        # Normalize and combine metrics
        efficiency_score = min(recent_efficiency / 2.0, 1.0)  # Cap at 2.0 efficiency
        anxiety_score = max(0, 1.0 - recent_anxiety / 20.0)  # Inverse anxiety
        trust_score = recent_trust
        interaction_score = min(recent_interactions / 50.0, 1.0)  # Cap at 50 interactions
        
        return (efficiency_score + anxiety_score + trust_score + interaction_score) / 4.0

class TunedAdaptiveFieldNetwork:
    def __init__(self, nodes, capacitors, api_endpoints=None, config=None):
        self.nodes = nodes
        self.capacitors = capacitors
        self.time = 0
        self.api_endpoints = api_endpoints or {}
        self.metrics = NetworkMetrics()
        
        # Configuration
        default_config = {
            "api_fetch_interval": 5,  # Fetch external signals every N steps
            "parallel_processing": True,  # Enable parallel node processing
            "adaptive_timing": True,  # Adjust timing based on network state
            "energy_balancing": True,  # Enable network-wide energy balancing
            "emergency_protocols": True,  # Emergency responses to network issues
            "logging": True,  # Enable detailed logging
            "max_workers": 4,  # Thread pool size for parallel processing
            "signal_batching": True,  # Batch process signals for efficiency
            "network_optimization": True,  # Enable network-wide optimizations
            "ethics_auditing": True  # Enable AI ethics framework auditing
        }
        self.config = {**default_config, **(config or {})}
        
        # AI Ethics Framework integration
        self.ethics_framework = get_ethics_framework() if self.config["ethics_auditing"] else None
        
        # Performance tracking
        self.step_times = deque(maxlen=100)
        self.api_response_times = deque(maxlen=100)
        self.last_optimization_time = 0
        self.optimization_interval = 50  # Optimize every N steps
        
        # Thread pool for parallel processing
        if self.config["parallel_processing"]:
            self.executor = ThreadPoolExecutor(max_workers=self.config["max_workers"])
        
        # Signal processing queue for batching
        self.signal_queue = deque()
        self.pending_signals = {}
        
        # Emergency state tracking
        self.emergency_state = False
        self.emergency_threshold = 0.3  # Health score threshold
        
        # Setup logging
        if self.config["logging"]:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        
        self.logger.info(f"Network initialized with {len(nodes)} nodes and {len(capacitors)} capacitors")

    async def fetch_external_streams_async(self):
        """Asynchronous API fetching for better performance"""
        external_streams = {}
        fetch_start_time = time.time()
        
        async def fetch_signal(node_id, endpoint):
            try:
                loop = asyncio.get_event_loop()
                if endpoint["type"] == "human":
                    signal = await loop.run_in_executor(None, fetch_human_signal, endpoint["url"])
                elif endpoint["type"] == "AI":
                    signal = await loop.run_in_executor(None, fetch_ai_signal, endpoint["url"])
                elif endpoint["type"] == "world":
                    signal = await loop.run_in_executor(None, fetch_world_signal, endpoint["url"])
                else:
                    return node_id, None
                return node_id, signal
            except Exception as e:
                self.logger.warning(f"Failed to fetch signal for node {node_id}: {e}")
                return node_id, None
        
        # Fetch all signals concurrently
        tasks = []
        for node in self.nodes:
            if node.node_id in self.api_endpoints:
                endpoint = self.api_endpoints[node.node_id]
                tasks.append(fetch_signal(node.node_id, endpoint))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, tuple) and result[1] is not None:
                    node_id, signal = result
                    external_streams[node_id] = signal
        
        # Track API response time
        response_time = time.time() - fetch_start_time
        self.api_response_times.append(response_time)
        
        return external_streams

    def fetch_external_streams(self):
        """Synchronous wrapper for API fetching"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.fetch_external_streams_async())
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.fetch_external_streams_async())

    def process_nodes_parallel(self, operation, *args):
        """Process nodes in parallel for better performance"""
        if not self.config["parallel_processing"]:
            # Sequential processing fallback
            for node in self.nodes:
                getattr(node, operation)(*args)
            return
        
        def process_node(node):
            try:
                return getattr(node, operation)(*args)
            except Exception as e:
                self.logger.error(f"Error processing node {node.node_id}: {e}")
                return None
        
        # Submit all tasks
        futures = [self.executor.submit(process_node, node) for node in self.nodes]
        
        # Wait for completion
        for future in futures:
            future.result()  # This will raise any exceptions

    def balance_network_energy(self):
        """Distribute energy more evenly across the network"""
        if not self.config["energy_balancing"]:
            return
        
        # Calculate energy distribution
        node_energies = [node.energy for node in self.nodes]
        mean_energy = np.mean(node_energies)
        energy_std = np.std(node_energies)
        
        # Only balance if there's significant inequality
        if energy_std > mean_energy * 0.3:
            high_energy_nodes = [node for node in self.nodes if node.energy > mean_energy + energy_std]
            low_energy_nodes = [node for node in self.nodes if node.energy < mean_energy - energy_std]
            
            # Transfer small amounts from high to low energy nodes
            for i, high_node in enumerate(high_energy_nodes):
                if i < len(low_energy_nodes):
                    low_node = low_energy_nodes[i]
                    transfer = min(high_node.energy * 0.05, (mean_energy - low_node.energy) * 0.5)
                    
                    high_node.energy -= transfer
                    low_node.energy += transfer
                    
                    # Build trust between nodes that help each other
                    high_node.trust_network[low_node.node_id] = \
                        high_node.trust_network.get(low_node.node_id, 0.5) + 0.02
                    low_node.trust_network[high_node.node_id] = \
                        low_node.trust_network.get(high_node.node_id, 0.5) + 0.02

    def handle_emergency_state(self):
        """Emergency protocols for network health issues"""
        health_score = self.metrics.get_health_score()
        
        if health_score < self.emergency_threshold and not self.emergency_state:
            self.emergency_state = True
            self.logger.warning(f"Network entering emergency state! Health score: {health_score:.3f}")
            
            # Emergency measures
            for node in self.nodes:
                # Reduce energy consumption
                if hasattr(node, 'signal_processing_cost'):
                    node.signal_processing_cost *= 1.5
                
                # Force sleep for highly anxious nodes
                if node.anxiety > 15:
                    node.phase = "sleep"
                    node.sleep_stage = "deep"
                
                # Reset trust networks if they're causing issues
                if len(node.trust_network) > len(self.nodes) * 0.8:
                    node.trust_network = {k: v for k, v in node.trust_network.items() if v > 0.7}
        
        elif health_score > self.emergency_threshold * 1.5 and self.emergency_state:
            self.emergency_state = False
            self.logger.info(f"Network recovering from emergency state. Health score: {health_score:.3f}")
            
            # Restore normal operations
            for node in self.nodes:
                if hasattr(node, 'signal_processing_cost'):
                    node.signal_processing_cost /= 1.5

    def optimize_network(self):
        """Perform network-wide optimizations"""
        if not self.config["network_optimization"]:
            return
        
        # Optimize spatial distribution of nodes
        positions = np.array([node.position for node in self.nodes])
        if len(positions) > 1:
            # Calculate center of mass
            center = np.mean(positions, axis=0)
            
            # Apply gentle centering force to prevent nodes from spreading too far
            for node in self.nodes:
                if node.phase != "sleep":  # Don't move sleeping nodes
                    distance_from_center = np.linalg.norm(node.position - center)
                    if distance_from_center > 20:  # Configurable threshold
                        centering_force = (center - node.position) * 0.01
                        node.velocity += centering_force
        
        # Optimize capacitor placement based on node activity
        if len(self.capacitors) > 0 and len(self.nodes) > 0:
            # Find high-activity areas
            active_positions = [node.position for node in self.nodes 
                             if node.phase in ["active", "interactive", "inspired"]]
            
            if active_positions:
                activity_center = np.mean(active_positions, axis=0)
                
                # Move the most depleted capacitor toward activity center
                most_depleted_cap = min(self.capacitors, key=lambda c: c.energy / c.capacity)
                direction_to_activity = activity_center - most_depleted_cap.position
                movement = direction_to_activity * 0.02  # Slow movement
                most_depleted_cap.position += movement

    def batch_process_signals(self, external_streams):
        """Efficiently batch process external signals"""
        if not self.config["signal_batching"]:
            # Process immediately
            signals_processed = 0
            if external_streams:
                for node in self.nodes:
                    if node.node_id in external_streams:
                        signal_type, signal_energy = external_streams[node.node_id]
                        node.absorb_external_signal(signal_energy, signal_type)
                        signals_processed += 1
            return signals_processed
        
        # Add to signal queue
        for node_id, signal_data in external_streams.items():
            if node_id not in self.pending_signals:
                self.pending_signals[node_id] = []
            self.pending_signals[node_id].append(signal_data)
        
        # Process batched signals
        signals_processed = 0
        for node in self.nodes:
            if node.node_id in self.pending_signals:
                signals = self.pending_signals[node.node_id]
                
                # Combine multiple signals of the same type
                combined_signals = defaultdict(float)
                for signal_type, signal_energy in signals:
                    combined_signals[signal_type] += signal_energy
                
                # Process combined signals
                for signal_type, total_energy in combined_signals.items():
                    # Cap total energy per batch to prevent overload
                    capped_energy = min(total_energy, 15.0)
                    node.absorb_external_signal(capped_energy, signal_type)
                    signals_processed += 1
                
                # Clear processed signals
                del self.pending_signals[node.node_id]
        
        return signals_processed

    def step(self, external_streams=None):
        """Enhanced step function with performance optimizations and monitoring"""
        step_start_time = time.time()
        interactions_this_step = 0
        signals_processed = 0
        
        # AI Ethics Framework: Audit network step decision
        if self.ethics_framework:
            network_state = {
                "time": self.time,
                "health_score": self.metrics.get_health_score(),
                "emergency_state": self.emergency_state,
                "node_count": len(self.nodes),
                "capacitor_count": len(self.capacitors)
            }
            has_violations, violation_messages = audit_decision_simple(
                action_type="network_step",
                actor_id="network",
                external_streams=external_streams is not None,
                environment_state=network_state,
                logged=True,
                verified=True
            )
            
            if has_violations:
                self.logger.warning(f"Ethics violations in network step: {violation_messages}")
                # In critical cases, we might want to halt or alert
                for msg in violation_messages:
                    if "CRITICAL" in msg:
                        self.logger.error(f"CRITICAL ethics violation detected: {msg}")
        
        self.time += 1
        
        # Handle emergency states
        if self.config["emergency_protocols"]:
            self.handle_emergency_state()
        
        # Use the enhanced step_update method for each node
        if self.config["parallel_processing"]:
            # Prepare external signals for each node
            node_signals = {}
            if external_streams:
                for node in self.nodes:
                    if node.node_id in external_streams:
                        signal_type, signal_energy = external_streams[node.node_id]
                        node_signals[node.node_id] = [{
                            "signal_energy": signal_energy,
                            "signal_type": signal_type,
                            "source_id": f"api_{signal_type}"
                        }]
            
            # Process all nodes in parallel using their enhanced step_update method
            def process_node_step(node):
                node_external_signals = node_signals.get(node.node_id, None)
                node.step_update(self.time, self.capacitors, self.nodes, node_external_signals)
                return 1  # Count this as processed
            
            futures = [self.executor.submit(process_node_step, node) for node in self.nodes]
            for future in futures:
                future.result()
        else:
            # Sequential processing using enhanced step_update
            for node in self.nodes:
                node_external_signals = None
                if external_streams and node.node_id in external_streams:
                    signal_type, signal_energy = external_streams[node.node_id]
                    node_external_signals = [{
                        "signal_energy": signal_energy,
                        "signal_type": signal_type,
                        "source_id": f"api_{signal_type}"
                    }]
                
                node.step_update(self.time, self.capacitors, self.nodes, node_external_signals)
        
        # Fetch and process external signals
        if external_streams is None and self.time % self.config["api_fetch_interval"] == 0:
            try:
                external_streams = self.fetch_external_streams()
                signals_processed = self.batch_process_signals(external_streams)
            except Exception as e:
                self.logger.error(f"Error fetching external streams: {e}")
        
        # Count interactions (approximate)
        for node in self.nodes:
            for capacitor in self.capacitors:
                distance = np.linalg.norm(node.position - capacitor.position)
                if distance < 0.5 + node.radius:  # Interaction threshold
                    interactions_this_step += 1
        
        # Network-wide optimizations
        if self.config["energy_balancing"]:
            self.balance_network_energy()
        
        # Periodic optimizations
        if (self.config["network_optimization"] and 
            self.time - self.last_optimization_time > self.optimization_interval):
            self.optimize_network()
            self.last_optimization_time = self.time
        
        # Update metrics
        self.metrics.update(self.nodes, self.capacitors, interactions_this_step, signals_processed)
        
        # Track performance
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        
        # Adaptive timing adjustment
        if self.config["adaptive_timing"] and len(self.step_times) > 10:
            avg_step_time = np.mean(list(self.step_times)[-10:])
            if avg_step_time > 0.1:  # If steps are taking too long
                self.config["api_fetch_interval"] = min(10, self.config["api_fetch_interval"] + 1)
                self.logger.info(f"Increased API fetch interval to {self.config['api_fetch_interval']}")

    def print_states(self):
        """Enhanced state reporting with network metrics"""
        health_score = self.metrics.get_health_score()
        
        print(f"\n{'='*60}")
        print(f"Time: {self.time} | Network Health: {health_score:.3f}")
        if self.emergency_state:
            print("⚠️  EMERGENCY STATE ACTIVE")
        
        # Performance metrics
        if len(self.step_times) > 0:
            avg_step_time = np.mean(list(self.step_times)[-10:])
            print(f"Avg Step Time: {avg_step_time*1000:.1f}ms")
        
        if len(self.api_response_times) > 0:
            avg_api_time = np.mean(list(self.api_response_times)[-5:])
            print(f"Avg API Response: {avg_api_time*1000:.1f}ms")
        
        # Network-wide statistics
        total_energy = sum(node.energy for node in self.nodes) + sum(cap.energy for cap in self.capacitors)
        avg_efficiency = np.mean([node.energy_efficiency for node in self.nodes])
        avg_anxiety = np.mean([node.anxiety for node in self.nodes])
        
        phase_counts = defaultdict(int)
        for node in self.nodes:
            phase_counts[node.phase] += 1
        
        print(f"Total Energy: {total_energy:.1f} | Avg Efficiency: {avg_efficiency:.2f} | Avg Anxiety: {avg_anxiety:.1f}")
        print(f"Phase Distribution: {dict(phase_counts)}")
        
        print("\nNodes:")
        for node in self.nodes:
            status = node.get_status_dict()
            trust_str = f"Trust:{status['trust_network_size']}" if status['trust_network_size'] > 0 else ""
            efficiency_str = f"Eff:{status['energy_efficiency']:.2f}"
            print(f"  Node {status['node_id']}: E:{status['energy']:.1f} "
                  f"Phase:{status['phase']} Anxiety:{status['anxiety']:.1f} "
                  f"{efficiency_str} {trust_str}")
        
        print("Capacitors:")
        for j, capacitor in enumerate(self.capacitors):
            utilization = (capacitor.energy / capacitor.capacity) * 100
            print(f"  Cap {j}: {capacitor.energy:.1f}/{capacitor.capacity} ({utilization:.1f}%)")
        print(f"{'='*60}\n")

    def get_network_status(self):
        """Return comprehensive network status as dictionary"""
        health_score = self.metrics.get_health_score()
        
        # Aggregate statistics
        node_stats = [node.get_status_dict() for node in self.nodes]
        capacitor_stats = [{
            "id": i,
            "position": cap.position.tolist(),
            "energy": cap.energy,
            "capacity": cap.capacity,
            "utilization": (cap.energy / cap.capacity) * 100
        } for i, cap in enumerate(self.capacitors)]
        
        return {
            "time": self.time,
            "health_score": health_score,
            "emergency_state": self.emergency_state,
            "total_energy": sum(node["energy"] for node in node_stats) + sum(cap["energy"] for cap in capacitor_stats),
            "average_efficiency": np.mean([node["energy_efficiency"] for node in node_stats]),
            "average_anxiety": np.mean([node["anxiety"] for node in node_stats]),
            "phase_distribution": {phase: sum(1 for node in node_stats if node["phase"] == phase) 
                                 for phase in ["active", "sleep", "inspired", "interactive"]},
            "performance": {
                "avg_step_time_ms": np.mean(list(self.step_times)[-10:]) * 1000 if self.step_times else 0,
                "avg_api_response_ms": np.mean(list(self.api_response_times)[-5:]) * 1000 if self.api_response_times else 0
            },
            "nodes": node_stats,
            "capacitors": capacitor_stats
        }

    def save_state(self, filename):
        """Save current network state to file"""
        state = self.get_network_status()
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        self.logger.info(f"Network state saved to {filename}")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
