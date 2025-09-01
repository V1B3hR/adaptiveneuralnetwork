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
from api_integrations.human_api import fetch_human_signal
from api_integrations.ai_api import fetch_ai_signal  
from api_integrations.world_api import fetch_world_signal

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [DEBUG] %(message)s')

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
        
        logging.debug("Initialized NetworkMetrics with empty metrics deques.")

    def update(self, nodes, capacitors, interactions_this_step, signals_processed):
        """Update all metrics"""
        # Calculate total energy
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
        
        # Update metrics
        self.total_energy.append(total_energy)
        self.network_efficiency.append(avg_efficiency)
        self.interaction_count.append(interactions_this_step)
        self.signal_processing_rate.append(signals_processed)
        self.trust_network_density.append(trust_density)
        self.collective_anxiety.append(avg_anxiety)
        self.phase_distribution.append(dict(phase_counts))
        self.memory_usage.append(total_memories)

        # Debugging logs for updates
        logging.debug(f"Updated metrics: Total Energy={total_energy}, Avg Efficiency={avg_efficiency}, "
                      f"Avg Anxiety={avg_anxiety}, Trust Density={trust_density}, Total Memories={total_memories}, "
                      f"Interactions This Step={interactions_this_step}, Signals Processed={signals_processed}")
        logging.debug(f"Phase Distribution: {dict(phase_counts)}")

    def get_health_score(self):
        """Calculate overall network health (0-1)"""
        if len(self.total_energy) < 10:
            logging.warning("Not enough data to calculate health score. Returning default score of 0.5.")
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
        
        health_score = (efficiency_score + anxiety_score + trust_score + interaction_score) / 4.0

        # Debugging logs for health score
        logging.debug(f"Health Score Calculation: Efficiency={efficiency_score}, Anxiety={anxiety_score}, "
                      f"Trust={trust_score}, Interactions={interaction_score}, Health Score={health_score}")
        
        return health_score

class TunedAdaptiveFieldNetwork:
    def __init__(self, nodes, capacitors, api_endpoints=None, config=None):
        self.nodes = nodes
        self.capacitors = capacitors
        self.api_endpoints = api_endpoints or {}
        self.config = config or {}

        # Debugging initialization
        logging.debug(f"Initialized TunedAdaptiveFieldNetwork with {len(nodes)} nodes and {len(capacitors)} capacitors.")
        if self.api_endpoints:
            logging.debug(f"API Endpoints: {self.api_endpoints}")
        if self.config:
            logging.debug(f"Configuration: {self.config}")
