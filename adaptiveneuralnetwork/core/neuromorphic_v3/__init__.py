"""
3rd Generation Neuromorphic Computing Implementation.

This module provides advanced neuromorphic computing capabilities including:
- Multi-compartment neurons with dendritic processing
- Advanced synaptic plasticity (STDP, metaplasticity)
- Hierarchical network structures
- Temporal pattern encoding
- Hardware backend abstractions for 3rd generation platforms
"""

from .advanced_neurons import (
    AdaptiveThresholdNeuron,
    BurstingNeuron,
    MultiCompartmentNeuron,
    StochasticNeuron,
)
from .network_topology import (
    DynamicConnectivity,
    HierarchicalNetwork,
    PopulationLayer,
    RealisticDelays,
)
from .plasticity import (
    HomeostaticScaling,
    MetaplasticitySynapse,
    MultiTimescalePlasticity,
    STDPSynapse,
)
from .temporal_coding import (
    OscillatoryDynamics,
    PhaseEncoder,
    SparseDistributedRepresentation,
    TemporalPatternEncoder,
)

__all__ = [
    # Advanced neurons
    "MultiCompartmentNeuron",
    "AdaptiveThresholdNeuron",
    "BurstingNeuron",
    "StochasticNeuron",
    # Plasticity mechanisms
    "STDPSynapse",
    "MetaplasticitySynapse",
    "HomeostaticScaling",
    "MultiTimescalePlasticity",
    # Network topology
    "HierarchicalNetwork",
    "DynamicConnectivity",
    "PopulationLayer",
    "RealisticDelays",
    # Temporal coding
    "TemporalPatternEncoder",
    "PhaseEncoder",
    "OscillatoryDynamics",
    "SparseDistributedRepresentation",
]
