"""
Simple test to verify V3 components are working.
"""

import logging

import torch

from adaptiveneuralnetwork.core.neuromorphic import NeuromorphicConfig
from adaptiveneuralnetwork.core.neuromorphic_v3.advanced_neurons import (
    MultiCompartmentNeuron,
    NeuronV3Config,
)
from adaptiveneuralnetwork.neuromorphic import GenericV3Backend

logging.basicConfig(level=logging.INFO)


def test_basic_v3():
    print("Testing basic V3 components...")

    # Test Multi-compartment neuron
    base_config = NeuromorphicConfig(generation=3)
    neuron_config = NeuronV3Config(base_config=base_config)

    mc_neuron = MultiCompartmentNeuron(neuron_config, num_dendrites=4)

    batch_size = 2
    dendritic_input = torch.randn(batch_size, 4) * 0.5
    somatic_input = torch.randn(batch_size, 1) * 0.3

    spikes, states = mc_neuron(dendritic_input, somatic_input)
    print(f"✓ Multi-compartment neuron: {spikes.shape}")

    # Test backend
    backend = GenericV3Backend()
    backend.initialize(NeuromorphicConfig(generation=3))
    constraints = backend.get_constraints()
    print(f"✓ Generic V3 backend: {constraints.max_neurons_per_core} max neurons per core")

    print("Basic V3 components working!")


if __name__ == "__main__":
    test_basic_v3()
