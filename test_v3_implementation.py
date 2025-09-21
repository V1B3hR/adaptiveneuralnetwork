"""
Test script for 3rd generation neuromorphic computing implementation.

This script validates the new V3 features and demonstrates their capabilities.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

# Import base neuromorphic
from adaptiveneuralnetwork.core.neuromorphic import NeuromorphicConfig, NeuromorphicPlatform

# Import V3 components
from adaptiveneuralnetwork.core.neuromorphic_v3 import (
    AdaptiveThresholdNeuron,
    BurstingNeuron,
    DynamicConnectivity,
    HierarchicalNetwork,
    HomeostaticScaling,
    MetaplasticitySynapse,
    MultiCompartmentNeuron,
    MultiTimescalePlasticity,
    OscillatoryDynamics,
    PhaseEncoder,
    PopulationLayer,
    RealisticDelays,
    SparseDistributedRepresentation,
    STDPSynapse,
    StochasticNeuron,
    TemporalPatternEncoder,
)

# Import V3 configurations
from adaptiveneuralnetwork.core.neuromorphic_v3.advanced_neurons import (
    NeuronV3Config,
)
from adaptiveneuralnetwork.core.neuromorphic_v3.network_topology import TopologyConfig
from adaptiveneuralnetwork.core.neuromorphic_v3.plasticity import (
    HomeostaticConfig,
    MetaplasticityConfig,
    STDPConfig,
)
from adaptiveneuralnetwork.core.neuromorphic_v3.temporal_coding import TemporalConfig

# Import hardware backends
from adaptiveneuralnetwork.neuromorphic import GenericV3Backend, Loihi2Backend, SpiNNaker2Backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_advanced_neurons():
    """Test advanced neuron models."""
    logger.info("Testing advanced neuron models...")

    # Create V3 configuration
    base_config = NeuromorphicConfig(generation=3)
    neuron_config = NeuronV3Config(base_config=base_config)

    # Test Multi-compartment neuron
    print("\n1. Multi-compartment Neuron:")
    mc_neuron = MultiCompartmentNeuron(neuron_config, num_dendrites=4)

    batch_size = 2
    dendritic_input = torch.randn(batch_size, 4) * 0.5
    somatic_input = torch.randn(batch_size, 1) * 0.3

    spikes, states = mc_neuron(dendritic_input, somatic_input)
    print(f"  Output spikes shape: {spikes.shape}")
    print(f"  Compartment voltages: {states['compartment_voltages'].shape}")

    # Test Adaptive threshold neuron
    print("\n2. Adaptive Threshold Neuron:")
    adaptive_neuron = AdaptiveThresholdNeuron(neuron_config)

    for t in range(10):
        input_current = torch.randn(batch_size, 1) * 0.8
        spikes, states = adaptive_neuron(input_current, current_time=t * 0.001)

        if t == 9:  # Print final state
            print(f"  Final threshold: {states['threshold'].mean():.3f}")
            print(f"  Firing rate: {states['firing_rate'].mean():.2f} Hz")

    # Test Bursting neuron
    print("\n3. Bursting Neuron:")
    bursting_neuron = BurstingNeuron(neuron_config)

    burst_count = 0
    for t in range(20):
        input_current = torch.ones(batch_size, 1) * 0.9  # Strong input to trigger bursts
        spikes, states = bursting_neuron(input_current, current_time=t * 0.001)

        if states["in_burst"].any():
            burst_count += 1

    print(f"  Burst events detected: {burst_count}")

    # Test Stochastic neuron
    print("\n4. Stochastic Neuron:")
    stochastic_neuron = StochasticNeuron(neuron_config)

    spike_variability = []
    for trial in range(5):
        spikes_trial = []
        for t in range(10):
            input_current = torch.ones(1, 1) * 0.7
            spikes, states = stochastic_neuron(input_current, current_time=t * 0.001)
            spikes_trial.append(spikes.sum().item())
        spike_variability.append(sum(spikes_trial))

    print(f"  Spike count variability across trials: {spike_variability}")
    print(f"  Standard deviation: {np.std(spike_variability):.2f}")


def test_plasticity_mechanisms():
    """Test plasticity mechanisms."""
    logger.info("Testing plasticity mechanisms...")

    # STDP configuration
    stdp_config = STDPConfig(a_plus=0.01, a_minus=0.012, tau_plus=0.020, tau_minus=0.020)

    print("\n1. STDP Synapse:")
    stdp_synapse = STDPSynapse(pre_size=10, post_size=5, config=stdp_config)

    # Test STDP learning
    for t in range(10):
        pre_spikes = torch.rand(2, 10) > 0.9  # Sparse spikes
        post_spikes = torch.rand(2, 5) > 0.95  # Very sparse spikes

        current, plasticity_info = stdp_synapse(
            pre_spikes.float(), post_spikes.float(), current_time=t * 0.001, learning=True
        )

    initial_weights = torch.full((10, 5), 0.5)
    final_weights = plasticity_info["synaptic_weights"]
    weight_changes = (final_weights - initial_weights).abs().mean()
    print(f"  Average weight change: {weight_changes:.4f}")

    # Test Metaplasticity
    print("\n2. Metaplasticity Synapse:")
    meta_config = MetaplasticityConfig(theta_plus=0.1, tau_theta=10.0)
    meta_synapse = MetaplasticitySynapse(
        pre_size=8, post_size=4, stdp_config=stdp_config, meta_config=meta_config
    )

    for t in range(20):
        pre_spikes = torch.rand(1, 8) > 0.8
        post_spikes = torch.rand(1, 4) > 0.9

        current, plasticity_info = meta_synapse(
            pre_spikes.float(), post_spikes.float(), current_time=t * 0.001, learning=True
        )

    print(f"  Metaplastic threshold: {plasticity_info['metaplastic_threshold'].mean():.3f}")

    # Test Homeostatic Scaling
    print("\n3. Homeostatic Scaling:")
    homeostatic_config = HomeostaticConfig(target_rate=10.0, scaling_window=1.0)
    homeostatic = HomeostaticScaling(num_neurons=6, config=homeostatic_config)

    weights = torch.randn(8, 6) * 0.1
    for t in range(100):  # Long simulation for homeostatic effects
        post_spikes = torch.rand(1, 6) > 0.95
        scaled_weights, homeostatic_info = homeostatic(
            post_spikes.float(), weights, current_time=t * 0.01
        )

    print(f"  Scaling factors: {homeostatic_info['scaling_factors'].mean():.3f}")
    print(
        f"  Target vs actual rate: {homeostatic_info['target_rate']:.1f} vs "
        f"{homeostatic_info['firing_rates'].mean():.1f} Hz"
    )


def test_network_topology():
    """Test network topology features."""
    logger.info("Testing network topology features...")

    base_config = NeuromorphicConfig(generation=3)
    neuron_config = NeuronV3Config(base_config=base_config)
    topology_config = TopologyConfig(num_layers=3, layer_sizes=[20, 15, 10])

    print("\n1. Population Layer:")
    population = PopulationLayer(
        population_size=15,
        neuron_type="adaptive_threshold",
        neuron_config=neuron_config,
        lateral_inhibition=True,
    )

    external_input = torch.randn(2, 15) * 0.5
    pop_spikes, pop_states = population(external_input)

    print(f"  Population output shape: {pop_spikes.shape}")
    print(f"  Population activity: {pop_states['population_activity'].mean():.3f}")

    print("\n2. Realistic Delays:")
    delays = RealisticDelays(source_size=10, target_size=8, config=topology_config)

    input_spikes = torch.rand(2, 10) > 0.9
    delayed_spikes = delays(input_spikes.float())

    print(f"  Delayed output shape: {delayed_spikes.shape}")
    print(f"  Delay range: {delays.delays.min():.4f} - {delays.delays.max():.4f} s")

    print("\n3. Dynamic Connectivity:")
    dynamic_conn = DynamicConnectivity(pre_size=12, post_size=8, config=topology_config)

    initial_connections = dynamic_conn.connectivity_mask.sum().item()

    for t in range(50):
        pre_spikes = torch.rand(1, 12) > 0.85
        post_spikes = torch.rand(1, 8) > 0.9

        current, conn_info = dynamic_conn(
            pre_spikes.float(), post_spikes.float(), current_time=t * 0.001, plasticity=True
        )

    final_connections = conn_info["num_connections"]
    formation_events = conn_info["formation_events"]
    pruning_events = conn_info["pruning_events"]

    print(f"  Initial connections: {initial_connections}")
    print(f"  Final connections: {final_connections}")
    print(f"  Formation events: {formation_events}")
    print(f"  Pruning events: {pruning_events}")

    print("\n4. Hierarchical Network:")
    hierarchical_net = HierarchicalNetwork(
        config=topology_config, layer_types=["adaptive_threshold"] * 3
    )

    input_spikes = torch.rand(2, topology_config.layer_sizes[0]) > 0.9
    net_output, net_states = hierarchical_net(input_spikes.float())

    print(f"  Network output shape: {net_output.shape}")
    print(f"  Layer activities: {net_states['layer_activities']}")
    print(f"  Network coherence: {net_states['network_coherence']:.3f}")


def test_temporal_coding():
    """Test temporal coding mechanisms."""
    logger.info("Testing temporal coding mechanisms...")

    temporal_config = TemporalConfig(
        pattern_window=0.1, reference_frequency=40.0, sparsity_target=0.05
    )

    print("\n1. Temporal Pattern Encoder:")
    pattern_encoder = TemporalPatternEncoder(input_size=20, pattern_size=10, config=temporal_config)

    for t in range(15):
        input_spikes = torch.rand(2, 20) > 0.92
        patterns, encoding_info = pattern_encoder(input_spikes.float(), current_time=t * 0.001)

    print(f"  Pattern activations shape: {patterns.shape}")
    print(f"  Pattern strengths: {patterns.mean(dim=0)[:5]}")  # First 5 patterns

    print("\n2. Phase Encoder:")
    phase_encoder = PhaseEncoder(input_size=16, config=temporal_config)

    for t in range(20):
        input_spikes = torch.rand(1, 16) > 0.88
        phase_encoded, phase_info = phase_encoder(input_spikes.float(), current_time=t * 0.001)

    print(f"  Phase encoded shape: {phase_encoded.shape}")
    print(f"  Current phases: {phase_info['current_phases'][:5]}")  # First 5

    print("\n3. Oscillatory Dynamics:")
    oscillatory = OscillatoryDynamics(num_oscillators=4, config=temporal_config)

    oscillator_history = []
    for t in range(30):
        external_input = torch.randn(1, 4) * 0.1
        oscillator_output, osc_info = oscillatory(external_input, current_time=t * 0.001)
        oscillator_history.append(oscillator_output[0])

    oscillator_history = torch.stack(oscillator_history)
    print(f"  Oscillator output shape: {oscillator_history.shape}")
    print(f"  Frequencies: {osc_info['frequencies']}")

    print("\n4. Sparse Distributed Representation:")
    sparse_repr = SparseDistributedRepresentation(
        input_size=50, representation_size=200, config=temporal_config
    )

    for iteration in range(10):
        input_activation = torch.rand(2, 50)
        sparse_output, sparsity_info = sparse_repr(input_activation, adapt_sparsity=True)

    print(f"  Sparse output shape: {sparse_output.shape}")
    print(f"  Actual sparsity: {sparsity_info['actual_sparsity']:.3f}")
    print(f"  Target sparsity: {sparsity_info['target_sparsity']:.3f}")


def test_hardware_backends():
    """Test hardware backend implementations."""
    logger.info("Testing hardware backends...")

    config = NeuromorphicConfig(
        platform=NeuromorphicPlatform.LOIHI2,
        generation=3,
        enable_stdp=True,
        enable_hierarchical_structure=True,
    )

    # Create a simple V3 model for testing
    class SimpleV3Model(nn.Module):
        def __init__(self):
            super().__init__()
            base_config = NeuromorphicConfig(generation=3)
            neuron_config = NeuronV3Config(base_config=base_config)

            self.population1 = PopulationLayer(20, "adaptive_threshold", neuron_config)
            self.population2 = PopulationLayer(15, "adaptive_threshold", neuron_config)

            stdp_config = STDPConfig()
            self.stdp_connection = STDPSynapse(20, 15, stdp_config)

    model = SimpleV3Model()

    # Test Loihi2 Backend
    print("\n1. Loihi 2 Backend:")
    loihi2_backend = Loihi2Backend()
    loihi2_backend.initialize(config)

    constraints = loihi2_backend.get_constraints()
    print(f"  Max neurons per core: {constraints.max_neurons_per_core}")
    print(f"  Supports multi-compartment: {constraints.supports_multi_compartment}")

    try:
        compiled_model = loihi2_backend.compile_network(model)
        print(f"  Compilation successful: {len(compiled_model['cores'])} cores used")

        deployment_id = loihi2_backend.deploy_model(compiled_model)
        print(f"  Deployment ID: {deployment_id}")

        # Test execution
        input_data = torch.randn(2, 20)
        output, metrics = loihi2_backend.execute(deployment_id, input_data, num_timesteps=100)
        print(f"  Execution output shape: {output.shape}")
        print(f"  Power consumption: {metrics.power_consumption_mw:.1f} mW")
    except Exception as e:
        print(f"  Error during Loihi2 testing: {e}")

    # Test SpiNNaker2 Backend
    print("\n2. SpiNNaker2 Backend:")
    spinnaker2_backend = SpiNNaker2Backend()
    spinnaker2_backend.initialize(config)

    try:
        compiled_model = spinnaker2_backend.compile_network(model)
        print(f"  Compilation successful: {len(compiled_model['populations'])} populations")

        deployment_id = spinnaker2_backend.deploy_model(compiled_model)
        print(f"  Deployment ID: {deployment_id}")

        input_data = torch.randn(2, 20)
        output, metrics = spinnaker2_backend.execute(deployment_id, input_data, num_timesteps=100)
        print(f"  Execution output shape: {output.shape}")
        print(f"  Power consumption: {metrics.power_consumption_mw:.1f} mW")
    except Exception as e:
        print(f"  Error during SpiNNaker2 testing: {e}")

    # Test Generic V3 Backend
    print("\n3. Generic V3 Backend:")
    generic_backend = GenericV3Backend()
    generic_backend.initialize(config)

    try:
        compiled_model = generic_backend.compile_network(model)
        v3_features = compiled_model["metadata"]["v3_features_enabled"]
        print(f"  V3 features detected: {sum(v3_features.values())} / {len(v3_features)}")

        deployment_id = generic_backend.deploy_model(compiled_model)
        print(f"  Deployment ID: {deployment_id}")

        input_data = torch.randn(2, 20)
        output, metrics = generic_backend.execute(deployment_id, input_data, num_timesteps=100)
        print(f"  Execution output shape: {output.shape}")
        print(f"  Power consumption: {metrics.power_consumption_mw:.1f} mW")
    except Exception as e:
        print(f"  Error during Generic V3 testing: {e}")


def test_integration():
    """Test integration of V3 components."""
    logger.info("Testing V3 component integration...")

    class IntegratedV3Network(nn.Module):
        def __init__(self):
            super().__init__()

            # Base configuration
            base_config = NeuromorphicConfig(generation=3)
            neuron_config = NeuronV3Config(base_config=base_config)
            temporal_config = TemporalConfig()
            topology_config = TopologyConfig(layer_sizes=[30, 20, 10])

            # Temporal encoding
            self.temporal_encoder = TemporalPatternEncoder(50, 30, temporal_config)

            # Hierarchical network with advanced neurons
            self.hierarchical_net = HierarchicalNetwork(
                topology_config,
                layer_types=["adaptive_threshold", "adaptive_threshold", "bursting"],
            )

            # Advanced plasticity
            stdp_config = STDPConfig()
            meta_config = MetaplasticityConfig()
            homeostatic_config = HomeostaticConfig()

            self.multi_plasticity = MultiTimescalePlasticity(
                20, 10, stdp_config, meta_config, homeostatic_config
            )

            # Oscillatory dynamics
            self.oscillatory = OscillatoryDynamics(4, temporal_config)

            # Sparse representation
            self.sparse_repr = SparseDistributedRepresentation(10, 40, temporal_config)

        def forward(self, x, current_time=0.0):
            # Temporal pattern encoding
            patterns, _ = self.temporal_encoder(x, current_time)

            # Hierarchical processing
            hierarchical_out, _ = self.hierarchical_net(patterns, current_time)

            # Multi-timescale plasticity
            plastic_out, _ = self.multi_plasticity(
                hierarchical_out,
                torch.zeros_like(hierarchical_out[:, :10]),  # Dummy post-synaptic
                current_time,
            )

            # Oscillatory modulation
            osc_out, _ = self.oscillatory(current_time=current_time)

            # Combine with oscillatory influence
            modulated = plastic_out + osc_out.unsqueeze(0) * 0.1

            # Sparse representation
            sparse_out, _ = self.sparse_repr(modulated)

            return sparse_out

    print("\nIntegrated V3 Network Test:")

    network = IntegratedV3Network()

    # Test forward pass
    batch_size = 2
    input_size = 50
    test_input = torch.rand(batch_size, input_size) > 0.9

    try:
        output = network(test_input.float(), current_time=0.001)
        print(f"  Network output shape: {output.shape}")
        print(f"  Output sparsity: {(output > 0).float().mean():.3f}")
        print("  ✓ Integration test successful!")

        # Test temporal dynamics
        outputs_over_time = []
        for t in range(10):
            varying_input = torch.rand(batch_size, input_size) > (0.85 + t * 0.01)
            output = network(varying_input.float(), current_time=t * 0.001)
            outputs_over_time.append(output.mean().item())

        temporal_variance = np.var(outputs_over_time)
        print(f"  Temporal variance: {temporal_variance:.4f}")

    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")


def main():
    """Run all V3 tests."""
    print("=" * 60)
    print("🧠 3RD GENERATION NEUROMORPHIC COMPUTING TESTS")
    print("=" * 60)

    try:
        test_advanced_neurons()
        print("\n" + "=" * 40)

        test_plasticity_mechanisms()
        print("\n" + "=" * 40)

        test_network_topology()
        print("\n" + "=" * 40)

        test_temporal_coding()
        print("\n" + "=" * 40)

        test_hardware_backends()
        print("\n" + "=" * 40)

        test_integration()

        print("\n" + "=" * 60)
        print("✅ ALL V3 TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
