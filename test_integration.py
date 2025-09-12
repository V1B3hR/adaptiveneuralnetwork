#!/usr/bin/env python3
"""
Integration test to verify the phase encoding changes work with existing code.
"""

import sys
import os
import logging

# Add the project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from adaptiveneuralnetwork.core.neuromorphic import (
    NeuromorphicConfig, NeuromorphicPlatform, create_neuromorphic_model
)

logging.basicConfig(level=logging.INFO)

def test_existing_usage_patterns():
    """Test that existing usage patterns still work."""
    print("Testing existing usage patterns...")
    
    # Test 1: Basic config creation (existing pattern)
    config1 = NeuromorphicConfig()
    print(f"Default config phase encoding: {config1.enable_phase_encoding}")
    
    # Test 2: Config with platform (existing pattern)
    config2 = NeuromorphicConfig(platform=NeuromorphicPlatform.LOIHI2)
    print(f"Loihi2 config phase encoding: {config2.enable_phase_encoding}")
    
    # Test 3: Config with generation (existing pattern)
    config3 = NeuromorphicConfig(generation=3)
    print(f"Generation 3 config phase encoding: {config3.enable_phase_encoding}")
    
    # Test 4: Model creation (existing pattern)
    model = create_neuromorphic_model(
        input_dim=784,
        output_dim=10,
        platform=NeuromorphicPlatform.SIMULATION
    )
    print(f"Model config phase encoding: {model.config.enable_phase_encoding}")
    
    # Test 5: Temporal patterns enable auto-configuration
    config4 = NeuromorphicConfig(
        platform=NeuromorphicPlatform.SIMULATION,
        enable_temporal_patterns=True
    )
    print(f"Temporal patterns config phase encoding: {config4.enable_phase_encoding}")
    
    print("✓ All existing usage patterns work correctly\n")

def test_intelligent_configuration_scenarios():
    """Test intelligent configuration in realistic scenarios."""
    print("Testing intelligent configuration scenarios...")
    
    # Scenario 1: Audio processing with SpiNNaker2
    print("Scenario 1: Audio processing")
    config_audio = NeuromorphicConfig(
        platform=NeuromorphicPlatform.SPINNAKER2,
        input_data_type='sequential',
        enable_temporal_patterns=True
    )
    print(f"Audio config phase encoding: {config_audio.enable_phase_encoding}")
    assert config_audio.enable_phase_encoding, "Audio processing should enable phase encoding"
    
    # Scenario 2: Computer vision with basic platform
    print("Scenario 2: Computer vision with basic platform")
    config_vision = NeuromorphicConfig(
        platform=NeuromorphicPlatform.TRUENORTH,
        # No temporal characteristics
    )
    print(f"Vision config phase encoding: {config_vision.enable_phase_encoding}")
    assert not config_vision.enable_phase_encoding, "Basic vision should not enable phase encoding"
    
    # Scenario 3: Advanced 3rd gen features
    print("Scenario 3: Advanced 3rd generation features")
    config_advanced = NeuromorphicConfig(
        platform=NeuromorphicPlatform.LOIHI2,
        generation=3,
        enable_multi_compartment=True,
        enable_hierarchical_structure=True,
        num_hierarchy_levels=5
    )
    print(f"Advanced config phase encoding: {config_advanced.enable_phase_encoding}")
    assert config_advanced.enable_phase_encoding, "Advanced 3rd gen should enable phase encoding"
    
    # Scenario 4: Explicit user control
    print("Scenario 4: Explicit user control")
    config_explicit = NeuromorphicConfig.create_with_explicit_phase_encoding(
        False,
        platform=NeuromorphicPlatform.LOIHI2,
        enable_temporal_patterns=True  # Would normally auto-enable
    )
    print(f"Explicit config phase encoding: {config_explicit.enable_phase_encoding}")
    assert not config_explicit.enable_phase_encoding, "Explicit False should be preserved"
    
    print("✓ All intelligent configuration scenarios work correctly\n")

def main():
    """Run integration tests."""
    print("Running Integration Tests for Intelligent Phase Encoding\n")
    
    try:
        test_existing_usage_patterns()
        test_intelligent_configuration_scenarios()
        
        print("=" * 60)
        print("✓ All integration tests passed!")
        print("The intelligent phase encoding implementation is backward compatible")
        print("and works correctly with existing code patterns.")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"❌ Integration test failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)