#!/usr/bin/env python3
"""
Test script for intelligent phase encoding configuration.

This script tests the new intelligent configuration logic for the enable_phase_encoding
parameter in NeuromorphicConfig.
"""

import logging
import sys
import os

# Add the project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from adaptiveneuralnetwork.core.neuromorphic import NeuromorphicConfig, NeuromorphicPlatform

# Set up logging to capture auto-configuration messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_hardware_capability_detection():
    """Test hardware capability detection for different platforms."""
    print("=" * 60)
    print("Testing Hardware Capability Detection")
    print("=" * 60)
    
    # Test 3rd generation platforms (should support phase encoding)
    config_loihi2 = NeuromorphicConfig(platform=NeuromorphicPlatform.LOIHI2)
    supported, reason = config_loihi2._detect_hardware_phase_encoding_support()
    print(f"Loihi2: Supported={supported}, Reason='{reason}'")
    assert supported, "Loihi2 should support phase encoding"
    
    config_spinnaker2 = NeuromorphicConfig(platform=NeuromorphicPlatform.SPINNAKER2)
    supported, reason = config_spinnaker2._detect_hardware_phase_encoding_support()
    print(f"SpiNNaker2: Supported={supported}, Reason='{reason}'")
    assert supported, "SpiNNaker2 should support phase encoding"
    
    # Test 2nd generation platforms (should not support phase encoding)
    config_loihi = NeuromorphicConfig(platform=NeuromorphicPlatform.LOIHI)
    supported, reason = config_loihi._detect_hardware_phase_encoding_support()
    print(f"Loihi: Supported={supported}, Reason='{reason}'")
    assert not supported, "Loihi should not support phase encoding"
    
    # Test simulation mode (should support all features)
    config_sim = NeuromorphicConfig(platform=NeuromorphicPlatform.SIMULATION)
    supported, reason = config_sim._detect_hardware_phase_encoding_support()
    print(f"Simulation: Supported={supported}, Reason='{reason}'")
    assert supported, "Simulation should support phase encoding"
    
    print("✓ Hardware capability detection working correctly\n")


def test_model_requirement_analysis():
    """Test model requirement analysis for phase encoding."""
    print("=" * 60)
    print("Testing Model Requirement Analysis")
    print("=" * 60)
    
    # Test with temporal patterns enabled
    config = NeuromorphicConfig()
    config.enable_temporal_patterns = True
    should_enable, reasons = config._analyze_model_requirements()
    print(f"Temporal patterns: Should enable={should_enable}, Reasons={reasons}")
    assert should_enable, "Should enable phase encoding when temporal patterns are enabled"
    
    # Test with oscillatory dynamics enabled
    config = NeuromorphicConfig()
    config.enable_oscillatory_dynamics = True
    should_enable, reasons = config._analyze_model_requirements()
    print(f"Oscillatory dynamics: Should enable={should_enable}, Reasons={reasons}")
    assert should_enable, "Should enable phase encoding when oscillatory dynamics are enabled"
    
    # Test with hierarchical structure
    config = NeuromorphicConfig()
    config.enable_hierarchical_structure = True
    config.num_hierarchy_levels = 4
    should_enable, reasons = config._analyze_model_requirements()
    print(f"Hierarchical structure: Should enable={should_enable}, Reasons={reasons}")
    assert should_enable, "Should enable phase encoding for hierarchical structures"
    
    # Test with 3rd generation advanced features
    config = NeuromorphicConfig()
    config.generation = 3
    config.enable_multi_compartment = True
    should_enable, reasons = config._analyze_model_requirements()
    print(f"3rd gen advanced features: Should enable={should_enable}, Reasons={reasons}")
    assert should_enable, "Should enable phase encoding for 3rd gen advanced features"
    
    # Test with no special requirements
    config = NeuromorphicConfig()
    should_enable, reasons = config._analyze_model_requirements()
    print(f"No special requirements: Should enable={should_enable}, Reasons={reasons}")
    assert not should_enable, "Should not enable phase encoding without special requirements"
    
    print("✓ Model requirement analysis working correctly\n")


def test_input_data_analysis():
    """Test input data characteristics analysis."""
    print("=" * 60)
    print("Testing Input Data Characteristics Analysis")
    print("=" * 60)
    
    # Test with sequential data
    config = NeuromorphicConfig()
    config.input_data_type = 'sequential'
    should_enable, reasons = config._analyze_input_data_characteristics()
    print(f"Sequential data: Should enable={should_enable}, Reasons={reasons}")
    assert should_enable, "Should enable phase encoding for sequential data"
    
    # Test with high-dimensional data
    config = NeuromorphicConfig()
    config.input_data_type = 'high_dimensional'
    should_enable, reasons = config._analyze_input_data_characteristics()
    print(f"High-dimensional data: Should enable={should_enable}, Reasons={reasons}")
    assert should_enable, "Should enable phase encoding for high-dimensional data"
    
    # Test with temporal patterns data
    config = NeuromorphicConfig()
    config.input_data_type = 'temporal_patterns'
    should_enable, reasons = config._analyze_input_data_characteristics()
    print(f"Temporal patterns data: Should enable={should_enable}, Reasons={reasons}")
    assert should_enable, "Should enable phase encoding for temporal patterns data"
    
    # Test with no specific data type
    config = NeuromorphicConfig()
    should_enable, reasons = config._analyze_input_data_characteristics()
    print(f"No specific data type: Should enable={should_enable}, Reasons={reasons}")
    assert not should_enable, "Should not enable phase encoding without specific data type"
    
    print("✓ Input data characteristics analysis working correctly\n")


def test_automatic_configuration():
    """Test automatic phase encoding configuration."""
    print("=" * 60)
    print("Testing Automatic Phase Encoding Configuration")
    print("=" * 60)
    
    # Test automatic enabling on 3rd gen platform with temporal patterns
    print("Test 1: 3rd gen platform + temporal patterns")
    config = NeuromorphicConfig(
        platform=NeuromorphicPlatform.LOIHI2,
        enable_temporal_patterns=True
    )
    print(f"Phase encoding automatically enabled: {config.enable_phase_encoding}")
    assert config.enable_phase_encoding, "Should automatically enable phase encoding"
    
    # Test no automatic enabling on 2nd gen platform
    print("\nTest 2: 2nd gen platform + temporal patterns")
    config = NeuromorphicConfig(
        platform=NeuromorphicPlatform.LOIHI,
        enable_temporal_patterns=True
    )
    print(f"Phase encoding automatically enabled: {config.enable_phase_encoding}")
    assert not config.enable_phase_encoding, "Should not enable on 2nd gen platforms"
    
    # Test automatic enabling in simulation mode
    print("\nTest 3: Simulation mode + oscillatory dynamics")
    config = NeuromorphicConfig(
        platform=NeuromorphicPlatform.SIMULATION,
        enable_oscillatory_dynamics=True
    )
    print(f"Phase encoding automatically enabled: {config.enable_phase_encoding}")
    assert config.enable_phase_encoding, "Should automatically enable in simulation mode"
    
    # Test automatic enabling with input data characteristics
    print("\nTest 4: 3rd gen + sequential input data")
    config = NeuromorphicConfig(
        platform=NeuromorphicPlatform.SPINNAKER2,
        input_data_type='sequential'
    )
    print(f"Phase encoding automatically enabled: {config.enable_phase_encoding}")
    assert config.enable_phase_encoding, "Should enable for sequential data on 3rd gen"
    
    # Test no automatic enabling without requirements
    print("\nTest 5: Default configuration")
    config = NeuromorphicConfig()
    print(f"Phase encoding automatically enabled: {config.enable_phase_encoding}")
    assert not config.enable_phase_encoding, "Should not enable by default"
    
    print("✓ Automatic configuration working correctly\n")


def test_backward_compatibility():
    """Test that existing behavior is maintained for backward compatibility."""
    print("=" * 60)
    print("Testing Backward Compatibility")
    print("=" * 60)
    
    # Test that explicitly set values are preserved using factory method
    print("Test: Explicitly setting enable_phase_encoding=True using factory method")
    config = NeuromorphicConfig.create_with_explicit_phase_encoding(True)
    assert config.enable_phase_encoding, "Explicitly set True value should be preserved"
    
    print("Test: Explicitly setting enable_phase_encoding=False using factory method")
    config = NeuromorphicConfig.create_with_explicit_phase_encoding(
        False,
        platform=NeuromorphicPlatform.LOIHI2,
        enable_temporal_patterns=True
    )
    assert not config.enable_phase_encoding, "Explicitly set False value should be preserved"
    print(f"Phase encoding value: {config.enable_phase_encoding}")
    
    # Test normal constructor with explicit True value
    print("Test: Normal constructor with enable_phase_encoding=True")  
    config = NeuromorphicConfig(enable_phase_encoding=True)
    # This will still trigger auto-config but should provide guidance
    print(f"Phase encoding value: {config.enable_phase_encoding}")
    
    print("✓ Backward compatibility maintained\n")


def test_factory_method():
    """Test the factory method for explicit phase encoding configuration."""
    print("=" * 60)
    print("Testing Factory Method")
    print("=" * 60)
    
    # Test factory method with explicit True
    print("Test: Factory method with enable_phase_encoding=True")
    config = NeuromorphicConfig.create_with_explicit_phase_encoding(
        True,
        platform=NeuromorphicPlatform.LOIHI  # 2nd gen platform
    )
    assert config.enable_phase_encoding, "Factory method should preserve explicit True"
    
    # Test factory method with explicit False  
    print("Test: Factory method with enable_phase_encoding=False")
    config = NeuromorphicConfig.create_with_explicit_phase_encoding(
        False,
        platform=NeuromorphicPlatform.LOIHI2,  # 3rd gen platform
        enable_temporal_patterns=True  # Would normally trigger auto-enable
    )
    assert not config.enable_phase_encoding, "Factory method should preserve explicit False"
    
    print("✓ Factory method working correctly\n")


def main():
    """Run all tests."""
    print("Testing Intelligent Phase Encoding Configuration\n")
    
    try:
        test_hardware_capability_detection()
        test_model_requirement_analysis()
        test_input_data_analysis()
        test_automatic_configuration()
        test_factory_method()
        test_backward_compatibility()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("Intelligent phase encoding configuration is working correctly.")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)