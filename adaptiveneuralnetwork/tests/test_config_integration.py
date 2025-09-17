"""
Tests for configuration integration with AliveLoopNode and other system components.
"""

import pytest
import tempfile
import json
from pathlib import Path

from adaptiveneuralnetwork.config import AdaptiveNeuralNetworkConfig, load_config


class TestAliveLoopNodeConfigIntegration:
    """Test configuration integration with AliveLoopNode."""

    def test_node_uses_default_config_values(self):
        """Test that nodes use default configuration values when no config provided."""
        from core.alive_node import AliveLoopNode
        
        node = AliveLoopNode(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            initial_energy=10.0,
            node_id=1
        )
        
        # Should use default values
        assert node.anxiety_threshold == 8.0
        assert node.energy_drain_resistance == 0.7
        assert node.signal_redundancy_level == 2
        assert node.anxiety_history.maxlen == 20

    def test_node_uses_provided_config_values(self):
        """Test that nodes use provided configuration values."""
        from core.alive_node import AliveLoopNode
        
        config = AdaptiveNeuralNetworkConfig()
        config.proactive_interventions.anxiety_threshold = 12.0  
        config.attack_resilience.energy_drain_resistance = 0.9
        config.attack_resilience.signal_redundancy_level = 4
        config.rolling_history.max_len = 50
        
        node = AliveLoopNode(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0], 
            initial_energy=10.0,
            node_id=1,
            config=config
        )
        
        # Should use config values
        assert node.anxiety_threshold == 12.0
        assert node.energy_drain_resistance == 0.9
        assert node.signal_redundancy_level == 4
        assert node.anxiety_history.maxlen == 50

    def test_node_config_from_json_file(self):
        """Test that nodes can use configuration loaded from JSON file."""
        from core.alive_node import AliveLoopNode
        
        # Create test config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
            
        try:
            config_dict = {
                "proactive_interventions": {
                    "anxiety_threshold": 6.0,
                    "max_help_signals_per_period": 5,
                    "help_signal_cooldown": 5
                },
                "attack_resilience": {
                    "energy_drain_resistance": 0.85,
                    "signal_redundancy_level": 3,
                    "attack_detection_threshold": 2
                },
                "rolling_history": {
                    "max_len": 30
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(config_dict, f)
            
            # Load config and create node
            config = AdaptiveNeuralNetworkConfig.from_json(json_path)
            node = AliveLoopNode(
                position=[1.0, 1.0],
                velocity=[0.1, 0.1],
                initial_energy=15.0,
                node_id=2,
                config=config
            )
            
            # Verify config values are applied
            assert node.anxiety_threshold == 6.0
            assert node.max_help_signals_per_period == 5
            assert node.help_signal_cooldown == 5
            assert node.energy_drain_resistance == 0.85
            assert node.signal_redundancy_level == 3
            assert node.attack_detection_threshold == 2
            assert node.anxiety_history.maxlen == 30
            
        finally:
            Path(json_path).unlink()

    def test_proactive_interventions_can_be_disabled(self):
        """Test that proactive interventions can be disabled via configuration."""
        from core.alive_node import AliveLoopNode
        
        config = AdaptiveNeuralNetworkConfig()
        config.proactive_interventions.anxiety_enabled = False
        config.proactive_interventions.calm_enabled = False
        config.proactive_interventions.energy_enabled = False
        
        node = AliveLoopNode(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            initial_energy=10.0,
            node_id=1,
            config=config
        )
        
        # Verify configuration is stored
        assert node.config is not None
        assert node.config.proactive_interventions.anxiety_enabled is False
        assert node.config.proactive_interventions.calm_enabled is False
        assert node.config.proactive_interventions.energy_enabled is False

    def test_attack_resilience_parameters_configurable(self):
        """Test that attack resilience parameters are configurable."""
        from core.alive_node import AliveLoopNode
        
        config = AdaptiveNeuralNetworkConfig()
        config.attack_resilience.energy_drain_resistance = 0.95
        config.attack_resilience.max_drain_per_attacker_ratio = 0.03
        config.attack_resilience.signal_redundancy_level = 5
        config.attack_resilience.frequency_hopping_enabled = False
        config.attack_resilience.jamming_detection_sensitivity = 0.1
        
        node = AliveLoopNode(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            initial_energy=10.0,
            node_id=1,
            config=config
        )
        
        # Verify attack resilience parameters are applied
        assert node.energy_drain_resistance == 0.95
        assert node.signal_redundancy_level == 5
        assert node.jamming_detection_sensitivity == 0.1
        
        # Verify configuration object contains the values
        assert node.config.attack_resilience.max_drain_per_attacker_ratio == 0.03
        assert node.config.attack_resilience.frequency_hopping_enabled is False

    def test_trend_analysis_window_affects_node_behavior(self):
        """Test that trend analysis window configuration affects node behavior."""
        from core.alive_node import AliveLoopNode
        
        # Create configs with different trend windows
        config1 = AdaptiveNeuralNetworkConfig()
        config1.trend_analysis.window = 3
        
        config2 = AdaptiveNeuralNetworkConfig()
        config2.trend_analysis.window = 10
        
        node1 = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config1)
        node2 = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config2)
        
        # Verify different configurations are stored
        assert node1.config.trend_analysis.window == 3
        assert node2.config.trend_analysis.window == 10

    def test_rolling_history_max_len_affects_deques(self):
        """Test that rolling history max length affects deque sizes."""
        from core.alive_node import AliveLoopNode
        
        config = AdaptiveNeuralNetworkConfig()
        config.rolling_history.max_len = 100
        
        node = AliveLoopNode(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            initial_energy=10.0,
            node_id=1,
            config=config
        )
        
        # Verify all history deques use the configured max length
        assert node.anxiety_history.maxlen == 100
        assert node.joy_history.maxlen == 100
        assert node.energy_history.maxlen == 100
        assert node.calm_history.maxlen == 100
        
        # Verify emotion histories also use the configured max length  
        for emotion_name, history_deque in node.emotion_histories.items():
            assert history_deque.maxlen == 100

    def test_config_validation_warnings_in_node_creation(self):
        """Test that config validation warnings are triggered during config creation."""
        from core.alive_node import AliveLoopNode
        
        # Create config with invalid values - this triggers validation warnings
        with pytest.warns(UserWarning):
            config = AdaptiveNeuralNetworkConfig()
            config.trend_analysis.window = -5  # Invalid value
            config.rolling_history.max_len = 2  # Too small
            config.attack_resilience.energy_drain_resistance = 2.0  # Too large
            config._validate_and_clamp()  # Explicitly call validation
        
        # Now create node with the validated config
        node = AliveLoopNode(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            initial_energy=10.0,
            node_id=1,
            config=config
        )
        
        # Values should be clamped to safe ranges
        assert node.config.trend_analysis.window == 1
        assert node.config.rolling_history.max_len == 5
        assert node.config.attack_resilience.energy_drain_resistance == 1.0

    def test_backward_compatibility_without_config(self):
        """Test that nodes work without configuration for backward compatibility.""" 
        from core.alive_node import AliveLoopNode
        
        # Create node without config parameter
        node = AliveLoopNode(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            initial_energy=10.0,
            node_id=1
            # No config parameter
        )
        
        # Should use default values and not crash
        assert node.anxiety_threshold == 8.0  # Default value
        assert node.energy_drain_resistance == 0.7  # Default value
        assert hasattr(node, 'config')  # Should have config attribute


class TestConfigurationMonotonicity:
    """Test monotonic behavior changes with configuration parameters."""

    def test_trend_window_monotonicity(self):
        """Test that larger trend windows generally smooth out trend detection."""
        from core.alive_node import AliveLoopNode
        
        # Create synthetic emotional history data
        test_data = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0]
        
        # Test with small window
        config_small = AdaptiveNeuralNetworkConfig()
        config_small.trend_analysis.window = 3
        
        # Test with large window  
        config_large = AdaptiveNeuralNetworkConfig()
        config_large.trend_analysis.window = 6
        
        node_small = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_small)
        node_large = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_large)
        
        # Add test data to histories
        for value in test_data:
            node_small.anxiety_history.append(value)
            node_large.anxiety_history.append(value)
        
        # Verify configurations are different
        assert node_small.config.trend_analysis.window < node_large.config.trend_analysis.window

    def test_anxiety_threshold_monotonicity(self):
        """Test that higher anxiety thresholds reduce intervention frequency."""
        from core.alive_node import AliveLoopNode
        
        # Create configs with different anxiety thresholds
        config_low = AdaptiveNeuralNetworkConfig()
        config_low.proactive_interventions.anxiety_threshold = 5.0
        
        config_high = AdaptiveNeuralNetworkConfig()
        config_high.proactive_interventions.anxiety_threshold = 10.0
        
        node_low = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_low)
        node_high = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_high)
        
        # Verify thresholds are different
        assert node_low.anxiety_threshold < node_high.anxiety_threshold
        
        # Set same anxiety level
        test_anxiety = 7.0  # Between the two thresholds
        node_low.anxiety = test_anxiety
        node_high.anxiety = test_anxiety
        
        # Low threshold node should be above threshold, high threshold node should be below
        assert node_low.anxiety > node_low.anxiety_threshold
        assert node_high.anxiety < node_high.anxiety_threshold

    def test_energy_drain_resistance_monotonicity(self):
        """Test that higher energy drain resistance reduces drain effects."""
        from core.alive_node import AliveLoopNode
        
        # Create configs with different energy drain resistance
        config_low = AdaptiveNeuralNetworkConfig()
        config_low.attack_resilience.energy_drain_resistance = 0.3
        
        config_high = AdaptiveNeuralNetworkConfig() 
        config_high.attack_resilience.energy_drain_resistance = 0.9
        
        node_low = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_low)
        node_high = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_high)
        
        # Verify resistance values are monotonic
        assert node_low.energy_drain_resistance < node_high.energy_drain_resistance


class TestConfigurationBounds:
    """Test that configuration bounds are respected."""

    def test_max_drain_per_attacker_ratio_bounds(self):
        """Test that max drain per attacker ratio respects bounds."""
        config = AdaptiveNeuralNetworkConfig()
        
        # Test upper bound
        config.attack_resilience.max_drain_per_attacker_ratio = 2.0  # Above 1.0
        config._validate_and_clamp()
        assert config.attack_resilience.max_drain_per_attacker_ratio <= 1.0
        
        # Test lower bound
        config.attack_resilience.max_drain_per_attacker_ratio = -0.5  # Below 0.01
        config._validate_and_clamp()
        assert config.attack_resilience.max_drain_per_attacker_ratio >= 0.01

    def test_jamming_detection_sensitivity_bounds(self):
        """Test that jamming detection sensitivity respects bounds."""
        config = AdaptiveNeuralNetworkConfig()
        
        # Test upper bound
        config.attack_resilience.jamming_detection_sensitivity = 2.0  # Above 1.0
        config._validate_and_clamp()
        assert config.attack_resilience.jamming_detection_sensitivity <= 1.0
        
        # Test lower bound
        config.attack_resilience.jamming_detection_sensitivity = -0.5  # Below 0.0
        config._validate_and_clamp()
        assert config.attack_resilience.jamming_detection_sensitivity >= 0.0

    def test_energy_drain_resistance_bounds(self):
        """Test that energy drain resistance respects bounds."""
        config = AdaptiveNeuralNetworkConfig()
        
        # Test upper bound
        config.attack_resilience.energy_drain_resistance = 1.5  # Above 1.0
        config._validate_and_clamp()
        assert config.attack_resilience.energy_drain_resistance <= 1.0
        
        # Test lower bound  
        config.attack_resilience.energy_drain_resistance = -0.5  # Below 0.0
        config._validate_and_clamp()
        assert config.attack_resilience.energy_drain_resistance >= 0.0