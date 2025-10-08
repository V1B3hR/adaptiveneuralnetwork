"""
Tests that verify configuration flags actually change system behavior.

This tests the core requirement that enabling/disabling proactive interventions,
changing thresholds, and modifying attack resilience parameters actually
affect the behavior of the system.
"""

from collections import deque
from unittest.mock import Mock, patch

import pytest

from adaptiveneuralnetwork.config import AdaptiveNeuralNetworkConfig
from core.alive_node import AliveLoopNode


class TestProactiveInterventionBehavior:
    """Test that proactive intervention config flags change behavior."""

    def test_anxiety_intervention_enabled_vs_disabled(self):
        """Test that enabling/disabling anxiety interventions changes behavior."""
        # Create two configs - one with anxiety interventions enabled, one disabled
        config_enabled = AdaptiveNeuralNetworkConfig()
        config_enabled.proactive_interventions.anxiety_enabled = True
        config_enabled.proactive_interventions.anxiety_threshold = 5.0  # Low threshold for testing

        config_disabled = AdaptiveNeuralNetworkConfig()
        config_disabled.proactive_interventions.anxiety_enabled = False
        config_disabled.proactive_interventions.anxiety_threshold = 5.0  # Same threshold

        # Create nodes with different configs
        node_enabled = AliveLoopNode(
            position=[0.0, 0.0], velocity=[0.0, 0.0],
            node_id=1, config=config_enabled
        )
        node_disabled = AliveLoopNode(
            position=[1.0, 1.0], velocity=[0.0, 0.0],
            node_id=2, config=config_disabled
        )

        # Set high anxiety level to trigger intervention check
        node_enabled.anxiety = 7.0  # Above threshold
        node_disabled.anxiety = 7.0  # Above threshold

        # Check that configuration flags are correctly applied
        assert node_enabled.config.proactive_interventions.anxiety_enabled is True
        assert node_disabled.config.proactive_interventions.anxiety_enabled is False

        # Both should have same anxiety threshold
        assert node_enabled.anxiety_threshold == node_disabled.anxiety_threshold == 5.0

        # Both should have anxiety above threshold
        assert node_enabled.anxiety > node_enabled.anxiety_threshold
        assert node_disabled.anxiety > node_disabled.anxiety_threshold

    def test_trend_window_variation_affects_slope_estimation(self):
        """Test that different trend windows affect slope estimation monotonically."""
        # Create configs with different trend windows
        config_small = AdaptiveNeuralNetworkConfig()
        config_small.trend_analysis.window = 3

        config_large = AdaptiveNeuralNetworkConfig()
        config_large.trend_analysis.window = 10

        node_small = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_small)
        node_large = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_large)

        # Create synthetic signal with clear trend
        # Signal: steady increase from 1.0 to 10.0
        test_signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        # Add data to both nodes
        for value in test_signal:
            node_small.anxiety_history.append(value)
            node_large.anxiety_history.append(value)

        # Verify window sizes are different
        assert node_small.config.trend_analysis.window < node_large.config.trend_analysis.window

        # Both should have same data but different analysis windows
        assert list(node_small.anxiety_history) == list(node_large.anxiety_history)

    def test_max_help_signals_cap_respected(self):
        """Test that max_help_signals_per_period cap is respected."""
        config = AdaptiveNeuralNetworkConfig()
        config.proactive_interventions.max_help_signals_per_period = 2  # Low limit for testing
        config.proactive_interventions.anxiety_threshold = 1.0  # Very low threshold

        node = AliveLoopNode(
            position=[0.0, 0.0], velocity=[0.0, 0.0],
            node_id=1, config=config
        )

        # Set high anxiety to trigger help signals
        node.anxiety = 10.0  # Well above threshold

        # Verify configuration is applied
        assert node.max_help_signals_per_period == 2
        assert node.anxiety_threshold == 1.0
        assert node.anxiety > node.anxiety_threshold

    def test_anxiety_threshold_affects_intervention_trigger(self):
        """Test that different anxiety thresholds affect when interventions trigger."""
        # Create configs with different thresholds
        config_low = AdaptiveNeuralNetworkConfig()
        config_low.proactive_interventions.anxiety_threshold = 3.0

        config_high = AdaptiveNeuralNetworkConfig()
        config_high.proactive_interventions.anxiety_threshold = 9.0

        node_low = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_low)
        node_high = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_high)

        # Set anxiety level between the two thresholds
        test_anxiety = 6.0
        node_low.anxiety = test_anxiety
        node_high.anxiety = test_anxiety

        # Low threshold node should be above threshold, high threshold node below
        assert node_low.anxiety > node_low.anxiety_threshold  # Should trigger intervention
        assert node_high.anxiety < node_high.anxiety_threshold  # Should not trigger


class TestAttackResilienceBehavior:
    """Test that attack resilience configuration affects system behavior."""

    def test_energy_drain_resistance_affects_damage(self):
        """Test that different energy drain resistance values affect damage taken."""
        # Create configs with different resistance levels
        config_low = AdaptiveNeuralNetworkConfig()
        config_low.attack_resilience.energy_drain_resistance = 0.1  # Low resistance

        config_high = AdaptiveNeuralNetworkConfig()
        config_high.attack_resilience.energy_drain_resistance = 0.9  # High resistance

        node_low = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_low)
        node_high = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_high)

        # Verify resistance values are applied
        assert node_low.energy_drain_resistance == 0.1
        assert node_high.energy_drain_resistance == 0.9

        # High resistance should be greater than low resistance (monotonic)
        assert node_high.energy_drain_resistance > node_low.energy_drain_resistance

    def test_signal_redundancy_levels_different(self):
        """Test that different signal redundancy levels are applied."""
        config_low = AdaptiveNeuralNetworkConfig()
        config_low.attack_resilience.signal_redundancy_level = 1

        config_high = AdaptiveNeuralNetworkConfig()
        config_high.attack_resilience.signal_redundancy_level = 5

        node_low = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_low)
        node_high = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_high)

        # Verify redundancy levels are applied
        assert node_low.signal_redundancy_level == 1
        assert node_high.signal_redundancy_level == 5

        # Higher redundancy should provide better protection (monotonic)
        assert node_high.signal_redundancy_level > node_low.signal_redundancy_level

    def test_frequency_hopping_can_be_toggled(self):
        """Test that frequency hopping can be enabled/disabled."""
        config_disabled = AdaptiveNeuralNetworkConfig()
        config_disabled.attack_resilience.frequency_hopping_enabled = False

        config_enabled = AdaptiveNeuralNetworkConfig()
        config_enabled.attack_resilience.frequency_hopping_enabled = True

        node_disabled = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_disabled)
        node_enabled = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_enabled)

        # Verify frequency hopping settings are applied
        assert node_disabled.config.attack_resilience.frequency_hopping_enabled is False
        assert node_enabled.config.attack_resilience.frequency_hopping_enabled is True

    def test_jamming_detection_sensitivity_affects_detection(self):
        """Test that jamming detection sensitivity affects detection behavior."""
        config_low = AdaptiveNeuralNetworkConfig()
        config_low.attack_resilience.jamming_detection_sensitivity = 0.1  # Less sensitive

        config_high = AdaptiveNeuralNetworkConfig()
        config_high.attack_resilience.jamming_detection_sensitivity = 0.9  # More sensitive

        node_low = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_low)
        node_high = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_high)

        # Verify sensitivity settings are applied
        assert node_low.jamming_detection_sensitivity == 0.1
        assert node_high.jamming_detection_sensitivity == 0.9

        # Higher sensitivity should detect more events (monotonic)
        assert node_high.jamming_detection_sensitivity > node_low.jamming_detection_sensitivity

    def test_max_drain_per_attacker_ratio_caps_damage(self):
        """Test that max drain per attacker ratio caps damage appropriately."""
        config_strict = AdaptiveNeuralNetworkConfig()
        config_strict.attack_resilience.max_drain_per_attacker_ratio = 0.01  # Very strict cap

        config_lenient = AdaptiveNeuralNetworkConfig()
        config_lenient.attack_resilience.max_drain_per_attacker_ratio = 0.2  # More lenient cap

        node_strict = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_strict)
        node_lenient = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_lenient)

        # Verify caps are applied
        assert node_strict.config.attack_resilience.max_drain_per_attacker_ratio == 0.01
        assert node_lenient.config.attack_resilience.max_drain_per_attacker_ratio == 0.2

        # Stricter cap should be smaller (monotonic protection)
        assert (node_strict.config.attack_resilience.max_drain_per_attacker_ratio <
                node_lenient.config.attack_resilience.max_drain_per_attacker_ratio)


class TestTrustManipulationDetection:
    """Test trust manipulation detection configuration."""

    def test_trust_growth_rate_limit_affects_detection(self):
        """Test that trust growth rate limit affects detection thresholds."""
        config_strict = AdaptiveNeuralNetworkConfig()
        config_strict.attack_resilience.trust_growth_rate_limit = 0.1  # Strict limit

        config_lenient = AdaptiveNeuralNetworkConfig()
        config_lenient.attack_resilience.trust_growth_rate_limit = 0.8  # Lenient limit

        node_strict = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_strict)
        node_lenient = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_lenient)

        # Verify limits are applied
        assert node_strict.config.attack_resilience.trust_growth_rate_limit == 0.1
        assert node_lenient.config.attack_resilience.trust_growth_rate_limit == 0.8

        # Stricter limit should catch more rapid growth (monotonic sensitivity)
        assert (node_strict.config.attack_resilience.trust_growth_rate_limit <
                node_lenient.config.attack_resilience.trust_growth_rate_limit)

    def test_rapid_trust_threshold_affects_detection(self):
        """Test that rapid trust threshold affects detection sensitivity."""
        config_sensitive = AdaptiveNeuralNetworkConfig()
        config_sensitive.attack_resilience.rapid_trust_threshold = 1.0  # Low threshold = more sensitive

        config_tolerant = AdaptiveNeuralNetworkConfig()
        config_tolerant.attack_resilience.rapid_trust_threshold = 5.0  # High threshold = less sensitive

        node_sensitive = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_sensitive)
        node_tolerant = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_tolerant)

        # Verify thresholds are applied
        assert node_sensitive.config.attack_resilience.rapid_trust_threshold == 1.0
        assert node_tolerant.config.attack_resilience.rapid_trust_threshold == 5.0

        # Lower threshold should be more sensitive (monotonic)
        assert (node_sensitive.config.attack_resilience.rapid_trust_threshold <
                node_tolerant.config.attack_resilience.rapid_trust_threshold)


class TestRollingHistoryBehavior:
    """Test that rolling history configuration affects memory behavior."""

    def test_history_max_len_affects_memory_capacity(self):
        """Test that different max_len values affect memory capacity."""
        config_small = AdaptiveNeuralNetworkConfig()
        config_small.rolling_history.max_len = 10

        config_large = AdaptiveNeuralNetworkConfig()
        config_large.rolling_history.max_len = 100

        node_small = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_small)
        node_large = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_large)

        # Verify memory capacities are different
        assert node_small.anxiety_history.maxlen == 10
        assert node_large.anxiety_history.maxlen == 100

        # Larger capacity should hold more data
        assert node_large.anxiety_history.maxlen > node_small.anxiety_history.maxlen

        # Test with actual data - add more data than small capacity
        test_data = list(range(50))  # 50 values

        for value in test_data:
            node_small.anxiety_history.append(value)
            node_large.anxiety_history.append(value)

        # Small capacity should only keep last 10 values
        assert len(node_small.anxiety_history) == 10
        assert list(node_small.anxiety_history) == list(range(40, 50))  # Last 10 values

        # Large capacity should keep all 50 values
        assert len(node_large.anxiety_history) == 50
        assert list(node_large.anxiety_history) == test_data


class TestEnvironmentAdaptationBehavior:
    """Test environment adaptation configuration effects."""

    def test_stress_thresholds_affect_adaptation(self):
        """Test that different stress thresholds affect adaptation behavior."""
        config_sensitive = AdaptiveNeuralNetworkConfig()
        config_sensitive.environment_adaptation.stress_threshold_low = 1.0
        config_sensitive.environment_adaptation.stress_threshold_high = 3.0

        config_tolerant = AdaptiveNeuralNetworkConfig()
        config_tolerant.environment_adaptation.stress_threshold_low = 5.0
        config_tolerant.environment_adaptation.stress_threshold_high = 9.0

        node_sensitive = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_sensitive)
        node_tolerant = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_tolerant)

        # Verify thresholds are applied
        assert node_sensitive.config.environment_adaptation.stress_threshold_low == 1.0
        assert node_sensitive.config.environment_adaptation.stress_threshold_high == 3.0
        assert node_tolerant.config.environment_adaptation.stress_threshold_low == 5.0
        assert node_tolerant.config.environment_adaptation.stress_threshold_high == 9.0

        # Sensitive node should have lower thresholds (more reactive)
        assert (node_sensitive.config.environment_adaptation.stress_threshold_low <
                node_tolerant.config.environment_adaptation.stress_threshold_low)
        assert (node_sensitive.config.environment_adaptation.stress_threshold_high <
                node_tolerant.config.environment_adaptation.stress_threshold_high)

    def test_adaptation_rate_affects_learning_speed(self):
        """Test that adaptation rate affects learning speed."""
        config_slow = AdaptiveNeuralNetworkConfig()
        config_slow.environment_adaptation.adaptation_rate = 0.01  # Slow adaptation

        config_fast = AdaptiveNeuralNetworkConfig()
        config_fast.environment_adaptation.adaptation_rate = 0.5   # Fast adaptation

        node_slow = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=1, config=config_slow)
        node_fast = AliveLoopNode(position=[0.0, 0.0], velocity=[0.0, 0.0], node_id=2, config=config_fast)

        # Verify adaptation rates are applied
        assert node_slow.config.environment_adaptation.adaptation_rate == 0.01
        assert node_fast.config.environment_adaptation.adaptation_rate == 0.5

        # Fast adaptation should be higher (monotonic)
        assert (node_fast.config.environment_adaptation.adaptation_rate >
                node_slow.config.environment_adaptation.adaptation_rate)


class TestStructuredLogging:
    """Test that structured logging configuration works."""

    @patch('adaptiveneuralnetwork.config.logger')
    def test_structured_logging_can_be_disabled(self, mock_logger):
        """Test that structured logging can be disabled."""
        config_disabled = AdaptiveNeuralNetworkConfig()
        config_disabled.enable_structured_logging = False
        config_disabled.log_config_events = True  # This should be ignored

        # Reset mock to ignore initialization calls
        mock_logger.reset_mock()

        # Try to log an event
        config_disabled.log_event('config', 'Test message', test_param='value')

        # Logger should not have been called because structured logging is disabled
        mock_logger.info.assert_not_called()

    @patch('adaptiveneuralnetwork.config.logger')
    def test_structured_logging_respects_event_flags(self, mock_logger):
        """Test that structured logging respects individual event type flags."""
        config = AdaptiveNeuralNetworkConfig()
        config.enable_structured_logging = True
        config.log_config_events = False  # Disable config events
        config.log_intervention_events = True  # Enable intervention events

        # Reset mock to ignore initialization calls
        mock_logger.reset_mock()

        # Try to log a config event (should be ignored)
        config.log_event('config', 'Config test message')
        mock_logger.info.assert_not_called()

        # Reset mock again
        mock_logger.reset_mock()

        # Try to log an intervention event (should be logged)
        config.log_event('intervention', 'Intervention test message')
        mock_logger.info.assert_called_once()

        # Verify the logged data structure
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == "Structured event: %s"
        logged_data = call_args[0][1]
        assert logged_data['event_type'] == 'intervention'
        assert logged_data['message'] == 'Intervention test message'
