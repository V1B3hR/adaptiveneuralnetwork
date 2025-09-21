"""
Tests for the central configuration system.
"""

import os
import tempfile
from pathlib import Path

import pytest

from adaptiveneuralnetwork.config import (
    AdaptiveNeuralNetworkConfig,
    AttackResilienceConfig,
    EnvironmentAdaptationConfig,
    ProactiveInterventionsConfig,
    RollingHistoryConfig,
    TrendAnalysisConfig,
    get_global_config,
    load_config,
    reset_global_config,
    set_global_config,
)


class TestTrendAnalysisConfig:
    """Test TrendAnalysisConfig functionality."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrendAnalysisConfig()
        assert config.window == 5
        assert config.enable_prediction is True
        assert config.prediction_steps == 3


class TestProactiveInterventionsConfig:
    """Test ProactiveInterventionsConfig functionality."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProactiveInterventionsConfig()
        assert config.anxiety_enabled is True
        assert config.calm_enabled is True
        assert config.anxiety_threshold == 8.0
        assert config.max_help_signals_per_period == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProactiveInterventionsConfig(
            anxiety_enabled=False, anxiety_threshold=6.0, max_help_signals_per_period=5
        )
        assert config.anxiety_enabled is False
        assert config.anxiety_threshold == 6.0
        assert config.max_help_signals_per_period == 5


class TestAttackResilienceConfig:
    """Test AttackResilienceConfig functionality."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AttackResilienceConfig()
        assert config.energy_drain_resistance == 0.7
        assert config.max_drain_per_attacker_ratio == 0.07
        assert config.signal_redundancy_level == 2
        assert config.frequency_hopping_enabled is True
        assert config.attack_detection_threshold == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AttackResilienceConfig(
            energy_drain_resistance=0.9, signal_redundancy_level=4, frequency_hopping_enabled=False
        )
        assert config.energy_drain_resistance == 0.9
        assert config.signal_redundancy_level == 4
        assert config.frequency_hopping_enabled is False


class TestAdaptiveNeuralNetworkConfig:
    """Test main configuration class."""

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = AdaptiveNeuralNetworkConfig()

        assert config.trend_analysis.window == 5
        assert config.rolling_history.max_len == 20
        assert config.proactive_interventions.anxiety_enabled is True
        assert config.attack_resilience.energy_drain_resistance == 0.7
        assert config.enable_structured_logging is True

    def test_validation_and_clamping(self):
        """Test configuration validation and clamping."""
        # Test trend analysis window clamping
        config = AdaptiveNeuralNetworkConfig()
        config.trend_analysis.window = -1
        config._validate_and_clamp()
        assert config.trend_analysis.window == 1

        config.trend_analysis.window = 200
        config._validate_and_clamp()
        assert config.trend_analysis.window == 100

        # Test rolling history clamping
        config.rolling_history.max_len = 2
        config._validate_and_clamp()
        assert config.rolling_history.max_len == 5

        config.rolling_history.max_len = 2000
        config._validate_and_clamp()
        assert config.rolling_history.max_len == 1000

        # Test attack resilience clamping
        config.attack_resilience.energy_drain_resistance = -0.5
        config._validate_and_clamp()
        assert config.attack_resilience.energy_drain_resistance == 0.0

        config.attack_resilience.energy_drain_resistance = 1.5
        config._validate_and_clamp()
        assert config.attack_resilience.energy_drain_resistance == 1.0

    def test_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = AdaptiveNeuralNetworkConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "trend_analysis" in config_dict
        assert "rolling_history" in config_dict
        assert "proactive_interventions" in config_dict
        assert "attack_resilience" in config_dict

        assert config_dict["trend_analysis"]["window"] == 5
        assert config_dict["rolling_history"]["max_len"] == 20
        assert config_dict["proactive_interventions"]["anxiety_enabled"] is True
        assert config_dict["attack_resilience"]["energy_drain_resistance"] == 0.7

    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "trend_analysis": {"window": 10},
            "proactive_interventions": {"anxiety_threshold": 6.0},
            "attack_resilience": {"energy_drain_resistance": 0.9},
            "enable_structured_logging": False,
        }

        config = AdaptiveNeuralNetworkConfig._from_dict(config_dict)

        assert config.trend_analysis.window == 10
        assert config.proactive_interventions.anxiety_threshold == 6.0
        assert config.attack_resilience.energy_drain_resistance == 0.9
        assert config.enable_structured_logging is False

    def test_yaml_roundtrip(self):
        """Test YAML save and load roundtrip."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name

        try:
            # Create config and save to YAML
            original_config = AdaptiveNeuralNetworkConfig()
            original_config.trend_analysis.window = 10
            original_config.proactive_interventions.anxiety_threshold = 6.0
            original_config.to_yaml(yaml_path)

            # Load from YAML
            loaded_config = AdaptiveNeuralNetworkConfig.from_yaml(yaml_path)

            assert loaded_config.trend_analysis.window == 10
            assert loaded_config.proactive_interventions.anxiety_threshold == 6.0
            assert loaded_config.rolling_history.max_len == 20  # default value
        finally:
            os.unlink(yaml_path)

    def test_json_roundtrip(self):
        """Test JSON save and load roundtrip."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            # Create config dictionary and save to JSON
            import json

            config_dict = {
                "trend_analysis": {"window": 15},
                "proactive_interventions": {"anxiety_threshold": 7.0},
                "attack_resilience": {"energy_drain_resistance": 0.8},
            }
            with open(json_path, "w") as f:
                json.dump(config_dict, f)

            # Load from JSON
            loaded_config = AdaptiveNeuralNetworkConfig.from_json(json_path)

            assert loaded_config.trend_analysis.window == 15
            assert loaded_config.proactive_interventions.anxiety_threshold == 7.0
            assert loaded_config.attack_resilience.energy_drain_resistance == 0.8
        finally:
            os.unlink(json_path)

    def test_update(self):
        """Test configuration update functionality."""
        config = AdaptiveNeuralNetworkConfig()

        # Test simple update
        updated_config = config.update(enable_structured_logging=False)
        assert updated_config.enable_structured_logging is False
        assert config.enable_structured_logging is True  # Original unchanged

        # Test nested update
        updated_config = config.update(**{"trend_analysis.window": 15})
        assert updated_config.trend_analysis.window == 15
        assert config.trend_analysis.window == 5  # Original unchanged

    def test_from_env(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        test_env = {
            "ANN_TREND_WINDOW": "12",
            "ANN_HISTORY_MAX_LEN": "30",
            "ANN_ANXIETY_ENABLED": "false",
            "ANN_ENERGY_DRAIN_RESISTANCE": "0.8",
            "ANN_SIGNAL_REDUNDANCY": "4",
            "ANN_FREQUENCY_HOPPING": "false",
            "ANN_LOG_LEVEL": "DEBUG",
            "ANN_STRUCTURED_LOGGING": "false",
        }

        # Temporarily set environment variables
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            config = AdaptiveNeuralNetworkConfig.from_env("ANN_")

            assert config.trend_analysis.window == 12
            assert config.rolling_history.max_len == 30
            assert config.proactive_interventions.anxiety_enabled is False
            assert config.attack_resilience.energy_drain_resistance == 0.8
            assert config.attack_resilience.signal_redundancy_level == 4
            assert config.attack_resilience.frequency_hopping_enabled is False
            assert config.log_level == "DEBUG"
            assert config.enable_structured_logging is False
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_with_file(self):
        """Test loading config from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            # Create test config file
            import json

            config_dict = {
                "proactive_interventions": {"anxiety_threshold": 5.0},
                "attack_resilience": {"energy_drain_resistance": 0.95},
            }
            with open(json_path, "w") as f:
                json.dump(config_dict, f)

            # Load config
            config = load_config(json_path, from_env=False)

            assert config.proactive_interventions.anxiety_threshold == 5.0
            assert config.attack_resilience.energy_drain_resistance == 0.95
        finally:
            os.unlink(json_path)

    def test_load_config_with_overrides(self):
        """Test loading config with overrides."""
        config = load_config(config_path=None, from_env=False, **{"trend_analysis.window": 20})

        assert config.trend_analysis.window == 20

    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                load_config(txt_path)
        finally:
            os.unlink(txt_path)


class TestGlobalConfig:
    """Test global configuration management."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_global_config()

    def test_get_global_config(self):
        """Test getting global configuration."""
        config = get_global_config()
        assert isinstance(config, AdaptiveNeuralNetworkConfig)

    def test_set_global_config(self):
        """Test setting global configuration."""
        custom_config = AdaptiveNeuralNetworkConfig()
        custom_config.trend_analysis.window = 25

        set_global_config(custom_config)
        retrieved_config = get_global_config()

        assert retrieved_config.trend_analysis.window == 25

    def test_reset_global_config(self):
        """Test resetting global configuration."""
        # Set custom config
        custom_config = AdaptiveNeuralNetworkConfig()
        custom_config.trend_analysis.window = 25
        set_global_config(custom_config)

        # Reset and get new config
        reset_global_config()
        new_config = get_global_config()

        # Should be back to defaults
        assert new_config.trend_analysis.window == 5


class TestConfigIntegration:
    """Test configuration integration with system components."""

    def test_config_flags_enable_disable(self):
        """Test that config flags can enable/disable features."""
        # Create config with some interventions disabled
        config = AdaptiveNeuralNetworkConfig()
        config.proactive_interventions.anxiety_enabled = False
        config.proactive_interventions.joy_enabled = False

        enabled = config._get_enabled_interventions()

        assert enabled["anxiety"] is False
        assert enabled["joy"] is False
        assert enabled["calm"] is True  # Should remain enabled

    def test_config_thresholds_respected(self):
        """Test that configuration thresholds are respected."""
        config = AdaptiveNeuralNetworkConfig()

        # Test custom thresholds
        config.proactive_interventions.anxiety_threshold = 12.0
        config.attack_resilience.max_drain_per_attacker_ratio = 0.05

        assert config.proactive_interventions.anxiety_threshold == 12.0
        assert config.attack_resilience.max_drain_per_attacker_ratio == 0.05

        # Test clamping
        config.attack_resilience.max_drain_per_attacker_ratio = 2.0  # Should be clamped
        config._validate_and_clamp()
        assert config.attack_resilience.max_drain_per_attacker_ratio == 1.0
