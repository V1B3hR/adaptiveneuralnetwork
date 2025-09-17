"""
Central configuration system for Adaptive Neural Network.

This module provides a unified configuration layer to make all system parameters
configurable via YAML/JSON files, environment variables, constructor args, and CLI flags.
"""

import os
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path

import yaml


# Configure logging for configuration events
logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysisConfig:
    """Configuration for trend analysis parameters."""
    window: int = 5  # Trend analysis window size
    enable_prediction: bool = True
    prediction_steps: int = 3


@dataclass
class RollingHistoryConfig:
    """Configuration for rolling history parameters."""
    max_len: int = 20  # Maximum length of rolling histories


@dataclass
class ProactiveInterventionsConfig:
    """Configuration for proactive interventions."""
    # Per-subsystem enable flags
    anxiety_enabled: bool = True
    calm_enabled: bool = True
    energy_enabled: bool = True
    joy_enabled: bool = True
    grief_enabled: bool = True
    hope_enabled: bool = True
    anger_enabled: bool = True
    resilience_enabled: bool = True
    
    # Intervention thresholds
    anxiety_threshold: float = 8.0
    max_help_signals_per_period: int = 3
    help_signal_cooldown: int = 10
    anxiety_unload_capacity: float = 2.0


@dataclass
class AttackResilienceConfig:
    """Configuration for attack resilience features."""
    # Energy drain resistance
    energy_drain_resistance: float = 0.7  # 0.0 to 1.0
    max_drain_per_attacker_ratio: float = 0.07
    
    # Jamming resistance  
    signal_redundancy_level: int = 2
    frequency_hopping_enabled: bool = True
    jamming_detection_sensitivity: float = 0.3
    
    # Trust manipulation detection
    trust_growth_rate_limit: float = 0.5  # Max trust growth per interaction
    rapid_trust_threshold: float = 2.0    # Threshold for rapid trust detection
    
    # Attack detection
    attack_detection_threshold: int = 3
    suspicious_events_max_len: int = 10


@dataclass
class EnvironmentAdaptationConfig:
    """Configuration for environment adaptation parameters."""
    stress_threshold_low: float = 3.0
    stress_threshold_high: float = 7.0
    adaptation_rate: float = 0.1
    learning_rate_adaptation: bool = True


@dataclass
class AdaptiveNeuralNetworkConfig:
    """
    Central configuration class for the Adaptive Neural Network system.
    
    This class unifies all configuration parameters and provides methods for
    loading from files, environment variables, and runtime updates.
    """
    
    # Sub-configurations
    trend_analysis: TrendAnalysisConfig = field(default_factory=TrendAnalysisConfig)
    rolling_history: RollingHistoryConfig = field(default_factory=RollingHistoryConfig)
    proactive_interventions: ProactiveInterventionsConfig = field(default_factory=ProactiveInterventionsConfig)
    attack_resilience: AttackResilienceConfig = field(default_factory=AttackResilienceConfig)
    environment_adaptation: EnvironmentAdaptationConfig = field(default_factory=EnvironmentAdaptationConfig)
    
    # Logging configuration
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_config_events: bool = True
    log_intervention_events: bool = True
    log_attack_events: bool = True
    
    def __post_init__(self):
        """Validate and clamp configuration values to safe ranges."""
        self._validate_and_clamp()
        if self.log_config_events:
            self._log_applied_config()
    
    def _validate_and_clamp(self):
        """Validate and clamp all configuration values to safe ranges."""
        # Validate trend analysis
        if self.trend_analysis.window < 1:
            warnings.warn("trend_analysis.window must be >= 1, clamping to 1")
            self.trend_analysis.window = 1
        elif self.trend_analysis.window > 100:
            warnings.warn("trend_analysis.window > 100 may cause performance issues, clamping to 100")
            self.trend_analysis.window = 100
            
        # Validate rolling history
        if self.rolling_history.max_len < 5:
            warnings.warn("rolling_history.max_len should be >= 5 for meaningful trends, clamping to 5")
            self.rolling_history.max_len = 5
        elif self.rolling_history.max_len > 1000:
            warnings.warn("rolling_history.max_len > 1000 may cause memory issues, clamping to 1000")
            self.rolling_history.max_len = 1000
            
        # Validate attack resilience
        self.attack_resilience.energy_drain_resistance = max(0.0, min(1.0, self.attack_resilience.energy_drain_resistance))
        self.attack_resilience.max_drain_per_attacker_ratio = max(0.01, min(1.0, self.attack_resilience.max_drain_per_attacker_ratio))
        self.attack_resilience.jamming_detection_sensitivity = max(0.0, min(1.0, self.attack_resilience.jamming_detection_sensitivity))
        
        # Validate thresholds
        if self.proactive_interventions.anxiety_threshold <= 0:
            warnings.warn("anxiety_threshold must be > 0, clamping to 1.0")
            self.proactive_interventions.anxiety_threshold = 1.0
    
    def _log_applied_config(self):
        """Log the applied configuration for debugging and audit purposes."""
        logger.info("Applied configuration:")
        logger.info(f"  Trend analysis window: {self.trend_analysis.window}")
        logger.info(f"  Rolling history max length: {self.rolling_history.max_len}")
        logger.info(f"  Proactive interventions enabled: {self._get_enabled_interventions()}")
        logger.info(f"  Attack resilience - energy drain resistance: {self.attack_resilience.energy_drain_resistance}")
        logger.info(f"  Attack resilience - signal redundancy level: {self.attack_resilience.signal_redundancy_level}")
        logger.info(f"  Attack resilience - frequency hopping: {self.attack_resilience.frequency_hopping_enabled}")
    
    def _get_enabled_interventions(self) -> Dict[str, bool]:
        """Get a dictionary of enabled intervention subsystems."""
        interventions = self.proactive_interventions
        return {
            'anxiety': interventions.anxiety_enabled,
            'calm': interventions.calm_enabled,
            'energy': interventions.energy_enabled,
            'joy': interventions.joy_enabled,
            'grief': interventions.grief_enabled,
            'hope': interventions.hope_enabled,
            'anger': interventions.anger_enabled,
            'resilience': interventions.resilience_enabled,
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AdaptiveNeuralNetworkConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path) as f:
                config_dict = yaml.safe_load(f)
            
            return cls._from_dict(config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {yaml_path}: {e}")
    
    @classmethod  
    def from_json(cls, json_path: Union[str, Path]) -> "AdaptiveNeuralNetworkConfig":
        """Load configuration from JSON file."""
        import json
        
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        try:
            with open(json_path) as f:
                config_dict = json.load(f)
            
            return cls._from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file {json_path}: {e}")
    
    @classmethod
    def from_env(cls, prefix: str = "ANN_") -> "AdaptiveNeuralNetworkConfig":
        """Load configuration from environment variables."""
        config_dict = {}
        
        # Map environment variables to config structure
        env_mappings = {
            f"{prefix}TREND_WINDOW": ("trend_analysis", "window", int),
            f"{prefix}HISTORY_MAX_LEN": ("rolling_history", "max_len", int),
            f"{prefix}ANXIETY_ENABLED": ("proactive_interventions", "anxiety_enabled", bool),
            f"{prefix}CALM_ENABLED": ("proactive_interventions", "calm_enabled", bool),
            f"{prefix}ENERGY_ENABLED": ("proactive_interventions", "energy_enabled", bool),
            f"{prefix}ENERGY_DRAIN_RESISTANCE": ("attack_resilience", "energy_drain_resistance", float),
            f"{prefix}SIGNAL_REDUNDANCY": ("attack_resilience", "signal_redundancy_level", int),
            f"{prefix}FREQUENCY_HOPPING": ("attack_resilience", "frequency_hopping_enabled", bool),
            f"{prefix}LOG_LEVEL": (None, "log_level", str),
            f"{prefix}STRUCTURED_LOGGING": (None, "enable_structured_logging", bool),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert string values to appropriate types
                    if type_func == bool:
                        converted_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        converted_value = type_func(value)
                    
                    # Set nested configuration
                    if section is None:
                        config_dict[key] = converted_value
                    else:
                        if section not in config_dict:
                            config_dict[section] = {}
                        config_dict[section][key] = converted_value
                        
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Could not parse environment variable {env_var}={value}: {e}")
        
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "AdaptiveNeuralNetworkConfig":
        """Create configuration from dictionary, handling nested structures."""
        # Extract nested configurations
        trend_analysis_dict = config_dict.pop("trend_analysis", {})
        rolling_history_dict = config_dict.pop("rolling_history", {})
        proactive_interventions_dict = config_dict.pop("proactive_interventions", {})
        attack_resilience_dict = config_dict.pop("attack_resilience", {})
        environment_adaptation_dict = config_dict.pop("environment_adaptation", {})
        
        return cls(
            trend_analysis=TrendAnalysisConfig(**trend_analysis_dict),
            rolling_history=RollingHistoryConfig(**rolling_history_dict),
            proactive_interventions=ProactiveInterventionsConfig(**proactive_interventions_dict),
            attack_resilience=AttackResilienceConfig(**attack_resilience_dict),
            environment_adaptation=EnvironmentAdaptationConfig(**environment_adaptation_dict),
            **config_dict
        )
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "trend_analysis": {
                "window": self.trend_analysis.window,
                "enable_prediction": self.trend_analysis.enable_prediction,
                "prediction_steps": self.trend_analysis.prediction_steps,
            },
            "rolling_history": {
                "max_len": self.rolling_history.max_len,
            },
            "proactive_interventions": {
                "anxiety_enabled": self.proactive_interventions.anxiety_enabled,
                "calm_enabled": self.proactive_interventions.calm_enabled,
                "energy_enabled": self.proactive_interventions.energy_enabled,
                "joy_enabled": self.proactive_interventions.joy_enabled,
                "grief_enabled": self.proactive_interventions.grief_enabled,
                "hope_enabled": self.proactive_interventions.hope_enabled,
                "anger_enabled": self.proactive_interventions.anger_enabled,
                "resilience_enabled": self.proactive_interventions.resilience_enabled,
                "anxiety_threshold": self.proactive_interventions.anxiety_threshold,
                "max_help_signals_per_period": self.proactive_interventions.max_help_signals_per_period,
                "help_signal_cooldown": self.proactive_interventions.help_signal_cooldown,
                "anxiety_unload_capacity": self.proactive_interventions.anxiety_unload_capacity,
            },
            "attack_resilience": {
                "energy_drain_resistance": self.attack_resilience.energy_drain_resistance,
                "max_drain_per_attacker_ratio": self.attack_resilience.max_drain_per_attacker_ratio,
                "signal_redundancy_level": self.attack_resilience.signal_redundancy_level,
                "frequency_hopping_enabled": self.attack_resilience.frequency_hopping_enabled,
                "jamming_detection_sensitivity": self.attack_resilience.jamming_detection_sensitivity,
                "trust_growth_rate_limit": self.attack_resilience.trust_growth_rate_limit,
                "rapid_trust_threshold": self.attack_resilience.rapid_trust_threshold,
                "attack_detection_threshold": self.attack_resilience.attack_detection_threshold,
                "suspicious_events_max_len": self.attack_resilience.suspicious_events_max_len,
            },
            "environment_adaptation": {
                "stress_threshold_low": self.environment_adaptation.stress_threshold_low,
                "stress_threshold_high": self.environment_adaptation.stress_threshold_high,
                "adaptation_rate": self.environment_adaptation.adaptation_rate,
                "learning_rate_adaptation": self.environment_adaptation.learning_rate_adaptation,
            },
            "enable_structured_logging": self.enable_structured_logging,
            "log_level": self.log_level,
            "log_config_events": self.log_config_events,
            "log_intervention_events": self.log_intervention_events,
            "log_attack_events": self.log_attack_events,
        }
    
    def update(self, **kwargs) -> "AdaptiveNeuralNetworkConfig":
        """Create new configuration with updated parameters."""
        config_dict = self.to_dict()
        
        # Handle nested updates
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "trend_analysis.window"
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        
        return self._from_dict(config_dict)

    def log_event(self, event_type: str, message: str, **kwargs):
        """Log structured events based on configuration settings."""
        if not self.enable_structured_logging:
            return
            
        should_log = False
        if event_type == "config" and self.log_config_events:
            should_log = True
        elif event_type == "intervention" and self.log_intervention_events:
            should_log = True
        elif event_type == "attack" and self.log_attack_events:
            should_log = True
        elif event_type in ["trend_detection", "threshold_violation"]:
            should_log = True
            
        if should_log:
            log_data = {
                "event_type": event_type,
                "message": message,
                **kwargs
            }
            logger.info("Structured event: %s", log_data)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    from_env: bool = True,
    env_prefix: str = "ANN_",
    **overrides
) -> AdaptiveNeuralNetworkConfig:
    """
    Load configuration from multiple sources with precedence:
    1. Keyword arguments (highest precedence)
    2. Environment variables 
    3. Configuration file (lowest precedence)
    4. Defaults
    """
    config = AdaptiveNeuralNetworkConfig()
    
    # Load from file if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            config = AdaptiveNeuralNetworkConfig.from_yaml(config_path)
        elif config_path.suffix.lower() == '.json':
            config = AdaptiveNeuralNetworkConfig.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Apply environment variables
    if from_env:
        env_config = AdaptiveNeuralNetworkConfig.from_env(env_prefix)
        env_dict = env_config.to_dict()
        config_dict = config.to_dict()
        
        # Merge environment config
        for key, value in env_dict.items():
            if isinstance(value, dict) and key in config_dict:
                config_dict[key].update(value)
            else:
                config_dict[key] = value
        
        config = AdaptiveNeuralNetworkConfig._from_dict(config_dict)
    
    # Apply overrides
    if overrides:
        config = config.update(**overrides)
    
    return config


# Global config instance for convenience
_global_config: Optional[AdaptiveNeuralNetworkConfig] = None


def get_global_config() -> AdaptiveNeuralNetworkConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_global_config(config: AdaptiveNeuralNetworkConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def reset_global_config() -> None:
    """Reset the global configuration to defaults."""
    global _global_config
    _global_config = None