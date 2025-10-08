"""
Enhanced AI signal API using the new signal adapter system.
Provides AI system monitoring and integration capabilities.
"""

import logging

import requests

from .signal_adapter import (
    SignalAdapter,
    SignalMapping,
    SignalSource,
    SignalType,
    StateVariable,
)

logger = logging.getLogger(__name__)


def fetch_ai_signal(api_url, params=None):
    """
    Legacy function for backwards compatibility.
    Fetches AI-generated signal (e.g. model prediction, system status) from an external API.
    Returns a tuple: (signal_type, signal_energy)
    """
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Example: map AI prediction confidence to energy signal
        confidence = data.get("confidence", 0.5)
        energy_signal = confidence * 7
        return ("AI", energy_signal)
    except Exception as e:
        logger.error(f"[API ERROR] AI signal fetch failed: {e}")
        return ("AI", 1.0)


class AISignalManager:
    """Enhanced AI signal manager for AI system monitoring and integration"""

    def __init__(self, security_enabled: bool = True):
        self.adapter = SignalAdapter(security_enabled=security_enabled)
        self._setup_default_sources()

    def _setup_default_sources(self):
        """Setup common AI signal sources"""

        # ML model performance source
        model_source = SignalSource(
            name="ml_model_metrics",
            signal_type=SignalType.AI,
            api_url="https://api.example.com/ml/metrics",  # Placeholder
            mappings=[
                SignalMapping("accuracy", StateVariable.TRUST, "linear", 1.0, 0.0, 0.0, 1.0),
                SignalMapping("confidence", StateVariable.CALM, "sigmoid", 3.0, 0.0, 0.0, 4.0),
                SignalMapping("uncertainty", StateVariable.ANXIETY, "linear", 5.0, 0.0, 0.0, 8.0),
                SignalMapping("processing_speed", StateVariable.ENERGY, "logarithmic", 3.0, 0.0, 0.0, 15.0)
            ],
            privacy_level="protected",
            update_interval=60,
            data_retention_hours=24
        )

        # System health source
        system_source = SignalSource(
            name="ai_system_health",
            signal_type=SignalType.AI,
            api_url="https://api.example.com/system/health",  # Placeholder
            mappings=[
                SignalMapping("cpu_usage", StateVariable.ANXIETY, "threshold", 2.0, 80.0, 0.0, 6.0),
                SignalMapping("memory_usage", StateVariable.ANXIETY, "linear", 0.05, -4.0, 0.0, 4.0),
                SignalMapping("response_time", StateVariable.AROUSAL, "logarithmic", 0.5, 0.0, 0.0, 3.0),
                SignalMapping("uptime", StateVariable.TRUST, "sigmoid", 0.01, 0.0, 0.0, 1.0),
                SignalMapping("error_rate", StateVariable.ANXIETY, "linear", 10.0, 0.0, 0.0, 7.0)
            ],
            privacy_level="protected",
            update_interval=30,
            data_retention_hours=48
        )

        self.adapter.register_source(model_source)
        self.adapter.register_source(system_source)

    def fetch_ai_state_changes(self, source_name: str = None,
                              system_params: dict | None = None) -> dict[StateVariable, float]:
        """
        Fetch AI signal data and return mapped state variable changes.
        
        Args:
            source_name: Specific AI source to fetch from (None for all AI sources)
            system_params: System parameters for AI monitoring APIs
            
        Returns:
            Dictionary mapping state variables to their new values
        """
        params = system_params or {}

        if source_name:
            return self.adapter.fetch_signals(source_name, params)
        else:
            # Fetch from all AI sources and combine
            all_signals = self.adapter.fetch_all_signals(params)
            combined = {}

            for source_name, signals in all_signals.items():
                source = self.adapter.sources[source_name]
                if source.signal_type == SignalType.AI:
                    for var, value in signals.items():
                        if var in combined:
                            # Weight by recency and reliability
                            weight = 0.7 if source.error_count == 0 else 0.3
                            combined[var] = combined[var] * (1 - weight) + value * weight
                        else:
                            combined[var] = value

            return combined
