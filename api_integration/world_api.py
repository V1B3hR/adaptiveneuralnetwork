"""
Enhanced environmental signal API using the new signal adapter system.
Provides environmental and sensor data integration with proper error handling.
"""

import logging
from typing import Any, Dict, Optional

import requests

from .signal_adapter import (
    ApiCredentials,
    SignalAdapter,
    SignalMapping,
    SignalSource,
    SignalType,
    StateVariable,
)

logger = logging.getLogger(__name__)


def fetch_world_signal(api_url, params=None):
    """
    Legacy function for backwards compatibility.
    Fetches world/environmental signal (e.g. weather, news, sensor data) from an external API.
    Returns a tuple: (signal_type, signal_energy)
    """
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Example: map weather intensity or news sentiment to energy signal
        intensity = data.get("intensity", 0.3)
        energy_signal = intensity * 12
        return ("world", energy_signal)
    except Exception as e:
        logger.error(f"[API ERROR] World signal fetch failed: {e}")
        return ("world", 2.0)


class EnvironmentalSignalManager:
    """Enhanced environmental signal manager for multi-modal environmental data"""

    def __init__(self, security_enabled: bool = True):
        self.adapter = SignalAdapter(security_enabled=security_enabled)
        self._setup_default_sources()

    def _setup_default_sources(self):
        """Setup common environmental signal sources"""

        # Weather/climate source
        weather_source = SignalSource(
            name="weather_data",
            signal_type=SignalType.ENVIRONMENTAL,
            api_url="https://api.openweathermap.org/data/2.5/weather",  # Example
            mappings=[
                SignalMapping("temp", StateVariable.AROUSAL, "linear", 0.02, -0.7, -1.0, 1.0),
                SignalMapping("humidity", StateVariable.ANXIETY, "linear", 0.05, -2.5, 0.0, 3.0),
                SignalMapping(
                    "wind_speed", StateVariable.ENERGY, "logarithmic", 2.0, 0.0, 0.0, 10.0
                ),
                SignalMapping("pressure", StateVariable.CALM, "linear", 0.001, -1.0, 0.0, 2.0),
                SignalMapping("visibility", StateVariable.TRUST, "sigmoid", 0.1, 0.0, 0.0, 1.0),
            ],
            privacy_level="public",
            update_interval=300,  # 5 minutes
            data_retention_hours=24,
        )

        # Air quality source
        air_quality_source = SignalSource(
            name="air_quality",
            signal_type=SignalType.ENVIRONMENTAL,
            api_url="https://api.airvisual.com/v2/nearest_city",  # Example
            mappings=[
                SignalMapping("aqi", StateVariable.ANXIETY, "threshold", 3.0, 100.0, 0.0, 8.0),
                SignalMapping("pm25", StateVariable.ANXIETY, "linear", 0.1, 0.0, 0.0, 5.0),
                SignalMapping("pm10", StateVariable.ANXIETY, "linear", 0.05, 0.0, 0.0, 3.0),
                SignalMapping("o3", StateVariable.AROUSAL, "logarithmic", 0.5, 0.0, 0.0, 2.0),
            ],
            privacy_level="public",
            update_interval=900,  # 15 minutes
            data_retention_hours=48,
        )

        # News/social sentiment source
        news_source = SignalSource(
            name="news_sentiment",
            signal_type=SignalType.SOCIAL,
            api_url="https://api.newsapi.org/v2/top-headlines",  # Example
            mappings=[
                SignalMapping(
                    "sentiment_score",
                    StateVariable.EMOTIONAL_VALENCE,
                    "linear",
                    2.0,
                    -1.0,
                    -1.0,
                    1.0,
                ),
                SignalMapping(
                    "urgency_score", StateVariable.ANXIETY, "sigmoid", 3.0, 0.0, 0.0, 6.0
                ),
                SignalMapping(
                    "relevance_score", StateVariable.ATTENTION_FOCUS, "linear", 1.0, 0.0, 0.0, 2.0
                ),
                SignalMapping("trustworthiness", StateVariable.TRUST, "linear", 1.0, 0.0, 0.0, 1.0),
            ],
            privacy_level="public",
            update_interval=1800,  # 30 minutes
            data_retention_hours=72,
        )

        # Economic indicators source
        economic_source = SignalSource(
            name="economic_indicators",
            signal_type=SignalType.ECONOMIC,
            api_url="https://api.exchangerate.host/latest",  # Example
            mappings=[
                SignalMapping(
                    "volatility", StateVariable.ANXIETY, "logarithmic", 2.0, 0.0, 0.0, 4.0
                ),
                SignalMapping("growth_rate", StateVariable.ENERGY, "linear", 10.0, 0.0, 0.0, 15.0),
                SignalMapping("stability_index", StateVariable.CALM, "sigmoid", 2.0, 0.0, 0.0, 3.0),
                SignalMapping(
                    "confidence_index", StateVariable.TRUST, "linear", 0.01, 0.0, 0.0, 1.0
                ),
            ],
            privacy_level="public",
            update_interval=3600,  # 1 hour
            data_retention_hours=168,  # 1 week
        )

        self.adapter.register_source(weather_source)
        self.adapter.register_source(air_quality_source)
        self.adapter.register_source(news_source)
        self.adapter.register_source(economic_source)

    def add_sensor_source(
        self, name: str, api_url: str, mappings: list, credentials: Optional[ApiCredentials] = None
    ) -> None:
        """Add a custom sensor/environmental source"""
        sensor_source = SignalSource(
            name=name,
            signal_type=SignalType.SENSOR,
            api_url=api_url,
            mappings=mappings,
            credentials=credentials,
            privacy_level="protected",
            update_interval=60,
            data_retention_hours=24,
        )
        self.adapter.register_source(sensor_source)

    def configure_api_key(self, source_name: str, api_key: str) -> None:
        """Configure API key for a specific environmental source"""
        if source_name in self.adapter.sources:
            if not self.adapter.sources[source_name].credentials:
                self.adapter.sources[source_name].credentials = ApiCredentials()
            self.adapter.sources[source_name].credentials.api_key = api_key
            logger.info(f"Configured API key for environmental source: {source_name}")
        else:
            logger.error(f"Cannot configure API key for unknown source: {source_name}")

    def fetch_environmental_state_changes(
        self, source_name: str = None, location_params: Optional[Dict] = None
    ) -> Dict[StateVariable, float]:
        """
        Fetch environmental signal data and return mapped state variable changes.

        Args:
            source_name: Specific source to fetch from (None for all environmental sources)
            location_params: Location parameters (lat, lon, city, etc.) for APIs that need them

        Returns:
            Dictionary mapping state variables to their new values
        """
        params = location_params or {}

        if source_name:
            return self.adapter.fetch_signals(source_name, params)
        else:
            # Fetch from all environmental sources and combine
            all_signals = self.adapter.fetch_all_signals(params)
            combined = {}

            for source_name, signals in all_signals.items():
                source = self.adapter.sources[source_name]
                if source.signal_type in [
                    SignalType.ENVIRONMENTAL,
                    SignalType.SENSOR,
                    SignalType.ECONOMIC,
                ]:
                    for var, value in signals.items():
                        if var in combined:
                            # Weight by signal strength and combine
                            combined[var] = (combined[var] + value) / 2.0
                        else:
                            combined[var] = value

            return combined

    def get_environmental_summary(self) -> Dict[str, Any]:
        """Get a summary of current environmental conditions"""
        summary = {"sources": {}, "last_updates": {}, "alert_conditions": []}

        for name, source in self.adapter.sources.items():
            if source.signal_type in [
                SignalType.ENVIRONMENTAL,
                SignalType.SENSOR,
                SignalType.ECONOMIC,
            ]:
                status = self.adapter.get_source_status(name)
                summary["sources"][name] = status
                summary["last_updates"][name] = status.get("last_update", 0)

                # Check for alert conditions
                if source.cached_data:
                    for mapping in source.mappings:
                        if (
                            mapping.source_field in source.cached_data
                            and mapping.target_variable == StateVariable.ANXIETY
                        ):
                            raw_value = source.cached_data[mapping.source_field]
                            transformed_value = mapping.apply_transformation(float(raw_value))

                            if transformed_value > 5.0:  # High anxiety threshold
                                summary["alert_conditions"].append(
                                    {
                                        "source": name,
                                        "field": mapping.source_field,
                                        "value": raw_value,
                                        "anxiety_impact": transformed_value,
                                    }
                                )

        return summary

    def setup_emergency_protocols(self) -> None:
        """Setup emergency response protocols for extreme environmental conditions"""

        def emergency_handler(source, error):
            logger.warning(f"Emergency protocol activated for {source.name}: {error}")
            # Return safe values during emergencies
            return {
                StateVariable.ANXIETY: 2.0,  # Moderate anxiety
                StateVariable.ENERGY: 3.0,  # Conserve energy
                StateVariable.CALM: 1.0,  # Reduced calm
                StateVariable.TRUST: 0.3,  # Reduced trust
            }

        # Register emergency handlers for critical sources
        for source_name in ["weather_data", "air_quality"]:
            self.adapter.register_error_handler(source_name, emergency_handler)
