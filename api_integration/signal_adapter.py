"""
Enhanced signal adapter system for multi-modal external data integration.
Supports multiple signal sources and maps to various node state variables.
"""

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

# Setup logger
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of external signals"""
    HUMAN = "human"
    AI = "ai"
    ENVIRONMENTAL = "environmental"
    SENSOR = "sensor"
    SOCIAL = "social"
    ECONOMIC = "economic"


class StateVariable(Enum):
    """Node state variables that can be affected by external signals"""
    ENERGY = "energy"
    ANXIETY = "anxiety"
    CALM = "calm"
    TRUST = "trust"
    PHASE = "phase"
    ATTENTION_FOCUS = "attention_focus"
    EMOTIONAL_VALENCE = "emotional_valence"
    AROUSAL = "arousal"


@dataclass
class SignalMapping:
    """Defines how external signal data maps to node state variables"""
    source_field: str  # Field name in the external data
    target_variable: StateVariable  # Node state variable to modify
    transformation: str = "linear"  # "linear", "logarithmic", "sigmoid", "threshold"
    scaling_factor: float = 1.0
    offset: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def apply_transformation(self, value: float) -> float:
        """Apply the specified transformation to the input value"""
        if self.transformation == "linear":
            result = value * self.scaling_factor + self.offset
        elif self.transformation == "logarithmic":
            import math
            result = math.log(max(0.001, value)) * self.scaling_factor + self.offset
        elif self.transformation == "sigmoid":
            import math
            result = 1 / (1 + math.exp(-value * self.scaling_factor)) + self.offset
        elif self.transformation == "threshold":
            result = self.scaling_factor if value > self.offset else 0.0
        else:
            result = value * self.scaling_factor + self.offset

        # Apply bounds if specified
        if self.min_value is not None:
            result = max(self.min_value, result)
        if self.max_value is not None:
            result = min(self.max_value, result)

        return result


@dataclass
class ApiCredentials:
    """Authentication credentials for external APIs"""
    api_key: str | None = None
    secret_key: str | None = None
    token: str | None = None
    username: str | None = None
    password: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class SignalSource:
    """Configuration for an external signal source"""
    name: str
    signal_type: SignalType
    api_url: str
    mappings: list[SignalMapping]
    credentials: ApiCredentials | None = None
    update_interval: int = 10  # seconds
    timeout: int = 5
    retry_attempts: int = 3
    privacy_level: str = "public"  # "public", "protected", "private", "confidential"
    data_retention_hours: int = 24
    integrity_check: bool = True
    last_update: float = 0.0
    cached_data: dict[str, Any] | None = None
    error_count: int = 0

    def is_update_needed(self) -> bool:
        """Check if data needs to be refreshed"""
        return time.time() - self.last_update >= self.update_interval

    def verify_integrity(self, data: dict[str, Any]) -> bool:
        """Verify data integrity using checksums or signatures"""
        if not self.integrity_check:
            return True

        # Basic integrity check - verify expected fields exist
        required_fields = [mapping.source_field for mapping in self.mappings]
        return all(field in data for field in required_fields)


class SignalAdapter:
    """Enhanced signal adapter for external data integration"""

    def __init__(self, security_enabled: bool = True):
        self.sources: dict[str, SignalSource] = {}
        self.security_enabled = security_enabled
        self.session = requests.Session()
        self.error_handlers: dict[str, callable] = {}

        # Default error handler
        self.error_handlers["default"] = self._default_error_handler

    def register_source(self, source: SignalSource) -> None:
        """Register a new signal source"""
        self.sources[source.name] = source
        logger.info(f"Registered signal source: {source.name} ({source.signal_type.value})")

    def register_error_handler(self, source_name: str, handler: callable) -> None:
        """Register custom error handler for a specific source"""
        self.error_handlers[source_name] = handler

    def _authenticate_request(self, source: SignalSource) -> dict[str, str]:
        """Prepare authentication headers for API request"""
        headers = {"Content-Type": "application/json"}

        if source.credentials:
            if source.credentials.api_key:
                headers["X-API-Key"] = source.credentials.api_key
            if source.credentials.token:
                headers["Authorization"] = f"Bearer {source.credentials.token}"
            headers.update(source.credentials.custom_headers)

        return headers

    def _sign_request(self, source: SignalSource, data: str) -> str:
        """Create request signature for data integrity"""
        if not source.credentials or not source.credentials.secret_key:
            return ""

        signature = hmac.new(
            source.credentials.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _fetch_data(self, source: SignalSource, params: dict | None = None) -> dict[str, Any] | None:
        """Fetch data from external API with proper authentication and error handling"""
        try:
            headers = self._authenticate_request(source)

            # Add integrity signature if enabled
            if self.security_enabled and source.credentials and source.credentials.secret_key:
                request_data = json.dumps(params or {})
                signature = self._sign_request(source, request_data)
                headers["X-Signature"] = signature

            response = self.session.get(
                source.api_url,
                params=params,
                headers=headers,
                timeout=source.timeout
            )
            response.raise_for_status()

            data = response.json()

            # Verify data integrity
            if not source.verify_integrity(data):
                logger.warning(f"Data integrity check failed for source: {source.name}")
                return None

            # Cache successful response
            source.cached_data = data
            source.last_update = time.time()
            source.error_count = 0

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {source.name}: {e}")
            source.error_count += 1
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from {source.name}: {e}")
            source.error_count += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data from {source.name}: {e}")
            source.error_count += 1
            return None

    def _default_error_handler(self, source: SignalSource, error: Exception) -> dict[StateVariable, float]:
        """Default error handler returns safe baseline values"""
        return {
            StateVariable.ENERGY: 1.0,
            StateVariable.ANXIETY: 0.1,
            StateVariable.CALM: 0.5,
            StateVariable.TRUST: 0.5
        }

    def fetch_signals(self, source_name: str, params: dict | None = None) -> dict[StateVariable, float]:
        """Fetch and process signals from a specific source"""
        if source_name not in self.sources:
            logger.error(f"Unknown signal source: {source_name}")
            return {}

        source = self.sources[source_name]

        # Check if update is needed
        if not source.is_update_needed() and source.cached_data:
            data = source.cached_data
        else:
            data = self._fetch_data(source, params)

        if data is None:
            # Use error handler
            handler = self.error_handlers.get(source_name, self.error_handlers["default"])
            return handler(source, Exception("Data fetch failed"))

        # Apply mappings to convert external data to node state variables
        results = {}
        for mapping in source.mappings:
            if mapping.source_field in data:
                raw_value = data[mapping.source_field]
                if isinstance(raw_value, (int, float)):
                    transformed_value = mapping.apply_transformation(float(raw_value))
                    results[mapping.target_variable] = transformed_value
                else:
                    logger.warning(f"Non-numeric value for {mapping.source_field}: {raw_value}")

        return results

    def fetch_all_signals(self, params: dict | None = None) -> dict[str, dict[StateVariable, float]]:
        """Fetch signals from all registered sources"""
        all_signals = {}
        for source_name in self.sources:
            signals = self.fetch_signals(source_name, params)
            if signals:
                all_signals[source_name] = signals
        return all_signals

    def get_source_status(self, source_name: str) -> dict[str, Any]:
        """Get status information for a signal source"""
        if source_name not in self.sources:
            return {"error": "Source not found"}

        source = self.sources[source_name]
        return {
            "name": source.name,
            "type": source.signal_type.value,
            "last_update": source.last_update,
            "error_count": source.error_count,
            "cached_data_available": source.cached_data is not None,
            "update_needed": source.is_update_needed(),
            "privacy_level": source.privacy_level
        }

    def cleanup_expired_data(self) -> None:
        """Remove expired cached data based on retention policies"""
        current_time = time.time()
        for source in self.sources.values():
            if source.cached_data and source.data_retention_hours > 0:
                expiry_time = source.last_update + (source.data_retention_hours * 3600)
                if current_time > expiry_time:
                    source.cached_data = None
                    logger.info(f"Expired cached data for source: {source.name}")


# Pre-configured signal sources for common use cases
def create_human_emotion_source(api_url: str, api_key: str = None) -> SignalSource:
    """Create a pre-configured source for human emotion data"""
    return SignalSource(
        name="human_emotion",
        signal_type=SignalType.HUMAN,
        api_url=api_url,
        mappings=[
            SignalMapping("happiness", StateVariable.CALM, "linear", 2.0, 0.0, 0.0, 5.0),
            SignalMapping("stress", StateVariable.ANXIETY, "linear", 3.0, 0.0, 0.0, 10.0),
            SignalMapping("energy_level", StateVariable.ENERGY, "linear", 1.5, 0.0, 0.0, 20.0),
            SignalMapping("trust_level", StateVariable.TRUST, "sigmoid", 1.0, 0.0, 0.0, 1.0)
        ],
        credentials=ApiCredentials(api_key=api_key) if api_key else None,
        privacy_level="private",
        update_interval=5
    )


def create_environmental_source(api_url: str, api_key: str = None) -> SignalSource:
    """Create a pre-configured source for environmental data"""
    return SignalSource(
        name="environmental",
        signal_type=SignalType.ENVIRONMENTAL,
        api_url=api_url,
        mappings=[
            SignalMapping("temperature", StateVariable.AROUSAL, "linear", 0.1, -2.0, -1.0, 1.0),
            SignalMapping("air_quality", StateVariable.ANXIETY, "threshold", 5.0, 50.0, 0.0, 10.0),
            SignalMapping("noise_level", StateVariable.ANXIETY, "logarithmic", 0.5, 0.0, 0.0, 5.0),
            SignalMapping("light_intensity", StateVariable.ENERGY, "sigmoid", 2.0, 0.0, 0.0, 15.0)
        ],
        credentials=ApiCredentials(api_key=api_key) if api_key else None,
        privacy_level="public",
        update_interval=30
    )


def create_ai_system_source(api_url: str, api_key: str = None) -> SignalSource:
    """Create a pre-configured source for AI system metrics"""
    return SignalSource(
        name="ai_system",
        signal_type=SignalType.AI,
        api_url=api_url,
        mappings=[
            SignalMapping("model_confidence", StateVariable.TRUST, "linear", 1.0, 0.0, 0.0, 1.0),
            SignalMapping("system_load", StateVariable.ANXIETY, "linear", 0.1, 0.0, 0.0, 3.0),
            SignalMapping("prediction_accuracy", StateVariable.CALM, "linear", 2.0, 0.0, 0.0, 4.0),
            SignalMapping("processing_speed", StateVariable.ENERGY, "logarithmic", 5.0, 0.0, 0.0, 25.0)
        ],
        credentials=ApiCredentials(api_key=api_key) if api_key else None,
        privacy_level="protected",
        update_interval=15
    )


def create_sensor_source(api_url: str, api_key: str = None) -> SignalSource:
    """Create a pre-configured source for sensor data"""
    return SignalSource(
        name="sensor_data",
        signal_type=SignalType.SENSOR,
        api_url=api_url,
        mappings=[
            SignalMapping("motion_detected", StateVariable.AROUSAL, "threshold", 2.0, 0.5, 0.0, 2.0),
            SignalMapping("proximity", StateVariable.ANXIETY, "linear", -0.5, 3.0, 0.0, 3.0),
            SignalMapping("battery_level", StateVariable.ENERGY, "linear", 0.2, 0.0, 0.0, 20.0),
            SignalMapping("signal_strength", StateVariable.TRUST, "sigmoid", 2.0, -0.5, 0.0, 1.0)
        ],
        credentials=ApiCredentials(api_key=api_key) if api_key else None,
        privacy_level="protected",
        update_interval=20
    )
