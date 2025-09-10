"""
Enhanced human signal API using the new signal adapter system.
Provides secure, multi-modal human data integration with privacy controls.
"""

import requests
import logging
from typing import Dict, Any, Optional
from .signal_adapter import SignalAdapter, SignalSource, SignalType, StateVariable, SignalMapping, ApiCredentials

logger = logging.getLogger(__name__)


def fetch_human_signal(api_url, params=None):
    """
    Legacy function for backwards compatibility.
    Fetches human-related data (e.g. sentiment, emotion, biometrics) from an external API.
    Returns a tuple: (signal_type, signal_energy)
    """
    try:
        response = requests.get(api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Example: map sentiment score to energy signal
        sentiment = data.get("sentiment_score", 0.5)
        energy_signal = sentiment * 10  # Scale as needed
        return ("human", energy_signal)
    except Exception as e:
        logger.error(f"[API ERROR] Human signal fetch failed: {e}")
        # Return baseline signal if API fails
        return ("human", 1.0)


class HumanSignalManager:
    """Enhanced human signal manager with privacy and security features"""
    
    def __init__(self, security_enabled: bool = True):
        self.adapter = SignalAdapter(security_enabled=security_enabled)
        self._setup_default_sources()
        
    def _setup_default_sources(self):
        """Setup common human signal sources"""
        
        # Emotion/sentiment source
        emotion_source = SignalSource(
            name="human_emotion",
            signal_type=SignalType.HUMAN,
            api_url="https://api.example.com/emotion",  # Placeholder
            mappings=[
                SignalMapping("happiness", StateVariable.CALM, "linear", 2.0, 0.0, 0.0, 5.0),
                SignalMapping("stress", StateVariable.ANXIETY, "linear", 3.0, 0.0, 0.0, 10.0),
                SignalMapping("energy_level", StateVariable.ENERGY, "linear", 1.5, 0.0, 0.0, 20.0),
                SignalMapping("trust_level", StateVariable.TRUST, "sigmoid", 1.0, 0.0, 0.0, 1.0),
                SignalMapping("emotional_valence", StateVariable.EMOTIONAL_VALENCE, "linear", 1.0, 0.0, -1.0, 1.0)
            ],
            privacy_level="private",  # Human data is private by default
            update_interval=5,
            data_retention_hours=1  # Minimal retention for privacy
        )
        
        # Biometric source
        biometric_source = SignalSource(
            name="human_biometrics",
            signal_type=SignalType.HUMAN,
            api_url="https://api.example.com/biometrics",  # Placeholder
            mappings=[
                SignalMapping("heart_rate", StateVariable.AROUSAL, "linear", 0.02, -1.5, 0.0, 2.0),
                SignalMapping("skin_conductance", StateVariable.ANXIETY, "logarithmic", 2.0, 0.0, 0.0, 8.0),
                SignalMapping("activity_level", StateVariable.ENERGY, "sigmoid", 1.5, 0.0, 0.0, 15.0),
                SignalMapping("sleep_quality", StateVariable.CALM, "linear", 1.0, 0.0, 0.0, 3.0)
            ],
            privacy_level="confidential",  # Biometric data is highly sensitive
            update_interval=10,
            data_retention_hours=0.5  # Very short retention
        )
        
        # Social interaction source  
        social_source = SignalSource(
            name="human_social",
            signal_type=SignalType.SOCIAL,
            api_url="https://api.example.com/social",  # Placeholder
            mappings=[
                SignalMapping("social_engagement", StateVariable.TRUST, "linear", 0.5, 0.2, 0.0, 1.0),
                SignalMapping("communication_frequency", StateVariable.ENERGY, "logarithmic", 3.0, 0.0, 0.0, 12.0),
                SignalMapping("group_cohesion", StateVariable.CALM, "sigmoid", 2.0, 0.0, 0.0, 4.0),
                SignalMapping("conflict_level", StateVariable.ANXIETY, "linear", 2.5, 0.0, 0.0, 7.0)
            ],
            privacy_level="protected",
            update_interval=30,
            data_retention_hours=2
        )
        
        self.adapter.register_source(emotion_source)
        self.adapter.register_source(biometric_source)
        self.adapter.register_source(social_source)
        
    def add_custom_source(self, source: SignalSource) -> None:
        """Add a custom human signal source"""
        if source.signal_type not in [SignalType.HUMAN, SignalType.SOCIAL]:
            logger.warning(f"Non-human signal type registered in HumanSignalManager: {source.signal_type}")
        self.adapter.register_source(source)
        
    def configure_credentials(self, source_name: str, credentials: ApiCredentials) -> None:
        """Configure authentication credentials for a source"""
        if source_name in self.adapter.sources:
            self.adapter.sources[source_name].credentials = credentials
            logger.info(f"Configured credentials for human signal source: {source_name}")
        else:
            logger.error(f"Cannot configure credentials for unknown source: {source_name}")
            
    def fetch_human_state_changes(self, source_name: str = None, params: Optional[Dict] = None) -> Dict[StateVariable, float]:
        """
        Fetch human signal data and return mapped state variable changes.
        
        Args:
            source_name: Specific source to fetch from (None for all human sources)
            params: Additional parameters for API calls
            
        Returns:
            Dictionary mapping state variables to their new values
        """
        if source_name:
            return self.adapter.fetch_signals(source_name, params)
        else:
            # Fetch from all human/social sources and combine
            all_signals = self.adapter.fetch_all_signals(params)
            combined = {}
            
            for source_name, signals in all_signals.items():
                source = self.adapter.sources[source_name]
                if source.signal_type in [SignalType.HUMAN, SignalType.SOCIAL]:
                    for var, value in signals.items():
                        if var in combined:
                            # Average multiple sources for same variable
                            combined[var] = (combined[var] + value) / 2.0
                        else:
                            combined[var] = value
                            
            return combined
            
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate a privacy compliance report for human data sources"""
        report = {
            "sources": {},
            "privacy_levels": {},
            "data_retention": {},
            "last_cleanup": None
        }
        
        for name, source in self.adapter.sources.items():
            if source.signal_type in [SignalType.HUMAN, SignalType.SOCIAL]:
                report["sources"][name] = {
                    "privacy_level": source.privacy_level,
                    "retention_hours": source.data_retention_hours,
                    "last_update": source.last_update,
                    "has_cached_data": source.cached_data is not None,
                    "error_count": source.error_count
                }
                
                if source.privacy_level not in report["privacy_levels"]:
                    report["privacy_levels"][source.privacy_level] = 0
                report["privacy_levels"][source.privacy_level] += 1
                
        return report
        
    def cleanup_private_data(self) -> None:
        """Clean up expired private data according to retention policies"""
        self.adapter.cleanup_expired_data()
        logger.info("Cleaned up expired human signal data")
        
    def anonymize_source_data(self, source_name: str) -> bool:
        """Anonymize cached data for a specific source"""
        if source_name not in self.adapter.sources:
            return False
            
        source = self.adapter.sources[source_name]
        if source.cached_data:
            # Remove or anonymize identifying information
            anonymized_data = {}
            for key, value in source.cached_data.items():
                if key in ["user_id", "session_id", "device_id", "ip_address"]:
                    anonymized_data[key] = "anonymized"
                else:
                    anonymized_data[key] = value
            source.cached_data = anonymized_data
            logger.info(f"Anonymized cached data for source: {source_name}")
            return True
            
        return False
