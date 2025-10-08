"""
Production infrastructure components for adaptive neural networks.

This module provides production-ready components for deployment, scaling,
and enterprise integration of adaptive neural networks.
"""

from .deployment import AutoScaler, KubernetesDeployment

# Import optional components with error handling
_SERVING_AVAILABLE = False
_DATABASE_AVAILABLE = False
_MESSAGING_AVAILABLE = False
_AUTH_AVAILABLE = False

try:
    from .serving import FastAPIServer, ModelServer
    _SERVING_AVAILABLE = True
except ImportError:
    pass

try:
    from .database import DatabaseManager, NoSQLManager, SQLManager
    _DATABASE_AVAILABLE = True
except ImportError:
    pass

try:
    from .messaging import KafkaProducer, MessageQueue, RabbitMQProducer
    _MESSAGING_AVAILABLE = True
except ImportError:
    pass

try:
    from .auth import AuthManager, JWTAuth, OAuth2Auth
    _AUTH_AVAILABLE = True
except ImportError:
    pass

# Base exports (always available)
__all__ = [
    "KubernetesDeployment",
    "AutoScaler",
]

# Add conditional exports
if _SERVING_AVAILABLE:
    __all__.extend(["ModelServer", "FastAPIServer"])

if _DATABASE_AVAILABLE:
    __all__.extend(["DatabaseManager", "SQLManager", "NoSQLManager"])

if _MESSAGING_AVAILABLE:
    __all__.extend(["MessageQueue", "KafkaProducer", "RabbitMQProducer"])

if _AUTH_AVAILABLE:
    __all__.extend(["AuthManager", "JWTAuth", "OAuth2Auth"])
