# Enhanced Adaptive Neural Network Features

This document describes the new features implemented to address the requirements for expanded external signal integration, anxiety overwhelm safety protocols, time series tracking, and enhanced security.

## 1. Expanded External Signal Absorption Subsystem

### Overview
The enhanced signal adapter system provides comprehensive multi-modal external data integration with proper error handling, security, and privacy controls.

### Key Components

#### Signal Adapter (`api_integration/signal_adapter.py`)
- **Multi-source support**: Human, AI, Environmental, Sensor, Social, Economic
- **Flexible signal mapping**: Linear, logarithmic, sigmoid, and threshold transformations
- **Security features**: Authentication, data integrity checks, request signing
- **Privacy controls**: Data classification, retention policies, anonymization

#### Signal Managers
- **HumanSignalManager** (`api_integration/human_api.py`): Handles human emotion, biometric, and social data with strict privacy controls
- **EnvironmentalSignalManager** (`api_integration/world_api.py`): Processes weather, air quality, news sentiment, and economic indicators
- **AISignalManager** (`api_integration/ai_api.py`): Monitors AI system health, model performance, and decision confidence

### Usage Example

```python
from api_integration.signal_adapter import SignalAdapter, SignalSource, SignalType, StateVariable, SignalMapping
from api_integration.human_api import HumanSignalManager

# Create human signal manager
human_manager = HumanSignalManager(security_enabled=True)

# Configure API credentials
from api_integration.signal_adapter import ApiCredentials
credentials = ApiCredentials(api_key="your-api-key")
human_manager.configure_credentials("human_emotion", credentials)

# Fetch human state changes
state_changes = human_manager.fetch_human_state_changes()
# Returns: {StateVariable.ANXIETY: 2.5, StateVariable.CALM: 1.8, ...}
```

### Signal Mapping Configuration

```python
# Create custom signal source
source = SignalSource(
    name="custom_biometrics",
    signal_type=SignalType.HUMAN,
    api_url="https://api.myservice.com/biometrics",
    mappings=[
        SignalMapping("heart_rate", StateVariable.AROUSAL, "linear", 0.02, -1.5, 0.0, 2.0),
        SignalMapping("stress_level", StateVariable.ANXIETY, "sigmoid", 3.0, 0.0, 0.0, 8.0)
    ],
    privacy_level="confidential",
    data_retention_hours=1  # Minimal retention for privacy
)
```

## 2. Anxiety Overwhelm Safety Protocol

### Overview
Implements a comprehensive safety protocol for managing anxiety overwhelm in nodes, including help signal broadcasting, anxiety unloading, and cooperative recovery behaviors.

### Key Features

#### Anxiety Monitoring
- **Threshold-based detection**: Configurable anxiety thresholds trigger help protocols
- **Cooldown mechanisms**: Prevents spam help requests
- **Rate limiting**: Limits help signals per time period

#### Help Signal Protocol
- **Automatic help requests**: Overwhelmed nodes automatically request help from trusted nearby nodes
- **Trust-based filtering**: Only trusted nodes can provide/receive help
- **Energy requirements**: Helpers must have sufficient energy and low anxiety

#### Message Format
```python
help_signal = {
    "type": "anxiety_help_request",
    "anxiety_level": 9.5,
    "energy_level": 8.0,
    "urgency": 0.95,
    "requesting_node": 1,
    "timestamp": 1234567890,
    "unload_capacity_needed": 2.0
}
```

### Usage in AliveLoopNode

```python
# Create node with anxiety monitoring
node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1)

# Configure anxiety thresholds
node.anxiety_threshold = 8.0  # Trigger help at anxiety level 8.0
node.max_help_signals_per_period = 3
node.help_signal_cooldown = 10

# Check anxiety status
status = node.get_anxiety_status()
print(f"Anxiety level: {status['anxiety_level']}")
print(f"Is overwhelmed: {status['is_overwhelmed']}")
print(f"Can send help: {status['can_send_help']}")

# Manually send help signal if needed
nearby_nodes = [other_node1, other_node2]
if node.check_anxiety_overwhelm():
    helped_by = node.send_help_signal(nearby_nodes)
```

### Privacy and Trust Boundaries
- Help signals respect existing trust networks
- Minimum trust thresholds for help provision
- Memory of help interactions affects future trust levels
- Privacy classification of help requests

## 3. Time Series Tracking and Exposure

### Overview
Comprehensive time series tracking system for key node state variables with querying, visualization, and export capabilities.

### Key Features

#### Data Storage
- **In-memory storage**: Fast access to recent data
- **SQLite persistence**: Long-term storage with indexing
- **Automatic cleanup**: Configurable retention policies

#### Tracked Variables
- Energy, anxiety, calm, trust levels
- Communication activity and trust network metrics
- Phase states and emotional valence
- Custom node state variables

### Usage Example

```python
from core.time_series_tracker import TimeSeriesTracker, TimeSeriesQuery, track_node_automatically

# Create tracker
tracker = TimeSeriesTracker(persist_to_disk=True, db_path="network_data.db")

# Automatic tracking of nodes
for node in network.nodes:
    track_node_automatically(tracker, node)

# Query specific data
query = TimeSeriesQuery(
    node_ids=[1, 2, 3],
    variables=["anxiety", "energy"],
    start_time=time.time() - 3600,  # Last hour
    end_time=time.time()
)

results = tracker.query(query)

# Visualize data
fig = tracker.visualize_node_variables(
    node_id=1,
    variables=["anxiety", "energy", "calm"],
    time_range_hours=24,
    save_path="node_1_analysis.png"
)

# Export data
tracker.export_data(query, format="json", output_path="network_timeseries")
```

### Visualization Capabilities

```python
# Compare multiple nodes
fig = tracker.compare_nodes(
    node_ids=[1, 2, 3],
    variable="anxiety",
    time_range_hours=12,
    save_path="anxiety_comparison.png"
)

# Network-wide visualization
network.visualize_network_timeseries(time_range_hours=24, save_path="network_overview.png")
```

### Statistics and Analytics

```python
# Get comprehensive statistics
stats = tracker.get_statistics()
print(f"Total data points: {stats['total_points']}")
print(f"Unique nodes: {len(stats['unique_nodes'])}")
print(f"Time range: {stats['time_range_seconds']} seconds")

# Get latest values for a node
latest = tracker.get_latest_values(node_id=1, variables=["anxiety", "energy"])
```

## 4. Security and Privacy Enhancements

### Overview
Comprehensive security and privacy framework for external API integration with authentication, authorization, data integrity, and privacy controls.

### Security Features

#### Authentication and Authorization
- **API key management**: Secure storage and rotation of API keys
- **Token-based authentication**: Support for Bearer tokens and custom headers
- **Request signing**: HMAC-SHA256 signatures for data integrity
- **Multi-factor authentication**: Support for username/password + API key

#### Data Integrity
- **Request signatures**: Cryptographic signatures prevent tampering
- **Response validation**: Automatic verification of expected data structures
- **Error detection**: Comprehensive error handling and logging
- **Retry mechanisms**: Configurable retry logic with exponential backoff

### Privacy Controls

#### Data Classification
- **Public**: Weather, economic indicators (long retention)
- **Protected**: AI system metrics, sensor data (medium retention)
- **Private**: Human emotions, social data (short retention)
- **Confidential**: Biometric data (minimal retention)

#### Retention Policies
```python
# Configure retention for different privacy levels
source.data_retention_hours = {
    "public": 168,      # 1 week
    "protected": 24,    # 1 day
    "private": 2,       # 2 hours
    "confidential": 0.5 # 30 minutes
}
```

#### Data Anonymization
```python
# Anonymize sensitive data
human_manager.anonymize_source_data("human_emotion")

# Privacy compliance report
report = human_manager.get_privacy_report()
print(f"Privacy levels: {report['privacy_levels']}")
print(f"Data retention: {report['data_retention']}")
```

### Security Configuration

```python
# Enable comprehensive security
adapter = SignalAdapter(security_enabled=True)

# Configure secure credentials
credentials = ApiCredentials(
    api_key="your-api-key",
    secret_key="your-secret-key",  # For request signing
    custom_headers={"X-Client-ID": "adaptive-neural-network"}
)

# Register secure source
source = SignalSource(
    name="secure_human_data",
    signal_type=SignalType.HUMAN,
    api_url="https://secure-api.example.com/data",
    mappings=[...],
    credentials=credentials,
    privacy_level="confidential",
    integrity_check=True
)
```

### Audit and Compliance

```python
# Memory access audit logging
memory = Memory(
    content=sensitive_data,
    classification="private",
    audit_log=[]
)

# Access with logging
content = memory.access(accessor_node_id=2)
# Logs: "accessed_by_2_at_1234567890"

# Security compliance check
from core.ai_ethics import audit_decision
ethics_result = audit_decision({
    "action": "data_sharing",
    "preserve_life": True,
    "absolute_honesty": True,
    "privacy": True,
    "human_authority": True,
    "proportionality": True
})
```

## Integration with Enhanced Network

The `TunedAdaptiveFieldNetwork` class integrates all these features:

```python
# Create enhanced network
network = TunedAdaptiveFieldNetwork(
    nodes=nodes,
    capacitors=capacitors,
    enable_time_series=True,
    enable_security=True
)

# Step with external signal integration
network.step(location_params={"lat": 40.7128, "lon": -74.0060})

# Get comprehensive status
status = network.get_network_status()
print(f"Network anxiety: {status['performance_metrics']['average_network_anxiety']}")
print(f"Help signals sent: {status['performance_metrics']['total_help_signals']}")

# Export all network data
network.export_network_data(format="json", output_path="full_network_export")

# Cleanup resources
network.cleanup_resources()
```

## Best Practices

### Security
1. Always enable security for production deployments
2. Rotate API keys regularly
3. Use appropriate privacy classifications
4. Monitor audit logs for suspicious access patterns
5. Implement proper error handling for API failures

### Performance
1. Configure appropriate retention policies to manage storage
2. Use memory storage for frequently accessed recent data
3. Implement proper rate limiting for external API calls
4. Clean up expired data regularly

### Privacy
1. Minimize data retention for sensitive information
2. Anonymize data when possible
3. Respect user consent and data sovereignty
4. Document data flows and access patterns

### Monitoring
1. Track network stability metrics
2. Monitor anxiety patterns across nodes
3. Analyze help signal effectiveness
4. Review external signal integration health

This enhanced system provides a robust, secure, and privacy-aware foundation for adaptive neural network operations with comprehensive external signal integration and cooperative anxiety management.