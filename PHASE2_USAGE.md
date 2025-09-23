# Phase 2 Enhancements Usage Examples

## Quick Start

```python
from core.alive_node import AliveLoopNode

# Create a node with enhanced trust and energy systems
node = AliveLoopNode((0, 0), (0.1, 0.1), node_id=0)

# The trust attribute is now automatically maintained
print(f"Node trust level: {node.trust}")

# Energy system with hardened thresholds
print(f"Emergency threshold: {node.emergency_energy_threshold}")
```

## Trust Network Visualization

```python
# Get visualization data for your dashboard
graph_data = node.get_trust_network_visualization()
print(f"Nodes: {len(graph_data['nodes'])}")
print(f"Edges: {len(graph_data['edges'])}")

# Get comprehensive metrics
metrics = node.get_trust_network_metrics()
print(f"Network resilience: {metrics['network_resilience']:.2%}")
print(f"Suspicious ratio: {metrics['suspicious_ratio']:.2%}")
```

## Health Monitoring

```python
# Monitor network health with alerts
health_report = node.monitor_trust_network_health()
print(f"Overall health: {health_report['overall_health']:.3f}")

for alert in health_report['alerts']:
    print(f"{alert['level']}: {alert['message']}")
```

## Distributed Consensus

```python
# Create multiple nodes
nodes = [AliveLoopNode((i, i), (0.1, 0.1), node_id=i) for i in range(5)]

# Initiate consensus vote
vote_request = nodes[0].trust_network_system.initiate_consensus_vote(subject_node_id=2)

# Collect responses
responses = []
for voter in nodes[1:]:
    if voter.node_id != 2:  # Don't ask subject to vote on themselves
        response = voter.respond_to_trust_vote(vote_request)
        if response:
            responses.append(response)

# Process consensus
result = nodes[0].trust_network_system.process_consensus_vote(vote_request, responses)
print(f"Consensus trust: {result['consensus_trust']:.3f}")
print(f"Recommendation: {result['recommendation']}")
```

## Byzantine Fault Tolerance Testing

```python
# Run stress test to validate resilience
results = node.run_byzantine_stress_test(
    malicious_ratio=0.33,  # 33% malicious nodes
    num_simulations=50
)

print(f"Resilience score: {results['resilience_score']:.3f}")
print(f"Detection rate: {results['detection_rate']:.3f}")
print(f"False positive rate: {results['false_positive_rate']:.3f}")
```

## Energy System Hardening

```python
# Energy conservation activates automatically
node.energy = 0.15  # Low energy
node.adaptive_energy_allocation()

if node.emergency_mode:
    print("Emergency conservation activated!")
    print(f"Communication range reduced: {node.communication_range}")

# Request energy from trusted network
shared_energy = node.request_distributed_energy(2.0)
print(f"Received {shared_energy} energy from network")
```

## Run the Demo

```bash
python demos/demo_phase2_enhancements.py
```

This will showcase all Phase 2 enhancements with a rich interactive demonstration.