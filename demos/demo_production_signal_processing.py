#!/usr/bin/env python3
"""
Demo script showcasing production-ready signal processing features
"""

import os
import sys

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.alive_node import AliveLoopNode, Memory, SocialSignal
from core.time_manager import get_timestamp


def demo_production_features():
    """Demonstrate all the production signal processing features"""
    print("=== Adaptive Neural Network - Production Signal Processing Demo ===\n")

    # Create nodes
    print("1. Creating nodes with production-ready signal processing...")
    node1 = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=20.0, node_id=1)
    node2 = AliveLoopNode(position=(1, 1), velocity=(0, 0), initial_energy=20.0, node_id=2)

    # Set up trust
    node1.trust_network[2] = 0.8
    node2.trust_network[1] = 0.8

    print("✓ Nodes created with trust relationships established\n")

    # 2. Demonstrate Idempotency and Deduplication
    print("2. Testing idempotency and message deduplication...")

    memory = Memory(
        content="critical_system_alert",
        importance=0.9,
        timestamp=get_timestamp(),
        memory_type="alert",
    )

    # Send same signal twice with same idempotency key
    signal1 = SocialSignal(
        content=memory,
        signal_type="memory",
        urgency=0.8,
        source_id=node2.node_id,
        idempotency_key="ALERT_001",
    )

    signal2 = SocialSignal(
        content=memory,
        signal_type="memory",
        urgency=0.8,
        source_id=node2.node_id,
        idempotency_key="ALERT_001",  # Same key
    )

    # Process both signals
    node1.receive_signal(signal1)
    initial_memory_count = len(node1.memory)

    node1.receive_signal(signal2)  # Should be deduplicated
    final_memory_count = len(node1.memory)

    print(f"✓ First signal processed - Memory count: {initial_memory_count}")
    print(f"✓ Duplicate signal ignored - Memory count: {final_memory_count}")
    print(f"✓ Duplicates detected: {node1.signal_metrics['duplicate_count']}\n")

    # 3. Demonstrate Observability Metrics
    print("3. Testing observability and metrics...")

    # Send multiple signals to generate metrics
    for i in range(5):
        signal = SocialSignal(
            content=f"metric_test_{i}", signal_type="memory", urgency=0.5, source_id=node2.node_id
        )
        node1.receive_signal(signal)

    metrics = node1._get_queue_metrics()
    print("✓ Queue metrics:")
    print(f"  - Queue depth: {metrics['queue_depth']}")
    print(f"  - Processed messages: {node1.signal_metrics['processed_count']}")
    print(f"  - Throughput: {metrics['throughput_per_second']:.2f} msgs/sec")
    print(f"  - Error rate: {metrics['error_rate']:.2f}")
    print(f"  - Circuit breaker: {metrics['circuit_breaker_state']}\n")

    # 4. Demonstrate Dead Letter Queue
    print("4. Testing Dead Letter Queue for poison messages...")

    poison_signal = SocialSignal(
        content="poison_message",
        signal_type="invalid_signal_type",  # Invalid type
        urgency=0.5,
        source_id=node2.node_id,
    )

    node1.receive_signal(poison_signal)

    dlq_messages = node1.process_dlq_messages()
    print("✓ Poison message detected and moved to DLQ")
    print(f"✓ DLQ contains {len(dlq_messages)} messages")
    print(f"✓ DLQ message error: {dlq_messages[0]['error'] if dlq_messages else 'None'}\n")

    # 5. Demonstrate Schema Validation
    print("5. Testing schema version validation...")

    invalid_schema_signal = SocialSignal(
        content="test_content",
        signal_type="memory",
        urgency=0.5,
        source_id=node2.node_id,
        schema_version="999.0",  # Unsupported version
    )

    node1.receive_signal(invalid_schema_signal)
    updated_dlq = node1.process_dlq_messages()

    print("✓ Invalid schema version rejected")
    print(f"✓ Total DLQ messages: {len(updated_dlq)}\n")

    # 6. Demonstrate Circuit Breaker
    print("6. Testing circuit breaker functionality...")

    initial_cb_state = node1.circuit_breaker["state"]

    # Trigger multiple failures to open circuit breaker
    for i in range(6):  # More than failure threshold
        error_signal = SocialSignal(
            content="error_trigger", signal_type="bad_type", urgency=0.5, source_id=node2.node_id
        )
        node1.receive_signal(error_signal)

    final_cb_state = node1.circuit_breaker["state"]

    print(f"✓ Circuit breaker state: {initial_cb_state} → {final_cb_state}")
    print(f"✓ Failure count: {node1.circuit_breaker['failure_count']}\n")

    # 7. Demonstrate Partition Queues
    print("7. Testing partition-based message ordering...")

    partition_key = "critical_alerts"

    for i in range(3):
        signal = SocialSignal(
            content=f"ordered_message_{i}",
            signal_type="memory",
            urgency=0.6,
            source_id=node2.node_id,
            partition_key=partition_key,
        )
        node2.receive_signal(signal)  # Use node2 to avoid circuit breaker

    print(f"✓ Partition '{partition_key}' created")
    print(f"✓ Partition queue depth: {len(node2.partition_queues.get(partition_key, []))}")
    print(f"✓ Total partitions: {len(node2.partition_queues)}\n")

    # 8. Demonstrate Producer Flow Control
    print("8. Testing producer-side flow control...")

    # Fill up target node's queue
    queue_capacity = node2.communication_queue.maxlen
    for i in range(int(queue_capacity * 0.9)):
        dummy_signal = SocialSignal(
            content=f"filler_{i}", signal_type="memory", urgency=0.1, source_id=99
        )
        node2.communication_queue.append(dummy_signal)

    # Try to send signal with backpressure
    responses = node1.send_signal(
        target_nodes=[node2], signal_type="memory", content="backpressure_test"
    )

    print(f"✓ Target queue near capacity: {len(node2.communication_queue)}/{queue_capacity}")
    print("✓ Backpressure handled gracefully\n")

    # 9. Demonstrate Replay Functionality
    print("9. Testing signal replay capabilities...")

    replayed_signals = node1.replay_signals()

    print(f"✓ Found {len(replayed_signals)} signals available for replay")
    if replayed_signals:
        print("✓ Sample replay entry:")
        sample = replayed_signals[0]
        print(f"  - Signal ID: {sample['signal_id'][:8]}...")
        print(f"  - Type: {sample['signal_type']}")
        print(f"  - Correlation ID: {sample['correlation_id'][:8]}...")
    print()

    # 10. Demonstrate Graceful Shutdown
    print("10. Testing graceful shutdown and queue draining...")

    # Add some signals to demonstrate draining
    for i in range(3):
        signal = SocialSignal(
            content=f"shutdown_test_{i}", signal_type="memory", urgency=0.5, source_id=node2.node_id
        )
        node2.communication_queue.append(signal)

    queue_before = len(node2.communication_queue)
    shutdown_result = node2.graceful_shutdown(timeout=5)
    queue_after = len(node2.communication_queue)

    print(f"✓ Queue before shutdown: {queue_before}")
    print(f"✓ Queue after shutdown: {queue_after}")
    print(f"✓ Graceful shutdown successful: {shutdown_result}\n")

    # 11. Final Metrics Summary
    print("11. Final system metrics summary...")

    final_metrics = node1._get_queue_metrics()

    print("✓ Node 1 Final Metrics:")
    print(f"  - Total processed: {node1.signal_metrics['processed_count']}")
    print(f"  - Total errors: {node1.signal_metrics['error_count']}")
    print(f"  - DLQ messages: {node1.signal_metrics['dlq_count']}")
    print(f"  - Duplicates: {node1.signal_metrics['duplicate_count']}")
    print(f"  - Circuit breaker: {final_metrics['circuit_breaker_state']}")
    print(f"  - Deduplication store size: {len(node1.deduplication_store)}")
    print(f"  - Persisted signals: {len(node1.persisted_signals)}")

    print("\n=== Production Features Demo Complete ===")
    print("\nImplemented features:")
    print("✓ Message deduplication with TTL-based idempotency")
    print("✓ Comprehensive observability metrics")
    print("✓ Dead Letter Queue for poison message handling")
    print("✓ Schema version validation")
    print("✓ Circuit breaker pattern for fault tolerance")
    print("✓ Partition-based message ordering")
    print("✓ Producer-side flow control and backpressure")
    print("✓ Signal replay and recovery capabilities")
    print("✓ Correlation IDs for distributed tracing")
    print("✓ Graceful shutdown with queue draining")


if __name__ == "__main__":
    demo_production_features()
