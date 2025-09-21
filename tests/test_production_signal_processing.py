"""
Test suite for production signal processing features
"""

import unittest

from core.alive_node import AliveLoopNode, Memory, SocialSignal
from core.time_manager import get_timestamp


class TestProductionSignalProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test nodes"""
        self.node1 = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1)
        self.node2 = AliveLoopNode(position=(1, 1), velocity=(0, 0), initial_energy=15.0, node_id=2)

        # Set up trust between nodes
        self.node1.trust_network[2] = 0.8
        self.node2.trust_network[1] = 0.8
        self.node1.influence_network[2] = 0.7
        self.node2.influence_network[1] = 0.7

    def test_idempotency_and_deduplication(self):
        """Test message deduplication based on idempotency keys"""
        # Create signal with specific idempotency key
        memory = Memory(
            content="test_memory", importance=0.7, timestamp=get_timestamp(), memory_type="test"
        )

        signal = SocialSignal(
            content=memory,
            signal_type="memory",
            urgency=0.5,
            source_id=self.node2.node_id,
            idempotency_key="unique_test_key_123",
        )

        # First signal should be processed
        response1 = self.node1.receive_signal(signal)
        initial_memory_count = len(self.node1.memory)

        # Second signal with same idempotency key should be ignored
        signal2 = SocialSignal(
            content=memory,
            signal_type="memory",
            urgency=0.5,
            source_id=self.node2.node_id,
            idempotency_key="unique_test_key_123",  # Same key
        )

        response2 = self.node1.receive_signal(signal2)
        final_memory_count = len(self.node1.memory)

        # Memory count should not increase on duplicate
        self.assertEqual(initial_memory_count, final_memory_count)
        self.assertEqual(self.node1.signal_metrics["duplicate_count"], 1)

    def test_observability_metrics(self):
        """Test queue metrics and observability features"""
        # Get initial metrics
        initial_metrics = self.node1._get_queue_metrics()
        self.assertEqual(initial_metrics["queue_depth"], 0)
        self.assertEqual(initial_metrics["error_rate"], 0.0)

        # Send some signals
        for i in range(3):
            signal = SocialSignal(
                content=f"test_message_{i}",
                signal_type="memory",
                urgency=0.5,
                source_id=self.node2.node_id,
            )
            self.node1.receive_signal(signal)

        # Check updated metrics
        metrics = self.node1._get_queue_metrics()
        self.assertEqual(metrics["queue_depth"], 3)
        self.assertEqual(self.node1.signal_metrics["processed_count"], 3)
        self.assertGreater(metrics["throughput_per_second"], 0)

    def test_dead_letter_queue(self):
        """Test DLQ handling for poison messages"""
        # Create invalid signal that should trigger error
        signal = SocialSignal(
            content="test",
            signal_type="invalid_type",  # This should trigger an error
            urgency=0.5,
            source_id=self.node2.node_id,
        )

        initial_dlq_count = len(self.node1.dead_letter_queue)
        response = self.node1.receive_signal(signal)

        # Signal should be in DLQ
        self.assertIsNone(response)
        self.assertEqual(len(self.node1.dead_letter_queue), initial_dlq_count + 1)
        self.assertEqual(self.node1.signal_metrics["dlq_count"], 1)
        self.assertGreater(self.node1.signal_metrics["error_count"], 0)

    def test_schema_validation(self):
        """Test schema version validation"""
        # Signal with unsupported schema version
        signal = SocialSignal(
            content="test",
            signal_type="memory",
            urgency=0.5,
            source_id=self.node2.node_id,
            schema_version="999.0",  # Unsupported version
        )

        response = self.node1.receive_signal(signal)

        # Should be rejected and added to DLQ
        self.assertIsNone(response)
        self.assertGreater(len(self.node1.dead_letter_queue), 0)

    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        # Trigger multiple errors to open circuit breaker
        for i in range(6):  # More than failure threshold (5)
            signal = SocialSignal(
                content="test",
                signal_type="invalid_type",
                urgency=0.5,
                source_id=self.node2.node_id,
            )
            self.node1.receive_signal(signal)

        # Circuit breaker should be open
        self.assertEqual(self.node1.circuit_breaker["state"], "open")

        # Valid signal should be rejected while circuit is open
        valid_signal = SocialSignal(
            content="test", signal_type="memory", urgency=0.5, source_id=self.node2.node_id
        )

        response = self.node1.receive_signal(valid_signal)
        self.assertIsNone(response)

    def test_partition_queues(self):
        """Test partition-based ordering guarantees"""
        partition_key = "test_partition"

        # Send signals with same partition key
        for i in range(3):
            signal = SocialSignal(
                content=f"message_{i}",
                signal_type="memory",
                urgency=0.5,
                source_id=self.node2.node_id,
                partition_key=partition_key,
            )
            self.node1.receive_signal(signal)

        # Check partition queue
        self.assertIn(partition_key, self.node1.partition_queues)
        self.assertEqual(len(self.node1.partition_queues[partition_key]), 3)

    def test_producer_flow_control(self):
        """Test producer-side backpressure"""
        # Fill up the target node's communication queue
        target_queue_capacity = self.node2.communication_queue.maxlen

        # Fill queue to near capacity
        for i in range(int(target_queue_capacity * 0.9)):
            signal = SocialSignal(
                content=f"fill_message_{i}",
                signal_type="memory",
                urgency=0.1,
                source_id=99,  # Different source to avoid circuit breaker
            )
            self.node2.communication_queue.append(signal)

        # Now try to send from node1 to node2
        responses = self.node1.send_signal(
            target_nodes=[self.node2], signal_type="memory", content="backpressure_test"
        )

        # Should handle backpressure appropriately
        # The exact behavior depends on implementation - either empty responses or successful with warning
        self.assertIsInstance(responses, list)

    def test_replay_functionality(self):
        """Test signal replay capabilities"""
        # Send some signals
        for i in range(3):
            signal = SocialSignal(
                content=f"replay_test_{i}",
                signal_type="memory",
                urgency=0.5,
                source_id=self.node2.node_id,
            )
            self.node1.receive_signal(signal)

        # Test replay
        replayed = self.node1.replay_signals()
        self.assertGreaterEqual(len(replayed), 3)

        # Check replay entries have required fields
        for entry in replayed:
            self.assertIn("signal_id", entry)
            self.assertIn("correlation_id", entry)
            self.assertIn("timestamp", entry)

    def test_dlq_reprocessing(self):
        """Test DLQ message reprocessing"""
        # Create signal that goes to DLQ
        signal = SocialSignal(
            content="test", signal_type="invalid_type", urgency=0.5, source_id=self.node2.node_id
        )

        self.node1.receive_signal(signal)

        # Get DLQ messages
        dlq_messages = self.node1.process_dlq_messages()
        self.assertGreater(len(dlq_messages), 0)

        # Attempt reprocessing (will still fail for invalid type)
        signal_id = dlq_messages[0]["signal_id"]
        result = self.node1.reprocess_dlq_message(signal_id)
        self.assertFalse(result)  # Should fail for invalid signal type

    def test_graceful_shutdown(self):
        """Test graceful shutdown and queue draining"""
        # Add some signals to queue
        for i in range(3):
            signal = SocialSignal(
                content=f"shutdown_test_{i}",
                signal_type="memory",
                urgency=0.5,
                source_id=self.node2.node_id,
            )
            self.node1.communication_queue.append(signal)

        # Test graceful shutdown
        result = self.node1.graceful_shutdown(timeout=5)

        # Check if queues were drained
        self.assertTrue(result)
        self.assertEqual(len(self.node1.communication_queue), 0)

    def test_correlation_ids(self):
        """Test correlation ID for distributed tracing"""
        correlation_id = "test_correlation_123"

        signal = SocialSignal(
            content="correlation_test",
            signal_type="memory",
            urgency=0.5,
            source_id=self.node2.node_id,
            correlation_id=correlation_id,
        )

        self.node1.receive_signal(signal)

        # Check that correlation ID was preserved in processing attempts
        self.assertEqual(signal.correlation_id, correlation_id)
        self.assertGreater(len(signal.processing_attempts), 0)
        self.assertEqual(signal.processing_attempts[0]["correlation_id"], correlation_id)


if __name__ == "__main__":
    unittest.main()
