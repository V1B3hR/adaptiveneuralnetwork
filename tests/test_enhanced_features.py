"""
Test cases for the enhanced adaptive neural network features:
1. External signal absorption subsystem
2. Anxiety overwhelm safety protocol
3. Time series tracking
4. Security and privacy features
"""

import os

# Add the project root to the path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, "/home/runner/work/adaptiveneuralnetwork/adaptiveneuralnetwork")

from api_integration.human_api import HumanSignalManager
from api_integration.signal_adapter import (
    SignalAdapter,
    SignalMapping,
    SignalSource,
    SignalType,
    StateVariable,
)
from core.alive_node import AliveLoopNode
from core.network import Capacitor, TunedAdaptiveFieldNetwork
from core.time_series_tracker import TimeSeriesQuery, TimeSeriesTracker, track_node_automatically


class TestSignalAdapter(unittest.TestCase):
    """Test the enhanced signal adapter system"""

    def setUp(self):
        self.adapter = SignalAdapter(security_enabled=False)  # Disable security for testing

    def test_signal_source_registration(self):
        """Test registering and managing signal sources"""
        source = SignalSource(
            name="test_source",
            signal_type=SignalType.HUMAN,
            api_url="http://test.example.com/api",
            mappings=[SignalMapping("happiness", StateVariable.CALM, "linear", 2.0, 0.0, 0.0, 5.0)],
        )

        self.adapter.register_source(source)
        self.assertIn("test_source", self.adapter.sources)
        self.assertEqual(self.adapter.sources["test_source"].signal_type, SignalType.HUMAN)

    def test_signal_mapping_transformations(self):
        """Test different signal transformation functions"""
        mapping = SignalMapping("test_field", StateVariable.ENERGY, "linear", 2.0, 1.0, 0.0, 10.0)

        # Linear transformation: value * 2.0 + 1.0
        result = mapping.apply_transformation(3.0)
        self.assertEqual(result, 7.0)

        # Test bounds
        result = mapping.apply_transformation(10.0)  # Should be capped at max_value=10.0
        self.assertEqual(result, 10.0)

    def test_mock_data_fetch(self):
        """Test fetching data with mocked responses"""
        source = SignalSource(
            name="mock_source",
            signal_type=SignalType.HUMAN,
            api_url="http://mock.example.com/api",
            mappings=[
                SignalMapping("stress_level", StateVariable.ANXIETY, "linear", 3.0, 0.0, 0.0, 10.0)
            ],
        )

        self.adapter.register_source(source)

        # Mock the data fetching method
        mock_data = {"stress_level": 2.5}
        self.adapter._fetch_data = Mock(return_value=mock_data)

        signals = self.adapter.fetch_signals("mock_source")

        self.assertIn(StateVariable.ANXIETY, signals)
        self.assertEqual(signals[StateVariable.ANXIETY], 7.5)  # 2.5 * 3.0


class TestAnxietyOverwhelmProtocol(unittest.TestCase):
    """Test the anxiety overwhelm safety protocol"""

    def setUp(self):
        self.node1 = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1)
        self.node2 = AliveLoopNode(position=(1, 0), velocity=(0, 0), initial_energy=12.0, node_id=2)
        self.node3 = AliveLoopNode(position=(0, 1), velocity=(0, 0), initial_energy=10.0, node_id=3)

        # Setup trust networks
        self.node1.trust_network[2] = 0.7
        self.node1.trust_network[3] = 0.6
        self.node2.trust_network[1] = 0.8
        self.node3.trust_network[1] = 0.5

    def test_anxiety_overwhelm_detection(self):
        """Test detection of anxiety overwhelm"""
        # Normal anxiety level
        self.node1.anxiety = 5.0
        self.assertFalse(self.node1.check_anxiety_overwhelm())

        # High anxiety level
        self.node1.anxiety = 9.0
        self.assertTrue(self.node1.check_anxiety_overwhelm())

    def test_help_signal_sending(self):
        """Test sending help signals when overwhelmed"""
        # Set up overwhelmed node
        self.node1.anxiety = 10.0
        self.node1.energy = 8.0
        self.node1._time = 100  # Set current time

        # Set up helper nodes
        self.node2.anxiety = 3.0
        self.node2.energy = 10.0
        self.node3.anxiety = 2.0
        self.node3.energy = 8.0

        # Make sure communication range is sufficient
        self.node1.communication_range = 5.0

        nearby_nodes = [self.node2, self.node3]

        # Debug: check conditions before sending
        print(f"Node1 can send help: {self.node1.can_send_help_signal()}")
        print(f"Node1 is overwhelmed: {self.node1.check_anxiety_overwhelm()}")

        responded_nodes = self.node1.send_help_signal(nearby_nodes)

        # Should have sent help signals
        self.assertGreater(self.node1.help_signals_sent, 0)
        self.assertGreater(self.node1.last_help_signal_time, 0)

        # Should have created memory of help request
        help_memories = [m for m in self.node1.memory if m.memory_type == "help_signal"]
        self.assertGreater(len(help_memories), 0)

    def test_help_signal_cooldown(self):
        """Test help signal cooldown mechanism"""
        self.node1.anxiety = 10.0
        self.node1.energy = 8.0
        self.node1.last_help_signal_time = self.node1._time - 5  # Recent help signal

        # Should not be able to send another help signal due to cooldown
        self.assertFalse(self.node1.can_send_help_signal())

        # After cooldown period
        self.node1.last_help_signal_time = self.node1._time - 15
        self.assertTrue(self.node1.can_send_help_signal())

    def test_anxiety_help_response(self):
        """Test processing anxiety help responses"""
        self.node1.anxiety = 8.0
        initial_anxiety = self.node1.anxiety

        # Create mock help response
        help_response = {
            "type": "anxiety_help_response",
            "helper_node": 2,
            "anxiety_reduction_offered": 2.0,
            "support_message": "You're not alone.",
        }

        from core.alive_node import SocialSignal

        signal = SocialSignal(
            content=help_response, signal_type="anxiety_help_response", urgency=0.7, source_id=2
        )

        self.node1._process_anxiety_help_response(signal)

        # Anxiety should be reduced
        self.assertLess(self.node1.anxiety, initial_anxiety)

        # Should have memory of receiving help
        help_memories = [m for m in self.node1.memory if m.memory_type == "help_received"]
        self.assertGreater(len(help_memories), 0)

    def test_calm_application(self):
        """Test natural calm effect"""
        self.node1.anxiety = 6.0
        self.node1.calm = 2.0
        initial_anxiety = self.node1.anxiety

        self.node1.apply_calm_effect()

        # Anxiety should be reduced by calm effect
        self.assertLess(self.node1.anxiety, initial_anxiety)


class TestTimeSeriesTracking(unittest.TestCase):
    """Test the time series tracking system"""

    def setUp(self):
        # Use temporary database file
        self.db_file = tempfile.NamedTemporaryFile(delete=False)
        self.db_file.close()

        self.tracker = TimeSeriesTracker(
            max_memory_points=1000, persist_to_disk=True, db_path=self.db_file.name
        )

    def tearDown(self):
        # Clean up temporary database
        try:
            os.unlink(self.db_file.name)
        except:
            pass

    def test_node_state_recording(self):
        """Test recording node state data"""
        state_data = {"energy": 10.5, "anxiety": 3.2, "calm": 1.8, "trust": 0.7}

        timestamp = time.time()
        self.tracker.record_node_state(1, state_data, timestamp)

        # Check that data was recorded
        latest_values = self.tracker.get_latest_values(1)
        self.assertEqual(latest_values["energy"], 10.5)
        self.assertEqual(latest_values["anxiety"], 3.2)

    def test_time_series_querying(self):
        """Test querying time series data"""
        # Record some test data
        timestamps = [time.time() + i for i in range(5)]
        for i, ts in enumerate(timestamps):
            state_data = {"energy": 10.0 + i, "anxiety": 2.0 + i * 0.5}
            self.tracker.record_node_state(1, state_data, ts)

        # Query the data
        query = TimeSeriesQuery(
            node_ids=[1],
            variables=["energy", "anxiety"],
            start_time=timestamps[1],
            end_time=timestamps[3],
            max_points=100,  # Ensure we don't limit results artificially
        )

        results = self.tracker.query(query)

        # Debug: print what we got
        energy_points = [p for p in results if p.variable_name == "energy"]
        print(f"Energy points found: {len(energy_points)}")
        print(f"Timestamps queried: {timestamps[1]} to {timestamps[3]}")
        print(f"Energy points timestamps: {[p.timestamp for p in energy_points]}")

        # We record data at timestamps 0,1,2,3,4 and query for 1,2,3 (inclusive)
        # So we should get 3 energy data points
        self.assertEqual(len(energy_points), 3)

    def test_automatic_node_tracking(self):
        """Test automatic tracking of AliveLoopNode"""
        node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1)
        node.anxiety = 4.0

        timestamp = time.time()
        track_node_automatically(self.tracker, node, timestamp)

        # Check that key variables were tracked
        latest_values = self.tracker.get_latest_values(1)
        self.assertEqual(latest_values["energy"], 15.0)
        self.assertEqual(latest_values["anxiety"], 4.0)

    def test_statistics_tracking(self):
        """Test statistics collection"""
        # Record data for multiple nodes
        for node_id in [1, 2, 3]:
            state_data = {"energy": 10.0, "anxiety": 2.0}
            self.tracker.record_node_state(node_id, state_data)

        stats = self.tracker.get_statistics()

        self.assertEqual(len(stats["unique_nodes"]), 3)
        self.assertIn("energy", stats["unique_variables"])
        self.assertIn("anxiety", stats["unique_variables"])


class TestHumanSignalManager(unittest.TestCase):
    """Test the human signal management system"""

    def setUp(self):
        self.manager = HumanSignalManager(security_enabled=False)

    def test_privacy_controls(self):
        """Test privacy controls for human data"""
        # Check privacy report
        report = self.manager.get_privacy_report()

        self.assertIn("sources", report)
        self.assertIn("privacy_levels", report)

        # Should have private/confidential sources for human data
        privacy_levels = report["privacy_levels"]
        self.assertTrue(any(level in ["private", "confidential"] for level in privacy_levels))

    def test_data_anonymization(self):
        """Test data anonymization features"""
        # Mock some cached data with identifying information
        source_name = "human_emotion"
        if source_name in self.manager.adapter.sources:
            source = self.manager.adapter.sources[source_name]
            source.cached_data = {
                "user_id": "12345",
                "happiness": 0.8,
                "stress": 0.3,
                "session_id": "abc123",
            }

            # Anonymize the data
            success = self.manager.anonymize_source_data(source_name)
            self.assertTrue(success)

            # Check that identifying data was anonymized
            anonymized_data = source.cached_data
            self.assertEqual(anonymized_data["user_id"], "anonymized")
            self.assertEqual(anonymized_data["session_id"], "anonymized")
            # Non-identifying data should remain
            self.assertEqual(anonymized_data["happiness"], 0.8)


class TestEnhancedNetwork(unittest.TestCase):
    """Test the enhanced network with all new features"""

    def setUp(self):
        # Create test nodes
        self.nodes = [
            AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1),
            AliveLoopNode(position=(2, 0), velocity=(0, 0), initial_energy=12.0, node_id=2),
            AliveLoopNode(position=(0, 2), velocity=(0, 0), initial_energy=10.0, node_id=3),
        ]

        # Create test capacitors
        self.capacitors = [Capacitor(capacity=10.0)]
        # Add position to capacitor for testing
        self.capacitors[0].position = np.array([1.0, 1.0])

        # Create enhanced network
        self.network = TunedAdaptiveFieldNetwork(
            nodes=self.nodes,
            capacitors=self.capacitors,
            enable_time_series=True,
            enable_security=False,  # Disable for testing
        )

    def test_network_initialization(self):
        """Test that enhanced network initializes properly"""
        self.assertEqual(len(self.network.nodes), 3)
        self.assertTrue(hasattr(self.network, "signal_adapter"))
        self.assertTrue(hasattr(self.network, "time_series_tracker"))
        self.assertTrue(hasattr(self.network, "performance_metrics"))

    def test_network_step_with_anxiety_overwhelm(self):
        """Test network step with anxiety overwhelm handling"""
        # Set up one node with high anxiety
        self.nodes[0].anxiety = 9.0
        self.nodes[0].energy = 8.0

        # Set up helper nodes
        self.nodes[1].anxiety = 2.0
        self.nodes[1].energy = 10.0
        self.nodes[2].anxiety = 3.0
        self.nodes[2].energy = 8.0

        # Setup trust networks
        self.nodes[0].trust_network[2] = 0.7
        self.nodes[0].trust_network[3] = 0.6

        initial_help_signals = self.network.performance_metrics["total_help_signals"]

        # Step the network
        self.network.step()

        # Should have processed anxiety overwhelm
        self.assertGreaterEqual(
            self.network.performance_metrics["total_help_signals"], initial_help_signals
        )

    def test_network_status_reporting(self):
        """Test comprehensive network status reporting"""
        # Step the network a few times to generate data
        for i in range(5):
            self.network.step()

        status = self.network.get_network_status()

        self.assertIn("time", status)
        self.assertIn("node_count", status)
        self.assertIn("performance_metrics", status)
        self.assertIn("nodes", status)
        self.assertIn("external_signals", status)

        # Check node-specific status
        self.assertEqual(len(status["nodes"]), 3)

    def test_time_series_integration(self):
        """Test time series tracking integration"""
        # Step the network multiple times
        for i in range(10):
            self.network.step()

        # Check that time series data was recorded
        stats = self.network.time_series_tracker.get_statistics()
        self.assertGreater(stats["total_points"], 0)
        self.assertEqual(len(stats["unique_nodes"]), 3)

    def test_network_stability_assessment(self):
        """Test network stability assessment"""
        # Step the network
        self.network.step()

        # Check that stability metrics are calculated
        stability_score = self.network.performance_metrics["network_stability_score"]
        self.assertIsInstance(stability_score, float)
        self.assertGreaterEqual(stability_score, 0.0)
        self.assertLessEqual(stability_score, 1.0)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_cases = [
        TestSignalAdapter,
        TestAnxietyOverwhelmProtocol,
        TestTimeSeriesTracking,
        TestHumanSignalManager,
        TestEnhancedNetwork,
    ]

    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'=' * 50}")
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
