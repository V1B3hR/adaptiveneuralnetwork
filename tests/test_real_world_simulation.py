"""
Test for Real-World Simulation & Transfer Learning - Phase 2.3

Tests the real-world simulation capabilities, sensor networks,
and transfer learning validation systems.
"""

import json
import os

# Import from the core package
import sys
import time
import unittest
from unittest.mock import Mock

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.real_world_adapter import (
    RealWorldSimulator,
    SensorReading,
    SensorType,
    SimulatedSensor,
    TransferLearningValidator,
)


class TestSimulatedSensor(unittest.TestCase):
    def setUp(self):
        """Set up test environment for sensor tests."""
        self.sensor = SimulatedSensor(
            sensor_id="test_sensor_1",
            sensor_type=SensorType.TEMPERATURE,
            base_value=25.0,
            noise_stddev=0.5,
        )

    def test_sensor_initialization(self):
        """Test sensor initialization."""
        self.assertEqual(self.sensor.sensor_id, "test_sensor_1")
        self.assertEqual(self.sensor.sensor_type, SensorType.TEMPERATURE)
        self.assertEqual(self.sensor.base_value, 25.0)
        self.assertEqual(self.sensor.current_value, 25.0)
        self.assertTrue(self.sensor.is_functional)

    def test_sensor_reading(self):
        """Test sensor reading generation."""
        reading = self.sensor.read()

        self.assertIsInstance(reading, SensorReading)
        self.assertEqual(reading.sensor_id, "test_sensor_1")
        self.assertEqual(reading.sensor_type, SensorType.TEMPERATURE)
        self.assertIsInstance(reading.value, float)
        self.assertGreater(reading.timestamp, 0)
        self.assertGreaterEqual(reading.confidence, 0.0)
        self.assertLessEqual(reading.confidence, 1.0)

    def test_sensor_environmental_update(self):
        """Test sensor response to environmental changes."""
        initial_value = self.sensor.current_value

        # Apply environmental change
        self.sensor.update_environment(5.0)

        # Sensor value should have changed
        self.assertNotEqual(self.sensor.current_value, initial_value)
        self.assertEqual(self.sensor.current_value, initial_value + 5.0)

    def test_sensor_calibration(self):
        """Test sensor calibration functionality."""
        # Introduce some offset
        self.sensor.calibration_offset = 2.0

        # Calibrate
        self.sensor.calibrate()

        # Offset should be reduced
        self.assertLess(abs(self.sensor.calibration_offset), 2.0)

    def test_sensor_failure_and_repair(self):
        """Test sensor failure and repair mechanisms."""
        # Force sensor failure
        self.sensor.is_functional = False

        # Reading should indicate failure
        reading = self.sensor.read()
        self.assertEqual(reading.confidence, 0.0)
        self.assertTrue(reading.value != reading.value)  # NaN check

        # Repair sensor
        self.sensor.repair()
        self.assertTrue(self.sensor.is_functional)

        # Reading should be normal again
        reading = self.sensor.read()
        self.assertGreater(reading.confidence, 0.0)
        self.assertFalse(reading.value != reading.value)  # Not NaN


class TestRealWorldSimulator(unittest.TestCase):
    def setUp(self):
        """Set up test environment for simulator tests."""
        self.simulator = RealWorldSimulator()

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(len(self.simulator.sensors), 0)
        self.assertGreater(len(self.simulator.scenarios), 0)
        self.assertIsNone(self.simulator.current_scenario)

        # Check that default scenarios are loaded
        self.assertIn("urban", self.simulator.scenarios)
        self.assertIn("natural", self.simulator.scenarios)
        self.assertIn("industrial", self.simulator.scenarios)

    def test_scenario_startup(self):
        """Test starting a simulation scenario."""
        success = self.simulator.start_scenario("urban")

        self.assertTrue(success)
        self.assertIsNotNone(self.simulator.current_scenario)
        self.assertEqual(self.simulator.current_scenario.name, "urban_environment")
        self.assertIsNotNone(self.simulator.simulation_start_time)

        # Should have created sensors
        self.assertGreater(len(self.simulator.sensors), 0)

    def test_invalid_scenario_startup(self):
        """Test starting an invalid scenario."""
        success = self.simulator.start_scenario("nonexistent")

        self.assertFalse(success)
        self.assertIsNone(self.simulator.current_scenario)

    def test_sensor_network_creation(self):
        """Test creation of sensor networks for scenarios."""
        sensors = self.simulator.create_sensor_network("urban")

        self.assertIsInstance(sensors, dict)
        self.assertGreater(len(sensors), 0)

        # Check that sensors have correct types
        for sensor_id, sensor in sensors.items():
            self.assertIsInstance(sensor, SimulatedSensor)
            self.assertEqual(sensor.sensor_id, sensor_id)

    def test_sensor_readings_collection(self):
        """Test collection of sensor readings."""
        # Start a scenario first
        self.simulator.start_scenario("natural")

        # Get sensor readings
        readings = self.simulator.get_sensor_readings()

        self.assertIsInstance(readings, dict)
        self.assertGreater(len(readings), 0)

        # Check reading properties
        for sensor_id, reading in readings.items():
            self.assertIsInstance(reading, SensorReading)
            self.assertEqual(reading.sensor_id, sensor_id)

    def test_environmental_change_simulation(self):
        """Test simulation of environmental changes."""
        # Start scenario
        self.simulator.start_scenario("industrial")

        # Get initial sensor readings
        initial_readings = self.simulator.get_sensor_readings()

        # Simulate environmental change
        self.simulator.simulate_environmental_change("temperature_variance", 10.0)

        # Get readings after change
        changed_readings = self.simulator.get_sensor_readings()

        # Some sensors should show different readings
        # (This is probabilistic due to noise, so we check that readings are generated)
        self.assertEqual(len(initial_readings), len(changed_readings))

    def test_simulation_statistics(self):
        """Test generation of simulation statistics."""
        # Start scenario
        self.simulator.start_scenario("urban")

        # Let some time pass and get readings
        time.sleep(0.1)
        self.simulator.get_sensor_readings()

        # Get statistics
        stats = self.simulator.get_simulation_statistics()

        self.assertIn("scenario_name", stats)
        self.assertIn("elapsed_time", stats)
        self.assertIn("active_sensors", stats)
        self.assertIn("total_sensors", stats)
        self.assertIn("environmental_state", stats)

        self.assertEqual(stats["scenario_name"], "urban_environment")
        self.assertGreater(stats["elapsed_time"], 0)

    def test_sensor_data_export(self):
        """Test export of sensor data."""
        # Start scenario and collect some data
        self.simulator.start_scenario("natural")

        # Generate some readings
        for _ in range(3):
            self.simulator.get_sensor_readings()
            time.sleep(0.01)

        # Export data
        export_json = self.simulator.export_sensor_data("json")

        # Should be valid JSON
        data = json.loads(export_json)

        self.assertIn("scenario", data)
        self.assertIn("sensor_data", data)
        self.assertEqual(data["scenario"], "natural_environment")

        # Should have sensor data
        self.assertGreater(len(data["sensor_data"]), 0)

        # Each sensor should have readings
        for sensor_id, readings in data["sensor_data"].items():
            self.assertIsInstance(readings, list)
            if readings:  # If we have readings
                self.assertIn("timestamp", readings[0])
                self.assertIn("value", readings[0])


class TestTransferLearningValidator(unittest.TestCase):
    def setUp(self):
        """Set up test environment for transfer learning tests."""
        self.validator = TransferLearningValidator()
        self.mock_model = Mock()
        self.mock_test_data = Mock()

    def test_validator_initialization(self):
        """Test transfer learning validator initialization."""
        self.assertEqual(len(self.validator.baseline_performances), 0)
        self.assertEqual(len(self.validator.transfer_results), 0)
        self.assertEqual(len(self.validator.adaptation_metrics), 0)

    def test_baseline_establishment(self):
        """Test establishing baseline performance."""
        baseline = self.validator.establish_baseline(
            model=self.mock_model,
            source_environment="urban_environment",
            test_data=self.mock_test_data,
        )

        self.assertIn("accuracy", baseline)
        self.assertIn("precision", baseline)
        self.assertIn("recall", baseline)
        self.assertIn("f1_score", baseline)

        # Check that baseline was stored
        self.assertIn("urban_environment", self.validator.baseline_performances)

    def test_transfer_evaluation(self):
        """Test transfer evaluation between environments."""
        # Establish baseline first
        self.validator.establish_baseline(self.mock_model, "urban_environment", self.mock_test_data)

        # Evaluate transfer
        transfer_results = self.validator.evaluate_transfer(
            model=self.mock_model,
            source_env="urban_environment",
            target_env="industrial_environment",
            test_data=self.mock_test_data,
            adaptation_steps=5,
        )

        self.assertIn("source_environment", transfer_results)
        self.assertIn("target_environment", transfer_results)
        self.assertIn("baseline_metrics", transfer_results)
        self.assertIn("transfer_metrics", transfer_results)
        self.assertIn("domain_similarity", transfer_results)
        self.assertIn("performance_retention", transfer_results)
        self.assertIn("transfer_success", transfer_results)

        self.assertEqual(transfer_results["source_environment"], "urban_environment")
        self.assertEqual(transfer_results["target_environment"], "industrial_environment")
        self.assertIsInstance(transfer_results["transfer_success"], bool)

    def test_transfer_without_baseline(self):
        """Test transfer evaluation without established baseline."""
        with self.assertRaises(ValueError):
            self.validator.evaluate_transfer(
                model=self.mock_model,
                source_env="nonexistent_environment",
                target_env="industrial_environment",
                test_data=self.mock_test_data,
            )

    def test_adaptation_validation(self):
        """Test adaptation validation over multiple steps."""
        adaptation_results = self.validator.validate_adaptation(
            model=self.mock_model,
            environment="natural_environment",
            adaptation_data=self.mock_test_data,
            steps=5,
        )

        self.assertIn("environment", adaptation_results)
        self.assertIn("adaptation_steps", adaptation_results)
        self.assertIn("initial_accuracy", adaptation_results)
        self.assertIn("final_accuracy", adaptation_results)
        self.assertIn("accuracy_improvement", adaptation_results)
        self.assertIn("adaptation_efficiency", adaptation_results)
        self.assertIn("convergence_achieved", adaptation_results)
        self.assertIn("adaptation_history", adaptation_results)

        self.assertEqual(adaptation_results["adaptation_steps"], 5)
        self.assertEqual(len(adaptation_results["adaptation_history"]), 5)

        # Check that adaptation history has correct structure
        for step_data in adaptation_results["adaptation_history"]:
            self.assertIn("step", step_data)
            self.assertIn("accuracy", step_data)
            self.assertIn("loss", step_data)

    def test_domain_similarity_calculation(self):
        """Test domain similarity calculation."""
        # Test same domain
        sim_same = self.validator._calculate_domain_similarity(
            "urban_environment", "urban_environment"
        )
        self.assertEqual(sim_same, 1.0)

        # Test different domains
        sim_different = self.validator._calculate_domain_similarity(
            "urban_environment", "natural_environment"
        )
        self.assertLess(sim_different, 1.0)
        self.assertGreater(sim_different, 0.0)

    def test_transfer_report_generation(self):
        """Test comprehensive transfer report generation."""
        # Set up some test data
        self.validator.establish_baseline(self.mock_model, "urban_environment", self.mock_test_data)

        self.validator.evaluate_transfer(
            self.mock_model, "urban_environment", "industrial_environment", self.mock_test_data
        )

        self.validator.validate_adaptation(
            self.mock_model, "natural_environment", self.mock_test_data
        )

        # Generate report
        report = self.validator.generate_transfer_report()

        self.assertIsInstance(report, str)
        self.assertIn("Transfer Learning Validation Report", report)
        self.assertIn("Baseline Performances", report)
        self.assertIn("Transfer Results", report)
        self.assertIn("Adaptation Results", report)

    def test_integration_scenario(self):
        """Test complete integration scenario."""
        # Simulate a complete transfer learning validation workflow

        # 1. Establish baseline in source environment
        baseline = self.validator.establish_baseline(
            self.mock_model, "urban_environment", self.mock_test_data
        )
        self.assertGreater(baseline["accuracy"], 0.0)

        # 2. Evaluate transfer to target environment
        transfer_results = self.validator.evaluate_transfer(
            self.mock_model,
            "urban_environment",
            "natural_environment",
            self.mock_test_data,
            adaptation_steps=10,
        )
        self.assertIsNotNone(transfer_results)

        # 3. Validate adaptation in target environment
        adaptation_results = self.validator.validate_adaptation(
            self.mock_model, "natural_environment", self.mock_test_data, steps=10
        )
        self.assertIsNotNone(adaptation_results)

        # 4. Generate comprehensive report
        report = self.validator.generate_transfer_report()
        self.assertIn("urban_environment", report)
        self.assertIn("natural_environment", report)

        # 5. Verify that all results are stored
        self.assertEqual(len(self.validator.baseline_performances), 1)
        self.assertEqual(len(self.validator.transfer_results), 1)
        self.assertEqual(len(self.validator.adaptation_metrics), 1)


if __name__ == "__main__":
    unittest.main()
