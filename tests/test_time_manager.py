"""
Tests for the TimeManager system
"""

import time
import unittest

from core.time_manager import TimeConfig, TimeManager, get_time_manager, set_time_manager


class TestTimeManager(unittest.TestCase):

    def setUp(self):
        """Set up a fresh TimeManager for each test"""
        self.time_manager = TimeManager()

    def test_simulation_time_advancement(self):
        """Test that simulation time advances correctly"""
        initial_step = self.time_manager.simulation_step
        self.time_manager.advance_simulation(5)
        self.assertEqual(self.time_manager.simulation_step, initial_step + 5)

    def test_circadian_cycle(self):
        """Test that circadian time cycles properly"""
        self.time_manager.advance_simulation(25)  # More than 24 hours
        self.assertEqual(self.time_manager.circadian_time, 1)  # 25 % 24 = 1

        self.time_manager.advance_simulation(23)  # Total 48 hours
        self.assertEqual(self.time_manager.circadian_time, 0)  # 48 % 24 = 0

    def test_performance_measurement(self):
        """Test performance measurement functionality"""
        self.time_manager.start_performance_measurement()
        time.sleep(0.01)  # Small delay
        duration = self.time_manager.end_performance_measurement()
        self.assertGreater(duration, 0)
        self.assertLess(duration, 1.0)  # Should be much less than 1 second

    def test_network_tick(self):
        """Test network tick functionality"""
        initial_step = self.time_manager.simulation_step
        self.time_manager.network_tick()
        self.assertGreaterEqual(self.time_manager.simulation_step, initial_step + 1)

    def test_statistics(self):
        """Test statistics gathering"""
        self.time_manager.advance_simulation(10)
        stats = self.time_manager.get_statistics()

        self.assertIn("simulation_step", stats)
        self.assertIn("circadian_time", stats)
        self.assertIn("tick_count", stats)
        self.assertEqual(stats["simulation_step"], 10)
        self.assertEqual(stats["circadian_time"], 10)

    def test_time_config(self):
        """Test time configuration"""
        config = TimeConfig(simulation_time_scale=2.0, circadian_cycle_hours=12)
        tm = TimeManager(config)

        self.assertEqual(tm.config.simulation_time_scale, 2.0)
        self.assertEqual(tm.config.circadian_cycle_hours, 12)

        tm.advance_simulation(13)
        self.assertEqual(tm.circadian_time, 1)  # 13 % 12 = 1

    def test_global_time_manager(self):
        """Test global time manager functions"""
        # Test that we get a time manager
        tm = get_time_manager()
        self.assertIsInstance(tm, TimeManager)

        # Test setting a custom time manager
        custom_tm = TimeManager()
        set_time_manager(custom_tm)
        self.assertIs(get_time_manager(), custom_tm)

    def test_reset(self):
        """Test time manager reset functionality"""
        self.time_manager.advance_simulation(50)
        self.time_manager.start_performance_measurement()
        time.sleep(0.01)
        self.time_manager.end_performance_measurement()

        # Verify we have some data
        self.assertGreater(self.time_manager.simulation_step, 0)
        stats = self.time_manager.get_statistics()
        self.assertGreater(stats["total_real_time"], 0)

        # Reset and verify everything is cleared
        self.time_manager.reset()
        self.assertEqual(self.time_manager.simulation_step, 0)
        self.assertEqual(self.time_manager.circadian_time, 0)
        stats = self.time_manager.get_statistics()
        self.assertEqual(stats["total_real_time"], 0)


if __name__ == '__main__':
    unittest.main()
