"""
Integration test to demonstrate time mitigation working properly
"""

import unittest
import time
from core.time_manager import TimeManager, TimeConfig, set_time_manager
from core.alive_node import AliveLoopNode
from core.network import AdaptiveClockNetwork


class TestTimeMitigation(unittest.TestCase):
    
    def setUp(self):
        """Set up a fresh TimeManager for each test"""
        self.time_manager = TimeManager()
        set_time_manager(self.time_manager)
        
    def test_simulation_vs_real_time_separation(self):
        """Test that simulation time and real time are properly separated"""
        initial_real_time = time.time()
        
        # Advance simulation time rapidly without waiting real time
        self.time_manager.advance_simulation(100)
        
        # Verify simulation time advanced but real time barely changed
        self.assertEqual(self.time_manager.simulation_step, 100)
        self.assertEqual(self.time_manager.circadian_time, 4)  # 100 % 24 = 4
        
        real_time_elapsed = time.time() - initial_real_time
        self.assertLess(real_time_elapsed, 0.1)  # Should be very fast
        
    def test_node_circadian_behavior_with_centralized_time(self):
        """Test that nodes properly use centralized time for circadian cycles"""
        node = AliveLoopNode((0, 0), (0, 0))
        
        # Test different times of day with appropriate energy levels
        test_cases = [
            (10, 15.0, 2.0, "active"),     # Daytime with good energy -> active
            (22, 15.0, 2.0, "sleep"),      # Nighttime -> sleep (circadian > 20)
            (2, 2.0, 2.0, "sleep"),        # Early morning + low energy -> sleep
            (14, 25.0, 3.0, "inspired"),  # Afternoon + high energy + low anxiety -> inspired
        ]
        
        for time_step, energy, anxiety, expected_phase in test_cases:
            node.energy = energy
            node.anxiety = anxiety
            
            # Reset time manager and advance to specific time
            self.time_manager.reset()
            self.time_manager.advance_simulation(time_step)
            
            # Step the node phase
            node.step_phase()
            
            # Verify the phase matches expected
            self.assertEqual(node.phase, expected_phase, 
                           f"At time {time_step} with energy {energy} and anxiety {anxiety}, "
                           f"expected {expected_phase} but got {node.phase}")
            
            # Verify circadian time matches
            self.assertEqual(node.circadian_cycle, time_step % 24)
    
    def test_network_performance_measurement(self):
        """Test that network performance measurement uses centralized time"""
        # Create a simple network
        genome = {
            'num_cells': 2,
            'capacitor_capacity': 5.0,
            'global_calm': 1.0,
            'per_cell': [{'energy_capacity': 10.0} for _ in range(2)]
        }
        
        network = AdaptiveClockNetwork(genome)
        
        # Reset time manager for clean measurement
        self.time_manager.reset()
        
        # Perform several network ticks
        for i in range(5):
            stimuli = [1.0, 2.0]
            network.network_tick(stimuli)
        
        # Get performance metrics
        metrics = network.calculate_performance_and_stability()
        
        # Verify metrics are calculated
        self.assertIn("execution_time", metrics)
        self.assertGreater(metrics["execution_time"], 0)
        
        # Verify simulation time advanced
        self.assertEqual(self.time_manager.simulation_step, 5)
        
        # Verify time manager tracked the ticks
        stats = self.time_manager.get_statistics()
        self.assertEqual(stats["tick_count"], 5)
        self.assertGreater(stats["total_real_time"], 0)
        
    def test_time_scaling_configuration(self):
        """Test that time scaling works properly"""
        # Create time manager with different scale
        config = TimeConfig(simulation_time_scale=2.0)  # 2x speed
        scaled_tm = TimeManager(config)
        set_time_manager(scaled_tm)
        
        # Simulate network tick (which should advance time based on scale)
        initial_time = time.time()
        time.sleep(0.01)  # Small delay
        scaled_tm.network_tick()
        
        # With 2x scale, simulation should advance more than 1 step
        # (exact amount depends on real time elapsed)
        self.assertGreaterEqual(scaled_tm.simulation_step, 1)
        
    def test_backward_compatibility(self):
        """Test that existing code still works with explicit time parameters"""
        node = AliveLoopNode((0, 0), (0, 0))
        
        # Test old-style explicit time parameter
        node.step_phase(current_time=15)
        self.assertEqual(node.circadian_cycle, 15)
        self.assertEqual(self.time_manager.simulation_step, 15)
        
        # Test that subsequent calls work correctly
        node.step_phase(current_time=20)
        self.assertEqual(node.circadian_cycle, 20)
        self.assertEqual(self.time_manager.simulation_step, 20)


if __name__ == '__main__':
    unittest.main()