"""
Test suite for core.capacitor module
Tests the CapacitorInSpace class with all its improvements and features
"""

import unittest
import numpy as np
import logging
import threading
import time
from core.capacitor import CapacitorInSpace
from tests.test_utils import set_seed, run_with_seed, get_test_seed


class TestCapacitorInSpace(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        # Set deterministic seed for reproducible tests
        set_seed(get_test_seed())
        
        # Create a basic capacitor for general tests
        self.capacitor = CapacitorInSpace(
            position=[0.0, 0.0], 
            capacity=10.0, 
            initial_energy=5.0
        )

    def test_basic_creation(self):
        """Test basic capacitor creation and initialization"""
        cap = CapacitorInSpace(position=[1.0, 2.0], capacity=15.0, initial_energy=7.5)
        
        # Test position
        np.testing.assert_array_almost_equal(cap.position, [1.0, 2.0])
        
        # Test capacity and energy
        self.assertEqual(cap.capacity, 15.0)
        self.assertEqual(cap.energy, 7.5)
        
        # Test string representation
        self.assertIn("CapacitorInSpace", str(cap))
        self.assertIn("1.0", str(cap))
        self.assertIn("2.0", str(cap))

    def test_energy_clamping(self):
        """Test energy is properly clamped to valid range"""
        # Test negative initial energy is clamped to 0
        cap1 = CapacitorInSpace(position=[0, 0], capacity=10.0, initial_energy=-5.0)
        self.assertEqual(cap1.energy, 0.0)
        
        # Test initial energy above capacity is clamped to capacity
        cap2 = CapacitorInSpace(position=[0, 0], capacity=10.0, initial_energy=15.0)
        self.assertEqual(cap2.energy, 10.0)
        
        # Test negative capacity is clamped to 0
        cap3 = CapacitorInSpace(position=[0, 0], capacity=-5.0, initial_energy=2.0)
        self.assertEqual(cap3.capacity, 0.0)
        self.assertEqual(cap3.energy, 0.0)

    def test_charge_functionality(self):
        """Test charge method with various scenarios"""
        cap = CapacitorInSpace(position=[0, 0], capacity=10.0, initial_energy=3.0)
        
        # Test normal charging
        absorbed = cap.charge(4.0)
        self.assertEqual(absorbed, 4.0)
        self.assertEqual(cap.energy, 7.0)
        
        # Test charging beyond capacity
        absorbed = cap.charge(5.0)
        self.assertEqual(absorbed, 3.0)  # Only 3.0 can be absorbed
        self.assertEqual(cap.energy, 10.0)  # At full capacity
        
        # Test charging with zero or negative amount
        absorbed = cap.charge(0.0)
        self.assertEqual(absorbed, 0.0)
        
        absorbed = cap.charge(-2.0)
        self.assertEqual(absorbed, 0.0)
        self.assertEqual(cap.energy, 10.0)  # Energy unchanged

    def test_discharge_functionality(self):
        """Test discharge method with various scenarios"""
        cap = CapacitorInSpace(position=[0, 0], capacity=10.0, initial_energy=8.0)
        
        # Test normal discharge
        released = cap.discharge(3.0)
        self.assertEqual(released, 3.0)
        self.assertEqual(cap.energy, 5.0)
        
        # Test discharge more than available
        released = cap.discharge(7.0)
        self.assertEqual(released, 5.0)  # Only 5.0 available
        self.assertEqual(cap.energy, 0.0)  # Fully discharged
        
        # Test discharge with zero or negative amount
        released = cap.discharge(0.0)
        self.assertEqual(released, 0.0)
        
        released = cap.discharge(-2.0)
        self.assertEqual(released, 0.0)
        self.assertEqual(cap.energy, 0.0)  # Energy unchanged

    def test_position_validation(self):
        """Test position validation during creation and updates"""
        # Test valid positions
        cap = CapacitorInSpace(position=[1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(cap.position, [1.0, 2.0, 3.0])
        
        # Test position update
        new_pos = cap.update_position([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(new_pos, [4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(cap.position, [4.0, 5.0, 6.0])
        
        # Test invalid position (not 1D sequence)
        with self.assertRaises(ValueError):
            CapacitorInSpace(position=[[1, 2], [3, 4]])  # 2D array
        
        # Test invalid new position
        with self.assertRaises(ValueError):
            cap.update_position([[1, 2], [3, 4]])

    def test_expected_dims_validation(self):
        """Test dimension validation when expected_dims is set"""
        # Test creation with correct dimensions
        cap = CapacitorInSpace(position=[1.0, 2.0], expected_dims=2)
        self.assertEqual(cap.position.shape[0], 2)
        
        # Test creation with wrong dimensions
        with self.assertRaises(ValueError) as cm:
            CapacitorInSpace(position=[1.0, 2.0, 3.0], expected_dims=2)
        self.assertIn("dimension mismatch", str(cm.exception))
        
        # Test position update with wrong dimensions
        cap = CapacitorInSpace(position=[1.0, 2.0], expected_dims=2)
        with self.assertRaises(ValueError) as cm:
            cap.update_position([1.0, 2.0, 3.0])
        self.assertIn("dimension mismatch", str(cm.exception))

    def test_bounds_validation(self):
        """Test bounds validation for positions"""
        bounds = ((-5.0, 5.0), (-10.0, 10.0))
        
        # Test creation within bounds
        cap = CapacitorInSpace(position=[2.0, 8.0], bounds=bounds)
        np.testing.assert_array_almost_equal(cap.position, [2.0, 8.0])
        
        # Test creation outside bounds
        with self.assertRaises(ValueError) as cm:
            CapacitorInSpace(position=[6.0, 8.0], bounds=bounds)  # x > 5.0
        self.assertIn("outside bounds", str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            CapacitorInSpace(position=[2.0, 15.0], bounds=bounds)  # y > 10.0
        self.assertIn("outside bounds", str(cm.exception))
        
        # Test invalid bounds (low > high)
        invalid_bounds = ((5.0, 2.0), (-10.0, 10.0))
        with self.assertRaises(ValueError) as cm:
            CapacitorInSpace(position=[3.0, 0.0], bounds=invalid_bounds)
        self.assertIn("Invalid bounds", str(cm.exception))
        
        # Test bounds dimensionality mismatch
        wrong_bounds = ((-5.0, 5.0),)  # 1D bounds for 2D position
        with self.assertRaises(ValueError) as cm:
            CapacitorInSpace(position=[1.0, 2.0], bounds=wrong_bounds)
        self.assertIn("Bounds dimensionality", str(cm.exception))

    def test_bounds_validation_on_update(self):
        """Test bounds validation when updating position"""
        bounds = ((-5.0, 5.0), (-10.0, 10.0))
        cap = CapacitorInSpace(position=[0.0, 0.0], bounds=bounds)
        
        # Test valid update
        cap.update_position([3.0, -8.0])
        np.testing.assert_array_almost_equal(cap.position, [3.0, -8.0])
        
        # Test invalid update
        with self.assertRaises(ValueError) as cm:
            cap.update_position([7.0, 0.0])  # x > 5.0
        self.assertIn("outside bounds", str(cm.exception))

    def test_fixed_position_mode(self):
        """Test fixed position functionality"""
        cap = CapacitorInSpace(position=[1.0, 2.0], fixed_position=True)
        
        # Position should be set correctly initially
        np.testing.assert_array_almost_equal(cap.position, [1.0, 2.0])
        
        # Position update should raise RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            cap.update_position([3.0, 4.0])
        self.assertIn("Position is fixed", str(cm.exception))
        
        # Position should remain unchanged
        np.testing.assert_array_almost_equal(cap.position, [1.0, 2.0])

    def test_logging_functionality(self):
        """Test logging and verbosity control"""
        # Test with custom logger
        logger = logging.getLogger("test_capacitor")
        logger.setLevel(logging.DEBUG)
        
        cap = CapacitorInSpace(
            position=[0, 0], 
            capacity=10.0, 
            initial_energy=5.0,
            logger=logger,
            verbosity=logging.DEBUG
        )
        
        # Test verbosity setting
        cap.set_verbosity(logging.INFO)
        
        # Test verbosity with external override
        cap.set_verbosity(logging.WARNING, override_external=True)
        
        # Test print_status (should not raise errors)
        cap.print_status()

    def test_to_dict_functionality(self):
        """Test dictionary serialization"""
        bounds = ((-5.0, 5.0), (-10.0, 10.0))
        cap = CapacitorInSpace(
            position=[2.0, 3.0], 
            capacity=15.0, 
            initial_energy=12.0,
            expected_dims=2,
            bounds=bounds,
            fixed_position=True
        )
        
        data = cap.to_dict()
        
        # Test all expected keys
        expected_keys = {"position", "capacity", "energy", "soc", 
                        "fixed_position", "expected_dims", "bounds"}
        self.assertEqual(set(data.keys()), expected_keys)
        
        # Test values
        self.assertEqual(data["position"], [2.0, 3.0])
        self.assertEqual(data["capacity"], 15.0)
        self.assertEqual(data["energy"], 12.0)
        self.assertEqual(data["soc"], 0.8)  # 12.0 / 15.0
        self.assertTrue(data["fixed_position"])
        self.assertEqual(data["expected_dims"], 2)
        self.assertEqual(data["bounds"], bounds)

    def test_state_of_charge_calculation(self):
        """Test state of charge (SOC) calculation"""
        cap = CapacitorInSpace(position=[0, 0], capacity=20.0, initial_energy=15.0)
        data = cap.to_dict()
        self.assertEqual(data["soc"], 0.75)  # 15.0 / 20.0
        
        # Test with zero capacity
        cap_zero = CapacitorInSpace(position=[0, 0], capacity=0.0)
        data_zero = cap_zero.to_dict()
        self.assertEqual(data_zero["soc"], 0.0)

    def test_thread_safety_basic(self):
        """Test basic thread safety functionality"""
        cap = CapacitorInSpace(
            position=[0, 0], 
            capacity=100.0, 
            initial_energy=50.0,
            thread_safe=True
        )
        
        results = []
        
        def charge_worker():
            for _ in range(10):
                absorbed = cap.charge(1.0)
                results.append(('charge', absorbed))
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        def discharge_worker():
            for _ in range(10):
                released = cap.discharge(1.0)
                results.append(('discharge', released))
                time.sleep(0.001)
        
        # Run concurrent operations
        threads = []
        for _ in range(2):
            t1 = threading.Thread(target=charge_worker)
            t2 = threading.Thread(target=discharge_worker)
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify operations completed without exceptions
        self.assertEqual(len(results), 40)  # 4 threads * 10 operations each
        
        # Energy should be within valid range
        self.assertGreaterEqual(cap.energy, 0.0)
        self.assertLessEqual(cap.energy, cap.capacity)

    def test_edge_cases(self):
        """Test various edge cases and error conditions"""
        # Test with very small numbers
        cap = CapacitorInSpace(position=[0, 0], capacity=1e-10, initial_energy=1e-12)
        self.assertGreaterEqual(cap.energy, 0.0)
        self.assertLessEqual(cap.energy, cap.capacity)
        
        # Test with very large numbers
        cap_large = CapacitorInSpace(
            position=[0, 0], 
            capacity=1e10, 
            initial_energy=5e9
        )
        self.assertEqual(cap_large.capacity, 1e10)
        self.assertEqual(cap_large.energy, 5e9)
        
        # Test empty position (should create 0-dimensional array)
        cap_empty = CapacitorInSpace(position=[])
        self.assertEqual(cap_empty.position.shape, (0,))
        self.assertEqual(cap_empty.position.size, 0)

    @run_with_seed(42)
    def test_reproducible_behavior(self):
        """Test that behavior is reproducible with same seed"""
        cap1 = CapacitorInSpace(position=[0, 0], capacity=10.0, initial_energy=5.0)
        cap2 = CapacitorInSpace(position=[0, 0], capacity=10.0, initial_energy=5.0)
        
        # Both capacitors should behave identically
        for i in range(5):
            charge_amount = np.random.random() * 2.0
            discharge_amount = np.random.random() * 2.0
            
            absorbed1 = cap1.charge(charge_amount)
            absorbed2 = cap2.charge(charge_amount)
            self.assertAlmostEqual(absorbed1, absorbed2)
            
            released1 = cap1.discharge(discharge_amount)
            released2 = cap2.discharge(discharge_amount)
            self.assertAlmostEqual(released1, released2)
            
            self.assertAlmostEqual(cap1.energy, cap2.energy)

    def test_position_copy_safety(self):
        """Test that position creation is safe from external modifications"""
        original_pos = [1.0, 2.0]
        cap = CapacitorInSpace(position=original_pos)
        
        # Modify original position array
        original_pos[0] = 999.0
        
        # Capacitor position should be unaffected
        self.assertEqual(cap.position[0], 1.0)
        
        # Update position and verify that the method returns the internal array
        new_pos = [3.0, 4.0]
        returned_pos = cap.update_position(new_pos)
        
        # The returned position should be the same object as the internal position
        self.assertIs(returned_pos, cap.position)
        
        # Verify the position was updated correctly
        np.testing.assert_array_equal(cap.position, [3.0, 4.0])
        
        # Test that modifying the input array after update doesn't affect capacitor
        new_pos[0] = 888.0
        np.testing.assert_array_equal(cap.position, [3.0, 4.0])  # Should be unchanged


if __name__ == "__main__":
    unittest.main()