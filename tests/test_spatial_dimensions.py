"""
Tests for spatial dimension functionality.

This module tests the dimension-agnostic capabilities of the adaptive neural network,
ensuring that nodes and components work correctly in arbitrary spatial dimensions.
"""

import unittest
import numpy as np
from core.alive_node import AliveLoopNode
from core.capacitor import CapacitorInSpace
from core.spatial_utils import (
    zero_vector, rand_vector, distance, validate_spatial_dimensions,
    expand_bounds_to_dimensions, validate_position_in_bounds, create_random_positions
)
from config.network_config import load_network_config


class TestSpatialDimensions(unittest.TestCase):
    """Test suite for spatial dimension functionality."""
    
    def test_spatial_utils_zero_vector(self):
        """Test zero vector creation for different dimensions."""
        # Test various dimensions
        for dim in [1, 2, 3, 5, 10]:
            vec = zero_vector(dim)
            self.assertEqual(vec.shape, (dim,))
            self.assertTrue(np.allclose(vec, 0.0))
        
        # Test error for invalid dimension
        with self.assertRaises(ValueError):
            zero_vector(0)
        with self.assertRaises(ValueError):
            zero_vector(-1)
    
    def test_spatial_utils_rand_vector(self):
        """Test random vector generation."""
        # Test single bounds for all dimensions
        for dim in [2, 3, 5]:
            vec = rand_vector(dim, (-1, 1))
            self.assertEqual(vec.shape, (dim,))
            self.assertTrue(np.all(vec >= -1))
            self.assertTrue(np.all(vec <= 1))
        
        # Test per-dimension bounds
        bounds = [(-1, 1), (-2, 2), (-0.5, 0.5)]
        vec = rand_vector(3, bounds)
        self.assertEqual(vec.shape, (3,))
        self.assertTrue(-1 <= vec[0] <= 1)
        self.assertTrue(-2 <= vec[1] <= 2)
        self.assertTrue(-0.5 <= vec[2] <= 0.5)
        
        # Test error for mismatched bounds
        with self.assertRaises(ValueError):
            rand_vector(2, bounds)  # 2D with 3 bounds
    
    def test_spatial_utils_distance(self):
        """Test distance calculation."""
        # Test 2D
        a = np.array([0, 0])
        b = np.array([3, 4])
        self.assertAlmostEqual(distance(a, b), 5.0)
        
        # Test 3D
        a = np.array([0, 0, 0])
        b = np.array([1, 1, 1])
        self.assertAlmostEqual(distance(a, b), np.sqrt(3))
        
        # Test dimension mismatch
        with self.assertRaises(ValueError):
            distance([0, 0], [0, 0, 0])
    
    def test_spatial_utils_validation(self):
        """Test spatial dimension validation."""
        # Valid arrays
        arrays = [np.array([1, 2]), np.array([3, 4])]
        validate_spatial_dimensions(arrays, 2)  # Should not raise
        
        # Invalid dimension count
        with self.assertRaises(ValueError):
            validate_spatial_dimensions(arrays, 3)
        
        # Invalid array shape (2D array instead of 1D)
        arrays = [np.array([[1, 2], [3, 4]])]
        with self.assertRaises(ValueError):
            validate_spatial_dimensions(arrays, 2)
    
    def test_alive_node_2d_creation(self):
        """Test AliveLoopNode creation in 2D."""
        node = AliveLoopNode(
            position=[1, 2], 
            velocity=[0.1, 0.2], 
            initial_energy=10.0,
            spatial_dims=2
        )
        
        self.assertEqual(node.spatial_dims, 2)
        self.assertEqual(node.position.shape, (2,))
        self.assertEqual(node.velocity.shape, (2,))
        self.assertEqual(node.attention_focus.shape, (2,))
        np.testing.assert_array_equal(node.position, [1, 2])
        np.testing.assert_array_equal(node.velocity, [0.1, 0.2])
        np.testing.assert_array_equal(node.attention_focus, [0, 0])
    
    def test_alive_node_3d_creation(self):
        """Test AliveLoopNode creation in 3D."""
        node = AliveLoopNode(
            position=[1, 2, 3], 
            velocity=[0.1, 0.2, 0.3], 
            initial_energy=15.0,
            spatial_dims=3
        )
        
        self.assertEqual(node.spatial_dims, 3)
        self.assertEqual(node.position.shape, (3,))
        self.assertEqual(node.velocity.shape, (3,))
        self.assertEqual(node.attention_focus.shape, (3,))
        np.testing.assert_array_equal(node.position, [1, 2, 3])
        np.testing.assert_array_equal(node.velocity, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(node.attention_focus, [0, 0, 0])
    
    def test_alive_node_inferred_dimensions(self):
        """Test AliveLoopNode with inferred spatial dimensions."""
        # Should infer 2D from position length
        node = AliveLoopNode(position=[1, 2], velocity=[0.1, 0.2])
        self.assertEqual(node.spatial_dims, 2)
        
        # Should infer 3D from position length
        node = AliveLoopNode(position=[1, 2, 3], velocity=[0.1, 0.2, 0.3])
        self.assertEqual(node.spatial_dims, 3)
    
    def test_alive_node_dimension_validation(self):
        """Test AliveLoopNode dimension validation."""
        # Position and velocity dimension mismatch
        with self.assertRaises(ValueError) as cm:
            AliveLoopNode(position=[1, 2], velocity=[0.1, 0.2, 0.3], spatial_dims=2)
        self.assertIn("dimension validation failed", str(cm.exception))
        
        # Position length doesn't match specified spatial_dims
        with self.assertRaises(ValueError) as cm:
            AliveLoopNode(position=[1, 2, 3], velocity=[0.1, 0.2, 0.3], spatial_dims=2)
        self.assertIn("dimension validation failed", str(cm.exception))
    
    def test_capacitor_interaction_dimension_validation(self):
        """Test that node-capacitor interactions validate dimensions."""
        # Create matching 3D node and capacitor
        node_3d = AliveLoopNode(
            position=[0, 0, 0], 
            velocity=[0, 0, 0], 
            spatial_dims=3
        )
        cap_3d = CapacitorInSpace(
            position=[1, 1, 1], 
            capacity=5.0, 
            expected_dims=3
        )
        
        # Should work fine
        node_3d.interact_with_capacitor(cap_3d)
        
        # Create mismatched dimensions
        cap_2d = CapacitorInSpace(
            position=[1, 1], 
            capacity=5.0, 
            expected_dims=2
        )
        
        # Should raise dimension mismatch error
        with self.assertRaises(ValueError) as cm:
            node_3d.interact_with_capacitor(cap_2d)
        self.assertIn("spatial dimensions", str(cm.exception))
    
    def test_3d_smoke_simulation(self):
        """Test a simple 3D simulation runs without errors."""
        # Create 3D nodes
        nodes = []
        for i in range(3):
            node = AliveLoopNode(
                position=[i, i, i],
                velocity=[0.1, 0.1, 0.1],
                initial_energy=10.0,
                node_id=i,
                spatial_dims=3
            )
            nodes.append(node)
        
        # Create 3D capacitors
        capacitors = []
        for i in range(2):
            cap = CapacitorInSpace(
                position=[i+1, i+1, i+1],
                capacity=5.0,
                initial_energy=2.0,
                expected_dims=3
            )
            capacitors.append(cap)
        
        # Run a few simulation steps
        for step in range(10):
            for node in nodes:
                # Basic node operations
                node.step_phase(step)
                node.move()
                
                # Interact with capacitors
                for cap in capacitors:
                    node.interact_with_capacitor(cap)
        
        # Verify nodes are still functioning
        for node in nodes:
            self.assertEqual(node.spatial_dims, 3)
            self.assertEqual(node.position.shape, (3,))
            self.assertGreater(node.energy, 0)  # Should still have energy
    
    def test_configuration_spatial_dims(self):
        """Test that spatial dimensions are loaded from configuration."""
        cfg = load_network_config("config/network_config.yaml")
        self.assertIn("spatial_dims", cfg)
        self.assertEqual(cfg["spatial_dims"], 2)  # Default should be 2
    
    def test_bounds_expansion(self):
        """Test bounds expansion for multiple dimensions."""
        # Single bounds for all dimensions
        bounds = expand_bounds_to_dimensions((-5, 5), 3)
        expected = [(-5, 5), (-5, 5), (-5, 5)]
        self.assertEqual(bounds, expected)
        
        # Per-dimension bounds
        input_bounds = [(-1, 1), (-2, 2), (-3, 3)]
        bounds = expand_bounds_to_dimensions(input_bounds, 3)
        self.assertEqual(bounds, input_bounds)
        
        # Error for mismatched count
        with self.assertRaises(ValueError):
            expand_bounds_to_dimensions([(-1, 1), (-2, 2)], 3)
    
    def test_create_random_positions(self):
        """Test random position generation."""
        positions = create_random_positions(5, 3, (-10, 10))
        self.assertEqual(positions.shape, (5, 3))
        
        # Check bounds
        self.assertTrue(np.all(positions >= -10))
        self.assertTrue(np.all(positions <= 10))
        
        # Test with per-dimension bounds
        bounds = [(-1, 1), (-2, 2), (-3, 3)]
        positions = create_random_positions(3, 3, bounds)
        self.assertEqual(positions.shape, (3, 3))
        
        # Check per-dimension bounds
        self.assertTrue(np.all(positions[:, 0] >= -1) and np.all(positions[:, 0] <= 1))
        self.assertTrue(np.all(positions[:, 1] >= -2) and np.all(positions[:, 1] <= 2))
        self.assertTrue(np.all(positions[:, 2] >= -3) and np.all(positions[:, 2] <= 3))


class TestHighDimensionalScenarios(unittest.TestCase):
    """Test high-dimensional scenarios beyond 3D."""
    
    def test_5d_node_creation(self):
        """Test node creation in 5D space."""
        node = AliveLoopNode(
            position=[1, 2, 3, 4, 5],
            velocity=[0.1, 0.1, 0.1, 0.1, 0.1],
            spatial_dims=5
        )
        
        self.assertEqual(node.spatial_dims, 5)
        self.assertEqual(node.position.shape, (5,))
        self.assertEqual(node.attention_focus.shape, (5,))
    
    def test_10d_capacitor_interaction(self):
        """Test capacitor interaction in 10D space."""
        dim = 10
        node = AliveLoopNode(
            position=[0] * dim,
            velocity=[0] * dim,
            spatial_dims=dim
        )
        
        cap = CapacitorInSpace(
            position=[0.1] * dim,
            capacity=5.0,
            expected_dims=dim
        )
        
        # Should work without errors
        node.interact_with_capacitor(cap)
        
        # Verify distance calculation works in high dimensions
        distance = np.linalg.norm(node.position - cap.position)
        expected_distance = np.sqrt(dim * 0.1**2)
        self.assertAlmostEqual(distance, expected_distance, places=5)


if __name__ == '__main__':
    unittest.main()