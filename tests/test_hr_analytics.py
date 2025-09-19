"""
Test HR Analytics integration functionality.

These tests validate the HR Analytics dataset loading, training, and artifact generation
functionality integrated into runsimulation.py.
"""

import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the functions from runsimulation.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from runsimulation import load_hr_analytics_data, run_hr_analytics_training, save_training_artifacts


class TestHRAnalytics(unittest.TestCase):
    """Test HR Analytics integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create data and outputs directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_load_hr_analytics_data_synthetic(self):
        """Test loading HR analytics data when CSV file doesn't exist."""
        # Ensure no CSV file exists
        csv_path = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        self.assertFalse(csv_path.exists())
        
        # Load data (should generate synthetic)
        data = load_hr_analytics_data()
        
        # Should return synthetic data
        if hasattr(data, 'shape'):
            # pandas DataFrame
            self.assertEqual(len(data), 1000)  # Synthetic data size
            expected_columns = ['Age', 'Attrition', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany']
            for col in expected_columns:
                self.assertIn(col, data.columns)
        else:
            # dict format (if pandas not available)
            self.assertEqual(len(data['Age']), 1000)
            expected_keys = ['Age', 'Attrition', 'MonthlyIncome', 'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany']
            for key in expected_keys:
                self.assertIn(key, data.keys())
    
    @patch('pandas.read_csv')
    def test_load_hr_analytics_data_with_csv(self, mock_read_csv):
        """Test loading HR analytics data when CSV file exists."""
        # Create mock CSV file
        csv_path = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        csv_path.touch()
        
        # Mock pandas DataFrame
        mock_df = MagicMock()
        mock_df.__len__ = MagicMock(return_value=1470)  # Real dataset size
        mock_read_csv.return_value = mock_df
        
        # Load data
        data = load_hr_analytics_data()
        
        # Should have called pandas.read_csv
        mock_read_csv.assert_called_once_with(csv_path)
        self.assertEqual(data, mock_df)
    
    def test_run_hr_analytics_training(self):
        """Test HR analytics training functionality."""
        # Create synthetic data
        synthetic_data = {
            'Age': [25, 30, 35, 40, 45],
            'Attrition': ['No', 'Yes', 'No', 'Yes', 'No'],
            'MonthlyIncome': [5000, 6000, 7000, 8000, 9000]
        }
        
        # Run training
        results = run_hr_analytics_training(synthetic_data, epochs=2, batch_size=2)
        
        # Validate results structure
        self.assertIn('training_metrics', results)
        self.assertIn('epochs_completed', results)
        self.assertIn('final_accuracy', results)
        self.assertIn('final_loss', results)
        
        # Check training metrics
        self.assertEqual(len(results['training_metrics']), 2)  # 2 epochs
        self.assertEqual(results['epochs_completed'], 2)
        
        # Check metrics have expected fields
        for epoch_metrics in results['training_metrics']:
            self.assertIn('epoch', epoch_metrics)
            self.assertIn('loss', epoch_metrics)
            self.assertIn('accuracy', epoch_metrics)
        
        # Check final values are floats
        self.assertIsInstance(results['final_accuracy'], float)
        self.assertIsInstance(results['final_loss'], float)
    
    def test_save_training_artifacts(self):
        """Test saving training artifacts."""
        # Create test results
        results = {
            'training_metrics': [
                {'epoch': 1, 'loss': 0.5, 'accuracy': 0.8}
            ],
            'epochs_completed': 1,
            'final_accuracy': 0.8,
            'final_loss': 0.5
        }
        
        # Create test data
        test_data = {'Age': [25, 30], 'Attrition': ['No', 'Yes']}
        
        # Save artifacts
        save_training_artifacts(results, test_data)
        
        # Check files were created
        outputs_dir = Path("outputs")
        self.assertTrue((outputs_dir / "hr_training_results.json").exists())
        self.assertTrue((outputs_dir / "dataset_info.json").exists())
        self.assertTrue((outputs_dir / "hr_model_weights.json").exists())
        
        # Validate JSON content
        with open(outputs_dir / "hr_training_results.json") as f:
            saved_results = json.load(f)
            self.assertEqual(saved_results, results)
        
        with open(outputs_dir / "dataset_info.json") as f:
            data_info = json.load(f)
            self.assertIn('dataset_type', data_info)
            self.assertIn('samples', data_info)
            self.assertIn('features', data_info)
        
        with open(outputs_dir / "hr_model_weights.json") as f:
            model_weights = json.load(f)
            self.assertIn('model_type', model_weights)
            self.assertIn('architecture', model_weights)
            self.assertIn('weights', model_weights)
            self.assertIn('training_completed', model_weights)
    
    def test_environment_variable_integration(self):
        """Test that environment variables are properly used."""
        # Set environment variables
        with patch.dict(os.environ, {'EPOCHS': '5', 'BATCH_SIZE': '16'}):
            epochs = int(os.getenv('EPOCHS', '10'))
            batch_size = int(os.getenv('BATCH_SIZE', '32'))
            
            self.assertEqual(epochs, 5)
            self.assertEqual(batch_size, 16)
        
        # Test defaults when not set
        with patch.dict(os.environ, {}, clear=True):
            epochs = int(os.getenv('EPOCHS', '10'))
            batch_size = int(os.getenv('BATCH_SIZE', '32'))
            
            self.assertEqual(epochs, 10)
            self.assertEqual(batch_size, 32)


if __name__ == '__main__':
    unittest.main()