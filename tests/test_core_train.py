"""
Test suite for core.train module.

Tests the training entry point for adaptive neural network datasets.
"""

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


class TestCoreTrainModule(unittest.TestCase):
    """Test the core.train module functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_output_dir = Path("/tmp/test_core_train_outputs")
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
        self.test_output_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def test_core_train_help(self):
        """Test that core.train --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "core.train", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Training Script", result.stdout)
        self.assertIn("--dataset", result.stdout)
        self.assertIn("--epochs", result.stdout)

    def test_core_train_single_dataset(self):
        """Test training on a single dataset with minimal epochs."""
        result = subprocess.run(
            [
                sys.executable, "-m", "core.train",
                "--dataset", "vr_driving",
                "--epochs", "2",
                "--num-samples", "100",
                "--output-dir", str(self.test_output_dir)
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        self.assertEqual(result.returncode, 0, f"Training failed: {result.stderr}")
        self.assertIn("Training complete", result.stdout)

        # Check that output file was created
        output_file = self.test_output_dir / "vr_driving_training_results.json"
        self.assertTrue(output_file.exists(), "Output file not created")

        # Verify output file contains expected data
        with open(output_file) as f:
            results = json.load(f)

        self.assertTrue(results.get("success"), "Training should succeed")
        self.assertEqual(results.get("dataset_type"), "vr_driving")
        self.assertEqual(results.get("epochs_completed"), 2)
        self.assertIn("best_accuracy", results)
        self.assertIn("training_time", results)

    def test_core_train_all_datasets(self):
        """Test training on all datasets with minimal epochs."""
        result = subprocess.run(
            [
                sys.executable, "-m", "core.train",
                "--dataset", "all",
                "--epochs", "2",
                "--num-samples", "50",
                "--output-dir", str(self.test_output_dir)
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        self.assertEqual(result.returncode, 0, f"Training failed: {result.stderr}")
        self.assertIn("Training complete", result.stdout)
        self.assertIn("TRAINING SUMMARY", result.stdout)

        # Check that all individual output files were created
        datasets = ["vr_driving", "autvi", "digakust"]
        for dataset in datasets:
            output_file = self.test_output_dir / f"{dataset}_training_results.json"
            self.assertTrue(output_file.exists(), f"Output file for {dataset} not created")

        # Check that combined results file was created
        combined_file = self.test_output_dir / "all_datasets_results.json"
        self.assertTrue(combined_file.exists(), "Combined results file not created")

        # Verify combined results contain all datasets
        with open(combined_file) as f:
            all_results = json.load(f)

        for dataset in datasets:
            self.assertIn(dataset, all_results, f"{dataset} not in combined results")
            self.assertTrue(all_results[dataset].get("success"), f"{dataset} training failed")

    def test_core_train_with_30_epochs(self):
        """Test training with 30 epochs as specified in the problem statement."""
        result = subprocess.run(
            [
                sys.executable, "-m", "core.train",
                "--dataset", "vr_driving",
                "--epochs", "30",
                "--num-samples", "100",
                "--output-dir", str(self.test_output_dir)
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        self.assertEqual(result.returncode, 0, f"Training failed: {result.stderr}")
        self.assertIn("Training complete", result.stdout)

        # Verify training completed with 30 epochs
        output_file = self.test_output_dir / "vr_driving_training_results.json"
        with open(output_file) as f:
            results = json.load(f)

        self.assertEqual(results.get("epochs_completed"), 30)


if __name__ == "__main__":
    unittest.main()
