"""
Tests for unified training and evaluation entry points.

These tests validate the configuration system and CLI interfaces
without requiring heavy dependencies like torch.
"""

# Test configuration loading without importing torch-dependent modules
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the config directly from train.py to avoid torch dependency
import eval as eval_module
import train


class TestConfigurationSystem:
    """Test configuration classes and serialization."""

    def test_dataset_config_defaults(self):
        """Test DatasetConfig with default values."""
        config = train.DatasetConfig()
        assert config.name == "mnist"
        assert config.batch_size == 64
        assert config.seed == 42

    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = train.ModelConfig()
        assert config.name == "adaptive"
        assert config.hidden_dim == 128
        assert config.num_nodes == 64

    def test_workflow_config_to_dict(self):
        """Test WorkflowConfig conversion to dictionary."""
        config = train.WorkflowConfig()
        config_dict = config.to_dict()

        assert "dataset" in config_dict
        assert "model" in config_dict
        assert "optimizer" in config_dict
        assert "training" in config_dict
        assert "evaluation" in config_dict

    def test_workflow_config_from_dict(self):
        """Test WorkflowConfig creation from dictionary."""
        config_dict = {
            "dataset": {"name": "cifar10", "batch_size": 128},
            "model": {"hidden_dim": 256},
            "training": {"epochs": 20}
        }
        config = train.WorkflowConfig.from_dict(config_dict)

        assert config.dataset.name == "cifar10"
        assert config.dataset.batch_size == 128
        assert config.model.hidden_dim == 256
        assert config.training.epochs == 20

    def test_workflow_config_yaml_roundtrip(self):
        """Test saving and loading configuration from YAML."""
        config = train.WorkflowConfig()
        config.dataset.name = "test_dataset"
        config.model.hidden_dim = 512
        config.training.epochs = 15

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name

        try:
            # Save to YAML
            config.to_yaml(yaml_path)

            # Load from YAML
            loaded_config = train.WorkflowConfig.from_yaml(yaml_path)

            assert loaded_config.dataset.name == "test_dataset"
            assert loaded_config.model.hidden_dim == 512
            assert loaded_config.training.epochs == 15
        finally:
            Path(yaml_path).unlink()

    def test_config_file_templates_valid(self):
        """Test that configuration file templates are valid."""
        config_dir = Path(__file__).parent.parent / "config" / "training"

        if not config_dir.exists():
            pytest.skip("Config directory not found")

        config_files = list(config_dir.glob("*.yaml"))
        assert len(config_files) > 0, "No config files found"

        for config_file in config_files:
            # Should load without errors
            config = train.WorkflowConfig.from_yaml(config_file)
            assert config is not None
            assert config.dataset.name is not None
            assert config.training.epochs > 0


class TestTrainScript:
    """Test train.py functionality."""

    def test_available_datasets_defined(self):
        """Test that AVAILABLE_DATASETS is defined."""
        assert hasattr(train, 'AVAILABLE_DATASETS')
        assert len(train.AVAILABLE_DATASETS) > 0
        assert "mnist" in train.AVAILABLE_DATASETS

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = train.create_parser()
        assert parser is not None

        # Test help doesn't crash
        try:
            parser.parse_args(['--help'])
        except SystemExit:
            pass  # Expected for --help

    def test_load_config_with_dataset(self):
        """Test loading config with dataset argument."""
        # Create mock args
        class Args:
            config = None
            dataset = "mnist"
            data_path = None
            batch_size = None
            num_workers = None
            model = None
            hidden_dim = None
            num_nodes = None
            epochs = None
            learning_rate = None
            weight_decay = None
            device = None
            seed = None
            use_amp = False
            checkpoint_dir = None
            log_dir = None
            verbose = False
            output_dir = None

        config = train.load_config(Args())
        assert config.dataset.name == "mnist"

    def test_load_config_with_overrides(self):
        """Test loading config with CLI overrides."""
        class Args:
            config = None
            dataset = "cifar10"
            data_path = "/custom/path"
            batch_size = 256
            num_workers = 8
            model = None
            hidden_dim = 512
            num_nodes = None
            epochs = 25
            learning_rate = 0.01
            weight_decay = None
            device = "cpu"
            seed = 123
            use_amp = True
            checkpoint_dir = None
            log_dir = None
            verbose = True
            output_dir = None

        config = train.load_config(Args())
        assert config.dataset.name == "cifar10"
        assert config.dataset.data_path == "/custom/path"
        assert config.dataset.batch_size == 256
        assert config.dataset.num_workers == 8
        assert config.model.hidden_dim == 512
        assert config.training.epochs == 25
        assert config.optimizer.learning_rate == 0.01
        assert config.training.device == "cpu"
        assert config.training.seed == 123
        assert config.training.use_amp == True


class TestEvalScript:
    """Test eval.py functionality."""

    def test_create_parser(self):
        """Test argument parser creation."""
        parser = eval_module.create_parser()
        assert parser is not None

        # Test help doesn't crash
        try:
            parser.parse_args(['--help'])
        except SystemExit:
            pass  # Expected for --help

    def test_load_eval_config(self):
        """Test loading evaluation config."""
        class Args:
            config = None
            dataset = "mnist"
            data_path = None
            batch_size = 64
            device = "cpu"
            output_dir = None
            save_predictions = False
            metrics = None

        config = eval_module.load_eval_config(Args())
        assert config.dataset.name == "mnist"
        assert config.evaluation.batch_size == 64
        assert config.training.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
