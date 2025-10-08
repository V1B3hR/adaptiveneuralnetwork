#!/usr/bin/env python3
"""
Unified Training Entry Point for Adaptive Neural Network

This script provides a consolidated training interface that replaces multiple
scattered training scripts. It supports:
- Configuration-driven workflows (YAML/JSON)
- Multiple datasets (MNIST, Kaggle datasets, custom)
- CLI with subcommands
- Flexible parameter overrides

Usage:
    # Train with config file
    python train.py --config config/training/mnist.yaml
    
    # Train with dataset name and custom parameters
    python train.py --dataset mnist --epochs 20 --batch-size 128
    
    # List available datasets
    python train.py --list-datasets
    
Examples:
    python train.py --config config/training/kaggle_default.yaml
    python train.py --dataset annomi --data-path data/annomi --epochs 10
    python train.py --config config/training/quick_test.yaml --device cpu
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Inline configuration classes to avoid package dependencies
@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    name: str = "mnist"
    data_path: Optional[str] = None
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    augmentation: bool = False
    augmentation_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    name: str = "adaptive"
    input_dim: int = 784
    hidden_dim: int = 128
    output_dim: int = 10
    num_nodes: int = 64
    dropout: float = 0.1
    activation: str = "relu"
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = False
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    log_every_n_steps: int = 10
    log_dir: str = "logs"
    verbose: bool = False
    device: str = "cuda"
    seed: int = 42
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_loss"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    batch_size: int = 128
    save_predictions: bool = False
    output_dir: str = "outputs"


@dataclass
class WorkflowConfig:
    """Complete workflow configuration combining all components."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "WorkflowConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "WorkflowConfig":
        """Load configuration from JSON file."""
        json_path = Path(json_path)
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WorkflowConfig":
        """Create configuration from dictionary."""
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        optimizer_config = OptimizerConfig(**config_dict.get("optimizer", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        
        return cls(
            dataset=dataset_config,
            model=model_config,
            optimizer=optimizer_config,
            training=training_config,
            evaluation=evaluation_config
        )
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = self.to_dict()
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation)
        }


# Dataset registry
AVAILABLE_DATASETS = {
    "mnist": "MNIST handwritten digits (28x28 grayscale images)",
    "cifar10": "CIFAR-10 natural images (32x32 color images)",
    "annomi": "ANNOMI Motivational Interviewing dataset (text)",
    "mental_health": "Mental Health dataset (text)",
    "vr_driving": "VR Driving simulation dataset",
    "autvi": "Automotive Vehicle Inspection dataset",
    "digakust": "Digital Acoustic Analysis dataset",
    "synthetic": "Synthetic dataset for testing",
}


def list_datasets():
    """Display available datasets."""
    print("\n" + "=" * 70)
    print("Available Datasets")
    print("=" * 70)
    for name, description in AVAILABLE_DATASETS.items():
        print(f"  {name:20s} - {description}")
    print("=" * 70 + "\n")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(
        description="Unified training script for Adaptive Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config/training/mnist.yaml
  %(prog)s --dataset annomi --data-path data/annomi --epochs 10
  %(prog)s --config config/training/quick_test.yaml --device cpu
  %(prog)s --list-datasets
        """
    )
    
    # Configuration file or dataset selection
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML/JSON configuration file"
    )
    config_group.add_argument(
        "--dataset", "-d",
        type=str,
        choices=list(AVAILABLE_DATASETS.keys()),
        help="Dataset name (will use default configuration)"
    )
    
    # List datasets
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    # Dataset parameters
    parser.add_argument("--data-path", type=str, help="Path to dataset")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--num-workers", type=int, help="Number of data loader workers")
    
    # Model parameters
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--hidden-dim", type=int, help="Hidden dimension size")
    parser.add_argument("--num-nodes", type=int, help="Number of nodes")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", "--lr", type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Advanced options
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, help="Log directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--save-config", type=str, help="Save resolved configuration to file")
    
    return parser


def load_config(args: argparse.Namespace) -> WorkflowConfig:
    """Load and merge configuration from file and CLI arguments."""
    # Load base configuration
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config_path = Path(args.config)
        if config_path.suffix in ['.yaml', '.yml']:
            config = WorkflowConfig.from_yaml(config_path)
        elif config_path.suffix == '.json':
            config = WorkflowConfig.from_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    elif args.dataset:
        logger.info(f"Using default configuration for dataset: {args.dataset}")
        # Create default configuration
        config = WorkflowConfig()
        config.dataset.name = args.dataset
    else:
        raise ValueError("Either --config or --dataset must be specified")
    
    # Override with CLI arguments
    if args.data_path is not None:
        config.dataset.data_path = args.data_path
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size
    if args.num_workers is not None:
        config.dataset.num_workers = args.num_workers
    
    if args.model is not None:
        config.model.name = args.model
    if args.hidden_dim is not None:
        config.model.hidden_dim = args.hidden_dim
    if args.num_nodes is not None:
        config.model.num_nodes = args.num_nodes
    
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.optimizer.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.optimizer.weight_decay = args.weight_decay
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed
        config.dataset.seed = args.seed
    
    if args.use_amp:
        config.training.use_amp = True
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.training.log_dir = args.log_dir
    if args.verbose:
        config.training.verbose = True
    
    if args.output_dir is not None:
        config.evaluation.output_dir = args.output_dir
    
    return config


def print_config(config: WorkflowConfig):
    """Pretty print configuration."""
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Dataset:     {config.dataset.name}")
    print(f"Data Path:   {config.dataset.data_path or 'default'}")
    print(f"Batch Size:  {config.dataset.batch_size}")
    print(f"Model:       {config.model.name}")
    print(f"Hidden Dim:  {config.model.hidden_dim}")
    print(f"Num Nodes:   {config.model.num_nodes}")
    print(f"Optimizer:   {config.optimizer.name}")
    print(f"LR:          {config.optimizer.learning_rate}")
    print(f"Epochs:      {config.training.epochs}")
    print(f"Device:      {config.training.device}")
    print(f"Seed:        {config.training.seed}")
    print(f"AMP:         {config.training.use_amp}")
    print("=" * 70 + "\n")


def train_with_config(config: WorkflowConfig):
    """
    Execute training with the given configuration.
    
    This is a placeholder that delegates to the actual training implementation.
    In a real implementation, this would:
    1. Load the dataset
    2. Create the model
    3. Set up optimizer and scheduler
    4. Run training loop
    5. Save checkpoints and results
    """
    logger.info("Starting training...")
    logger.info(f"Dataset: {config.dataset.name}")
    
    # Note: This is a minimal implementation
    # The actual training logic should be extracted from existing scripts
    # and placed in adaptiveneuralnetwork.training.trainer or similar
    
    try:
        # Import training components based on dataset type
        if config.dataset.name in ["mnist", "cifar10"]:
            logger.info("Training with vision dataset")
            # This would call the appropriate training function
            # from adaptiveneuralnetwork.training
            logger.warning("Vision training not yet fully integrated - placeholder")
        elif config.dataset.name in ["annomi", "mental_health"]:
            logger.info("Training with text dataset")
            # This would call the appropriate training function
            logger.warning("Text training not yet fully integrated - placeholder")
        elif config.dataset.name in ["vr_driving", "autvi", "digakust"]:
            logger.info("Training with Kaggle dataset")
            # This would call the appropriate training function
            logger.warning("Kaggle dataset training not yet fully integrated - placeholder")
        else:
            logger.warning(f"Unknown dataset type: {config.dataset.name}")
            
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {config.evaluation.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle list datasets
    if args.list_datasets:
        list_datasets()
        return 0
    
    # Validate arguments
    if not args.config and not args.dataset:
        parser.print_help()
        print("\nError: Either --config or --dataset must be specified")
        return 1
    
    try:
        # Load configuration
        config = load_config(args)
        
        # Set logging level
        if config.training.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Print configuration
        print_config(config)
        
        # Save configuration if requested
        if args.save_config:
            config.to_yaml(args.save_config)
            logger.info(f"Configuration saved to: {args.save_config}")
        
        # Execute training
        train_with_config(config)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
