#!/usr/bin/env python3
"""
Unified Evaluation Entry Point for Adaptive Neural Network

This script provides a consolidated evaluation interface for trained models.

Usage:
    # Evaluate a checkpoint
    python eval.py --checkpoint checkpoints/model.pt --dataset mnist
    
    # Evaluate with config
    python eval.py --config config/training/mnist.yaml --checkpoint checkpoints/model.pt
    
    # Batch evaluation on multiple checkpoints
    python eval.py --checkpoint-dir checkpoints/ --dataset cifar10
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


# Inline configuration classes (same as train.py)
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


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for Adaptive Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --checkpoint checkpoints/model.pt --dataset mnist
  %(prog)s --config config/training/mnist.yaml --checkpoint checkpoints/model.pt
  %(prog)s --checkpoint-dir checkpoints/ --dataset cifar10
        """
    )
    
    # Checkpoint specification
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint file"
    )
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory containing checkpoints (evaluates all)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML/JSON configuration file"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset name for evaluation"
    )
    
    # Evaluation parameters
    parser.add_argument("--data-path", type=str, help="Path to evaluation dataset")
    parser.add_argument("--batch-size", type=int, help="Evaluation batch size")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--save-predictions", action="store_true", help="Save model predictions")
    
    # Metrics
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to compute (e.g., accuracy loss f1)"
    )
    
    # Advanced options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    return parser


def load_eval_config(args: argparse.Namespace) -> WorkflowConfig:
    """Load evaluation configuration."""
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
        config = WorkflowConfig()
        config.dataset.name = args.dataset
    else:
        raise ValueError("Either --config or --dataset must be specified")
    
    # Override with CLI arguments
    if args.data_path is not None:
        config.dataset.data_path = args.data_path
    if args.batch_size is not None:
        config.evaluation.batch_size = args.batch_size
    if args.device is not None:
        config.training.device = args.device
    if args.output_dir is not None:
        config.evaluation.output_dir = args.output_dir
    if args.save_predictions:
        config.evaluation.save_predictions = True
    if args.metrics is not None:
        config.evaluation.metrics = args.metrics
    
    return config


def evaluate_checkpoint(checkpoint_path: Path, config: WorkflowConfig):
    """
    Evaluate a single checkpoint.
    
    This is a placeholder for the actual evaluation logic.
    """
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {config.dataset.name}")
    
    # Placeholder implementation
    # In a real implementation, this would:
    # 1. Load the checkpoint
    # 2. Load the evaluation dataset
    # 3. Run inference
    # 4. Compute metrics
    # 5. Save results
    
    logger.warning("Evaluation not yet fully integrated - placeholder")
    
    results = {
        "checkpoint": str(checkpoint_path),
        "dataset": config.dataset.name,
        "metrics": {}
    }
    
    for metric in config.evaluation.metrics:
        results["metrics"][metric] = 0.0  # Placeholder
    
    logger.info(f"Results: {results['metrics']}")
    
    return results


def evaluate_all_checkpoints(checkpoint_dir: Path, config: WorkflowConfig):
    """Evaluate all checkpoints in a directory."""
    logger.info(f"Evaluating all checkpoints in: {checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")
        return []
    
    logger.info(f"Found {len(checkpoint_files)} checkpoints")
    
    all_results = []
    for checkpoint_path in checkpoint_files:
        try:
            results = evaluate_checkpoint(checkpoint_path, config)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
    
    return all_results


def print_results(results):
    """Pretty print evaluation results."""
    if isinstance(results, list):
        print("\n" + "=" * 70)
        print("Evaluation Results Summary")
        print("=" * 70)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {Path(result['checkpoint']).name}")
            for metric, value in result['metrics'].items():
                print(f"   {metric:15s}: {value:.4f}")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        for metric, value in results['metrics'].items():
            print(f"{metric:15s}: {value:.4f}")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_eval_config(args)
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Run evaluation
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return 1
            results = evaluate_checkpoint(checkpoint_path, config)
            print_results(results)
        else:
            checkpoint_dir = Path(args.checkpoint_dir)
            if not checkpoint_dir.exists():
                logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
                return 1
            results = evaluate_all_checkpoints(checkpoint_dir, config)
            print_results(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
