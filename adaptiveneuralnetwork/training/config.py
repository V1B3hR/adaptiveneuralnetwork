"""
Training configuration module for Adaptive Neural Network.

Provides configuration classes for training workflows, datasets, and model parameters.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


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
    
    # Data augmentation
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
    
    # Model-specific parameters
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
    
    # Scheduler configuration
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Mixed precision training
    use_amp: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 10
    log_dir: str = "logs"
    verbose: bool = False
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    # Early stopping
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
        # Parse sub-configurations
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
        
        # Convert tuples to lists for YAML serialization
        if isinstance(config_dict.get('optimizer', {}).get('betas'), tuple):
            config_dict['optimizer']['betas'] = list(config_dict['optimizer']['betas'])
        
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation)
        }
