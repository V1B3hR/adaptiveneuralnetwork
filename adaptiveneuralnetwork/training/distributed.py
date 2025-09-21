"""
Distributed training support for adaptive neural networks.

This module provides support for distributed training using both PyTorch's
native distributed training and Ray for large-scale parallel training.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ..api.config import AdaptiveConfig
from ..api.model import AdaptiveModel


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    init_method: Optional[str] = None
    use_ray: bool = False
    ray_num_workers: int = 2
    ray_resources_per_worker: Dict[str, float] = None


class DistributedTrainer:
    """Distributed trainer for adaptive neural networks."""

    def __init__(
        self, model: AdaptiveModel, config: DistributedConfig, device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = config.world_size > 1
        self.is_main_process = config.rank == 0
        self.ddp_model = None

        # Initialize distributed training if needed
        if self.is_distributed and not config.use_ray:
            self._init_pytorch_distributed()

    def _init_pytorch_distributed(self) -> None:
        """Initialize PyTorch distributed training."""
        if not dist.is_available():
            raise RuntimeError("Distributed training not available")

        # Set environment variables if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = self.config.master_addr
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = self.config.master_port
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(self.config.world_size)
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(self.config.rank)

        # Initialize process group
        init_method = self.config.init_method or "env://"

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                init_method=init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
            )

        # Setup device
        if torch.cuda.is_available() and self.config.backend == "nccl":
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f"cuda:{self.config.local_rank}")

        # Move model to device and wrap with DDP
        self.model = self.model.to(self.device)
        self.ddp_model = DDP(
            self.model, device_ids=[self.config.local_rank] if torch.cuda.is_available() else None
        )

        if self.is_main_process:
            print(f"Initialized distributed training with {self.config.world_size} processes")

    def create_distributed_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = True, **kwargs
    ) -> DataLoader:
        """Create distributed dataloader with proper sampling."""
        if self.is_distributed and not self.config.use_ray:
            sampler = DistributedSampler(
                dataset, num_replicas=self.config.world_size, rank=self.config.rank, shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **kwargs
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epoch: int,
        gradient_clip_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Train for one epoch with distributed coordination."""
        # Set sampler epoch for proper shuffling
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        # Use DDP model if available, otherwise regular model
        model = self.ddp_model if self.ddp_model is not None else self.model
        model.train()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += target.size(0)

            # Log progress (only main process)
            if self.is_main_process and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples

        # Aggregate metrics across processes if distributed
        if self.is_distributed and not self.config.use_ray:
            metrics_tensor = torch.tensor([avg_loss, accuracy, total_samples], device=self.device)
            dist.all_reduce(metrics_tensor)

            # Average loss and accuracy, sum total samples
            avg_loss = metrics_tensor[0].item() / self.config.world_size
            accuracy = (metrics_tensor[1].item() * metrics_tensor[2].item()) / metrics_tensor[
                2
            ].item()

        return {"loss": avg_loss, "accuracy": accuracy, "samples": total_samples}

    def evaluate(self, dataloader: DataLoader, criterion: torch.nn.Module) -> Dict[str, float]:
        """Evaluate model with distributed coordination."""
        model = self.ddp_model if self.ddp_model is not None else self.model
        model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total_samples

        # Aggregate metrics across processes if distributed
        if self.is_distributed and not self.config.use_ray:
            metrics_tensor = torch.tensor([avg_loss, accuracy, total_samples], device=self.device)
            dist.all_reduce(metrics_tensor)

            avg_loss = metrics_tensor[0].item() / self.config.world_size
            accuracy = (metrics_tensor[1].item() * metrics_tensor[2].item()) / metrics_tensor[
                2
            ].item()

        return {"loss": avg_loss, "accuracy": accuracy, "samples": total_samples}

    def save_checkpoint(self, filepath: Union[str, Path], **extra_data) -> None:
        """Save checkpoint (only from main process)."""
        if not self.is_main_process:
            return

        # Get state dict from underlying model (not DDP wrapper)
        if self.ddp_model is not None:
            model_state = self.ddp_model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
            "config": self.model.config,
            "distributed_config": self.config,
            **extra_data,
        }

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load into underlying model (not DDP wrapper)
        if self.ddp_model is not None:
            self.ddp_model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Checkpoint loaded: {filepath}")
        return checkpoint

    def cleanup(self) -> None:
        """Cleanup distributed training."""
        if self.is_distributed and not self.config.use_ray and dist.is_initialized():
            dist.destroy_process_group()


class RayDistributedTrainer:
    """Ray-based distributed trainer for large-scale training."""

    def __init__(
        self,
        model_config: AdaptiveConfig,
        distributed_config: DistributedConfig,
        training_function: Callable,
    ):
        self.model_config = model_config
        self.distributed_config = distributed_config
        self.training_function = training_function

        try:
            import ray
            from ray import train
            from ray.train import torch as ray_torch

            self.ray = ray
            self.ray_train = train
            self.ray_torch = ray_torch
        except ImportError:
            raise ImportError("Ray not installed. Install with: pip install ray[train]")

    def setup_ray_training(self) -> None:
        """Setup Ray training environment."""
        if not self.ray.is_initialized():
            self.ray.init()

        # Configure Ray training
        from ray.train import ScalingConfig
        from ray.train.torch import TorchTrainer

        scaling_config = ScalingConfig(
            num_workers=self.distributed_config.ray_num_workers,
            use_gpu=torch.cuda.is_available(),
            resources_per_worker=self.distributed_config.ray_resources_per_worker
            or {"CPU": 1, "GPU": 0.5 if torch.cuda.is_available() else 0},
        )

        # Create trainer
        self.trainer = TorchTrainer(
            train_loop_per_worker=self._ray_training_function,
            train_loop_config={
                "model_config": self.model_config,
                "distributed_config": self.distributed_config,
            },
            scaling_config=scaling_config,
        )

    def _ray_training_function(self, config: Dict[str, Any]) -> None:
        """Ray training function to run on each worker."""
        # This function runs on each Ray worker
        model_config = config["model_config"]

        # Create model on worker
        model = AdaptiveModel(model_config)

        # Prepare for distributed training
        model = self.ray_torch.prepare_model(model)

        # Run the user-provided training function
        results = self.training_function(model, config)

        # Report results back to Ray
        self.ray_train.report(results)

    def run_training(self, **training_kwargs) -> Any:
        """Run distributed training with Ray."""
        self.setup_ray_training()

        result = self.trainer.fit()
        return result

    def shutdown(self) -> None:
        """Shutdown Ray."""
        if self.ray.is_initialized():
            self.ray.shutdown()


def run_distributed_training(
    model_config: AdaptiveConfig,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    distributed_config: Optional[DistributedConfig] = None,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
) -> Dict[str, Any]:
    """
    Run distributed training for adaptive neural network.

    This function can be called directly for single-process training,
    or launched via torch.distributed.launch for multi-process training.
    """
    # Create distributed config
    if distributed_config is None:
        distributed_config = DistributedConfig(
            world_size=world_size, rank=rank, local_rank=local_rank
        )

    # Create model and trainer
    model = AdaptiveModel(model_config)
    trainer = DistributedTrainer(model, distributed_config)

    # Create distributed dataloaders
    train_loader = trainer.create_distributed_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = trainer.create_distributed_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    results = {"train_history": [], "val_history": []}

    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer, criterion, epoch)
        results["train_history"].append(train_metrics)

        # Validate
        if val_loader is not None:
            val_metrics = trainer.evaluate(val_loader, criterion)
            results["val_history"].append(val_metrics)

            # Save best model
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                trainer.save_checkpoint(
                    f"best_model_rank_{rank}.pt", epoch=epoch, best_val_acc=best_val_acc
                )

        # Print progress (main process only)
        if trainer.is_main_process:
            print(
                f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.4f}"
            )
            if val_loader is not None:
                print(
                    f"  Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
                )

    # Cleanup
    trainer.cleanup()

    return results


def launch_distributed_training(
    script_path: str,
    num_processes: int,
    num_nodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: str = "12355",
    **script_args,
) -> subprocess.CompletedProcess:
    """
    Launch distributed training using torch.distributed.launch.

    Args:
        script_path: Path to training script
        num_processes: Number of processes per node
        num_nodes: Number of nodes
        node_rank: Rank of current node
        master_addr: Master node address
        master_port: Master node port
        **script_args: Arguments to pass to training script

    Returns:
        Completed process result
    """
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        str(num_processes),
        "--nnodes",
        str(num_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        master_port,
        script_path,
    ]

    # Add script arguments
    for key, value in script_args.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"Launching distributed training: {' '.join(cmd)}")

    return subprocess.run(cmd, capture_output=True, text=True)


# Convenience functions for different backends
def create_pytorch_distributed_config(
    world_size: int,
    rank: int,
    local_rank: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> DistributedConfig:
    """Create PyTorch distributed configuration."""
    return DistributedConfig(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        use_ray=False,
    )


def create_ray_distributed_config(
    num_workers: int = 2, resources_per_worker: Optional[Dict[str, float]] = None
) -> DistributedConfig:
    """Create Ray distributed configuration."""
    return DistributedConfig(
        use_ray=True,
        ray_num_workers=num_workers,
        ray_resources_per_worker=resources_per_worker or {"CPU": 1, "GPU": 0.5},
    )
