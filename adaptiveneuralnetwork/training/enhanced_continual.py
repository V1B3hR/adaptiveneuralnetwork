"""
Enhanced continual learning scenarios with progressive domain shift.

This module implements sophisticated continual learning setups including
blurred → corrupted → adversarial progression for robust adaptation.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from ..api.model import AdaptiveModel


@dataclass
class DomainShiftConfig:
    """Configuration for domain shift scenarios."""

    scenario_name: str
    initial_corruption: float = 0.0
    final_corruption: float = 1.0
    num_stages: int = 5
    samples_per_stage: int = 1000
    overlap_ratio: float = 0.2  # Overlap between consecutive stages
    adaptation_episodes: int = 10  # Episodes per stage for adaptation


class ProgressiveDomainShift:
    """Manages progressive domain shift scenarios."""

    def __init__(self, base_dataset: Dataset, config: DomainShiftConfig):
        self.base_dataset = base_dataset
        self.config = config
        self.corruption_functions = self._get_corruption_functions()
        self.current_stage = 0
        self.stage_datasets: List[Dataset] = []
        self._prepare_stage_datasets()

    def _get_corruption_functions(self) -> Dict[str, Callable]:
        """Define corruption functions for different types of domain shift."""
        return {
            "blur": self._apply_blur,
            "noise": self._apply_noise,
            "rotation": self._apply_rotation,
            "brightness": self._apply_brightness,
            "contrast": self._apply_contrast,
            "adversarial": self._apply_adversarial_noise,
        }

    def _apply_blur(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply Gaussian blur corruption."""
        if intensity == 0:
            return x

        # Create Gaussian kernel
        kernel_size = int(1 + 8 * intensity)
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = intensity * 3.0
        blur_transform = transforms.GaussianBlur(kernel_size, sigma)

        # Apply to each image in batch
        if x.dim() == 4:  # Batch of images
            return torch.stack([blur_transform(img) for img in x])
        else:  # Single image
            return blur_transform(x)

    def _apply_noise(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply Gaussian noise corruption."""
        if intensity == 0:
            return x

        noise = torch.randn_like(x) * intensity * 0.3
        return torch.clamp(x + noise, 0, 1)

    def _apply_rotation(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply rotation corruption."""
        if intensity == 0:
            return x

        max_angle = intensity * 45  # Up to 45 degrees
        angle = (torch.rand(1).item() - 0.5) * 2 * max_angle

        rotation_transform = transforms.functional.rotate

        if x.dim() == 4:  # Batch of images
            return torch.stack([rotation_transform(img, angle) for img in x])
        else:  # Single image
            return rotation_transform(x, angle)

    def _apply_brightness(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply brightness corruption."""
        if intensity == 0:
            return x

        # Random brightness factor
        factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * intensity
        return torch.clamp(x * factor, 0, 1)

    def _apply_contrast(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply contrast corruption."""
        if intensity == 0:
            return x

        # Random contrast factor
        factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * intensity
        mean = x.mean()
        return torch.clamp((x - mean) * factor + mean, 0, 1)

    def _apply_adversarial_noise(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply adversarial-like noise corruption."""
        if intensity == 0:
            return x

        # Create structured noise (more realistic than pure Gaussian)
        noise = torch.randn_like(x)

        # Add some structure to the noise
        if x.dim() >= 3:  # Has spatial dimensions
            # Apply some smoothing to create structured adversarial-like patterns
            kernel = torch.ones(1, 1, 3, 3) / 9.0
            if x.dim() == 4:
                noise = F.conv2d(noise, kernel, padding=1, groups=x.shape[1])

        noise = noise * intensity * 0.2
        return torch.clamp(x + noise, 0, 1)

    def _prepare_stage_datasets(self) -> None:
        """Prepare datasets for each stage of domain shift."""
        for stage in range(self.config.num_stages):
            # Calculate corruption intensity for this stage
            progress = stage / (self.config.num_stages - 1) if self.config.num_stages > 1 else 0
            corruption_intensity = self.config.initial_corruption + progress * (
                self.config.final_corruption - self.config.initial_corruption
            )

            # Create corrupted dataset for this stage
            stage_data = []
            stage_labels = []

            # Sample data for this stage
            indices = torch.randperm(len(self.base_dataset))[: self.config.samples_per_stage]

            for idx in indices:
                x, y = self.base_dataset[idx]

                # Apply progressive corruptions based on scenario
                if self.config.scenario_name == "blur_to_adversarial":
                    if stage < self.config.num_stages // 2:
                        # First half: blur progression
                        x_corrupted = self._apply_blur(x, corruption_intensity)
                    else:
                        # Second half: adversarial progression
                        adv_intensity = (stage - self.config.num_stages // 2) / (
                            self.config.num_stages // 2
                        )
                        x_corrupted = self._apply_adversarial_noise(x, adv_intensity)

                elif self.config.scenario_name == "noise_to_rotation":
                    if stage < self.config.num_stages // 2:
                        x_corrupted = self._apply_noise(x, corruption_intensity)
                    else:
                        rot_intensity = (stage - self.config.num_stages // 2) / (
                            self.config.num_stages // 2
                        )
                        x_corrupted = self._apply_rotation(x, rot_intensity)

                elif self.config.scenario_name == "multi_corruption":
                    # Apply multiple corruptions with different intensities
                    x_corrupted = x
                    corruptions = ["blur", "noise", "brightness"]
                    for i, corruption in enumerate(corruptions):
                        if stage >= i * (self.config.num_stages // len(corruptions)):
                            intensity = corruption_intensity * (
                                0.5 + 0.5 * (i + 1) / len(corruptions)
                            )
                            x_corrupted = self.corruption_functions[corruption](
                                x_corrupted, intensity
                            )

                else:  # Default: single corruption type
                    corruption_type = self.config.scenario_name.split("_")[0]
                    if corruption_type in self.corruption_functions:
                        x_corrupted = self.corruption_functions[corruption_type](
                            x, corruption_intensity
                        )
                    else:
                        x_corrupted = self._apply_noise(x, corruption_intensity)

                stage_data.append(x_corrupted)
                stage_labels.append(y)

            # Create dataset for this stage
            stage_dataset = TensorDataset(torch.stack(stage_data), torch.tensor(stage_labels))
            self.stage_datasets.append(stage_dataset)

    def get_stage_dataset(self, stage: int) -> Dataset:
        """Get dataset for a specific stage."""
        if stage < 0 or stage >= len(self.stage_datasets):
            raise ValueError(
                f"Stage {stage} not available. Valid range: 0-{len(self.stage_datasets) - 1}"
            )
        return self.stage_datasets[stage]

    def get_current_stage_dataset(self) -> Dataset:
        """Get dataset for current stage."""
        return self.get_stage_dataset(self.current_stage)

    def advance_stage(self) -> bool:
        """Advance to next stage. Returns True if successful, False if at end."""
        if self.current_stage < len(self.stage_datasets) - 1:
            self.current_stage += 1
            return True
        return False

    def get_stage_info(self, stage: Optional[int] = None) -> Dict[str, Any]:
        """Get information about a stage."""
        if stage is None:
            stage = self.current_stage

        progress = stage / (self.config.num_stages - 1) if self.config.num_stages > 1 else 0
        corruption_intensity = self.config.initial_corruption + progress * (
            self.config.final_corruption - self.config.initial_corruption
        )

        return {
            "stage": stage,
            "total_stages": len(self.stage_datasets),
            "scenario": self.config.scenario_name,
            "corruption_intensity": corruption_intensity,
            "dataset_size": (
                len(self.stage_datasets[stage]) if stage < len(self.stage_datasets) else 0
            ),
            "progress": progress,
        }


class ContinualLearningTrainer:
    """Trainer for enhanced continual learning scenarios."""

    def __init__(self, model: AdaptiveModel, optimizer: torch.optim.Optimizer, device: str = "cpu"):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.stage_history: List[Dict[str, Any]] = []
        self.adaptation_metrics: List[Dict[str, Any]] = []

    def train_stage(
        self,
        dataset: Dataset,
        stage_info: Dict[str, Any],
        epochs: int = 5,
        batch_size: int = 32,
        validation_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """Train on a single stage of continual learning."""

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None
        if validation_dataset:
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        stage_metrics = {
            "stage": stage_info["stage"],
            "scenario": stage_info["scenario"],
            "corruption_intensity": stage_info["corruption_intensity"],
            "epochs": epochs,
            "losses": [],
            "accuracies": [],
            "val_accuracies": [],
            "node_metrics": [],
        }

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

            # Calculate metrics
            epoch_accuracy = correct / total
            avg_loss = epoch_loss / len(dataloader)

            stage_metrics["losses"].append(avg_loss)
            stage_metrics["accuracies"].append(epoch_accuracy)

            # Validation
            if val_loader:
                val_acc = self._evaluate(val_loader)
                stage_metrics["val_accuracies"].append(val_acc)

            # Node metrics
            node_metrics = self.model.get_metrics()
            stage_metrics["node_metrics"].append(node_metrics)

            print(
                f"Stage {stage_info['stage']}, Epoch {epoch}: "
                f"Loss={avg_loss:.4f}, Acc={epoch_accuracy:.4f}"
            )

        # Store stage results
        self.stage_history.append(stage_metrics)

        return stage_metrics

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on given dataloader."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        self.model.train()
        return correct / total

    def measure_forgetting(self, previous_datasets: List[Dataset]) -> Dict[str, float]:
        """Measure catastrophic forgetting on previous tasks."""
        forgetting_metrics = {}

        for i, prev_dataset in enumerate(previous_datasets):
            prev_loader = DataLoader(prev_dataset, batch_size=64, shuffle=False)
            accuracy = self._evaluate(prev_loader)
            forgetting_metrics[f"stage_{i}_accuracy"] = accuracy

        return forgetting_metrics

    def run_continual_scenario(
        self,
        domain_shift: ProgressiveDomainShift,
        epochs_per_stage: int = 5,
        batch_size: int = 32,
        measure_forgetting: bool = True,
    ) -> Dict[str, Any]:
        """Run complete continual learning scenario."""

        scenario_results = {
            "scenario_name": domain_shift.config.scenario_name,
            "stages": [],
            "forgetting_metrics": [],
            "final_summary": {},
        }

        previous_datasets = []

        # Train on each stage
        for stage in range(len(domain_shift.stage_datasets)):
            domain_shift.current_stage = stage
            stage_dataset = domain_shift.get_current_stage_dataset()
            stage_info = domain_shift.get_stage_info()

            print(f"\n--- Training Stage {stage} ---")
            print(f"Scenario: {stage_info['scenario']}")
            print(f"Corruption intensity: {stage_info['corruption_intensity']:.3f}")

            # Train on current stage
            stage_metrics = self.train_stage(
                stage_dataset, stage_info, epochs=epochs_per_stage, batch_size=batch_size
            )

            scenario_results["stages"].append(stage_metrics)

            # Measure forgetting on previous stages
            if measure_forgetting and previous_datasets:
                forgetting = self.measure_forgetting(previous_datasets)
                scenario_results["forgetting_metrics"].append(
                    {"after_stage": stage, "metrics": forgetting}
                )

                print(f"Forgetting metrics: {forgetting}")

            previous_datasets.append(stage_dataset)

        # Final summary
        final_accuracies = [stage["accuracies"][-1] for stage in scenario_results["stages"]]
        scenario_results["final_summary"] = {
            "final_stage_accuracies": final_accuracies,
            "mean_final_accuracy": np.mean(final_accuracies),
            "accuracy_trend": np.polyfit(range(len(final_accuracies)), final_accuracies, 1)[0],
            "total_stages": len(domain_shift.stage_datasets),
        }

        return scenario_results


def create_enhanced_continual_scenario(
    dataset_name: str = "mnist",
    scenario_name: str = "blur_to_adversarial",
    num_stages: int = 5,
    samples_per_stage: int = 1000,
) -> ProgressiveDomainShift:
    """
    Create an enhanced continual learning scenario.

    Args:
        dataset_name: Base dataset to use ("mnist", "cifar10")
        scenario_name: Type of domain shift scenario
        num_stages: Number of progression stages
        samples_per_stage: Samples per stage

    Returns:
        Configured ProgressiveDomainShift instance
    """

    # Load base dataset
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        base_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        base_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create domain shift configuration
    config = DomainShiftConfig(
        scenario_name=scenario_name,
        num_stages=num_stages,
        samples_per_stage=samples_per_stage,
        initial_corruption=0.0,
        final_corruption=1.0,
    )

    return ProgressiveDomainShift(base_dataset, config)


# Example usage functions
def run_blur_to_adversarial_experiment(
    model: AdaptiveModel, optimizer: torch.optim.Optimizer, device: str = "cpu"
) -> Dict[str, Any]:
    """Run blur → adversarial continual learning experiment."""

    # Create scenario
    domain_shift = create_enhanced_continual_scenario(
        dataset_name="mnist",
        scenario_name="blur_to_adversarial",
        num_stages=5,
        samples_per_stage=800,
    )

    # Create trainer
    trainer = ContinualLearningTrainer(model, optimizer, device)

    # Run experiment
    results = trainer.run_continual_scenario(
        domain_shift, epochs_per_stage=3, batch_size=32, measure_forgetting=True
    )

    return results
