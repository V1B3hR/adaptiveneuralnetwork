"""
Dataset utilities for adaptive neural network training.

This module provides dataset loading and preprocessing for various
benchmarks including MNIST.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


def load_mnist(
    batch_size: int = 64,
    root: str = "./data",
    download: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset with standard preprocessing.
    
    Args:
        batch_size: Batch size for data loaders
        root: Root directory for dataset storage
        download: Whether to download dataset if not present
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Standard MNIST preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load train and test sets
    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=download
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=download
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


def load_mnist_subset(
    batch_size: int = 64,
    subset_size: int = 1000,
    root: str = "./data",
    download: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Load a subset of MNIST for quick testing and development.
    
    Args:
        batch_size: Batch size for data loaders
        subset_size: Number of samples to use from train set
        root: Root directory for dataset storage  
        download: Whether to download dataset if not present
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get full loaders first
    full_train_loader, test_loader = load_mnist(
        batch_size=batch_size,
        root=root,
        download=download,
        num_workers=num_workers
    )
    
    # Create subset of training data
    train_dataset = full_train_loader.dataset
    subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing and development."""
    
    def __init__(
        self, 
        num_samples: int = 1000,
        input_dim: int = 784,
        num_classes: int = 10,
        noise_level: float = 0.1
    ):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        torch.manual_seed(42)  # For reproducibility
        
        # Create class-specific patterns
        self.class_patterns = torch.randn(num_classes, input_dim)
        
        # Generate samples
        self.data = []
        self.targets = []
        
        for i in range(num_samples):
            class_id = torch.randint(0, num_classes, (1,)).item()
            
            # Start with class pattern and add noise
            sample = self.class_patterns[class_id] + noise_level * torch.randn(input_dim)
            
            self.data.append(sample)
            self.targets.append(class_id)
            
        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def create_synthetic_loaders(
    num_samples: int = 1000,
    batch_size: int = 64,
    input_dim: int = 784,
    num_classes: int = 10,
    test_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create synthetic data loaders for testing.
    
    Args:
        num_samples: Total number of samples
        batch_size: Batch size for data loaders
        input_dim: Input dimension
        num_classes: Number of classes
        test_split: Fraction of data to use for testing
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_size = int(num_samples * (1 - test_split))
    test_size = num_samples - train_size
    
    train_dataset = SyntheticDataset(
        num_samples=train_size,
        input_dim=input_dim,
        num_classes=num_classes
    )
    
    test_dataset = SyntheticDataset(
        num_samples=test_size,
        input_dim=input_dim,
        num_classes=num_classes,
        noise_level=0.05  # Less noise for test set
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader