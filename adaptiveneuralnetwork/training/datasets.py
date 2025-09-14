"""
Dataset utilities for adaptive neural network training.

This module provides dataset loading and preprocessing for various
benchmarks including MNIST, with domain randomization for cross-domain
generalization as required by Phase 2.1.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Callable, Union, List, Dict, Any
import random


class DomainRandomizedDataset(Dataset):
    """Dataset with domain randomization for cross-domain generalization."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        domain_configs: List[Dict[str, Any]],
        randomization_prob: float = 0.5
    ):
        """
        Initialize domain randomized dataset.
        
        Args:
            base_dataset: Base dataset to apply domain randomization to
            domain_configs: List of domain configuration dictionaries
            randomization_prob: Probability of applying domain randomization
        """
        self.base_dataset = base_dataset
        self.domain_configs = domain_configs
        self.randomization_prob = randomization_prob
        
    def __len__(self) -> int:
        return len(self.base_dataset)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data, target = self.base_dataset[idx]
        
        # Apply domain randomization with probability
        if random.random() < self.randomization_prob and self.domain_configs:
            domain_config = random.choice(self.domain_configs)
            data = self._apply_domain_transform(data, domain_config)
            
        return data, target
        
    def _apply_domain_transform(self, data: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """Apply domain-specific transformations to data."""
        transformed_data = data.clone()
        
        # Noise injection
        if 'noise_level' in config:
            noise = torch.randn_like(transformed_data) * config['noise_level']
            transformed_data += noise
            
        # Brightness/contrast adjustment
        if 'brightness_factor' in config:
            transformed_data *= config['brightness_factor']
            
        if 'contrast_factor' in config:
            mean = transformed_data.mean()
            transformed_data = (transformed_data - mean) * config['contrast_factor'] + mean
            
        # Blur simulation
        if 'blur_kernel_size' in config and config['blur_kernel_size'] > 1:
            # Simple blur approximation
            kernel_size = config['blur_kernel_size']
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            # Apply simple averaging filter as blur approximation
            if len(transformed_data.shape) >= 2:
                padded = torch.nn.functional.pad(transformed_data, 
                                               (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                                               mode='reflect')
                # Simple box filter
                blurred = torch.nn.functional.avg_pool2d(
                    padded.unsqueeze(0).unsqueeze(0), 
                    kernel_size, 
                    stride=1,
                    padding=0
                ).squeeze()
                transformed_data = blurred
        
        # Clamp values to valid range
        transformed_data = torch.clamp(transformed_data, 0.0, 1.0)
        
        return transformed_data


def create_cross_domain_loaders(
    base_dataset: Dataset,
    batch_size: int = 64,
    num_domains: int = 3
) -> List[DataLoader]:
    """
    Create data loaders with different domain configurations for cross-domain generalization.
    
    Args:
        base_dataset: Base dataset to create domains from
        batch_size: Batch size for data loaders
        num_domains: Number of different domains to create
        
    Returns:
        List of data loaders with different domain configurations
    """
    domain_configs = [
        # Clean domain (no modifications)
        {},
        # Noisy domain
        {'noise_level': 0.1, 'brightness_factor': 0.8},
        # High contrast domain
        {'contrast_factor': 1.5, 'brightness_factor': 1.2},
        # Blurred domain
        {'blur_kernel_size': 3, 'noise_level': 0.05},
        # Low light domain
        {'brightness_factor': 0.5, 'contrast_factor': 1.3},
        # High noise domain
        {'noise_level': 0.2, 'brightness_factor': 0.9, 'contrast_factor': 0.8}
    ]
    
    # Select configurations for requested number of domains
    selected_configs = domain_configs[:num_domains] if num_domains <= len(domain_configs) else domain_configs
    
    loaders = []
    for i, config in enumerate(selected_configs):
        if config:  # If config is not empty, apply domain randomization
            domain_dataset = DomainRandomizedDataset(
                base_dataset=base_dataset,
                domain_configs=[config],
                randomization_prob=1.0  # Always apply for domain-specific loaders
            )
        else:
            domain_dataset = base_dataset
            
        loader = DataLoader(
            domain_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for consistent evaluation
            num_workers=0
        )
        loaders.append(loader)
        
    return loaders


def load_mnist(
    batch_size: int = 64, root: str = "./data", download: bool = True, num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
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
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]  # MNIST mean and std
    )

    # Load train and test sets
    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, transform=transform, download=download
    )

    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, transform=transform, download=download
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader


def load_mnist_subset(
    batch_size: int = 64,
    subset_size: int = 1000,
    root: str = "./data",
    download: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
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
        batch_size=batch_size, root=root, download=download, num_workers=num_workers
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
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing and development."""

    def __init__(
        self,
        num_samples: int = 1000,
        input_dim: int = 784,
        num_classes: int = 10,
        noise_level: float = 0.1,
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def create_synthetic_loaders(
    num_samples: int = 1000,
    batch_size: int = 64,
    input_dim: int = 784,
    num_classes: int = 10,
    test_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
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
        num_samples=train_size, input_dim=input_dim, num_classes=num_classes
    )

    test_dataset = SyntheticDataset(
        num_samples=test_size,
        input_dim=input_dim,
        num_classes=num_classes,
        noise_level=0.05,  # Less noise for test set
    )

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_cifar10(
    batch_size: int = 64, root: str = "./data", download: bool = True, num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with standard preprocessing.

    Args:
        batch_size: Batch size for data loaders
        root: Root directory for dataset storage
        download: Whether to download dataset if not present
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Standard CIFAR-10 preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load train and test sets
    train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, transform=transform_train, download=download
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, transform=transform_test, download=download
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader


class CIFAR10Corrupted(Dataset):
    """
    CIFAR-10 dataset with various corruption types for domain shift robustness testing.
    
    Supports various corruption types like noise, blur, weather effects, etc.
    """
    
    CORRUPTION_TYPES = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]
    
    def __init__(
        self, 
        root: str = "./data",
        train: bool = True,
        corruption_type: str = 'gaussian_noise',
        severity: int = 1,
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        """
        Initialize corrupted CIFAR-10 dataset.
        
        Args:
            root: Root directory for dataset storage
            train: If True, use training set, otherwise test set
            corruption_type: Type of corruption to apply
            severity: Severity level (1-5, where 1 is mild, 5 is severe)
            transform: Additional transforms to apply
            download: Whether to download dataset if not present
        """
        self.root = root
        self.train = train
        self.corruption_type = corruption_type
        self.severity = max(1, min(5, severity))  # Clamp to [1, 5]
        self.transform = transform
        
        if corruption_type not in self.CORRUPTION_TYPES:
            raise ValueError(f"Corruption type {corruption_type} not supported. "
                           f"Choose from: {self.CORRUPTION_TYPES}")
        
        # Load base CIFAR-10 dataset
        self.base_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = self.base_dataset[idx]
        
        # Apply corruption
        corrupted_image = self._apply_corruption(image)
        
        # Apply additional transforms if provided
        if self.transform:
            corrupted_image = self.transform(corrupted_image)
        
        return corrupted_image, target
    
    def _apply_corruption(self, image) -> torch.Tensor:
        """Apply the specified corruption to an image."""
        # Convert PIL Image to numpy array
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        else:
            image = np.array(image)
            
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply corruption based on type
        if self.corruption_type == 'gaussian_noise':
            noise_std = 0.08 + (self.severity - 1) * 0.04
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            corrupted = image + noise
            
        elif self.corruption_type == 'shot_noise':
            lambda_param = 60.0 - (self.severity - 1) * 12.0
            corrupted = np.random.poisson(image * lambda_param) / lambda_param
            
        elif self.corruption_type == 'impulse_noise':
            prob = 0.03 + (self.severity - 1) * 0.02
            mask = np.random.random(image.shape) < prob
            corrupted = image.copy()
            corrupted[mask] = np.random.random(np.sum(mask))
            
        elif self.corruption_type in ['defocus_blur', 'gaussian_blur']:
            # Simple blur implementation without scipy dependency for now
            kernel_size = 1 + (self.severity - 1) * 2
            corrupted = image.copy()
            # Apply simple averaging for blur effect
            if kernel_size > 1:
                from torch.nn.functional import conv2d
                # Convert to tensor for convolution
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
                blurred = torch.zeros_like(img_tensor)
                for c in range(3):
                    blurred[:, c:c+1] = conv2d(img_tensor[:, c:c+1], kernel, padding=kernel_size//2)
                corrupted = blurred.squeeze(0).permute(1, 2, 0).numpy()
                
        elif self.corruption_type == 'brightness':
            factor = 0.1 + (self.severity - 1) * 0.3
            corrupted = image + factor
            
        elif self.corruption_type == 'contrast':
            factor = 0.75 + (self.severity - 1) * 0.1
            mean = np.mean(image)
            corrupted = (image - mean) * factor + mean
            
        elif self.corruption_type == 'pixelate':
            # Simple pixelation by downsampling and upsampling
            corrupted = image.copy()
            step = max(1, 6 - self.severity)
            corrupted[::step, ::step] = corrupted[::step, ::step]
            
        elif self.corruption_type == 'saturate':
            factor = 1.0 + (self.severity - 1) * 0.5
            corrupted = image * factor
            
        else:
            # Fallback to gaussian noise for unsupported types
            noise_std = 0.08 + (self.severity - 1) * 0.04
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            corrupted = image + noise
        
        # Clip to valid range and convert back to tensor
        corrupted = np.clip(corrupted, 0, 1)
        corrupted = torch.from_numpy(corrupted).permute(2, 0, 1)  # HWC to CHW
        
        return corrupted


def load_cifar10_corrupted(
    corruption_type: str = 'gaussian_noise',
    severity: int = 1,
    batch_size: int = 64,
    root: str = "./data",
    download: bool = True,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Load corrupted CIFAR-10 dataset for domain shift robustness testing.
    
    Args:
        corruption_type: Type of corruption to apply
        severity: Severity level (1-5)
        batch_size: Batch size for data loaders
        root: Root directory for dataset storage
        download: Whether to download dataset if not present
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, test_loader) with corrupted data
    """
    # Create datasets with corruption
    train_dataset = CIFAR10Corrupted(
        root=root, train=True, corruption_type=corruption_type,
        severity=severity, download=download
    )
    
    test_dataset = CIFAR10Corrupted(
        root=root, train=False, corruption_type=corruption_type,
        severity=severity, download=download
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader
