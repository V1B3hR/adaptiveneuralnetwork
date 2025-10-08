"""
Dataset abstraction unification and streaming support.

This module provides unified interfaces for various dataset formats including
WebDataset, HuggingFace Datasets, and custom streaming implementations.
"""

import json
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


@dataclass
class StreamingConfig:
    """Configuration for streaming datasets."""
    buffer_size: int = 1000
    prefetch_factor: int = 2
    num_workers: int = 0
    batch_size: int = 32
    shuffle_buffer_size: int = 10000
    cache_size_mb: int = 100
    compression: str | None = None  # "gzip", "lz4", etc.


class UnifiedDatasetInterface(ABC):
    """Abstract interface for unified dataset access."""

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length (may be approximate for streaming)."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        pass

    @abstractmethod
    def get_batch(self, indices: list[int]) -> Any:
        """Get batch of items."""
        pass

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Get dataset information and metadata."""
        pass

    @abstractmethod
    def stream(self, shuffle: bool = False) -> Iterator[Any]:
        """Create streaming iterator over dataset."""
        pass


class StreamingDatasetWrapper(UnifiedDatasetInterface, IterableDataset):
    """Streaming wrapper for large datasets."""

    def __init__(
        self,
        data_source: str | Path | Callable,
        config: StreamingConfig,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        self.data_source = data_source
        self.config = config
        self.transform = transform
        self.target_transform = target_transform

        self._length = None
        self._cache = {}
        self._cache_size_bytes = 0
        self._info = self._load_info()

    def _load_info(self) -> dict[str, Any]:
        """Load dataset information."""
        if isinstance(self.data_source, (str, Path)):
            info_path = Path(self.data_source) / "info.json"
            if info_path.exists():
                with open(info_path) as f:
                    return json.load(f)

        return {
            "name": "streaming_dataset",
            "description": "Streaming dataset",
            "approximate_size": "unknown"
        }

    def __len__(self) -> int:
        """Return approximate length."""
        if self._length is None:
            # Try to estimate length
            if isinstance(self.data_source, (str, Path)):
                data_path = Path(self.data_source)
                if data_path.is_dir():
                    # Count files
                    self._length = len(list(data_path.glob("*.json"))) + len(list(data_path.glob("*.pkl")))
                else:
                    self._length = 1000  # Default estimate
            else:
                self._length = 1000

        return self._length

    def __getitem__(self, index: int) -> Any:
        """Get item by index (may be approximate for streaming)."""
        # Check cache first
        if index in self._cache:
            return self._cache[index]

        # Load item
        item = self._load_item(index)

        # Cache if space available
        if self._cache_size_bytes < self.config.cache_size_mb * 1024 * 1024:
            self._cache[index] = item
            self._cache_size_bytes += self._estimate_item_size(item)

        return item

    def _load_item(self, index: int) -> Any:
        """Load individual item."""
        if isinstance(self.data_source, (str, Path)):
            # Load from file system
            data_path = Path(self.data_source)

            if data_path.is_dir():
                # Look for indexed files
                json_path = data_path / f"{index:06d}.json"
                pkl_path = data_path / f"{index:06d}.pkl"

                if json_path.exists():
                    with open(json_path) as f:
                        data = json.load(f)
                elif pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    # Generate synthetic data
                    data = self._generate_synthetic_item(index)
            else:
                # Single file - generate synthetic
                data = self._generate_synthetic_item(index)
        elif callable(self.data_source):
            # Data source is a function
            data = self.data_source(index)
        else:
            data = self._generate_synthetic_item(index)

        # Apply transforms
        if isinstance(data, dict) and 'x' in data and 'y' in data:
            x, y = data['x'], data['y']
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            x, y = data
        else:
            x, y = data, 0  # Default

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def _generate_synthetic_item(self, index: int) -> dict[str, Any]:
        """Generate synthetic data item."""
        np.random.seed(index)  # Deterministic based on index

        return {
            'x': np.random.randn(32, 32, 3).astype(np.float32),
            'y': np.random.randint(0, 10),
            'metadata': {'index': index, 'source': 'synthetic'}
        }

    def _estimate_item_size(self, item: Any) -> int:
        """Estimate memory size of item in bytes."""
        if isinstance(item, tuple):
            return sum(self._estimate_item_size(x) for x in item)
        elif isinstance(item, torch.Tensor):
            return item.nelement() * item.element_size()
        elif isinstance(item, np.ndarray):
            return item.nbytes
        else:
            # Rough estimate
            return 1024

    def get_batch(self, indices: list[int]) -> Any:
        """Get batch of items."""
        items = [self[i] for i in indices]

        # Stack into batch
        if items and isinstance(items[0], tuple):
            xs, ys = zip(*items, strict=False)
            if isinstance(xs[0], torch.Tensor):
                x_batch = torch.stack(xs)
            else:
                x_batch = torch.tensor(np.stack(xs))

            if isinstance(ys[0], torch.Tensor):
                y_batch = torch.stack(ys)
            else:
                y_batch = torch.tensor(ys)

            return x_batch, y_batch

        return items

    def get_info(self) -> dict[str, Any]:
        """Get dataset information."""
        return {
            **self._info,
            "length": len(self),
            "cache_size_mb": self._cache_size_bytes / (1024 * 1024),
            "config": self.config.__dict__,
        }

    def stream(self, shuffle: bool = False) -> Iterator[Any]:
        """Create streaming iterator."""
        indices = list(range(len(self)))

        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            yield self[i]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over dataset."""
        return self.stream(shuffle=True)


class WebDatasetWrapper(UnifiedDatasetInterface):
    """Wrapper for WebDataset format."""

    def __init__(
        self,
        urls: str | list[str],
        config: StreamingConfig,
        decode_handlers: dict[str, Callable] | None = None
    ):
        self.urls = urls if isinstance(urls, list) else [urls]
        self.config = config
        self.decode_handlers = decode_handlers or {}

        try:
            import webdataset as wds
            self.wds = wds
        except ImportError:
            raise ImportError("WebDataset not installed. Install with: pip install webdataset")

        self._dataset = None
        self._length = None
        self._setup_dataset()

    def _setup_dataset(self):
        """Setup WebDataset pipeline."""
        dataset = self.wds.WebDataset(self.urls)

        # Add decode handlers
        if self.decode_handlers:
            dataset = dataset.decode(**self.decode_handlers)
        else:
            dataset = dataset.decode("rgb8", "cls")

        # Add transforms
        dataset = dataset.to_tuple("jpg", "cls")

        self._dataset = dataset

    def __len__(self) -> int:
        """Estimate length from shard info."""
        if self._length is None:
            # Try to estimate from URL pattern or metadata
            self._length = 10000  # Default estimate
        return self._length

    def __getitem__(self, index: int) -> Any:
        """Not directly supported for WebDataset - use stream instead."""
        raise NotImplementedError("WebDataset doesn't support direct indexing. Use stream() instead.")

    def get_batch(self, indices: list[int]) -> Any:
        """Not directly supported for WebDataset."""
        raise NotImplementedError("WebDataset doesn't support direct indexing. Use stream() instead.")

    def get_info(self) -> dict[str, Any]:
        """Get WebDataset information."""
        return {
            "name": "webdataset",
            "urls": self.urls,
            "length": len(self),
            "config": self.config.__dict__,
        }

    def stream(self, shuffle: bool = False) -> Iterator[Any]:
        """Stream from WebDataset."""
        dataset = self._dataset

        if shuffle:
            dataset = dataset.shuffle(self.config.shuffle_buffer_size)

        return iter(dataset)


class HuggingFaceDatasetWrapper(UnifiedDatasetInterface):
    """Wrapper for HuggingFace Datasets."""

    def __init__(
        self,
        dataset_name: str,
        config: StreamingConfig,
        split: str = "train",
        streaming: bool = True,
        **dataset_kwargs
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.streaming = streaming
        self.dataset_kwargs = dataset_kwargs

        try:
            from datasets import load_dataset
            self.load_dataset = load_dataset
        except ImportError:
            raise ImportError("HuggingFace Datasets not installed. Install with: pip install datasets")

        self._dataset = None
        self._load_dataset()

    def _load_dataset(self):
        """Load HuggingFace dataset."""
        self._dataset = self.load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
            **self.dataset_kwargs
        )

    def __len__(self) -> int:
        """Get dataset length."""
        if hasattr(self._dataset, '__len__'):
            return len(self._dataset)
        else:
            # Streaming dataset - return estimate
            return getattr(self._dataset, 'info', {}).get('splits', {}).get(self.split, {}).get('num_examples', 10000)

    def __getitem__(self, index: int) -> Any:
        """Get item by index."""
        if self.streaming:
            raise NotImplementedError("Streaming HF datasets don't support direct indexing. Use stream() instead.")

        item = self._dataset[index]
        return self._process_hf_item(item)

    def _process_hf_item(self, item: dict[str, Any]) -> tuple[Any, Any]:
        """Process HuggingFace dataset item into (x, y) format."""
        # Common HF dataset field mappings
        x_fields = ['image', 'text', 'input', 'pixel_values', 'input_ids']
        y_fields = ['label', 'labels', 'target', 'ground_truth']

        x = None
        y = None

        # Find x (input) field
        for field in x_fields:
            if field in item:
                x = item[field]
                break

        # Find y (target) field
        for field in y_fields:
            if field in item:
                y = item[field]
                break

        # Fallback
        if x is None:
            x = item
        if y is None:
            y = 0

        return x, y

    def get_batch(self, indices: list[int]) -> Any:
        """Get batch of items."""
        items = [self[i] for i in indices]

        # Process batch similar to StreamingDatasetWrapper
        if items and isinstance(items[0], tuple):
            xs, ys = zip(*items, strict=False)
            return xs, ys

        return items

    def get_info(self) -> dict[str, Any]:
        """Get dataset information."""
        info = {
            "name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
            "length": len(self),
        }

        if hasattr(self._dataset, 'info'):
            info.update({
                "description": self._dataset.info.description,
                "features": str(self._dataset.info.features),
                "dataset_size": self._dataset.info.dataset_size,
            })

        return info

    def stream(self, shuffle: bool = False) -> Iterator[Any]:
        """Stream from HuggingFace dataset."""
        if self.streaming:
            # Already streaming
            dataset = self._dataset
            if shuffle:
                dataset = dataset.shuffle(seed=42, buffer_size=self.config.shuffle_buffer_size)

            for item in dataset:
                yield self._process_hf_item(item)
        else:
            # Convert to streaming
            indices = list(range(len(self)))
            if shuffle:
                np.random.shuffle(indices)

            for i in indices:
                yield self[i]


class UnifiedDatasetManager:
    """Manager for unified dataset access across different formats."""

    def __init__(self, default_config: StreamingConfig | None = None):
        self.default_config = default_config or StreamingConfig()
        self.registered_datasets: dict[str, UnifiedDatasetInterface] = {}

    def register_dataset(self, name: str, dataset: UnifiedDatasetInterface) -> None:
        """Register a dataset with the manager."""
        self.registered_datasets[name] = dataset
        print(f"Registered dataset: {name}")

    def create_streaming_dataset(
        self,
        data_source: str | Path | Callable,
        config: StreamingConfig | None = None,
        **kwargs
    ) -> StreamingDatasetWrapper:
        """Create streaming dataset wrapper."""
        config = config or self.default_config
        return StreamingDatasetWrapper(data_source, config, **kwargs)

    def create_webdataset(
        self,
        urls: str | list[str],
        config: StreamingConfig | None = None,
        **kwargs
    ) -> WebDatasetWrapper:
        """Create WebDataset wrapper."""
        config = config or self.default_config
        return WebDatasetWrapper(urls, config, **kwargs)

    def create_huggingface_dataset(
        self,
        dataset_name: str,
        config: StreamingConfig | None = None,
        **kwargs
    ) -> HuggingFaceDatasetWrapper:
        """Create HuggingFace dataset wrapper."""
        config = config or self.default_config
        return HuggingFaceDatasetWrapper(dataset_name, config, **kwargs)

    def create_unified_dataloader(
        self,
        dataset: UnifiedDatasetInterface,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs
    ) -> DataLoader:
        """Create unified dataloader that works with all dataset types."""

        if isinstance(dataset, (StreamingDatasetWrapper, IterableDataset)):
            # Use streaming approach
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                **kwargs
            )
        else:
            # Use regular approach
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                **kwargs
            )

    def get_dataset_info(self, name: str) -> dict[str, Any]:
        """Get information about a registered dataset."""
        if name not in self.registered_datasets:
            raise ValueError(f"Dataset '{name}' not registered")

        return self.registered_datasets[name].get_info()

    def list_datasets(self) -> list[str]:
        """List all registered datasets."""
        return list(self.registered_datasets.keys())

    def create_multi_dataset_loader(
        self,
        dataset_configs: list[dict[str, Any]],
        sampling_strategy: str = "round_robin",
        **loader_kwargs
    ) -> DataLoader:
        """Create dataloader that combines multiple datasets."""

        datasets = []
        for config in dataset_configs:
            dataset_type = config.pop("type")

            if dataset_type == "streaming":
                dataset = self.create_streaming_dataset(**config)
            elif dataset_type == "webdataset":
                dataset = self.create_webdataset(**config)
            elif dataset_type == "huggingface":
                dataset = self.create_huggingface_dataset(**config)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            datasets.append(dataset)

        # Create combined dataset
        combined_dataset = MultiDatasetWrapper(datasets, sampling_strategy)

        return self.create_unified_dataloader(combined_dataset, **loader_kwargs)


class MultiDatasetWrapper(UnifiedDatasetInterface, IterableDataset):
    """Wrapper that combines multiple datasets with different sampling strategies."""

    def __init__(self, datasets: list[UnifiedDatasetInterface], sampling_strategy: str = "round_robin"):
        self.datasets = datasets
        self.sampling_strategy = sampling_strategy
        self._length = sum(len(d) for d in datasets)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Any:
        # Find which dataset the index belongs to
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)

        raise IndexError("Index out of range")

    def get_batch(self, indices: list[int]) -> Any:
        return [self[i] for i in indices]

    def get_info(self) -> dict[str, Any]:
        return {
            "name": "multi_dataset",
            "num_datasets": len(self.datasets),
            "total_length": len(self),
            "sampling_strategy": self.sampling_strategy,
            "datasets": [d.get_info() for d in self.datasets]
        }

    def stream(self, shuffle: bool = False) -> Iterator[Any]:
        if self.sampling_strategy == "round_robin":
            # Round-robin sampling
            iterators = [d.stream(shuffle) for d in self.datasets]

            while iterators:
                for i, iterator in enumerate(iterators):
                    try:
                        yield next(iterator)
                    except StopIteration:
                        iterators.pop(i)
                        break
        else:
            # Sequential sampling
            for dataset in self.datasets:
                yield from dataset.stream(shuffle)

    def __iter__(self) -> Iterator[Any]:
        return self.stream(shuffle=True)


# Convenience functions
def create_unified_dataset_manager() -> UnifiedDatasetManager:
    """Create a unified dataset manager with default configuration."""
    return UnifiedDatasetManager()


def quick_stream_dataset(
    data_source: str | Path | Callable,
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """Quickly create a streaming dataloader."""
    manager = create_unified_dataset_manager()
    dataset = manager.create_streaming_dataset(data_source, **kwargs)
    return manager.create_unified_dataloader(dataset, batch_size, shuffle)


def quick_huggingface_stream(
    dataset_name: str,
    batch_size: int = 32,
    split: str = "train",
    **kwargs
) -> DataLoader:
    """Quickly create a HuggingFace streaming dataloader."""
    manager = create_unified_dataset_manager()
    dataset = manager.create_huggingface_dataset(dataset_name, split=split, **kwargs)
    return manager.create_unified_dataloader(dataset, batch_size)
