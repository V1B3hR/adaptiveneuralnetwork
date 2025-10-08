"""
Optimized Dataset and Buffer API for Phase 1 - Data Layer Rework.

This module provides vectorized batch collation, pinned memory support,
and efficient index-based sampling to eliminate data loading bottlenecks.

Key optimizations:
- Vectorized batch collation (no per-sample Python loops)
- Pinned memory for faster GPU transfers
- Pre-allocated tensor buffers to reduce allocations
- Index-based sampling without data copying
"""


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VectorizedDataset(Dataset):
    """
    Optimized dataset wrapper with vectorized operations.
    
    This dataset wrapper stores data in pre-allocated tensor buffers
    and provides vectorized batch access to eliminate per-sample loops.
    
    Key features:
    - Pre-allocated tensor storage
    - Vectorized indexing (no loops)
    - Optional pinned memory for GPU transfer
    - Minimal copying overhead
    """

    def __init__(
        self,
        data: torch.Tensor | np.ndarray | list,
        targets: torch.Tensor | np.ndarray | list,
        pin_memory: bool = False,
        device: str = 'cpu'
    ):
        """
        Initialize vectorized dataset.
        
        Args:
            data: Input data as tensor, array, or list
            targets: Target labels as tensor, array, or list
            pin_memory: If True, allocate tensors in pinned memory for faster GPU transfer
            device: Device to store tensors ('cpu' or 'cuda')
        """
        # Convert to tensors if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(np.array(data), dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(np.array(targets), dtype=torch.long)

        # Store in appropriate memory
        if pin_memory and device == 'cpu':
            self.data = data.pin_memory()
            self.targets = targets.pin_memory()
        else:
            self.data = data.to(device)
            self.targets = targets.to(device)

        self.pin_memory = pin_memory
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get single item - returns views, not copies."""
        return self.data[idx], self.targets[idx]

    def get_batch(self, indices: list[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized batch retrieval - no Python loops.
        
        Args:
            indices: List or tensor of indices
            
        Returns:
            Tuple of (batch_data, batch_targets)
        """
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)

        # Vectorized indexing - single operation, no loops
        batch_data = self.data[indices]
        batch_targets = self.targets[indices]

        return batch_data, batch_targets


class PreallocatedBuffer:
    """
    Pre-allocated tensor buffer for batch collation.
    
    Reduces memory allocations by reusing buffers across batches.
    """

    def __init__(
        self,
        batch_size: int,
        data_shape: tuple[int, ...],
        target_shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.long,
        pin_memory: bool = False
    ):
        """
        Initialize pre-allocated buffer.
        
        Args:
            batch_size: Maximum batch size
            data_shape: Shape of single data sample
            target_shape: Shape of single target sample
            dtype: Data type for data tensor
            target_dtype: Data type for target tensor
            pin_memory: If True, allocate in pinned memory
        """
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.target_shape = target_shape

        # Pre-allocate buffers
        full_data_shape = (batch_size,) + data_shape
        full_target_shape = (batch_size,) + target_shape

        if pin_memory:
            self.data_buffer = torch.empty(full_data_shape, dtype=dtype).pin_memory()
            self.target_buffer = torch.empty(full_target_shape, dtype=target_dtype).pin_memory()
        else:
            self.data_buffer = torch.empty(full_data_shape, dtype=dtype)
            self.target_buffer = torch.empty(full_target_shape, dtype=target_dtype)

    def fill_batch(
        self,
        data_list: list[torch.Tensor],
        target_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fill buffer with batch data using vectorized operations.
        
        Args:
            data_list: List of data tensors
            target_list: List of target tensors
            
        Returns:
            Tuple of (batch_data, batch_targets) views from buffer
        """
        actual_batch_size = len(data_list)

        # Stack directly into buffer using vectorized operations
        if data_list:
            # Use torch.stack for efficiency
            stacked_data = torch.stack(data_list)
            stacked_targets = torch.stack(target_list) if target_list else torch.tensor([], dtype=self.target_buffer.dtype)

            # Copy into buffer
            self.data_buffer[:actual_batch_size].copy_(stacked_data)
            if len(stacked_targets) > 0:
                self.target_buffer[:actual_batch_size].copy_(stacked_targets)

        # Return view of filled portion
        return (
            self.data_buffer[:actual_batch_size],
            self.target_buffer[:actual_batch_size]
        )


def vectorized_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pin_memory: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized collate function - no Python loops over samples.
    
    This function uses torch.stack and vectorized operations instead of
    iterating over individual samples in Python.
    
    Args:
        batch: List of (data, target) tuples
        pin_memory: If True, allocate result in pinned memory
        
    Returns:
        Tuple of (batched_data, batched_targets)
    """
    if not batch:
        return torch.empty(0), torch.empty(0)

    # Separate data and targets - single list comprehension
    data_list, target_list = zip(*batch, strict=False)

    # Stack using vectorized torch operations (not Python loops)
    batched_data = torch.stack(data_list)
    batched_targets = torch.stack(target_list) if isinstance(target_list[0], torch.Tensor) else torch.tensor(target_list)

    # Pin memory if requested
    if pin_memory:
        batched_data = batched_data.pin_memory()
        batched_targets = batched_targets.pin_memory()

    return batched_data, batched_targets


def create_optimized_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    drop_last: bool = False
) -> DataLoader:
    """
    Create optimized DataLoader with best practices for Phase 1.
    
    This function sets up a DataLoader with:
    - Vectorized collation
    - Pinned memory for GPU transfers
    - Async prefetch for overlapping I/O with compute
    - Persistent workers to reduce startup overhead
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (0 = main process only)
        pin_memory: Enable pinned memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        drop_last: Drop incomplete final batch
        
    Returns:
        Optimized DataLoader instance
    """
    # Only use prefetch_factor and persistent_workers if num_workers > 0
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'collate_fn': lambda batch: vectorized_collate_fn(batch, pin_memory=False),
        'drop_last': drop_last,
    }

    # Only add prefetch_factor if num_workers > 0
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = persistent_workers

    return DataLoader(dataset, **loader_kwargs)


class OptimizedDatasetWrapper(Dataset):
    """
    Wrapper to convert any dataset to use optimized access patterns.
    
    This wrapper can wrap existing datasets and provide:
    - Vectorized batch access via get_batch()
    - Optional pre-loading into memory
    - Index-based sampling without copying
    """

    def __init__(
        self,
        base_dataset: Dataset,
        preload: bool = False,
        pin_memory: bool = False
    ):
        """
        Initialize optimized wrapper.
        
        Args:
            base_dataset: Base dataset to wrap
            preload: If True, load entire dataset into memory
            pin_memory: If True and preload is True, use pinned memory
        """
        self.base_dataset = base_dataset
        self.preload = preload
        self.pin_memory = pin_memory

        if preload:
            self._preload_data(pin_memory)
        else:
            self.data = None
            self.targets = None

    def _preload_data(self, pin_memory: bool):
        """Pre-load entire dataset into memory."""
        print(f"Pre-loading dataset with {len(self.base_dataset)} samples...")

        data_list = []
        target_list = []

        for i in range(len(self.base_dataset)):
            data, target = self.base_dataset[i]
            data_list.append(data)
            target_list.append(target)

        # Convert to tensors
        self.data = torch.stack(data_list)
        if isinstance(target_list[0], torch.Tensor):
            self.targets = torch.stack(target_list)
        else:
            self.targets = torch.tensor(target_list)

        # Pin memory if requested and CUDA is available
        if pin_memory and torch.cuda.is_available():
            self.data = self.data.pin_memory()
            self.targets = self.targets.pin_memory()
            print(f"Pre-loaded {len(self)} samples into pinned memory")
        else:
            print(f"Pre-loaded {len(self)} samples into memory (pinned=False - CUDA not available)")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get single item."""
        if self.preload:
            return self.data[idx], self.targets[idx]
        else:
            return self.base_dataset[idx]

    def get_batch(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get batch using vectorized indexing.
        
        Args:
            indices: List of indices
            
        Returns:
            Tuple of (batch_data, batch_targets)
        """
        if self.preload:
            # Vectorized indexing on pre-loaded data
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            return self.data[idx_tensor], self.targets[idx_tensor]
        else:
            # Fall back to per-sample access for non-preloaded data
            batch = [self.base_dataset[i] for i in indices]
            return vectorized_collate_fn(batch)


# Utility function to convert existing datasets
def optimize_dataset(
    dataset: Dataset,
    preload: bool = False,
    pin_memory: bool = False
) -> OptimizedDatasetWrapper:
    """
    Convert any dataset to optimized version.
    
    Args:
        dataset: Dataset to optimize
        preload: Whether to pre-load into memory
        pin_memory: Whether to use pinned memory (only with preload=True)
        
    Returns:
        Optimized dataset wrapper
    """
    return OptimizedDatasetWrapper(dataset, preload=preload, pin_memory=pin_memory)
