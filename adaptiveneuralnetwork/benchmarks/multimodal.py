"""
Multi-modal benchmark infrastructure for adaptive neural networks.

This module provides benchmarking capabilities for multi-modal learning tasks,
including text-image pairs, audio-visual, and other cross-modal scenarios.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import logging
import numpy as np
from pathlib import Path
import json

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..api import AdaptiveModel, AdaptiveConfig
from ..training.loops import train_epoch, evaluate_model

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    Generic multi-modal dataset combining different data modalities.
    
    Supports combinations like:
    - Text + Image
    - Audio + Image  
    - Text + Audio
    - Text + Image + Audio
    """
    
    def __init__(
        self,
        text_data: Optional[List[str]] = None,
        image_data: Optional[torch.Tensor] = None,
        audio_data: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tokenizer: Optional[Any] = None,
        max_text_length: int = 128
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            text_data: List of text strings
            image_data: Image tensor data [N, C, H, W]
            audio_data: Audio tensor data [N, audio_features]
            labels: Target labels [N]
            tokenizer: Text tokenizer (e.g., from transformers)
            max_text_length: Maximum text sequence length
        """
        self.text_data = text_data
        self.image_data = image_data
        self.audio_data = audio_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Determine dataset size
        sizes = []
        if text_data is not None:
            sizes.append(len(text_data))
        if image_data is not None:
            sizes.append(image_data.shape[0])
        if audio_data is not None:
            sizes.append(audio_data.shape[0])
        if labels is not None:
            sizes.append(labels.shape[0])
        
        if not sizes:
            raise ValueError("At least one modality must be provided")
        
        self.size = min(sizes)  # Use minimum size to handle mismatches
        
        # Validate modalities
        self._validate_data()
    
    def _validate_data(self):
        """Validate that all provided modalities have consistent sizes."""
        if self.text_data is not None and len(self.text_data) < self.size:
            logger.warning(f"Text data size ({len(self.text_data)}) < dataset size ({self.size})")
        
        if self.image_data is not None and self.image_data.shape[0] < self.size:
            logger.warning(f"Image data size ({self.image_data.shape[0]}) < dataset size ({self.size})")
        
        if self.audio_data is not None and self.audio_data.shape[0] < self.size:
            logger.warning(f"Audio data size ({self.audio_data.shape[0]}) < dataset size ({self.size})")
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get multi-modal sample.
        
        Returns:
            (modality_dict, label) where modality_dict contains available modalities
        """
        sample = {}
        
        # Text modality
        if self.text_data is not None:
            text = self.text_data[idx]
            if self.tokenizer is not None:
                # Use tokenizer for encoding
                encoded = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors='pt'
                )
                sample['text'] = encoded['input_ids'].squeeze(0)
                if 'attention_mask' in encoded:
                    sample['text_mask'] = encoded['attention_mask'].squeeze(0)
            else:
                # Simple character-level encoding
                char_ids = [ord(c) - ord('a') + 1 if 'a' <= c.lower() <= 'z' else 0 for c in text[:self.max_text_length]]
                char_ids += [0] * (self.max_text_length - len(char_ids))  # Pad
                sample['text'] = torch.tensor(char_ids, dtype=torch.long)
        
        # Image modality
        if self.image_data is not None:
            sample['image'] = self.image_data[idx]
        
        # Audio modality
        if self.audio_data is not None:
            sample['audio'] = self.audio_data[idx]
        
        # Label
        label = self.labels[idx] if self.labels is not None else torch.tensor(0, dtype=torch.long)
        
        return sample, label


class SyntheticTextImageDataset(MultiModalDataset):
    """Synthetic text-image dataset for development and testing."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 10,
        image_size: int = 32,
        vocab_size: int = 1000,
        text_length: int = 64,
        correlation_strength: float = 0.8
    ):
        """
        Create synthetic text-image dataset with controllable correlation.
        
        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes
            image_size: Height/width of square images
            vocab_size: Text vocabulary size
            text_length: Length of text sequences
            correlation_strength: How correlated text and images are (0-1)
        """
        torch.manual_seed(42)  # For reproducibility
        
        # Generate class-specific patterns
        self.num_classes = num_classes
        
        # Image patterns (one per class)
        image_patterns = torch.randn(num_classes, 3, image_size, image_size)
        
        # Text patterns (one per class)
        text_patterns = torch.randint(0, vocab_size, (num_classes, text_length))  # Start from 0
        
        # Generate samples
        images = []
        texts = []
        labels = []
        
        for i in range(num_samples):
            class_id = torch.randint(0, num_classes, (1,)).item()
            labels.append(class_id)
            
            # Generate image
            if torch.rand(1).item() < correlation_strength:
                # Use class pattern + noise
                image = image_patterns[class_id] + 0.3 * torch.randn(3, image_size, image_size)
            else:
                # Use random pattern (decorrelated)
                random_class = torch.randint(0, num_classes, (1,)).item()
                image = image_patterns[random_class] + 0.3 * torch.randn(3, image_size, image_size)
            
            images.append(image)
            
            # Generate text
            if torch.rand(1).item() < correlation_strength:
                # Use class pattern + some random tokens
                text = text_patterns[class_id].clone()
                # Replace 20% of tokens with random ones
                mask = torch.rand(text_length) < 0.2
                text[mask] = torch.randint(0, vocab_size, (mask.sum(),))  # Start from 0
            else:
                # Use completely random text
                text = torch.randint(0, vocab_size, (text_length,))  # Start from 0
            
            texts.append(text)
        
        # Convert to tensors
        image_data = torch.stack(images)
        text_data = torch.stack(texts)
        label_data = torch.tensor(labels, dtype=torch.long)
        
        # Initialize parent class with tensor data
        # Convert text tensor to list of dummy strings for compatibility
        text_strings = [f"dummy_text_{i}" for i in range(num_samples)]
        
        super().__init__(
            text_data=text_strings,
            image_data=image_data,
            audio_data=None,
            labels=label_data,
            tokenizer=None
        )
        
        # Override text handling to use pre-generated tensors
        self._text_tensors = text_data
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Override to use pre-generated text tensors."""
        sample = {}
        
        # Use pre-generated text tensor instead of string processing
        sample['text'] = self._text_tensors[idx]
        sample['image'] = self.image_data[idx]
        
        label = self.labels[idx]
        
        return sample, label


class MultiModalAdaptiveModel(nn.Module):
    """
    Adaptive neural network model extended for multi-modal inputs.
    
    Combines different modality encoders with the adaptive core.
    """
    
    def __init__(
        self,
        config: AdaptiveConfig,
        text_vocab_size: Optional[int] = None,
        text_embed_dim: int = 128,
        image_channels: int = 3,
        audio_dim: Optional[int] = None,
        modalities: Optional[List[str]] = None
    ):
        """
        Initialize multi-modal adaptive model.
        
        Args:
            config: Base adaptive configuration
            text_vocab_size: Text vocabulary size (if text modality used)
            text_embed_dim: Text embedding dimension
            image_channels: Number of image channels
            audio_dim: Audio feature dimension
            modalities: List of modalities to support ['text', 'image', 'audio']
        """
        super().__init__()
        
        self.config = config
        self.modalities = modalities or ['text', 'image']
        
        # Initialize modality encoders
        self.text_encoder = None
        self.image_encoder = None
        self.audio_encoder = None
        
        modality_dims = []
        
        if 'text' in self.modalities:
            if text_vocab_size is None:
                raise ValueError("text_vocab_size required when using text modality")
            
            self.text_encoder = nn.Sequential(
                nn.Embedding(text_vocab_size, text_embed_dim),
                nn.LSTM(text_embed_dim, text_embed_dim, batch_first=True),
            )
            modality_dims.append(text_embed_dim)
        
        if 'image' in self.modalities:
            # Simple CNN encoder for images
            self.image_encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128)
            )
            modality_dims.append(128)
        
        if 'audio' in self.modalities:
            if audio_dim is None:
                raise ValueError("audio_dim required when using audio modality")
            
            self.audio_encoder = nn.Sequential(
                nn.Linear(audio_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            modality_dims.append(128)
        
        # Fusion layer to combine modalities
        total_dim = sum(modality_dims)
        self.fusion_layer = nn.Linear(total_dim, config.input_dim)
        
        # Base adaptive model
        self.adaptive_core = AdaptiveModel(config)
    
    def encode_modalities(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode all modalities and fuse them."""
        encoded_features = []
        
        if 'text' in self.modalities and 'text' in batch:
            text_input = batch['text']
            text_emb = self.text_encoder[0](text_input)  # Embedding
            lstm_out, (hidden, _) = self.text_encoder[1](text_emb)  # LSTM
            # Use last hidden state
            text_features = hidden[-1]  # Take last layer's hidden state
            encoded_features.append(text_features)
        
        if 'image' in self.modalities and 'image' in batch:
            image_input = batch['image']
            image_features = self.image_encoder(image_input)
            encoded_features.append(image_features)
        
        if 'audio' in self.modalities and 'audio' in batch:
            audio_input = batch['audio'] 
            audio_features = self.audio_encoder(audio_input)
            encoded_features.append(audio_features)
        
        if not encoded_features:
            raise ValueError("No valid modalities found in input batch")
        
        # Concatenate and fuse
        fused_features = torch.cat(encoded_features, dim=-1)
        fused_output = self.fusion_layer(fused_features)
        
        return fused_output
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multi-modal model."""
        # Encode and fuse modalities
        fused_input = self.encode_modalities(batch)
        
        # Pass through adaptive core
        output = self.adaptive_core(fused_input)
        
        return output


class MultiModalBenchmark:
    """Benchmark for multi-modal adaptive neural networks."""
    
    def __init__(
        self,
        config: AdaptiveConfig,
        modalities: List[str] = ['text', 'image'],
        device: Optional[torch.device] = None
    ):
        """Initialize multi-modal benchmark."""
        self.config = config
        self.modalities = modalities
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initialized multi-modal benchmark with modalities: {modalities}")
    
    def run_synthetic_benchmark(
        self,
        num_samples: int = 1000,
        num_classes: int = 10,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        correlation_strength: float = 0.8,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run benchmark on synthetic multi-modal data.
        
        Args:
            num_samples: Number of synthetic samples
            num_classes: Number of classes
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            correlation_strength: Cross-modal correlation (0-1)
            save_results: Whether to save results
        
        Returns:
            Benchmark results dictionary
        """
        logger.info("Starting synthetic multi-modal benchmark...")
        
        # Create synthetic dataset
        dataset = SyntheticTextImageDataset(
            num_samples=num_samples,
            num_classes=num_classes,
            correlation_strength=correlation_strength
        )
        
        # Split train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create multi-modal model
        model = MultiModalAdaptiveModel(
            config=self.config,
            text_vocab_size=1000,
            modalities=self.modalities
        ).to(self.device)
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_data, targets in train_loader:
                targets = targets.to(self.device)
                
                # Move batch data to device
                batch_dict = {}
                for key, value in batch_data.items():
                    batch_dict[key] = value.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_dict)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            
            # Evaluate
            model.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch_data, targets in test_loader:
                    targets = targets.to(self.device)
                    batch_dict = {}
                    for key, value in batch_data.items():
                        batch_dict[key] = value.to(self.device)
                    
                    outputs = model(batch_dict)
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            
            test_acc = test_correct / test_total
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}, "
                       f"Test Acc: {test_acc:.4f}")
        
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            'benchmark_type': 'multimodal_synthetic',
            'modalities': self.modalities,
            'final_train_accuracy': train_accuracies[-1],
            'final_test_accuracy': test_accuracies[-1],
            'best_test_accuracy': max(test_accuracies),
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'training_time': total_time,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'correlation_strength': correlation_strength,
            'num_samples': num_samples,
            'num_classes': num_classes,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'config': self.config.to_dict()
        }
        
        if save_results:
            output_path = Path("multimodal_benchmark_results.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        
        logger.info(f"Multi-modal benchmark completed. "
                   f"Final test accuracy: {results['final_test_accuracy']:.4f}")
        
        return results


def run_multimodal_benchmark(
    config: Optional[AdaptiveConfig] = None,
    modalities: List[str] = ['text', 'image'],
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    num_classes: int = None  # Add num_classes parameter
) -> Dict[str, Any]:
    """
    Convenience function to run multi-modal benchmark.
    
    Args:
        config: Model configuration
        modalities: List of modalities to use
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to run on
    
    Returns:
        Benchmark results
    """
    if config is None:
        config = AdaptiveConfig(
            num_nodes=64,
            hidden_dim=32,
            input_dim=256,  # Will be set by fusion layer
            output_dim=num_classes or 10  # Use provided num_classes or default to 10
        )
    
    # Update output_dim if num_classes provided
    if num_classes is not None:
        config.output_dim = num_classes
    
    benchmark = MultiModalBenchmark(config, modalities, device)
    
    return benchmark.run_synthetic_benchmark(
        num_classes=config.output_dim,  # Use config's output_dim
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    config = AdaptiveConfig(
        num_nodes=32,
        hidden_dim=32,
        input_dim=256,
        output_dim=5
    )
    
    results = run_multimodal_benchmark(
        config=config,
        modalities=['text', 'image'],
        epochs=3,
        batch_size=16
    )
    
    print(f"Multi-modal benchmark accuracy: {results['final_test_accuracy']:.4f}")