"""
Text classification benchmark for Human vs AI Generated Essays.

This module provides benchmarking capabilities for text classification tasks,
specifically for distinguishing between human-written and AI-generated essays.
"""

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from ..api import AdaptiveConfig

logger = logging.getLogger(__name__)


class EssayDataset(Dataset):
    """
    Dataset for Human vs AI Generated Essays classification.

    Supports both real data loading and synthetic data generation for testing.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab_size: int = 10000,
        max_length: int = 512,
        vocab: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the essay dataset.

        Args:
            texts: List of essay texts
            labels: List of binary labels (0=human, 1=AI)
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            vocab: Pre-built vocabulary dictionary
        """
        self.texts = texts
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self._build_vocabulary()
        else:
            self.vocab = vocab

        # Tokenize all texts
        self.tokenized_texts = [self._tokenize_text(text) for text in texts]

    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from all texts."""
        # Simple tokenization - split by whitespace and basic punctuation
        all_words = []
        for text in self.texts:
            words = self._simple_tokenize(text)
            all_words.extend(words)

        # Count words and keep most frequent
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 2)  # Reserve 2 for special tokens

        # Build vocabulary with special tokens
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in most_common:
            vocab[word] = len(vocab)

        return vocab

    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization for text."""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()

        # Split on whitespace and punctuation
        words = re.findall(r"\b\w+\b", text)
        return words

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Convert text to token indices."""
        words = self._simple_tokenize(text)

        # Convert words to indices
        indices = []
        for word in words[: self.max_length]:  # Truncate if too long
            indices.append(self.vocab.get(word, self.vocab["<UNK>"]))

        # Pad if too short
        while len(indices) < self.max_length:
            indices.append(self.vocab["<PAD>"])

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokenized_texts[idx], torch.tensor(self.labels[idx], dtype=torch.long)


class SyntheticEssayDataset(EssayDataset):
    """Synthetic essay dataset for development and testing."""

    def __init__(
        self,
        num_samples: int = 1000,
        vocab_size: int = 5000,
        max_length: int = 256,
        human_vs_ai_ratio: float = 0.5,
    ):
        """
        Generate synthetic essay data for testing.

        Args:
            num_samples: Total number of samples to generate
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            human_vs_ai_ratio: Ratio of human vs AI essays
        """
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Generate synthetic vocabulary
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for i in range(2, vocab_size):
            vocab[f"word_{i}"] = i

        # Create synthetic patterns for human vs AI essays
        human_words = [f"word_{i}" for i in range(2, vocab_size // 2)]
        ai_words = [f"word_{i}" for i in range(vocab_size // 2, vocab_size)]

        texts = []
        labels = []

        num_human = int(num_samples * human_vs_ai_ratio)
        num_ai = num_samples - num_human

        # Generate human-like essays (use first half of vocabulary more)
        for _ in range(num_human):
            length = np.random.randint(max_length // 4, max_length)
            words = np.random.choice(human_words, length, replace=True)
            # Add some AI words for realism
            if np.random.random() < 0.2:
                num_ai_words = np.random.randint(1, length // 4)
                positions = np.random.choice(length, num_ai_words, replace=False)
                for pos in positions:
                    words[pos] = np.random.choice(ai_words)

            text = " ".join(words)
            texts.append(text)
            labels.append(0)  # Human

        # Generate AI-like essays (use second half of vocabulary more)
        for _ in range(num_ai):
            length = np.random.randint(max_length // 4, max_length)
            words = np.random.choice(ai_words, length, replace=True)
            # Add some human words for realism
            if np.random.random() < 0.2:
                num_human_words = np.random.randint(1, length // 4)
                positions = np.random.choice(length, num_human_words, replace=False)
                for pos in positions:
                    words[pos] = np.random.choice(human_words)

            text = " ".join(words)
            texts.append(text)
            labels.append(1)  # AI

        # Initialize parent class
        super().__init__(texts, labels, vocab_size, max_length, vocab)


class TextClassificationBenchmark:
    """Benchmark for text classification tasks using adaptive neural networks."""

    def __init__(self, config: AdaptiveConfig, device: Optional[torch.device] = None):
        """
        Initialize text classification benchmark.

        Args:
            config: Adaptive neural network configuration
            device: Torch device for computation
        """
        self.config = self._update_config_for_text(config)
        self.device = device or torch.device(config.device)
        # Initialize model placeholder - will be created when needed with proper input dimensions
        self.model = None

    def _update_config_for_text(self, config: AdaptiveConfig) -> AdaptiveConfig:
        """Update configuration for text classification."""
        # Create a copy to avoid modifying original
        new_config = AdaptiveConfig(**config.to_dict())
        # Will be set based on dataset
        new_config.output_dim = 2  # Binary classification
        return new_config

    def _create_model(self, vocab_size: int, max_length: int) -> None:
        """Create the adaptive model with proper dimensions."""
        # Update config with dataset-specific parameters
        self.config.input_dim = max_length
        self.config.vocab_size = vocab_size  # Store for reference

        # Create model
        from ..api import create_adaptive_model

        self.model = create_adaptive_model(self.config)
        self.model.to(self.device)

    def run_essay_classification_benchmark(
        self,
        dataset: Optional[EssayDataset] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_split: float = 0.2,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run human vs AI essay classification benchmark.

        Args:
            dataset: Essay dataset (if None, uses synthetic data)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            test_split: Fraction of data for testing
            save_results: Whether to save results to file

        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Starting Human vs AI Essay Classification Benchmark")

        # Create synthetic dataset if none provided
        if dataset is None:
            logger.info("Creating synthetic essay dataset for testing")
            dataset = SyntheticEssayDataset(num_samples=2000, vocab_size=5000, max_length=256)

        # Split dataset
        total_size = len(dataset)
        test_size = int(total_size * test_split)
        train_size = total_size - test_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Create model with proper dimensions
        self._create_model(len(dataset.vocab), dataset.max_length)

        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training metrics
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        training_time = 0.0

        # Training loop
        start_time = time.time()

        logger.info(f"Training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train epoch
            train_loss, train_acc = self._train_text_epoch(
                self.model, train_loader, optimizer, criterion, self.device
            )

            # Evaluate
            test_acc = self._evaluate_text_model(self.model, test_loader, self.device)

            epoch_time = time.time() - epoch_start
            training_time += epoch_time

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, "
                    f"Test Acc: {test_acc:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )

        total_time = time.time() - start_time

        # Final evaluation
        final_test_accuracy = test_accuracies[-1]

        # Prepare results
        results = {
            "final_test_accuracy": final_test_accuracy,
            "best_test_accuracy": max(test_accuracies),
            "final_train_accuracy": train_accuracies[-1],
            "best_train_accuracy": max(train_accuracies),
            "final_train_loss": train_losses[-1],
            "min_train_loss": min(train_losses),
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "training_time": training_time,
            "total_time": total_time,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "vocab_size": len(dataset.vocab),
            "max_length": dataset.max_length,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "config": self.config.to_dict(),
        }

        if save_results:
            self._save_results(results, "essay_classification_results.json")

        logger.info(
            f"Essay classification benchmark completed. "
            f"Final test accuracy: {results['final_test_accuracy']:.4f}"
        )

        return results

    def _train_text_epoch(self, model, train_loader, optimizer, criterion, device):
        """Train one epoch for text classification."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Reset model state before each forward pass to avoid gradient issues
            model.reset_state()

            # Convert token indices to float for the model
            # The data shape is [batch_size, seq_length] with token indices
            output = model(data.float())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return total_loss / len(train_loader), correct / total

    def _evaluate_text_model(self, model, test_loader, device):
        """Evaluate model on test set."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # Convert token indices to float for the model
                output = model(data.float())
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return correct / total

    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save benchmark results to JSON file."""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {filepath}")


def run_essay_classification_benchmark(
    config: Optional[AdaptiveConfig] = None,
    dataset: Optional[EssayDataset] = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run essay classification benchmark.

    Args:
        config: Model configuration
        dataset: Essay dataset (uses synthetic if None)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Torch device

    Returns:
        Benchmark results dictionary
    """
    if config is None:
        config = AdaptiveConfig(
            num_epochs=epochs, learning_rate=learning_rate, batch_size=batch_size
        )

    # Initialize benchmark
    benchmark = TextClassificationBenchmark(config, device)

    # Run benchmark
    results = benchmark.run_essay_classification_benchmark(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_results=True,
    )

    return results
