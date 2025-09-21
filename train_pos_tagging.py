#!/usr/bin/env python3
"""
Training script for Part-of-Speech Tagging with Adaptive Epoch/Sample Heuristics.

This script provides a complete training pipeline for sequence labeling (POS tagging)
with dynamic epoch and sampling heuristics based on dataset characteristics.

Usage:
    python train_pos_tagging.py --data-path /path/to/dataset
    python train_pos_tagging.py --data-path /path/to/dataset --model transformer --epochs 20
    python train_pos_tagging.py --max-sentences 1000 --auto  # synthetic small-scale test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from adaptiveneuralnetwork.data.kaggle_datasets import (
    get_pos_dataset_statistics,
    load_pos_tagging_dataset,
)
from adaptiveneuralnetwork.models.pos_tagger import (
    POSTagger,
    POSTaggerConfig,
    compute_token_accuracy,
    masked_cross_entropy_loss,
)

# Optional dependencies
try:
    from seqeval.metrics import classification_report, f1_score
    from sklearn.metrics import confusion_matrix

    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False
    print("Warning: seqeval not available. Install with: pip install seqeval")

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def get_dynamic_heuristics(stats: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Compute dynamic training heuristics based on dataset characteristics.

    Args:
        stats: Dataset statistics
        args: Command line arguments

    Returns:
        Dictionary with recommended training parameters
    """
    num_sentences = stats.get("num_sentences", 0)
    total_tokens = stats.get("total_tokens", 0)

    # Dynamic epoch selection based on dataset size
    if args.epochs is None:
        if num_sentences <= 5000:
            epochs = 40
        elif num_sentences <= 15000:
            epochs = 30
        elif num_sentences <= 40000:
            epochs = 20
        else:
            epochs = 12
    else:
        epochs = args.epochs

    # Dynamic batch size based on token count
    if args.batch_size is None:
        if total_tokens > 800000:
            batch_size = 16  # Reduce for memory
        else:
            batch_size = 32
    else:
        batch_size = args.batch_size

    # Gradient accumulation for large effective batch sizes
    effective_tokens_per_batch = batch_size * stats.get("avg_sentence_length", 20)
    if effective_tokens_per_batch > 8000:
        grad_accumulation_steps = max(1, int(effective_tokens_per_batch / 4000))
    else:
        grad_accumulation_steps = args.gradient_accumulation_steps

    heuristics = {
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "reasoning": {
            "epoch_rule": f"Sentences: {num_sentences} -> {epochs} epochs",
            "batch_rule": f"Tokens: {total_tokens} -> batch_size {batch_size}",
            "grad_accum_rule": f"Effective tokens/batch: {effective_tokens_per_batch:.0f} -> {grad_accumulation_steps} steps",
        },
    }

    return heuristics


def create_data_loaders(datasets: Dict[str, Any], batch_size: int) -> Dict[str, DataLoader]:
    """Create PyTorch data loaders for POS datasets."""

    def collate_fn(batch_items):
        """Collate function for POS tagging batches."""
        indices = list(range(len(batch_items)))
        return datasets["train"].get_batch(indices)

    # Simple wrapper to make POSDataset compatible with DataLoader
    class DatasetWrapper:
        def __init__(self, pos_dataset):
            self.pos_dataset = pos_dataset

        def __len__(self):
            return len(self.pos_dataset)

        def __getitem__(self, idx):
            return idx  # Return index, actual data fetched in collate_fn

    loaders = {}
    for split, dataset in datasets.items():
        wrapper = DatasetWrapper(dataset)
        loaders[split] = DataLoader(
            wrapper,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=lambda indices, ds=dataset: ds.get_batch(indices),
        )

    return loaders


def compute_metrics(
    predictions: List[List[str]], labels: List[List[str]], tag_vocab: Dict[str, int]
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for POS tagging.

    Args:
        predictions: List of predicted tag sequences
        labels: List of true tag sequences
        tag_vocab: Tag vocabulary

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    if HAS_SEQEVAL:
        # Use seqeval for sequence labeling metrics
        macro_f1 = f1_score(labels, predictions, average="macro")
        micro_f1 = f1_score(labels, predictions, average="micro")

        # Get detailed classification report
        report = classification_report(labels, predictions, output_dict=True)

        metrics.update(
            {
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "per_tag_f1": {
                    tag: report.get(tag, {}).get("f1-score", 0.0)
                    for tag in tag_vocab.keys()
                    if tag != "<PAD>"
                },
            }
        )
    else:
        # Fallback token-level accuracy
        total_correct = 0
        total_tokens = 0

        for pred_seq, true_seq in zip(predictions, labels):
            for pred, true in zip(pred_seq, true_seq):
                total_tokens += 1
                if pred == true:
                    total_correct += 1

        metrics.update(
            {
                "token_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
                "macro_f1": 0.0,  # Placeholder
                "micro_f1": 0.0,  # Placeholder
            }
        )

    return metrics


def save_results(
    output_dir: Path,
    metrics_history: List[Dict[str, Any]],
    best_metrics: Dict[str, Any],
    config: Dict[str, Any],
    tag_vocab: Dict[str, int],
) -> None:
    """Save training results and metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics history
    with open(output_dir / "metrics_pos_tagging.json", "w") as f:
        json.dump(
            {
                "training_history": metrics_history,
                "best_validation": best_metrics,
                "final_epoch": len(metrics_history),
            },
            f,
            indent=2,
        )

    # Save per-tag report
    if "per_tag_f1" in best_metrics:
        with open(output_dir / "tag_report.json", "w") as f:
            json.dump(
                {
                    "per_tag_f1": best_metrics["per_tag_f1"],
                    "tag_vocabulary": tag_vocab,
                    "num_tags": len(tag_vocab),
                },
                f,
                indent=2,
            )

    # Save configuration
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


def train_epoch(
    model: POSTagger,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_accumulation_steps: int = 1,
    progress_bar: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    num_batches = 0

    iterator = tqdm(data_loader, desc="Training") if progress_bar and HAS_TQDM else data_loader

    for batch_idx, batch in enumerate(iterator):
        # Convert to tensors
        input_ids = torch.tensor(batch["input_ids"], device=device)
        tag_ids = torch.tensor(batch["tag_ids"], device=device)
        attention_mask = torch.tensor(batch["attention_mask"], device=device)
        lengths = torch.tensor(batch["lengths"], device=device)

        # Forward pass
        outputs = model(input_ids, attention_mask, lengths)
        logits = outputs["logits"]

        # Compute loss
        loss = masked_cross_entropy_loss(logits, tag_ids, attention_mask)

        # Scale loss for gradient accumulation
        loss = loss / grad_accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute metrics
        predictions = torch.argmax(logits, dim=-1)
        batch_accuracy = compute_token_accuracy(predictions, tag_ids, attention_mask)
        batch_tokens = attention_mask.sum().item()

        total_loss += loss.item() * grad_accumulation_steps
        total_tokens += batch_tokens
        total_correct += batch_accuracy * batch_tokens
        num_batches += 1

        if progress_bar and HAS_TQDM and isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_accuracy:.4f}"})

    # Handle remaining gradients
    if num_batches % grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return {
        "train_loss": total_loss / num_batches,
        "train_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
    }


def evaluate_model(
    model: POSTagger,
    data_loader: DataLoader,
    device: torch.device,
    tag_vocab: Dict[str, int],
    idx_to_tag: Dict[int, str],
) -> Dict[str, Any]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            tag_ids = torch.tensor(batch["tag_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            lengths = torch.tensor(batch["lengths"], device=device)

            outputs = model(input_ids, attention_mask, lengths)
            logits = outputs["logits"]

            loss = masked_cross_entropy_loss(logits, tag_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)

            # Convert to sequences for seqeval
            for i, length in enumerate(lengths):
                length = length.item()
                pred_tags = [idx_to_tag[predictions[i, j].item()] for j in range(length)]
                true_tags = [idx_to_tag[tag_ids[i, j].item()] for j in range(length)]

                all_predictions.append(pred_tags)
                all_labels.append(true_tags)

            # Token-level metrics
            batch_accuracy = compute_token_accuracy(predictions, tag_ids, attention_mask)
            batch_tokens = attention_mask.sum().item()

            total_loss += loss.item()
            total_tokens += batch_tokens
            total_correct += batch_accuracy * batch_tokens

    # Compute comprehensive metrics
    metrics = compute_metrics(all_predictions, all_labels, tag_vocab)
    metrics.update(
        {
            "val_loss": total_loss / len(data_loader),
            "token_accuracy": total_correct / total_tokens if total_tokens > 0 else 0.0,
        }
    )

    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train POS Tagger with Adaptive Heuristics")

    # Data arguments
    parser.add_argument("--data-path", type=str, help="Path to POS tagging dataset")
    parser.add_argument("--max-sentences", type=int, help="Maximum sentences to load")
    parser.add_argument("--min-token-length", type=int, default=1, help="Minimum token length")
    parser.add_argument(
        "--filter-punctuation", action="store_true", help="Filter punctuation tokens"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs (auto if not specified)")
    parser.add_argument("--batch-size", type=int, help="Batch size (auto if not specified)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--min-improve", type=float, default=0.001, help="Minimum improvement threshold"
    )

    # Model arguments
    parser.add_argument(
        "--model", choices=["bilstm", "transformer"], default="bilstm", help="Model type"
    )
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--max-len", type=int, default=512, help="Maximum sequence length")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--output-dir", type=str, default="./pos_tagging_output", help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--auto", action="store_true", help="Use automatic heuristics")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Handle synthetic data for testing
    if args.synthetic or not args.data_path:
        logger.info("Using synthetic POS data for testing")
        # Create simple synthetic data
        synthetic_sentences = [
            ["The", "cat", "sat", "on", "the", "mat"],
            ["Dogs", "bark", "loudly"],
            ["I", "love", "machine", "learning"],
            ["Python", "is", "a", "programming", "language"],
        ] * 10  # Repeat for more data

        synthetic_tags = [
            ["DT", "NN", "VBD", "IN", "DT", "NN"],
            ["NNS", "VBP", "RB"],
            ["PRP", "VBP", "NN", "NN"],
            ["NNP", "VBZ", "DT", "NN", "NN"],
        ] * 10

        # Create vocabularies
        all_tokens = set()
        all_tags = set()
        for sent in synthetic_sentences:
            all_tokens.update(sent)
        for tags in synthetic_tags:
            all_tags.update(tags)

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for token in sorted(all_tokens):
            vocab[token] = len(vocab)

        tag_vocab = {"<PAD>": 0}
        for tag in sorted(all_tags):
            tag_vocab[tag] = len(tag_vocab)

        # Create datasets
        from adaptiveneuralnetwork.data.kaggle_datasets import POSDataset

        train_size = int(0.8 * len(synthetic_sentences))
        val_size = int(0.1 * len(synthetic_sentences))

        datasets = {
            "train": POSDataset(
                synthetic_sentences[:train_size], synthetic_tags[:train_size], vocab, tag_vocab
            ),
            "val": POSDataset(
                synthetic_sentences[train_size : train_size + val_size],
                synthetic_tags[train_size : train_size + val_size],
                vocab,
                tag_vocab,
            ),
            "test": POSDataset(
                synthetic_sentences[train_size + val_size :],
                synthetic_tags[train_size + val_size :],
                vocab,
                tag_vocab,
            ),
        }

        stats = {
            "num_sentences": len(synthetic_sentences),
            "total_tokens": sum(len(s) for s in synthetic_sentences),
            "unique_tokens": len(vocab),
            "unique_tags": len(tag_vocab),
            "avg_sentence_length": sum(len(s) for s in synthetic_sentences)
            / len(synthetic_sentences),
        }

    else:
        # Load real dataset
        logger.info(f"Loading POS dataset from {args.data_path}")

        # Get statistics for heuristics
        stats = get_pos_dataset_statistics(args.data_path)
        logger.info(f"Dataset statistics: {stats}")

        # Load full dataset
        dataset_info = load_pos_tagging_dataset(
            args.data_path,
            max_sentences=args.max_sentences,
            min_token_length=args.min_token_length,
            filter_punctuation=args.filter_punctuation,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )

        datasets = dataset_info["datasets"]
        vocab = dataset_info["vocab"]
        tag_vocab = dataset_info["tag_vocab"]
        stats = dataset_info["stats"]

    # Get dynamic heuristics
    heuristics = get_dynamic_heuristics(stats, args)
    logger.info(f"Training heuristics: {heuristics}")

    # Create model
    config = POSTaggerConfig(
        vocab_size=len(vocab),
        num_tags=len(tag_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model,
        max_length=args.max_len,
    )

    model = POSTagger(config).to(device)
    logger.info(
        f"Created {args.model} model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create data loaders
    data_loaders = create_data_loaders(datasets, heuristics["batch_size"])

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create reverse tag mapping
    idx_to_tag = {idx: tag for tag, idx in tag_vocab.items()}

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    metrics_history = []

    output_dir = Path(args.output_dir)

    logger.info(f"Starting training for {heuristics['epochs']} epochs...")

    for epoch in range(heuristics["epochs"]):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model,
            data_loaders["train"],
            optimizer,
            device,
            heuristics["gradient_accumulation_steps"],
            progress_bar=not args.verbose,
        )

        # Validate
        val_metrics = evaluate_model(model, data_loaders["val"], device, tag_vocab, idx_to_tag)

        # Combine metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "time": time.time() - start_time,
            **train_metrics,
            **val_metrics,
        }

        metrics_history.append(epoch_metrics)

        # Check for improvement
        current_f1 = val_metrics.get("macro_f1", val_metrics.get("token_accuracy", 0.0))

        if current_f1 > best_f1 + args.min_improve:
            best_f1 = current_f1
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "vocab": vocab,
                    "tag_vocab": tag_vocab,
                    "metrics": epoch_metrics,
                },
                output_dir / "best_model.pt",
            )

        else:
            patience_counter += 1

        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{heuristics['epochs']} - "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val F1: {current_f1:.4f}, "
            f"Best F1: {best_f1:.4f}"
        )

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break

    # Final evaluation on test set
    if "test" in data_loaders:
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(model, data_loaders["test"], device, tag_vocab, idx_to_tag)
        logger.info(f"Test metrics: {test_metrics}")

    # Save results
    config_dict = {
        "model_config": vars(config),
        "training_args": vars(args),
        "heuristics": heuristics,
        "dataset_stats": stats,
    }

    save_results(output_dir, metrics_history, val_metrics, config_dict, tag_vocab)

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab": vocab,
            "tag_vocab": tag_vocab,
            "metrics": metrics_history[-1] if metrics_history else {},
        },
        output_dir / "final_model.pt",
    )

    logger.info(f"Training completed! Results saved to {output_dir}")
    logger.info(f"Best validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
