#!/usr/bin/env python3
"""
Evaluation script for Part-of-Speech Tagging models.

This script loads a trained POS tagging model and evaluates it on a test dataset,
providing detailed metrics and classification reports.

Usage:
    python evaluate_pos_tagging.py --checkpoint /path/to/model.pt --data-path /path/to/test_data
    python evaluate_pos_tagging.py --checkpoint /path/to/model.pt --synthetic  # Test with synthetic data
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from adaptiveneuralnetwork.data.kaggle_datasets import POSDataset, load_pos_tagging_dataset
from adaptiveneuralnetwork.models.pos_tagger import POSTagger

# Optional dependencies
try:
    from seqeval.metrics import (
        accuracy_score,
        classification_report,
        precision_recall_fscore_support,
    )
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load trained model from checkpoint.
    
    Returns:
        (model, vocab, tag_vocab, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    vocab = checkpoint['vocab']
    tag_vocab = checkpoint['tag_vocab']

    model = POSTagger(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab, tag_vocab, config


def evaluate_model(
    model: POSTagger,
    dataset: POSDataset,
    device: torch.device,
    batch_size: int = 32
) -> dict[str, Any]:
    """
    Evaluate model on dataset.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_tokens = []

    # Create reverse tag mapping
    idx_to_tag = {idx: tag for tag, idx in dataset.tag_vocab.items()}

    # Process in batches
    num_samples = len(dataset)
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = list(range(start_idx, end_idx))

            batch = dataset.get_batch(batch_indices)

            input_ids = torch.tensor(batch['input_ids'], device=device)
            tag_ids = torch.tensor(batch['tag_ids'], device=device)
            attention_mask = torch.tensor(batch['attention_mask'], device=device)
            lengths = torch.tensor(batch['lengths'], device=device)

            # Forward pass
            outputs = model(input_ids, attention_mask, lengths)
            logits = outputs['logits']

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Convert to sequences for evaluation
            for i, length in enumerate(lengths):
                length = length.item()
                pred_tags = [idx_to_tag[predictions[i, j].item()] for j in range(length)]
                true_tags = [idx_to_tag[tag_ids[i, j].item()] for j in range(length)]
                tokens = batch['tokens'][i][:length]

                all_predictions.append(pred_tags)
                all_labels.append(true_tags)
                all_tokens.append(tokens)

            num_batches += 1

    # Compute metrics
    metrics = compute_detailed_metrics(all_predictions, all_labels, dataset.tag_vocab)

    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'tokens': all_tokens,
        'metrics': metrics
    }


def compute_detailed_metrics(
    predictions: list[list[str]],
    labels: list[list[str]],
    tag_vocab: dict[str, int]
) -> dict[str, Any]:
    """Compute detailed evaluation metrics."""
    metrics = {}

    if HAS_SEQEVAL:
        # Sequence-level metrics using seqeval
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        # Get unique labels for per-tag metrics
        unique_labels = sorted(set(tag for sent in labels for tag in sent))

        per_tag_metrics = {}
        for i, tag in enumerate(unique_labels):
            if i < len(precision):
                per_tag_metrics[tag] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i]) if i < len(support) else 0
                }

        # Overall metrics
        macro_f1 = np.mean([m['f1'] for m in per_tag_metrics.values()])
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )

        metrics.update({
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1),
            'per_tag_metrics': per_tag_metrics
        })

        # Classification report
        try:
            report = classification_report(labels, predictions, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        except:
            pass

    else:
        # Fallback token-level metrics
        total_tokens = 0
        correct_tokens = 0

        for pred_seq, true_seq in zip(predictions, labels, strict=False):
            for pred, true in zip(pred_seq, true_seq, strict=False):
                total_tokens += 1
                if pred == true:
                    correct_tokens += 1

        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        metrics.update({
            'token_accuracy': accuracy,
            'total_tokens': total_tokens,
            'correct_tokens': correct_tokens
        })

    return metrics


def create_confusion_matrix(
    predictions: list[list[str]],
    labels: list[list[str]],
    tag_vocab: dict[str, int],
    output_path: str = None
) -> np.ndarray:
    """Create and optionally plot confusion matrix."""
    # Flatten sequences
    flat_preds = [tag for seq in predictions for tag in seq]
    flat_labels = [tag for seq in labels for tag in seq]

    # Get unique tags (excluding padding)
    unique_tags = sorted([tag for tag in tag_vocab.keys() if tag != '<PAD>'])

    if len(unique_tags) > 60:
        print("Warning: Too many tags for confusion matrix visualization")
        return None

    # Create confusion matrix
    cm = confusion_matrix(flat_labels, flat_preds, labels=unique_tags)

    # Plot if matplotlib available
    if HAS_PLOTTING and output_path:
        plt.figure(figsize=(max(8, len(unique_tags) * 0.5), max(6, len(unique_tags) * 0.4)))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_tags, yticklabels=unique_tags)
        plt.title('POS Tagging Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {output_path}")

    return cm


def print_evaluation_report(results: dict[str, Any], tag_vocab: dict[str, int]) -> None:
    """Print detailed evaluation report."""
    metrics = results['metrics']

    print("\n" + "="*80)
    print("POS TAGGING EVALUATION REPORT")
    print("="*80)

    # Overall metrics
    if 'accuracy' in metrics:
        print(f"Sequence Accuracy: {metrics['accuracy']:.4f}")
    if 'token_accuracy' in metrics:
        print(f"Token Accuracy: {metrics['token_accuracy']:.4f}")

    if 'micro_f1' in metrics:
        print(f"Micro F1: {metrics['micro_f1']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")

    # Per-tag metrics
    if 'per_tag_metrics' in metrics:
        print("\nPer-Tag Performance:")
        print("-" * 60)
        print(f"{'Tag':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)

        for tag, tag_metrics in sorted(metrics['per_tag_metrics'].items()):
            if tag != '<PAD>':
                print(f"{tag:<10} {tag_metrics['precision']:<10.3f} "
                      f"{tag_metrics['recall']:<10.3f} {tag_metrics['f1']:<10.3f} "
                      f"{tag_metrics['support']:<10}")

    # Dataset statistics
    print("\nDataset Statistics:")
    print(f"Total Sentences: {len(results['predictions'])}")
    print(f"Total Tokens: {sum(len(seq) for seq in results['predictions'])}")
    print(f"Unique Tags: {len(tag_vocab) - 1}")  # Exclude <PAD>

    # Examples of errors (first 5)
    print("\nExample Predictions (first 5 sentences):")
    print("-" * 80)

    for i in range(min(5, len(results['predictions']))):
        tokens = results['tokens'][i]
        preds = results['predictions'][i]
        labels = results['labels'][i]

        print(f"\nSentence {i+1}:")
        for token, pred, label in zip(tokens, preds, labels, strict=False):
            status = "✓" if pred == label else "✗"
            print(f"  {token:<15} {pred:<8} {label:<8} {status}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate POS Tagging Model"
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str, help="Path to test dataset")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test data")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions to file")
    parser.add_argument("--confusion-matrix", action="store_true", help="Generate confusion matrix")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

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

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, vocab, tag_vocab, config = load_model(args.checkpoint, device)
    logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")

    # Load test data
    if args.synthetic:
        logger.info("Using synthetic test data")
        # Create synthetic test data
        test_sentences = [
            ["The", "cat", "sat", "on", "the", "mat"],
            ["Dogs", "bark", "loudly", "at", "night"],
            ["I", "love", "machine", "learning", "very", "much"],
            ["Python", "is", "a", "great", "programming", "language"],
            ["Natural", "language", "processing", "is", "fascinating"]
        ]
        test_tags = [
            ["DT", "NN", "VBD", "IN", "DT", "NN"],
            ["NNS", "VBP", "RB", "IN", "NN"],
            ["PRP", "VBP", "NN", "NN", "RB", "RB"],
            ["NNP", "VBZ", "DT", "JJ", "NN", "NN"],
            ["JJ", "NN", "NN", "VBZ", "JJ"]
        ]

        test_dataset = POSDataset(test_sentences, test_tags, vocab, tag_vocab)

    elif args.data_path:
        logger.info(f"Loading test data from {args.data_path}")
        dataset_info = load_pos_tagging_dataset(args.data_path, seed=42)
        test_dataset = dataset_info['datasets']['test']

    else:
        logger.error("Must specify either --data-path or --synthetic")
        return

    logger.info(f"Test dataset size: {len(test_dataset)} sentences")

    # Evaluate model
    logger.info("Running evaluation...")
    results = evaluate_model(model, test_dataset, device, args.batch_size)

    # Print report
    print_evaluation_report(results, tag_vocab)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        metrics_serializable = json.loads(json.dumps(results['metrics'], default=convert_numpy))
        json.dump(metrics_serializable, f, indent=2)

    logger.info(f"Evaluation metrics saved to {output_dir / 'evaluation_metrics.json'}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_data = {
            'predictions': results['predictions'],
            'labels': results['labels'],
            'tokens': results['tokens']
        }

        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(predictions_data, f, indent=2)

        logger.info(f"Predictions saved to {output_dir / 'predictions.json'}")

    # Generate confusion matrix if requested
    if args.confusion_matrix:
        cm_path = output_dir / 'confusion_matrix.png'
        cm = create_confusion_matrix(
            results['predictions'],
            results['labels'],
            tag_vocab,
            str(cm_path) if HAS_PLOTTING else None
        )

        if cm is not None:
            # Save confusion matrix data
            np.save(output_dir / 'confusion_matrix.npy', cm)
            logger.info(f"Confusion matrix data saved to {output_dir / 'confusion_matrix.npy'}")

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
