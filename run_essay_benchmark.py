#!/usr/bin/env python3
"""
Script to run Human vs AI Generated Essays classification benchmark.

This script demonstrates the text classification capabilities of the adaptive neural network
on the task of distinguishing between human-written and AI-generated essays.
"""

import argparse
import logging
import torch
from pathlib import Path

# Add the current directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from adaptiveneuralnetwork.api import AdaptiveConfig
from adaptiveneuralnetwork.benchmarks.text_classification import (
    run_essay_classification_benchmark,
    SyntheticEssayDataset,
    EssayDataset
)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_kaggle_dataset(data_path: str, dataset_type: str = "auto") -> EssayDataset:
    """
    Load Kaggle datasets specified in the problem statement.
    
    Supports:
    1. ANNOMI Motivational Interviewing Dataset
    2. Mental Health FAQs Dataset
    3. Social Media Sentiments Analysis Dataset
    
    Args:
        data_path: Path to the dataset files
        dataset_type: Type of dataset ("annomi", "mental_health", "social_media_sentiment", or "auto")
        
    Returns:
        EssayDataset instance
    """
    from adaptiveneuralnetwork.data import (
        load_annomi_dataset, 
        load_mental_health_faqs_dataset,
        load_social_media_sentiment_dataset,
        print_dataset_info
    )
    
    try:
        if dataset_type == "auto":
            # Try to detect dataset type from path
            path_lower = data_path.lower()
            if "annomi" in path_lower or "motivational" in path_lower:
                dataset_type = "annomi"
            elif "mental" in path_lower or "health" in path_lower or "faq" in path_lower:
                dataset_type = "mental_health"
            elif "sentiment" in path_lower or "social" in path_lower or "media" in path_lower:
                dataset_type = "social_media_sentiment"
            else:
                print(f"Could not auto-detect dataset type from path: {data_path}")
                print("Please specify --dataset-type manually")
                print_dataset_info()
                return None
        
        print(f"Loading {dataset_type} dataset from {data_path}")
        
        if dataset_type == "annomi":
            return load_annomi_dataset(data_path)
        elif dataset_type == "mental_health":
            return load_mental_health_faqs_dataset(data_path)
        elif dataset_type == "social_media_sentiment":
            return load_social_media_sentiment_dataset(data_path)
        else:
            print(f"Unknown dataset type: {dataset_type}")
            print("Supported types: annomi, mental_health, social_media_sentiment")
            return None
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to synthetic dataset for demonstration.")
        return None

def main():
    """Main function to run the essay classification benchmark."""
    parser = argparse.ArgumentParser(
        description="Run Human vs AI Generated Essays Classification Benchmark"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--data-path", 
        type=str, 
        help="Path to the Kaggle dataset directory"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "annomi", "mental_health", "social_media_sentiment"],
        default="auto",
        help="Type of Kaggle dataset (auto-detect by default)"
    )
    parser.add_argument(
        "--synthetic", 
        action="store_true", 
        help="Use synthetic data instead of real dataset"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=2000, 
        help="Number of synthetic samples to generate"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,  # Changed default to 100 as per problem statement
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001, 
        help="Learning rate"
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden-dim", 
        type=int, 
        default=128, 
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-nodes", 
        type=int, 
        default=100, 
        help="Number of adaptive nodes"
    )
    
    # Other arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Determine device
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU.")
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = None
    if args.data_path and not args.synthetic:
        dataset = load_kaggle_dataset(args.data_path, args.dataset_type)
    
    if dataset is None:
        logger.info("Creating synthetic dataset...")
        dataset = SyntheticEssayDataset(
            num_samples=args.samples,
            vocab_size=5000,
            max_length=256
        )
        logger.info(f"Created synthetic dataset with {len(dataset)} samples")
    
    # Create configuration
    config = AdaptiveConfig(
        # Model architecture
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        output_dim=2,  # Binary classification
        
        # Training parameters
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        
        # Device
        device=str(device)
    )
    
    logger.info("Starting essay classification benchmark...")
    logger.info(f"Configuration: {args.epochs} epochs, batch size {args.batch_size}, lr {args.learning_rate}")
    
    try:
        # Run benchmark
        results = run_essay_classification_benchmark(
            config=config,
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device
        )
        
        # Print summary results
        print("\n" + "="*60)
        print("ESSAY CLASSIFICATION BENCHMARK RESULTS")
        print("="*60)
        print(f"Final Test Accuracy: {results['final_test_accuracy']:.4f}")
        print(f"Best Test Accuracy: {results['best_test_accuracy']:.4f}")
        print(f"Final Train Accuracy: {results['final_train_accuracy']:.4f}")
        print(f"Training Time: {results['training_time']:.2f} seconds")
        print(f"Total Time: {results['total_time']:.2f} seconds")
        print(f"Model Parameters: {results['model_parameters']:,}")
        print(f"Vocabulary Size: {results['vocab_size']:,}")
        print(f"Training Samples: {results['train_samples']:,}")
        print(f"Test Samples: {results['test_samples']:,}")
        print("="*60)
        
        # Success message
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        raise

if __name__ == "__main__":
    main()