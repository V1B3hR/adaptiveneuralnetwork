#!/usr/bin/env python3
"""
Usability-Enhanced Unified Training Script for Adaptive Neural Network

Features:
- Works as both CLI script and Jupyter/Colab cell.
- Dynamic argument defaults if run in a notebook/cell.
- Friendly dataset selection, with clear error messages.
- Pretty output and error highlighting.
- Training summary table.
- Output file path notification.
- Verbose logging toggle.
- Safe error handling (does not crash on one dataset).
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional: Pretty printing and color
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class Dummy:
        def __getattr__(self, name): return ""
    Fore = Style = Dummy()

# --- Import project modules ---
try:
    from training.scripts.train_new_datasets import (
        train_dataset,
        save_results,
        create_synthetic_dataset
    )
except ImportError as e:
    print(Fore.RED + "[ERROR] Could not import the core training components.\n"
          "Make sure you run this script in the root directory of your project, or adjust the import paths.")
    raise e

# Detect CLI vs Notebook/Cell
def get_args():
    available_datasets = ["vr_driving", "autvi", "digakust"]

    if hasattr(sys, 'argv') and len(sys.argv) > 1 and sys.argv[0].endswith('.py'):
        parser = argparse.ArgumentParser(
            description="Usability-Enhanced Training Script for Adaptive Neural Network\n"
                        "Example: python usability_enhanced_train.py --dataset all --epochs 10"
        )
        parser.add_argument("--dataset", choices=available_datasets + ["all"], default="all",
                            help=f"Dataset to train on (default: all). Choices: {available_datasets + ['all']}")
        parser.add_argument("--data-path", type=str, help="Path to dataset file or directory")
        parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
        parser.add_argument("--num-samples", type=int, default=1000, help="Number of synthetic samples to generate")
        parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save results")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
        parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
        return parser.parse_args()
    else:
        # Defaults for notebook/cell
        class Args:
            dataset = "all"
            data_path = None
            epochs = 10
            num_samples = 1000
            output_dir = "outputs"
            verbose = True
            seed = 42
        return Args()

def set_seed(seed):
    try:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    except Exception:
        pass

def print_header(args):
    print(Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "ADAPTIVE NEURAL NETWORK - USABILITY ENHANCED TRAINING SCRIPT")
    print(Fore.CYAN + "=" * 80)
    print(Fore.YELLOW + "Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Synthetic Samples: {args.num_samples}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Verbose: {args.verbose}")
    print(f"  Seed: {args.seed}")
    print(Fore.CYAN + "=" * 80 + "\n")

def print_summary(all_results):
    print(Fore.MAGENTA + "\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for dataset, results in all_results.items():
        status = (Fore.GREEN + "✓ SUCCESS") if results.get("success") else (Fore.RED + "✗ FAILED")
        print(f"{dataset:20s} - {status}{Style.RESET_ALL}")
        if results.get("success"):
            print(f"  Best Accuracy: {Fore.YELLOW}{results.get('best_accuracy', 0):.4f}{Style.RESET_ALL}")
            print(f"  Training Time: {results.get('training_time', 0):.2f}s")
        else:
            print(Fore.RED + f"  Error: {results.get('error', 'Unknown error')}" + Style.RESET_ALL)
    print("=" * 80 + Style.RESET_ALL)

def main():
    args = get_args()

    # Set logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, 
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # Set random seed
    set_seed(args.seed)

    # Print configuration
    print_header(args)

    # Training workflow
    available_datasets = ["vr_driving", "autvi", "digakust"]

    if args.dataset == "all":
        logger.info("Training on all supported datasets...")
        all_results = {}
        for i, dataset in enumerate(available_datasets, 1):
            print(Fore.BLUE + f"\n{'='*60}\nTraining on {dataset} ({i}/{len(available_datasets)})\n{'='*60}")
            try:
                results = train_dataset(dataset, args)
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(f"✗ {dataset} training crashed: {exc}")
                results = {"success": False, "error": str(exc), "traceback": tb}
            all_results[dataset] = results
            try:
                save_results(results, args.output_dir)
            except Exception as exc:
                logger.error(f"Could not save results for {dataset}: {exc}")
        # Save combined results
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        combined_file = output_path / "all_datasets_results.json"
        try:
            with open(combined_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"✓ Combined results saved to {combined_file}")
        except Exception as exc:
            logger.error(f"Could not save combined results: {exc}")
        print_summary(all_results)
        print(Fore.GREEN + f"\nResults saved in: {combined_file.absolute()}")
    else:
        # Validate dataset
        if args.dataset not in available_datasets:
            print(Fore.RED + f"Dataset '{args.dataset}' is not supported.")
            print(Fore.YELLOW + f"Available datasets: {available_datasets}")
            sys.exit(1)
        logger.info(f"Training on {args.dataset} dataset...")
        try:
            results = train_dataset(args.dataset, args)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"✗ Training crashed: {exc}")
            results = {"success": False, "error": str(exc), "traceback": tb}
        try:
            save_results(results, args.output_dir)
        except Exception as exc:
            logger.error(f"Could not save results: {exc}")
        print_summary({args.dataset: results})
        output_file = Path(args.output_dir) / f"{args.dataset}_results.json"
        print(Fore.GREEN + f"\nResults saved in: {output_file.absolute()}")

    print(Fore.GREEN + "\n✅ Training complete!" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
