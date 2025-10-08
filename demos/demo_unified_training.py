#!/usr/bin/env python3
"""
Demonstration of the new unified training interface.

This script shows the key features of the consolidated training system:
1. Configuration-driven workflows
2. CLI parameter overrides
3. Multiple dataset support
4. Config save/load
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*70}")
    print(f"Demo: {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    print(f"{'-'*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print(f"Exit code: {result.returncode}")
    return result.returncode == 0

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Unified Training Interface Demonstration                      ║
║                                                                        ║
║  This demo showcases the new configuration-driven training system     ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    demos = [
        ("python train.py --help", 
         "Show help message and available options"),
        
        ("python train.py --list-datasets", 
         "List all available datasets"),
        
        ("python train.py --config config/training/quick_test.yaml --save-config /tmp/demo_config.yaml",
         "Load config from file and save resolved config"),
        
        ("cat /tmp/demo_config.yaml",
         "Display the saved configuration file"),
        
        ("python train.py --dataset mnist --epochs 5 --batch-size 32 --device cpu",
         "Train with CLI arguments (dry run)"),
        
        ("python eval.py --help",
         "Show evaluation script help"),
    ]

    successful = 0
    failed = 0

    for cmd, description in demos:
        if run_command(cmd, description):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"Demo Summary")
    print(f"{'='*70}")
    print(f"Successful demos: {successful}/{len(demos)}")
    print(f"Failed demos: {failed}/{len(demos)}")
    
    if failed == 0:
        print("\n✅ All demos completed successfully!")
    else:
        print(f"\n⚠️  {failed} demo(s) failed")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Review the configuration templates in config/training/")
    print("2. Create your own configuration file for your experiment")
    print("3. Run: python train.py --config your_config.yaml")
    print("4. Evaluate: python eval.py --checkpoint checkpoints/model.pt --dataset mnist")
    print("\nSee docs/SCRIPT_CONSOLIDATION.md for complete documentation.")
    print("="*70)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
