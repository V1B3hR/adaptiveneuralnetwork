#!/usr/bin/env python3
"""
Quick Smoke Test Runner

This script runs the smoke tests for quick validation of the Adaptive Neural Network
text classification features. It's designed for CI/CD pipelines and quick validation.

Usage:
    python run_smoke_tests.py              # Run all smoke tests
    python run_smoke_tests.py --cli-only   # Run CLI test only
    python run_smoke_tests.py --suite-only # Run test suite only
"""

import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    try:
        import pandas
        import sklearn
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstalling required dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[nlp]", "-q"],
            check=True
        )
        print("✓ Dependencies installed successfully")
        return True


def run_cli_smoke_test():
    """Run smoke test via CLI interface."""
    print("\n" + "=" * 60)
    print("CLI Smoke Test")
    print("=" * 60)
    
    output_dir = Path("outputs/smoke_test_cli")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = subprocess.run(
        [
            sys.executable, "-m", 
            "adaptiveneuralnetwork.training.scripts.run_bitext_training",
            "--mode", "smoke",
            "--output-dir", str(output_dir)
        ],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✓ CLI smoke test passed")
        
        # Verify output files
        results_file = output_dir / "smoke_test_results.json"
        model_file = output_dir / "smoke_test_model.pkl"
        
        if results_file.exists() and model_file.exists():
            print(f"✓ Output files created:")
            print(f"  - {results_file}")
            print(f"  - {model_file}")
        return True
    else:
        print("\n✗ CLI smoke test failed")
        return False


def run_test_suite():
    """Run comprehensive smoke test suite."""
    print("\n" + "=" * 60)
    print("Comprehensive Smoke Test Suite")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "tests/test_quickstart_features.py"],
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n✓ Test suite passed")
        return True
    else:
        print("\n✗ Test suite failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run smoke tests for quick validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_smoke_tests.py              # Run all smoke tests
  python run_smoke_tests.py --cli-only   # Run CLI test only
  python run_smoke_tests.py --suite-only # Run test suite only
        """
    )
    
    parser.add_argument(
        "--cli-only",
        action="store_true",
        help="Run only CLI smoke test"
    )
    
    parser.add_argument(
        "--suite-only",
        action="store_true",
        help="Run only test suite"
    )
    
    parser.add_argument(
        "--skip-deps-check",
        action="store_true",
        help="Skip dependency check"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Smoke Tests for Quick Validation")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_deps_check:
        try:
            check_dependencies()
        except Exception as e:
            print(f"\n✗ Failed to install dependencies: {e}")
            return 1
    
    # Run tests
    results = []
    
    if args.cli_only:
        results.append(("CLI Test", run_cli_smoke_test()))
    elif args.suite_only:
        results.append(("Test Suite", run_test_suite()))
    else:
        # Run both
        results.append(("CLI Test", run_cli_smoke_test()))
        results.append(("Test Suite", run_test_suite()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ All smoke tests completed successfully!")
        return 0
    else:
        print(f"\n❌ {total - passed} smoke test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
