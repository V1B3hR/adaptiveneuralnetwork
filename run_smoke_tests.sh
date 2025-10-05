#!/bin/bash
# Quick smoke test runner for validation
# This script runs the smoke tests for quick validation of the system
#
# Usage:
#   ./run_smoke_tests.sh              # Run all smoke tests
#   ./run_smoke_tests.sh --cli-only   # Run CLI test only
#   ./run_smoke_tests.sh --suite-only # Run test suite only

set -e

echo "========================================"
echo "Running Smoke Tests for Quick Validation"
echo "========================================"
echo ""

# Parse arguments
CLI_ONLY=false
SUITE_ONLY=false

for arg in "$@"; do
    case $arg in
        --cli-only)
            CLI_ONLY=true
            shift
            ;;
        --suite-only)
            SUITE_ONLY=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Run CLI test
if [ "$SUITE_ONLY" = false ]; then
    echo "----------------------------------------"
    echo "Test 1: CLI Smoke Test (Default)"
    echo "----------------------------------------"
    python -m adaptiveneuralnetwork.training.scripts.run_bitext_training \
        --mode smoke \
        --output-dir outputs/smoke_test_cli
    echo ""
fi

# Run test suite
if [ "$CLI_ONLY" = false ]; then
    echo "----------------------------------------"
    echo "Test 2: Programmatic Smoke Test Suite"
    echo "----------------------------------------"
    python tests/test_quickstart_features.py
    echo ""
fi

echo "========================================"
echo "✅ All Smoke Tests Completed Successfully!"
echo "========================================"
echo ""
echo "Output files created:"
echo "  - outputs/smoke_test_cli/smoke_test_results.json"
echo "  - outputs/smoke_test_cli/smoke_test_model.pkl"
echo ""
