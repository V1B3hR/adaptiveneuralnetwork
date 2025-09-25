# Archived Benchmark Results

This directory contains historical benchmark results and analysis reports.

## Contents

- `main_report_20250910_105644.md` - Main benchmark analysis report
- `failure_analysis_20250910_105644.md` - Detailed failure analysis
- `improvement_proposals_20250910_105644.md` - Suggested improvements
- `trends_analysis_20250910_105644.md` - Performance trend analysis

## Current Benchmarks

For current benchmark capabilities and how to run them, see:
- [Training Guide](../docs/training/TRAINING_GUIDE.md) - Training benchmarks and datasets
- [Testing Guide](../docs/testing/TESTING_GUIDE.md) - Intelligence and robustness benchmarks
- [Implementation Guide](../docs/implementation/IMPLEMENTATION_GUIDE.md) - Performance optimization

## Running New Benchmarks

```bash
# Intelligence benchmarks
python scripts/run_intelligence_benchmark.py

# Performance benchmarks  
python benchmark_cli.py --full-suite

# Training benchmarks
python run_essay_benchmark.py --epochs 50
```

Results from new benchmark runs are saved to the repository root as JSON files:
- `benchmark_results.json` - Performance benchmarks
- `final_validation.json` - Validation results
- Various training result files from specific benchmark runs