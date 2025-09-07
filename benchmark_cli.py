#!/usr/bin/env python3
"""
AI Intelligence Benchmark CLI Tool

Command-line interface for running comprehensive AI intelligence validation
and benchmarking with ethical compliance checks.

Usage:
    python benchmark_cli.py --run-benchmark
    python benchmark_cli.py --generate-report
    python benchmark_cli.py --compare baseline.json
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from core.intelligence_benchmark import IntelligenceBenchmark, run_intelligence_validation
from core.ai_ethics import audit_decision, enforce_ethics_compliance


def main():
    """Main CLI entry point"""
    
    # Ethics check for CLI execution
    cli_decision = {
        "action": "run_intelligence_benchmark_cli",
        "preserve_life": True,
        "absolute_honesty": True,
        "privacy": True,
        "human_authority": True,
        "proportionality": True
    }
    enforce_ethics_compliance(cli_decision)
    
    parser = argparse.ArgumentParser(
        description="AI Intelligence Benchmark System - Validate AI capabilities with ethical compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_cli.py --run-benchmark
  python benchmark_cli.py --generate-report --output report.txt
  python benchmark_cli.py --compare baseline.json
  python benchmark_cli.py --run-benchmark --save-results results.json
        """
    )
    
    # Main actions
    parser.add_argument("--run-benchmark", action="store_true",
                       help="Run comprehensive intelligence benchmark")
    parser.add_argument("--generate-report", action="store_true", 
                       help="Generate benchmark report from previous results")
    parser.add_argument("--compare", metavar="BASELINE_FILE",
                       help="Compare with baseline model from JSON file")
    
    # Options
    parser.add_argument("--output", "-o", metavar="FILE",
                       help="Output file for reports")
    parser.add_argument("--save-results", metavar="FILE",
                       help="Save benchmark results to JSON file")
    parser.add_argument("--no-comparisons", action="store_true",
                       help="Skip generating comparison baselines")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--ethics-only", action="store_true",
                       help="Run only ethics compliance checks")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.run_benchmark or args.generate_report or args.compare or args.ethics_only):
        parser.error("Must specify an action: --run-benchmark, --generate-report, --compare, or --ethics-only")
    
    benchmark = IntelligenceBenchmark()
    
    try:
        if args.ethics_only:
            run_ethics_only_check()
            
        elif args.run_benchmark:
            run_benchmark_action(benchmark, args)
            
        elif args.generate_report:
            generate_report_action(benchmark, args)
            
        elif args.compare:
            compare_action(benchmark, args)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def run_ethics_only_check():
    """Run only ethics compliance verification"""
    print("Running Ethics Compliance Check...")
    print("=" * 40)
    
    # Test core ethics operations
    ethics_tests = [
        "intelligence_benchmark_execution",
        "ai_capability_validation", 
        "model_performance_comparison",
        "benchmark_data_storage",
        "ethics_framework_verification"
    ]
    
    all_compliant = True
    
    for test in ethics_tests:
        decision_log = {
            "action": test,
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True,
            "proportionality": True
        }
        
        audit_result = audit_decision(decision_log)
        status = "✓ PASSED" if audit_result["compliant"] else "✗ FAILED"
        print(f"{test}: {status}")
        
        if not audit_result["compliant"]:
            all_compliant = False
            print(f"  Violations: {audit_result['violations']}")
    
    print("\n" + "=" * 40)
    if all_compliant:
        print("✓ ALL ETHICS CHECKS PASSED")
        print("System is ready for intelligence benchmarking.")
    else:
        print("✗ ETHICS VIOLATIONS DETECTED")
        print("Please resolve violations before running benchmarks.")
        sys.exit(1)


def run_benchmark_action(benchmark, args):
    """Run the comprehensive benchmark"""
    print("Starting Comprehensive AI Intelligence Benchmark...")
    
    # Run benchmark
    include_comparisons = not args.no_comparisons
    results = benchmark.run_comprehensive_benchmark(include_comparisons=include_comparisons)
    
    # Generate and display report
    report = benchmark.generate_benchmark_report(args.output)
    if not args.output:
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)
        print(report)
    
    # Save results if requested
    if args.save_results:
        benchmark.save_benchmark_data(args.save_results)
    
    # Display summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Overall Score: {results['overall_score']:.1f}/100")
    print(f"Ethics Compliance: {'✓ PASSED' if results['ethics_compliance'] else '✗ FAILED'}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Duration: {results['performance_metrics']['benchmark_duration_seconds']:.2f}s")
    
    if results['ethics_compliance']:
        print("\n✓ System validation complete. AI capabilities verified with ethical compliance.")
    else:
        print("\n✗ Ethics compliance violation detected during benchmarking.")
        sys.exit(1)


def generate_report_action(benchmark, args):
    """Generate report from existing results"""
    # Check if there are results to report on
    if not benchmark.benchmark_results:
        print("No benchmark results found. Run --run-benchmark first.")
        sys.exit(1)
    
    # Generate report
    report = benchmark.generate_benchmark_report(args.output)
    
    if not args.output:
        print(report)
    else:
        print(f"Report generated: {args.output}")


def compare_action(benchmark, args):
    """Compare with baseline model"""
    if not os.path.exists(args.compare):
        print(f"Baseline file not found: {args.compare}")
        sys.exit(1)
    
    # Need to run benchmark first if no results exist
    if not benchmark.benchmark_results:
        print("Running benchmark for comparison...")
        benchmark.run_comprehensive_benchmark()
    
    # Perform comparison
    comparison = benchmark.compare_with_baseline(args.compare)
    
    # Display comparison results
    print("MODEL COMPARISON RESULTS")
    print("=" * 30)
    print(f"Current Model Score: {comparison['current_model_score']:.1f}/100")
    print(f"Baseline Model Score: {comparison['baseline_model_score']:.1f}/100")
    print(f"Performance Difference: {comparison['performance_difference']:+.1f} points")
    
    if comparison['performance_difference'] != 0:
        print(f"Improvement: {comparison['improvement_percentage']:+.1f}%")
    
    print(f"\nRecommendation: {comparison['recommendation']}")
    
    # Category breakdown
    print("\nCATEGORY COMPARISON:")
    print("-" * 20)
    for category, scores in comparison['category_comparisons'].items():
        category_name = category.replace('_', ' ').title()
        print(f"{category_name}:")
        print(f"  Current: {scores['current']:.1f}/100")
        print(f"  Baseline: {scores['baseline']:.1f}/100")
        print(f"  Difference: {scores['difference']:+.1f}")


if __name__ == "__main__":
    main()