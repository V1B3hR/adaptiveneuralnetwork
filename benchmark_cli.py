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
from core.robustness_validator import RobustnessValidator, run_robustness_validation
from core.adversarial_benchmark import AdversarialSignalTester
from core.benchmark_documentor import BenchmarkDocumentor
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
  python benchmark_cli.py --run-robustness
  python benchmark_cli.py --run-combined
  python benchmark_cli.py --generate-report --output report.txt
  python benchmark_cli.py --compare baseline.json
  python benchmark_cli.py --run-benchmark --save-results results.json
  python benchmark_cli.py --run-robustness --no-stress-tests
        """
    )
    
    # Main actions
    parser.add_argument("--run-benchmark", action="store_true",
                       help="Run comprehensive intelligence benchmark")
    parser.add_argument("--run-robustness", action="store_true",
                       help="Run comprehensive robustness validation")
    parser.add_argument("--run-adversarial", action="store_true",
                       help="Run adversarial signal benchmark tests")
    parser.add_argument("--run-combined", action="store_true",
                       help="Run both intelligence benchmark and robustness validation")
    parser.add_argument("--generate-comprehensive-docs", action="store_true",
                       help="Generate comprehensive benchmark documentation")
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
    parser.add_argument("--no-stress-tests", action="store_true",
                       help="Skip stress testing in robustness validation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--ethics-only", action="store_true",
                       help="Run only ethics compliance checks")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.run_benchmark or args.run_robustness or args.run_adversarial or args.run_combined or 
            args.generate_report or args.generate_comprehensive_docs or args.compare or args.ethics_only):
        parser.error("Must specify an action: --run-benchmark, --run-robustness, --run-adversarial, --run-combined, --generate-report, --generate-comprehensive-docs, --compare, or --ethics-only")
    
    benchmark = IntelligenceBenchmark()
    
    try:
        if args.ethics_only:
            run_ethics_only_check()
            
        elif args.run_benchmark:
            run_benchmark_action(benchmark, args)
            
        elif args.run_robustness:
            run_robustness_action(args)
            
        elif args.run_adversarial:
            run_adversarial_action(args)
            
        elif args.run_combined:
            run_combined_action(benchmark, args)
        
        elif args.generate_comprehensive_docs:
            run_comprehensive_docs_action(args)
            
        elif args.generate_report:
            generate_report_action(benchmark, args)
            
        elif args.compare:
            compare_action(benchmark, args)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def run_robustness_action(args):
    """Run comprehensive robustness validation"""
    print("Starting Comprehensive AI System Robustness Validation...")
    
    # Run robustness validation
    include_stress_tests = not args.no_stress_tests
    validator = RobustnessValidator()
    results = validator.run_comprehensive_robustness_validation(include_stress_tests=include_stress_tests)
    
    # Generate and display report
    report = validator.generate_robustness_report(args.output)
    if not args.output:
        print("\n" + "=" * 60)
        print("ROBUSTNESS VALIDATION REPORT")
        print("=" * 60)
        print(report)
    
    # Save results if requested
    if args.save_results:
        validator.save_validation_data(args.save_results)
    
    # Display summary
    print(f"\n{'=' * 60}")
    print("ROBUSTNESS VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Overall Robustness Score: {results['overall_robustness_score']:.1f}/100")
    print(f"Deployment Readiness: {results['deployment_readiness']}")
    print(f"Ethics Compliance: {'✓ PASSED' if results['ethics_compliance']['compliant'] else '✗ FAILED'}")
    print(f"Duration: {results['total_duration_seconds']:.2f}s")
    
    if results['ethics_compliance']['compliant']:
        print("\n✓ Robustness validation complete. System validated across deployment scenarios.")
    else:
        print("\n✗ Ethics compliance violation detected during robustness validation.")
        sys.exit(1)


def run_combined_action(benchmark, args):
    """Run both intelligence benchmark and robustness validation"""
    print("Starting Combined Intelligence + Robustness Validation...")
    print("=" * 70)
    
    # Run combined validation
    include_comparisons = not args.no_comparisons
    results = benchmark.run_comprehensive_benchmark(
        include_comparisons=include_comparisons, 
        include_robustness=True
    )
    
    # Generate and display report
    report = benchmark.generate_benchmark_report(args.output)
    if not args.output:
        print("\n" + "=" * 60)
        print("COMBINED VALIDATION REPORT")
        print("=" * 60)
        print(report)
    
    # Save results if requested
    if args.save_results:
        benchmark.save_benchmark_data(args.save_results)
    
    # Display summary
    print(f"\n{'=' * 70}")
    print("COMBINED VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    
    if 'combined_intelligence_robustness_score' in results:
        print(f"Combined Intelligence + Robustness Score: {results['combined_intelligence_robustness_score']:.1f}/100")
    
    print(f"Intelligence Score: {results['overall_score']:.1f}/100")
    
    if 'robustness_validation' in results:
        robustness_data = results['robustness_validation']
        print(f"Robustness Score: {robustness_data['overall_robustness_score']:.1f}/100")
        print(f"Deployment Readiness: {robustness_data['deployment_readiness']}")
    
    print(f"Ethics Compliance: {'✓ PASSED' if results['ethics_compliance'] else '✗ FAILED'}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Duration: {results['performance_metrics']['benchmark_duration_seconds']:.2f}s")
    
    if results['ethics_compliance']:
        print("\n✓ Combined validation complete. System validated for intelligence and robustness.")
    else:
        print("\n✗ Ethics compliance violation detected during validation.")
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


def run_adversarial_action(args):
    """Run adversarial signal benchmark tests"""
    print("Starting Adversarial Signal Benchmark Tests...")
    print("=" * 60)
    
    # Run adversarial testing
    adversarial_tester = AdversarialSignalTester()
    results = adversarial_tester.run_adversarial_benchmark()
    
    # Display summary
    print(f"\n{'=' * 60}")
    print("ADVERSARIAL BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Adversarial Resilience Score: {results['adversarial_resilience_score']:.1f}/100")
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    print(f"Average Performance Degradation: {results['average_performance_degradation']:.1f}%")
    
    if results['failure_modes']:
        print("\nFailure Modes Detected:")
        for scenario, mode in results['failure_modes'].items():
            print(f"  - {scenario}: {mode}")
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.save_results}")
    
    print(f"\n✓ Adversarial benchmark complete.")


def run_comprehensive_docs_action(args):
    """Generate comprehensive benchmark documentation"""
    print("Generating Comprehensive Benchmark Documentation...")
    print("=" * 60)
    
    # Check if results file is provided
    if not args.save_results:
        print("Error: --save-results must be specified to provide benchmark data for documentation")
        sys.exit(1)
    
    try:
        # Load benchmark results
        import json
        with open(args.save_results, 'r') as f:
            results = json.load(f)
        
        # Generate comprehensive documentation
        documentor = BenchmarkDocumentor()
        output_dir = documentor.document_benchmark_results(results)
        
        print(f"✓ Comprehensive documentation generated in: {output_dir}")
        print("\nGenerated reports:")
        print("  - main_report_*.md: Executive summary and key findings")
        print("  - failure_analysis_*.md: Detailed failure mode analysis")
        print("  - improvement_proposals_*.md: Actionable improvement recommendations")
        print("  - trends_analysis_*.md: Performance trends and future recommendations")
        print("  - raw_data_*.json: Complete benchmark data")
        
    except FileNotFoundError:
        print(f"Error: Results file '{args.save_results}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in results file '{args.save_results}'")
        sys.exit(1)


if __name__ == "__main__":
    main()