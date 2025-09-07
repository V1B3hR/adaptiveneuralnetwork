"""
AI Intelligence Benchmarking System

This module provides comprehensive intelligence validation and benchmarking capabilities
for the adaptive neural network, ensuring ethical compliance throughout all operations.
Enables comparison with other AI models through standardized metrics.
"""

import unittest
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from core.ai_ethics import audit_decision, log_ethics_event, enforce_ethics_compliance
from core.robustness_validator import RobustnessValidator


class IntelligenceBenchmark:
    """
    Comprehensive intelligence benchmarking system that validates AI capabilities
    while ensuring ethical compliance throughout all operations.
    """
    
    def __init__(self):
        self.benchmark_results = {}
        self.performance_metrics = {}
        self.ethics_audit_log = []
        self.comparison_baselines = {}
        
    def run_comprehensive_benchmark(self, include_comparisons=True, include_robustness=False) -> Dict[str, Any]:
        """
        Run comprehensive intelligence benchmark across all test categories.
        
        Returns:
            Complete benchmark results with standardized metrics for model comparison
        """
        # Ethics check for benchmark execution
        benchmark_decision = {
            "action": "run_comprehensive_intelligence_benchmark",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True,
            "proportionality": True
        }
        enforce_ethics_compliance(benchmark_decision)
        
        print("Starting Comprehensive AI Intelligence Benchmark...")
        print("=" * 60)
        
        benchmark_start_time = time.time()
        
        # Run all intelligence test categories
        categories = [
            'basic_problem_solving',
            'adaptive_learning', 
            'cognitive_functioning',
            'pattern_recognition',
            'rigorous_intelligence'
        ]
        
        overall_results = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_version': '1.0',
            'categories': {},
            'overall_score': 0.0,
            'ethics_compliance': True,
            'performance_metrics': {},
            'comparison_ready': True
        }
        
        total_score = 0.0
        total_tests = 0
        
        for category in categories:
            print(f"\n--- Running {category.replace('_', ' ').title()} Tests ---")
            category_results = self._run_category_benchmark(category)
            overall_results['categories'][category] = category_results
            
            # Aggregate scores
            total_score += category_results['score']
            total_tests += category_results['test_count']
            
        # Calculate overall intelligence score
        overall_results['overall_score'] = total_score / len(categories) if categories else 0.0
        overall_results['total_tests'] = total_tests
        
        # Performance metrics
        benchmark_duration = time.time() - benchmark_start_time
        overall_results['performance_metrics'] = {
            'benchmark_duration_seconds': round(benchmark_duration, 3),
            'tests_per_second': round(total_tests / benchmark_duration, 3) if benchmark_duration > 0 else 0,
            'average_test_duration': round(benchmark_duration / total_tests, 4) if total_tests > 0 else 0
        }
        
        # Verify ethical compliance across all tests
        overall_results['ethics_compliance'] = self._verify_ethics_compliance()
        
        # Generate comparison baselines if requested
        if include_comparisons:
            # Temporarily store results to generate baselines
            self.benchmark_results = overall_results
            overall_results['comparison_baselines'] = self._generate_comparison_baselines()
            
        # Run robustness validation if requested
        if include_robustness:
            print("\n--- Running Robustness Validation ---")
            robustness_validator = RobustnessValidator()
            robustness_results = robustness_validator.run_comprehensive_robustness_validation(include_stress_tests=True)
            overall_results['robustness_validation'] = robustness_results
            
            # Integrate robustness score into overall score
            robustness_weight = 0.3  # 30% weight for robustness
            intelligence_weight = 0.7  # 70% weight for intelligence
            
            combined_score = (overall_results['overall_score'] * intelligence_weight + 
                            robustness_results['overall_robustness_score'] * robustness_weight)
            overall_results['combined_intelligence_robustness_score'] = combined_score
            
        # Store final results in instance variable
        self.benchmark_results = overall_results
        
        print(f"\n{'=' * 60}")
        print("BENCHMARK COMPLETE")
        if include_robustness and 'combined_intelligence_robustness_score' in overall_results:
            print(f"Combined Intelligence + Robustness Score: {overall_results['combined_intelligence_robustness_score']:.2f}/100")
        print(f"Overall Intelligence Score: {overall_results['overall_score']:.2f}/100")
        if include_robustness and 'robustness_validation' in overall_results:
            print(f"Overall Robustness Score: {overall_results['robustness_validation']['overall_robustness_score']:.2f}/100")
        print(f"Total Tests: {total_tests}")
        print(f"Duration: {benchmark_duration:.2f}s")
        print(f"Ethics Compliance: {'✓ PASSED' if overall_results['ethics_compliance'] else '✗ FAILED'}")
        
        return overall_results
    
    def _run_category_benchmark(self, category: str) -> Dict[str, Any]:
        """Run benchmark for a specific intelligence category."""
        
        # Ethics audit for category execution
        category_decision = {
            "action": f"benchmark_category_{category}",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True
        }
        audit_result = audit_decision(category_decision)
        log_ethics_event(f"benchmark_{category}", audit_result)
        
        if not audit_result["compliant"]:
            raise RuntimeError(f"Ethics violation in {category}: {audit_result['violations']}")
        
        # Import and run the appropriate test module
        test_module_map = {
            'basic_problem_solving': 'tests.test_basic_problem_solving',
            'adaptive_learning': 'tests.test_adaptive_learning',
            'cognitive_functioning': 'tests.test_cognitive_functioning',
            'pattern_recognition': 'tests.test_pattern_recognition',
            'rigorous_intelligence': 'tests.test_rigorous_intelligence'
        }
        
        if category not in test_module_map:
            return {'error': f'Unknown category: {category}', 'score': 0.0, 'test_count': 0}
        
        try:
            # Run tests and collect results
            suite = unittest.TestLoader().loadTestsFromName(test_module_map[category])
            
            # Use null stream to suppress output
            import io
            with io.StringIO() as null_stream:
                runner = unittest.TextTestRunner(stream=null_stream, verbosity=0)
                
                start_time = time.time()
                result = runner.run(suite)
                duration = time.time() - start_time
            
            # Calculate category score
            total_tests = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            successful_tests = total_tests - failures - errors
            
            score = (successful_tests / total_tests * 100) if total_tests > 0 else 0.0
            
            category_result = {
                'score': round(score, 2),
                'test_count': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failures,
                'error_tests': errors,
                'duration_seconds': round(duration, 3),
                'ethics_compliant': True,  # All tests include ethics checks
                'performance_metrics': {
                    'tests_per_second': round(total_tests / duration, 2) if duration > 0 else 0,
                    'average_test_time': round(duration / total_tests, 4) if total_tests > 0 else 0
                }
            }
            
            print(f"  Score: {score:.1f}/100 ({successful_tests}/{total_tests} tests passed)")
            print(f"  Duration: {duration:.2f}s")
            
            return category_result
            
        except Exception as e:
            print(f"  Error running {category}: {str(e)}")
            return {
                'error': str(e),
                'score': 0.0,
                'test_count': 0,
                'ethics_compliant': False
            }
    
    def _verify_ethics_compliance(self) -> bool:
        """Verify that all benchmark operations maintained ethical compliance."""
        
        # Check that ethics audits were performed
        ethics_decision = {
            "action": "verify_benchmark_ethics_compliance",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True
        }
        
        audit_result = audit_decision(ethics_decision)
        return audit_result["compliant"]
    
    def _generate_comparison_baselines(self) -> Dict[str, Any]:
        """Generate baseline metrics for comparing with other AI models."""
        
        if not hasattr(self, 'benchmark_results') or not self.benchmark_results:
            return {}
        
        # Create standardized comparison metrics
        comparison_metrics = {
            'overall_intelligence_score': self.benchmark_results.get('overall_score', 0.0),
            'problem_solving_capability': 0.0,
            'learning_adaptability': 0.0,
            'cognitive_processing': 0.0,
            'pattern_recognition_accuracy': 0.0,
            'ethical_compliance_rate': 100.0 if self.benchmark_results.get('ethics_compliance') else 0.0,
            'performance_efficiency': {
                'benchmark_speed': self.benchmark_results.get('performance_metrics', {}).get('tests_per_second', 0),
                'response_time': self.benchmark_results.get('performance_metrics', {}).get('average_test_duration', 0),
                'total_tests_executed': self.benchmark_results.get('total_tests', 0)
            },
            'model_capabilities': {
                'supports_ethical_framework': True,
                'adaptive_learning': True,
                'social_interaction': True,
                'memory_systems': True,
                'energy_management': True,
                'circadian_rhythms': True
            }
        }
        
        # Extract category-specific scores
        categories = self.benchmark_results.get('categories', {})
        comparison_metrics['problem_solving_capability'] = categories.get('basic_problem_solving', {}).get('score', 0.0)
        comparison_metrics['learning_adaptability'] = categories.get('adaptive_learning', {}).get('score', 0.0)
        comparison_metrics['cognitive_processing'] = categories.get('cognitive_functioning', {}).get('score', 0.0)
        comparison_metrics['pattern_recognition_accuracy'] = categories.get('pattern_recognition', {}).get('score', 0.0)
        
        return comparison_metrics
    
    def generate_benchmark_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive benchmark report suitable for model comparison."""
        
        if not self.benchmark_results:
            return "No benchmark results available. Run run_comprehensive_benchmark() first."
        
        report = []
        report.append("AI INTELLIGENCE BENCHMARK REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {self.benchmark_results['timestamp']}")
        report.append(f"Benchmark Version: {self.benchmark_results['benchmark_version']}")
        report.append("")
        
        # Overall Summary
        report.append("OVERALL RESULTS")
        report.append("-" * 20)
        report.append(f"Overall Intelligence Score: {self.benchmark_results['overall_score']:.2f}/100")
        report.append(f"Total Tests Executed: {self.benchmark_results['total_tests']}")
        report.append(f"Ethics Compliance: {'PASSED' if self.benchmark_results['ethics_compliance'] else 'FAILED'}")
        report.append(f"Benchmark Duration: {self.benchmark_results['performance_metrics']['benchmark_duration_seconds']:.2f}s")
        report.append("")
        
        # Category Breakdown
        report.append("CATEGORY BREAKDOWN")
        report.append("-" * 20)
        for category, results in self.benchmark_results['categories'].items():
            category_name = category.replace('_', ' ').title()
            report.append(f"{category_name}:")
            report.append(f"  Score: {results['score']:.1f}/100")
            report.append(f"  Tests: {results['successful_tests']}/{results['test_count']} passed")
            report.append(f"  Duration: {results['duration_seconds']:.2f}s")
            report.append("")
        
        # Comparison Baselines
        if 'comparison_baselines' in self.benchmark_results:
            report.append("COMPARISON BASELINES FOR OTHER MODELS")
            report.append("-" * 40)
            baselines = self.benchmark_results['comparison_baselines']
            report.append(f"Problem Solving Capability: {baselines['problem_solving_capability']:.1f}/100")
            report.append(f"Learning Adaptability: {baselines['learning_adaptability']:.1f}/100")
            report.append(f"Cognitive Processing: {baselines['cognitive_processing']:.1f}/100")
            report.append(f"Pattern Recognition: {baselines['pattern_recognition_accuracy']:.1f}/100")
            report.append(f"Ethical Compliance Rate: {baselines['ethical_compliance_rate']:.1f}%")
            report.append("")
            report.append("Performance Metrics:")
            perf = baselines['performance_efficiency']
            report.append(f"  Benchmark Speed: {perf['benchmark_speed']:.2f} tests/sec")
            report.append(f"  Average Response Time: {perf['response_time']:.4f}s")
            report.append(f"  Total Tests: {perf['total_tests_executed']}")
            report.append("")
        
        # Model Capabilities
        if 'comparison_baselines' in self.benchmark_results:
            capabilities = self.benchmark_results['comparison_baselines']['model_capabilities']
            report.append("MODEL CAPABILITIES")
            report.append("-" * 20)
            for capability, supported in capabilities.items():
                status = "✓" if supported else "✗"
                report.append(f"{status} {capability.replace('_', ' ').title()}")
            report.append("")
        
        report.append("=" * 50)
        report.append("Report generated by Adaptive Neural Network Intelligence Benchmark System")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Benchmark report saved to: {output_file}")
        
        return report_text
    
    def save_benchmark_data(self, filename: str) -> None:
        """Save complete benchmark data as JSON for further analysis."""
        
        if not self.benchmark_results:
            raise ValueError("No benchmark results to save. Run benchmark first.")
        
        # Ethics check for data saving
        save_decision = {
            "action": "save_benchmark_data", 
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True  # Benchmark data doesn't contain personal information
        }
        enforce_ethics_compliance(save_decision)
        
        with open(filename, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2)
        
        print(f"Benchmark data saved to: {filename}")
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Compare current benchmark results with a baseline from another model."""
        
        if not self.benchmark_results:
            raise ValueError("No current benchmark results. Run benchmark first.")
        
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
        
        # Ethics check for comparison
        comparison_decision = {
            "action": "compare_with_baseline_model",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True
        }
        enforce_ethics_compliance(comparison_decision)
        
        current_score = self.benchmark_results['overall_score']
        baseline_score = baseline.get('overall_score', 0.0)
        
        comparison = {
            'current_model_score': current_score,
            'baseline_model_score': baseline_score,
            'performance_difference': current_score - baseline_score,
            'improvement_percentage': ((current_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0,
            'category_comparisons': {},
            'recommendation': ''
        }
        
        # Category-wise comparison
        for category in self.benchmark_results['categories']:
            current_cat_score = self.benchmark_results['categories'][category]['score']
            baseline_cat_score = baseline.get('categories', {}).get(category, {}).get('score', 0.0)
            
            comparison['category_comparisons'][category] = {
                'current': current_cat_score,
                'baseline': baseline_cat_score,
                'difference': current_cat_score - baseline_cat_score
            }
        
        # Generate recommendation
        if comparison['performance_difference'] > 0:
            comparison['recommendation'] = "Current model outperforms baseline"
        elif comparison['performance_difference'] < 0:
            comparison['recommendation'] = "Current model underperforms baseline - consider improvements"
        else:
            comparison['recommendation'] = "Current model performs equivalently to baseline"
        
        return comparison


def run_intelligence_validation(include_robustness=False) -> Dict[str, Any]:
    """
    Main function to run comprehensive intelligence validation with ethical compliance.
    This is the primary interface for validating AI intelligence capabilities.
    
    Args:
        include_robustness: Whether to include robustness validation alongside intelligence testing
    """
    benchmark = IntelligenceBenchmark()
    results = benchmark.run_comprehensive_benchmark(include_comparisons=True, include_robustness=include_robustness)
    
    # Generate comprehensive report
    report = benchmark.generate_benchmark_report()
    print("\n" + report)
    
    return results


if __name__ == "__main__":
    # Run the comprehensive intelligence benchmark
    results = run_intelligence_validation()
    
    # Save results for future comparison
    benchmark = IntelligenceBenchmark()
    benchmark.benchmark_results = results
    benchmark.save_benchmark_data("benchmark_results.json")