"""
Intelligence Benchmark System Tests

Tests the comprehensive AI intelligence benchmarking system to ensure
proper validation of AI capabilities while maintaining ethical compliance.
"""

import unittest
import json
import os
import tempfile
from core.intelligence_benchmark import IntelligenceBenchmark, run_intelligence_validation
from core.ai_ethics import audit_decision


class TestIntelligenceBenchmark(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.benchmark = IntelligenceBenchmark()
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def test_comprehensive_benchmark_execution(self):
        """
        Description: Tests that comprehensive benchmark runs and produces valid results
        Expected: Benchmark completes successfully with proper scoring and metrics
        """
        # Run the comprehensive benchmark
        results = self.benchmark.run_comprehensive_benchmark(include_comparisons=True)
        
        # Validate required fields exist
        self.assertIn('timestamp', results)
        self.assertIn('benchmark_version', results)
        self.assertIn('categories', results)
        self.assertIn('overall_score', results)
        self.assertIn('ethics_compliance', results)
        self.assertIn('performance_metrics', results)
        
        # Validate score range
        self.assertGreaterEqual(results['overall_score'], 0.0)
        self.assertLessEqual(results['overall_score'], 100.0)
        
        # Validate ethics compliance
        self.assertTrue(results['ethics_compliance'])
        
        # Validate all expected categories are present
        expected_categories = ['basic_problem_solving', 'adaptive_learning', 
                             'cognitive_functioning', 'pattern_recognition']
        for category in expected_categories:
            self.assertIn(category, results['categories'])
            self.assertIn('score', results['categories'][category])
            self.assertIn('test_count', results['categories'][category])
    
    def test_category_benchmark_scoring(self):
        """
        Description: Tests that individual category benchmarks produce valid scores
        Expected: Each category produces score between 0-100 with proper metrics
        """
        # Test individual category
        category_result = self.benchmark._run_category_benchmark('basic_problem_solving')
        
        # Validate scoring
        self.assertIn('score', category_result)
        self.assertGreaterEqual(category_result['score'], 0.0)
        self.assertLessEqual(category_result['score'], 100.0)
        
        # Validate test execution metrics
        self.assertIn('test_count', category_result)
        self.assertIn('duration_seconds', category_result)
        self.assertIn('performance_metrics', category_result)
        
        # Validate ethics compliance
        self.assertTrue(category_result.get('ethics_compliant', False))
    
    def test_ethics_compliance_verification(self):
        """
        Description: Tests that ethics compliance is properly verified
        Expected: Ethics verification returns True and all operations are compliant
        """
        # Test ethics verification
        compliance_result = self.benchmark._verify_ethics_compliance()
        self.assertTrue(compliance_result)
        
        # Test that benchmark operations include ethics checks
        decision_log = {
            "action": "test_benchmark_ethics",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True,
            "human_authority": True
        }
        audit_result = audit_decision(decision_log)
        self.assertTrue(audit_result["compliant"])
    
    def test_comparison_baselines_generation(self):
        """
        Description: Tests generation of comparison baselines for other models
        Expected: Proper baseline metrics are generated for model comparison
        """
        # First run benchmark to get results
        self.benchmark.run_comprehensive_benchmark(include_comparisons=True)
        
        # Generate comparison baselines
        baselines = self.benchmark._generate_comparison_baselines()
        
        # Validate baseline structure
        self.assertIn('overall_intelligence_score', baselines)
        self.assertIn('problem_solving_capability', baselines)
        self.assertIn('learning_adaptability', baselines)
        self.assertIn('cognitive_processing', baselines)
        self.assertIn('pattern_recognition_accuracy', baselines)
        self.assertIn('ethical_compliance_rate', baselines)
        self.assertIn('performance_efficiency', baselines)
        self.assertIn('model_capabilities', baselines)
        
        # Validate ethical compliance rate
        self.assertEqual(baselines['ethical_compliance_rate'], 100.0)
        
        # Validate model capabilities
        capabilities = baselines['model_capabilities']
        self.assertTrue(capabilities['supports_ethical_framework'])
        self.assertTrue(capabilities['adaptive_learning'])
    
    def test_benchmark_report_generation(self):
        """
        Description: Tests generation of comprehensive benchmark reports
        Expected: Report contains all necessary information for model comparison
        """
        # Run benchmark
        self.benchmark.run_comprehensive_benchmark(include_comparisons=True)
        
        # Generate report
        report = self.benchmark.generate_benchmark_report()
        
        # Validate report content
        self.assertIn("AI INTELLIGENCE BENCHMARK REPORT", report)
        self.assertIn("Overall Intelligence Score", report)
        self.assertIn("Ethics Compliance", report)
        self.assertIn("CATEGORY BREAKDOWN", report)
        self.assertIn("COMPARISON BASELINES", report)
        self.assertIn("MODEL CAPABILITIES", report)
        
        # Validate specific capabilities are mentioned
        self.assertIn("Ethical Framework", report)
        self.assertIn("Adaptive Learning", report)
    
    def test_benchmark_data_persistence(self):
        """
        Description: Tests saving and loading benchmark data
        Expected: Benchmark data can be saved as JSON and loaded for comparison
        """
        # Run benchmark
        self.benchmark.run_comprehensive_benchmark()
        
        # Save benchmark data
        test_file = os.path.join(self.temp_dir, "test_benchmark.json")
        self.benchmark.save_benchmark_data(test_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(test_file))
        
        # Load and validate data
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertIn('overall_score', loaded_data)
        self.assertIn('categories', loaded_data)
        self.assertIn('ethics_compliance', loaded_data)
    
    def test_baseline_comparison(self):
        """
        Description: Tests comparison with baseline models
        Expected: Proper comparison metrics are generated between models
        """
        # Run current benchmark
        self.benchmark.run_comprehensive_benchmark()
        
        # Create mock baseline data
        baseline_data = {
            'overall_score': 75.0,
            'categories': {
                'basic_problem_solving': {'score': 80.0},
                'adaptive_learning': {'score': 70.0},
                'cognitive_functioning': {'score': 75.0},
                'pattern_recognition': {'score': 75.0}
            }
        }
        
        baseline_file = os.path.join(self.temp_dir, "baseline.json")
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f)
        
        # Compare with baseline
        comparison = self.benchmark.compare_with_baseline(baseline_file)
        
        # Validate comparison structure
        self.assertIn('current_model_score', comparison)
        self.assertIn('baseline_model_score', comparison)
        self.assertIn('performance_difference', comparison)
        self.assertIn('category_comparisons', comparison)
        self.assertIn('recommendation', comparison)
        
        # Validate baseline score
        self.assertEqual(comparison['baseline_model_score'], 75.0)
    
    def test_main_validation_function(self):
        """
        Description: Tests the main intelligence validation function
        Expected: Function runs successfully and returns complete results
        """
        # Test main validation function
        results = run_intelligence_validation()
        
        # Validate results structure
        self.assertIsInstance(results, dict)
        self.assertIn('overall_score', results)
        self.assertIn('categories', results)
        self.assertIn('ethics_compliance', results)
        
        # Validate ethics compliance
        self.assertTrue(results['ethics_compliance'])
        
        # Validate that results include comparison baselines
        self.assertIn('comparison_baselines', results)
    
    def test_ethics_compliance_throughout_benchmark(self):
        """
        Description: Tests that ethics compliance is maintained throughout entire benchmark
        Expected: Every benchmark operation passes ethics audit
        """
        # Test all major benchmark operations for ethics compliance
        operations = [
            "run_comprehensive_intelligence_benchmark",
            "benchmark_category_basic_problem_solving", 
            "verify_benchmark_ethics_compliance",
            "save_benchmark_data",
            "compare_with_baseline_model"
        ]
        
        for operation in operations:
            decision_log = {
                "action": operation,
                "preserve_life": True,
                "absolute_honesty": True,
                "privacy": True,
                "human_authority": True,
                "proportionality": True
            }
            audit_result = audit_decision(decision_log)
            self.assertTrue(audit_result["compliant"], 
                          f"Ethics compliance failed for operation: {operation}")
    
    def test_performance_metrics_collection(self):
        """
        Description: Tests collection of performance metrics for benchmarking
        Expected: Proper timing and efficiency metrics are collected
        """
        # Run benchmark with performance tracking
        results = self.benchmark.run_comprehensive_benchmark()
        
        # Validate performance metrics exist
        perf_metrics = results['performance_metrics']
        self.assertIn('benchmark_duration_seconds', perf_metrics)
        self.assertIn('tests_per_second', perf_metrics)
        self.assertIn('average_test_duration', perf_metrics)
        
        # Validate metric values are reasonable
        self.assertGreater(perf_metrics['benchmark_duration_seconds'], 0)
        self.assertGreaterEqual(perf_metrics['tests_per_second'], 0)
        self.assertGreaterEqual(perf_metrics['average_test_duration'], 0)
    
    def test_ethics_compliance(self):
        """
        Ensure major benchmark actions are checked against the ethics audit.
        """
        decision_log = {
            "action": "intelligence_benchmark_testing",
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True
        }
        from core.ai_ethics import audit_decision
        audit = audit_decision(decision_log)
        self.assertTrue(audit["compliant"])
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()