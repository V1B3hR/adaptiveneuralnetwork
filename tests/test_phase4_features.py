"""
Tests for Phase 4: Explainability & Advanced Analytics Features

This module tests:
- Advanced Analytics Dashboard
- Decision Transparency System
- Neural Architecture Search
- Multi-objective Optimization
- Automated Feature Engineering
- Self-debugging System
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import Phase 4 components
from core.advanced_analytics import (
    AdvancedAnalyticsDashboard, NetworkTopologyNode, NetworkTopologyEdge,
    PerformanceMetric, EnergyDistribution
)

from core.decision_transparency import (
    DecisionTransparencySystem, MultiHeadAttentionTracker, CounterfactualReasoner,
    NaturalLanguageExplainer, AttentionMap, CounterfactualScenario, NaturalLanguageExplanation
)

from core.neural_architecture_search import (
    NeuralArchitectureSearchSystem, EvolutionaryGenerator, MultiObjectiveOptimizer,
    AutomatedFeatureEngineer, SelfDebuggingSystem, NetworkArchitecture, SearchSpace,
    MultiObjective
)


class TestAdvancedAnalyticsDashboard(unittest.TestCase):
    """Test Advanced Analytics Dashboard functionality"""
    
    def setUp(self):
        self.dashboard = AdvancedAnalyticsDashboard()
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        self.assertIsInstance(self.dashboard, AdvancedAnalyticsDashboard)
        self.assertEqual(len(self.dashboard.network_nodes), 0)
        self.assertEqual(len(self.dashboard.network_edges), 0)
        self.assertEqual(len(self.dashboard.performance_metrics), 0)
    
    def test_add_network_node(self):
        """Test adding network nodes"""
        self.dashboard.add_network_node(
            node_id="node1",
            node_type="agent",
            position=(1.0, 2.0, 3.0),
            connections=["node2"],
            activity_level=0.8,
            trust_score=0.9,
            energy_consumption=0.5
        )
        
        self.assertIn("node1", self.dashboard.network_nodes)
        node = self.dashboard.network_nodes["node1"]
        self.assertEqual(node.node_type, "agent")
        self.assertEqual(node.position, (1.0, 2.0, 3.0))
        self.assertEqual(node.activity_level, 0.8)
        self.assertEqual(node.trust_score, 0.9)
        self.assertEqual(node.energy_consumption, 0.5)
    
    def test_add_network_edge(self):
        """Test adding network edges"""
        # First add nodes
        self.dashboard.add_network_node("node1", "agent", (0, 0, 0))
        self.dashboard.add_network_node("node2", "agent", (1, 1, 1))
        
        self.dashboard.add_network_edge(
            edge_id="edge1",
            source_node="node1",
            target_node="node2",
            connection_type="trust",
            strength=0.7,
            data_flow_rate=0.3,
            latency=50.0
        )
        
        self.assertIn("edge1", self.dashboard.network_edges)
        edge = self.dashboard.network_edges["edge1"]
        self.assertEqual(edge.connection_type, "trust")
        self.assertEqual(edge.strength, 0.7)
        self.assertEqual(edge.data_flow_rate, 0.3)
        self.assertEqual(edge.latency, 50.0)
    
    def test_get_real_time_topology_visualization(self):
        """Test real-time topology visualization"""
        # Add some test data
        self.dashboard.add_network_node("node1", "agent", (0, 0, 0), activity_level=0.8)
        self.dashboard.add_network_node("node2", "decision_hub", (1, 1, 1), activity_level=0.6)
        self.dashboard.add_network_edge("edge1", "node1", "node2", "communication", strength=0.9)
        
        viz_data = self.dashboard.get_real_time_topology_visualization()
        
        self.assertIn("timestamp", viz_data)
        self.assertIn("nodes", viz_data)
        self.assertIn("edges", viz_data)
        self.assertIn("statistics", viz_data)
        
        self.assertEqual(len(viz_data["nodes"]), 2)
        self.assertEqual(len(viz_data["edges"]), 1)
        
        # Check node structure
        node = viz_data["nodes"][0]
        self.assertIn("id", node)
        self.assertIn("position", node)
        self.assertIn("activity_level", node)
        self.assertIn("trust_score", node)
    
    def test_add_performance_metric(self):
        """Test adding performance metrics"""
        self.dashboard.add_performance_metric(
            metric_name="cpu_usage",
            current_value=0.7,
            threshold_warning=0.8,
            threshold_critical=0.9
        )
        
        self.assertIn("cpu_usage", self.dashboard.performance_metrics)
        metric = self.dashboard.performance_metrics["cpu_usage"]
        self.assertEqual(metric.current_value, 0.7)
        self.assertEqual(metric.threshold_warning, 0.8)
        self.assertEqual(metric.threshold_critical, 0.9)
        self.assertEqual(len(metric.history), 1)
    
    def test_get_performance_degradation_warnings(self):
        """Test performance degradation warning system"""
        # Add metric that exceeds warning threshold
        self.dashboard.add_performance_metric("test_metric", 0.85, 0.8, 0.9)
        
        warnings = self.dashboard.get_performance_degradation_warnings()
        
        self.assertGreater(len(warnings), 0)
        warning = warnings[0]
        self.assertEqual(warning["metric_name"], "test_metric")
        self.assertEqual(warning["warning_level"], "warning")
        self.assertIn("recommended_actions", warning)
    
    def test_analyze_trust_network_flows(self):
        """Test trust network flow analysis"""
        # Create trust network
        self.dashboard.add_network_node("agent1", "agent", (0, 0, 0))
        self.dashboard.add_network_node("agent2", "agent", (1, 0, 0))
        self.dashboard.add_network_node("agent3", "agent", (0, 1, 0))
        
        self.dashboard.add_network_edge("trust1", "agent1", "agent2", "trust", strength=0.9, data_flow_rate=0.5)
        self.dashboard.add_network_edge("trust2", "agent2", "agent3", "trust", strength=0.7, data_flow_rate=0.3)
        
        analysis = self.dashboard.analyze_trust_network_flows()
        
        self.assertIn("timestamp", analysis)
        self.assertIn("trust_clusters", analysis)
        self.assertIn("flow_metrics", analysis)
        self.assertIn("visualization", analysis)
        
        # Check flow metrics
        metrics = analysis["flow_metrics"]
        self.assertIn("total_trust_connections", metrics)
        self.assertIn("average_trust_level", metrics)
        self.assertIn("trust_variance", metrics)
    
    def test_update_energy_distribution(self):
        """Test energy distribution updates"""
        self.dashboard.add_network_node("node1", "agent", (0, 0, 0))
        
        self.dashboard.update_energy_distribution(
            node_id="node1",
            energy_consumption=0.6,
            energy_efficiency=0.8,
            optimal_range=(0.3, 0.7)
        )
        
        self.assertIn("node1", self.dashboard.energy_distributions)
        energy_dist = self.dashboard.energy_distributions["node1"]
        self.assertEqual(energy_dist.energy_consumption, 0.6)
        self.assertEqual(energy_dist.energy_efficiency, 0.8)
        self.assertEqual(energy_dist.optimal_range, (0.3, 0.7))
    
    def test_get_energy_distribution_heatmap(self):
        """Test energy distribution heat map generation"""
        # Add nodes with energy data
        self.dashboard.add_network_node("node1", "agent", (0, 0, 0))
        self.dashboard.add_network_node("node2", "agent", (1, 1, 1))
        
        self.dashboard.update_energy_distribution("node1", 0.3, 0.9, (0.2, 0.8))
        self.dashboard.update_energy_distribution("node2", 0.9, 0.4, (0.2, 0.8))
        
        heatmap = self.dashboard.get_energy_distribution_heatmap()
        
        self.assertIn("timestamp", heatmap)
        self.assertIn("heatmap_data", heatmap)
        self.assertIn("statistics", heatmap)
        self.assertIn("color_scale", heatmap)
        
        self.assertEqual(len(heatmap["heatmap_data"]), 2)
        
        # Check heatmap point structure
        point = heatmap["heatmap_data"][0]
        self.assertIn("node_id", point)
        self.assertIn("position", point)
        self.assertIn("heat_level", point)
        self.assertIn("energy_consumption", point)
    
    def test_get_comprehensive_dashboard(self):
        """Test comprehensive dashboard generation"""
        # Add some test data
        self.dashboard.add_network_node("node1", "agent", (0, 0, 0), activity_level=0.7)
        self.dashboard.add_performance_metric("test_metric", 0.5, 0.8, 0.9)
        self.dashboard.update_energy_distribution("node1", 0.4, 0.8)
        
        dashboard = self.dashboard.get_comprehensive_dashboard()
        
        self.assertIn("timestamp", dashboard)
        self.assertIn("network_topology", dashboard)
        self.assertIn("performance_warnings", dashboard)
        self.assertIn("trust_analysis", dashboard)
        self.assertIn("energy_heatmap", dashboard)
        self.assertIn("system_health", dashboard)


class TestDecisionTransparencySystem(unittest.TestCase):
    """Test Decision Transparency System functionality"""
    
    def setUp(self):
        self.transparency_system = DecisionTransparencySystem()
    
    def test_transparency_system_initialization(self):
        """Test transparency system initialization"""
        self.assertIsInstance(self.transparency_system.attention_tracker, MultiHeadAttentionTracker)
        self.assertIsInstance(self.transparency_system.counterfactual_reasoner, CounterfactualReasoner)
        self.assertIsInstance(self.transparency_system.language_explainer, NaturalLanguageExplainer)
    
    def test_attention_mechanism(self):
        """Test attention mechanism functionality"""
        inputs = {
            "trust_level": 0.8,
            "energy_consumption": 0.6,
            "processing_time": 0.3,
            "complexity": 0.7
        }
        
        context = {
            "decision_id": "test_decision",
            "layer": "main"
        }
        
        attention_map = self.transparency_system.attention_tracker.calculate_attention(inputs, context)
        
        self.assertIsInstance(attention_map, AttentionMap)
        self.assertEqual(attention_map.decision_id, "test_decision")
        self.assertGreater(len(attention_map.attention_weights), 0)
        self.assertGreaterEqual(attention_map.overall_confidence, 0.0)
        self.assertLessEqual(attention_map.overall_confidence, 1.0)
        self.assertGreaterEqual(attention_map.attention_entropy, 0.0)
        self.assertLessEqual(attention_map.attention_entropy, 1.0)
    
    def test_attention_visualization(self):
        """Test attention visualization"""
        inputs = {"factor1": 0.5, "factor2": 0.8}
        context = {"decision_id": "viz_test"}
        
        attention_map = self.transparency_system.attention_tracker.calculate_attention(inputs, context)
        visualization = self.transparency_system.attention_tracker.visualize_attention(attention_map)
        
        self.assertIn("decision_id", visualization)
        self.assertIn("heatmap", visualization)
        self.assertIn("temporal_flow", visualization)
        self.assertIn("focus_regions", visualization)
        self.assertIn("summary", visualization)
    
    def test_counterfactual_reasoning(self):
        """Test counterfactual reasoning capabilities"""
        decision_id = "counterfactual_test"
        original_context = {"trust_level": 0.7, "resource_availability": 1.0}
        original_decision = "proceed"
        
        scenarios = self.transparency_system.counterfactual_reasoner.generate_counterfactual_scenarios(
            decision_id, original_context, original_decision, num_scenarios=3
        )
        
        self.assertEqual(len(scenarios), 3)
        
        for scenario in scenarios:
            self.assertIsInstance(scenario, CounterfactualScenario)
            self.assertEqual(scenario.original_decision_id, decision_id)
            self.assertIn("changed_factors", scenario.__dict__)
            self.assertGreaterEqual(scenario.confidence, 0.0)
            self.assertLessEqual(scenario.confidence, 1.0)
            self.assertGreaterEqual(scenario.probability, 0.0)
            self.assertLessEqual(scenario.probability, 1.0)
    
    def test_counterfactual_impact_analysis(self):
        """Test counterfactual impact analysis"""
        # Generate some test scenarios
        scenarios = []
        for i in range(5):
            scenario = CounterfactualScenario(
                scenario_id=f"test_scenario_{i}",
                original_decision_id="test_decision",
                changed_factors={"factor1": i * 0.2},
                predicted_outcome=f"outcome_{i}",
                confidence=0.5 + i * 0.1,
                probability=0.4 + i * 0.1,
                explanation=f"Test scenario {i}",
                impact_assessment={"impact1": i * 0.2, "impact2": (5-i) * 0.15},
                timestamp=time.time()
            )
            scenarios.append(scenario)
        
        impact_analysis = self.transparency_system.counterfactual_reasoner.analyze_counterfactual_impact(scenarios)
        
        self.assertIn("total_scenarios", impact_analysis)
        self.assertIn("average_confidence", impact_analysis)
        self.assertIn("average_probability", impact_analysis)
        self.assertIn("high_impact_scenarios", impact_analysis)
        self.assertIn("confidence_distribution", impact_analysis)
        
        self.assertEqual(impact_analysis["total_scenarios"], 5)
    
    def test_natural_language_explanation(self):
        """Test natural language explanation generation"""
        decision_data = {
            "decision_id": "nl_test",
            "decision": "approve_request",
            "confidence": 0.85,
            "reasoning_chain": [
                {"type": "analysis", "description": "Analyzed request parameters", "confidence": 0.9},
                {"type": "evaluation", "description": "Evaluated trust factors", "confidence": 0.8}
            ],
            "ethical_factors": [
                {"factor_name": "fairness", "compliance_score": 0.9, "assessment": "compliant"}
            ],
            "inputs": {"request_type": "urgent", "user_trust": 0.8},
            "overall_ethics_score": 0.85,
            "processing_duration": 2.5
        }
        
        explanation = self.transparency_system.language_explainer.generate_explanation(
            decision_data, "detailed"
        )
        
        self.assertIsInstance(explanation, NaturalLanguageExplanation)
        self.assertEqual(explanation.decision_id, "nl_test")
        self.assertEqual(explanation.explanation_type, "detailed")
        self.assertIsInstance(explanation.explanation_text, str)
        self.assertGreater(len(explanation.explanation_text), 0)
        self.assertGreater(len(explanation.key_factors), 0)
        self.assertGreaterEqual(explanation.readability_score, 0.0)
        self.assertLessEqual(explanation.readability_score, 1.0)
    
    def test_comprehensive_transparency_analysis(self):
        """Test comprehensive transparency analysis"""
        decision_data = {
            "decision_id": "comprehensive_test",
            "decision": "optimize_network",
            "confidence": 0.75,
            "inputs": {"network_load": 0.8, "energy_efficiency": 0.6},
            "context": {"priority": "high"},
            "reasoning_chain": [
                {"type": "observation", "description": "Network under stress", "confidence": 0.8}
            ],
            "ethical_factors": [
                {"factor_name": "efficiency", "compliance_score": 0.8, "assessment": "good"}
            ],
            "overall_ethics_score": 0.8,
            "processing_duration": 1.5
        }
        
        analysis = self.transparency_system.analyze_decision_transparency("comprehensive_test", decision_data)
        
        self.assertIn("decision_id", analysis)
        self.assertIn("attention_analysis", analysis)
        self.assertIn("counterfactual_analysis", analysis)
        self.assertIn("natural_language_explanations", analysis)
        self.assertIn("transparency_metrics", analysis)
        
        # Check attention analysis
        attention_analysis = analysis["attention_analysis"]
        self.assertIn("attention_map", attention_analysis)
        self.assertIn("visualization", attention_analysis)
        
        # Check counterfactual analysis
        counterfactual_analysis = analysis["counterfactual_analysis"]
        self.assertIn("scenarios", counterfactual_analysis)
        self.assertIn("impact_analysis", counterfactual_analysis)
        
        # Check natural language explanations
        nl_explanations = analysis["natural_language_explanations"]
        self.assertIn("summary", nl_explanations)
        self.assertIn("detailed", nl_explanations)
        self.assertIn("technical", nl_explanations)
        self.assertIn("ethical", nl_explanations)
    
    def test_transparency_dashboard(self):
        """Test transparency dashboard generation"""
        # First analyze a decision to populate data
        decision_data = {
            "decision_id": "dashboard_test",
            "decision": "test_decision",
            "confidence": 0.7,
            "inputs": {"test_input": 0.5},
            "context": {"test": True}
        }
        
        self.transparency_system.analyze_decision_transparency("dashboard_test", decision_data)
        
        dashboard = self.transparency_system.get_transparency_dashboard(["dashboard_test"])
        
        self.assertIn("timestamp", dashboard)
        self.assertIn("summary_metrics", dashboard)
        self.assertIn("recent_analyses", dashboard)
        self.assertIn("transparency_trends", dashboard)


class TestNeuralArchitectureSearch(unittest.TestCase):
    """Test Neural Architecture Search functionality"""
    
    def setUp(self):
        # Create search space
        self.search_space = SearchSpace(
            layer_types=["dense", "conv1d", "lstm"],
            layer_size_range=(16, 256),
            max_layers=10,
            activation_functions=["relu", "tanh", "sigmoid"],
            optimizer_options=["adam", "sgd", "rmsprop"],
            regularization_options={"dropout": [0.0, 0.1, 0.2, 0.3]},
            connection_patterns=["sequential", "skip"]
        )
        
        # Create objectives
        self.objectives = [
            MultiObjective(
                objective_id="accuracy",
                name="Accuracy",
                weight=0.4,
                maximize=True,
                target_range=(0.8, 1.0),
                evaluator=lambda data: data.get("accuracy", 0.5)
            ),
            MultiObjective(
                objective_id="efficiency",
                name="Energy Efficiency",
                weight=0.3,
                maximize=True,
                target_range=(0.7, 1.0),
                evaluator=lambda data: data.get("efficiency", 0.5)
            ),
            MultiObjective(
                objective_id="complexity",
                name="Model Complexity",
                weight=0.3,
                maximize=False,
                target_range=(0.0, 0.5),
                evaluator=lambda data: data.get("complexity", 0.5)
            )
        ]
        
        self.nas_system = NeuralArchitectureSearchSystem(self.search_space, self.objectives)
    
    def test_nas_system_initialization(self):
        """Test NAS system initialization"""
        self.assertEqual(self.nas_system.search_space, self.search_space)
        self.assertEqual(len(self.nas_system.objectives), 3)
        self.assertIsInstance(self.nas_system.architecture_generator, EvolutionaryGenerator)
        self.assertIsInstance(self.nas_system.multi_objective_optimizer, MultiObjectiveOptimizer)
    
    def test_evolutionary_generator(self):
        """Test evolutionary architecture generator"""
        generator = EvolutionaryGenerator()
        
        # Generate initial architecture
        arch = generator.generate_architecture(self.search_space)
        
        self.assertIsInstance(arch, NetworkArchitecture)
        self.assertGreater(len(arch.layers), 0)
        self.assertLessEqual(len(arch.layers), self.search_space.max_layers)
        self.assertEqual(len(arch.activation_functions), len(arch.layers))
        
        # Test mutation
        mutated_arch = generator.mutate_architecture(arch, mutation_rate=0.5)
        
        self.assertIsInstance(mutated_arch, NetworkArchitecture)
        self.assertNotEqual(mutated_arch.architecture_id, arch.architecture_id)
    
    def test_multi_objective_optimizer(self):
        """Test multi-objective optimizer"""
        optimizer = MultiObjectiveOptimizer(self.objectives)
        
        # Create test architecture
        arch = NetworkArchitecture(
            architecture_id="test_arch",
            layers=[{"layer_id": 0, "type": "dense", "size": 64}],
            connections=[(0, 1)],
            activation_functions=["relu"],
            optimizer_config={"name": "adam", "learning_rate": 0.001},
            regularization={},
            performance_metrics={},
            complexity_metrics={},
            energy_metrics={},
            timestamp=time.time()
        )
        
        # Evaluate architecture
        evaluation_data = {"accuracy": 0.85, "efficiency": 0.7, "complexity": 0.3}
        scores = optimizer.evaluate_architecture(arch, evaluation_data)
        
        self.assertIn("accuracy", scores)
        self.assertIn("efficiency", scores)
        self.assertIn("complexity", scores)
        
        # Update architecture with scores
        arch.performance_metrics.update(scores)
        
        # Update Pareto front
        added_to_pareto = optimizer.update_pareto_front(arch)
        self.assertTrue(added_to_pareto)
        self.assertEqual(len(optimizer.pareto_front), 1)
    
    def test_automated_feature_engineer(self):
        """Test automated feature engineering"""
        engineer = AutomatedFeatureEngineer()
        
        # Test data with various types
        data = {
            "numerical_feature": 0.75,
            "boolean_feature": True,
            "list_feature": [1, 2, 3, 4, 5],
            "string_feature": "test_string",
            "time_feature": time.time()
        }
        
        engineered_features = engineer.engineer_features(data)
        
        # Should include original features
        for key in data.keys():
            self.assertIn(key, engineered_features)
        
        # Should include engineered features
        self.assertGreater(len(engineered_features), len(data))
        
        # Check for specific engineered features
        self.assertIn("numerical_feature_squared", engineered_features)
        self.assertIn("mean_all_features", engineered_features)
    
    def test_self_debugging_system(self):
        """Test self-debugging system"""
        debug_system = SelfDebuggingSystem()
        
        # Create problematic architecture (very large)
        problematic_arch = NetworkArchitecture(
            architecture_id="problematic_arch",
            layers=[
                {"layer_id": 0, "type": "dense", "size": 50000},  # Very large layer
                {"layer_id": 1, "type": "dense", "size": 50000}
            ],
            connections=[(0, 1)],
            activation_functions=["relu", "relu"],
            optimizer_config={"name": "sgd", "learning_rate": 10.0},  # Very high learning rate
            regularization={},
            performance_metrics={"accuracy": float('nan')},  # NaN value
            complexity_metrics={},
            energy_metrics={},
            timestamp=time.time()
        )
        
        training_metrics = {
            "loss": 100.0,  # Very high loss
            "loss_history": [10.0, 10.1, 10.2, 10.3, 10.4]  # Not decreasing
        }
        
        corrected_arch, corrections = debug_system.detect_and_correct_errors(
            problematic_arch, training_metrics, {"memory_usage": 0.95}
        )
        
        self.assertIsInstance(corrected_arch, NetworkArchitecture)
        self.assertGreater(len(corrections), 0)
        
        # Check that corrections were applied
        self.assertNotEqual(corrected_arch.architecture_id, problematic_arch.architecture_id)
    
    def test_nas_search_optimization(self):
        """Test full NAS optimization process"""
        def mock_evaluation_function(architecture: NetworkArchitecture) -> Dict[str, Any]:
            """Mock evaluation function that returns random but reasonable values"""
            import random
            
            # Simulate evaluation based on architecture properties
            num_params = sum(layer.get("size", 32) for layer in architecture.layers)
            complexity = min(1.0, num_params / 10000.0)  # Normalize complexity
            
            return {
                "accuracy": random.uniform(0.5, 0.95),
                "efficiency": random.uniform(0.3, 0.9),
                "complexity": complexity,
                "loss": random.uniform(0.1, 2.0),
                "training_time": random.uniform(10, 100)
            }
        
        # Run search with limited evaluations for testing
        result = self.nas_system.search_optimal_architecture(
            evaluation_function=mock_evaluation_function,
            max_evaluations=10,
            population_size=5
        )
        
        self.assertIsNotNone(result.best_architecture)
        self.assertGreaterEqual(len(result.pareto_front), 1)
        self.assertEqual(result.total_evaluations, 10)
        self.assertGreater(result.optimization_time, 0)
        
        # Check that best architecture has performance metrics
        best_arch = result.best_architecture
        self.assertGreater(len(best_arch.performance_metrics), 0)
    
    def test_search_analytics(self):
        """Test search analytics generation"""
        # First run a small search to generate data
        def simple_eval(arch):
            return {"accuracy": 0.8, "efficiency": 0.7, "complexity": 0.3}
        
        # Add some architectures to history
        for i in range(3):
            arch = self.nas_system.architecture_generator.generate_architecture(self.search_space)
            arch.performance_metrics = {"accuracy": 0.7 + i * 0.1, "efficiency": 0.6 + i * 0.1}
            self.nas_system.search_history.append(arch)
        
        analytics = self.nas_system.get_search_analytics()
        
        self.assertIn("search_summary", analytics)
        self.assertIn("performance_trends", analytics)
        self.assertIn("diversity_metrics", analytics)
        
        # Check search summary
        summary = analytics["search_summary"]
        self.assertEqual(summary["total_architectures_evaluated"], 3)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple Phase 4 features"""
    
    def setUp(self):
        self.dashboard = AdvancedAnalyticsDashboard()
        self.transparency_system = DecisionTransparencySystem()
        
        # Set up a small NAS system
        search_space = SearchSpace(
            layer_types=["dense"],
            layer_size_range=(16, 64),
            max_layers=5,
            activation_functions=["relu", "tanh"],
            optimizer_options=["adam"],
            regularization_options={"dropout": [0.0, 0.1]},
            connection_patterns=["sequential"]
        )
        
        objectives = [
            MultiObjective("accuracy", "Accuracy", 0.6, True, None, lambda x: x.get("accuracy", 0.5)),
            MultiObjective("efficiency", "Efficiency", 0.4, True, None, lambda x: x.get("efficiency", 0.5))
        ]
        
        self.nas_system = NeuralArchitectureSearchSystem(search_space, objectives)
    
    def test_integrated_analytics_and_transparency(self):
        """Test integration between analytics dashboard and transparency system"""
        # Set up network topology
        self.dashboard.add_network_node("decision_agent", "agent", (0, 0, 0), activity_level=0.8)
        self.dashboard.add_network_node("analysis_hub", "decision_hub", (1, 1, 1), activity_level=0.6)
        self.dashboard.add_network_edge("comm_link", "decision_agent", "analysis_hub", "communication", strength=0.9)
        
        # Add performance metrics
        self.dashboard.add_performance_metric("decision_quality", 0.85, 0.7, 0.9)
        
        # Analyze a decision with transparency system
        decision_data = {
            "decision_id": "integrated_test",
            "decision": "network_optimization",
            "confidence": 0.8,
            "inputs": {"network_topology": "optimized", "performance_metric": 0.85},
            "context": {"integration_test": True}
        }
        
        transparency_analysis = self.transparency_system.analyze_decision_transparency(
            "integrated_test", decision_data
        )
        
        # Get comprehensive dashboard
        dashboard = self.dashboard.get_comprehensive_dashboard()
        
        # Verify integration
        self.assertIsNotNone(transparency_analysis)
        self.assertIsNotNone(dashboard)
        
        # Check that both systems have data
        self.assertGreater(len(self.dashboard.network_nodes), 0)
        self.assertGreater(len(self.transparency_system.transparency_logs), 0)
    
    def test_nas_with_transparency_feedback(self):
        """Test NAS system using transparency feedback for evaluation"""
        def transparency_aware_evaluation(architecture: NetworkArchitecture) -> Dict[str, Any]:
            """Evaluation function that considers transparency metrics"""
            
            # Simulate decision for this architecture
            decision_data = {
                "decision_id": f"nas_eval_{architecture.architecture_id}",
                "decision": "architecture_evaluation",
                "confidence": 0.7,
                "inputs": {
                    "layer_count": len(architecture.layers),
                    "complexity": sum(layer.get("size", 32) for layer in architecture.layers) / 1000.0
                },
                "context": {"architecture_evaluation": True}
            }
            
            # Get transparency analysis
            transparency_analysis = self.transparency_system.analyze_decision_transparency(
                decision_data["decision_id"], decision_data
            )
            
            # Extract transparency metrics
            transparency_score = transparency_analysis["transparency_metrics"]["overall_transparency_score"]
            attention_entropy = transparency_analysis["transparency_metrics"]["attention_entropy"]
            
            # Return evaluation including transparency metrics
            return {
                "accuracy": np.random.uniform(0.6, 0.9),
                "efficiency": 1.0 - (len(architecture.layers) / 10.0),  # Simpler is more efficient
                "transparency_score": transparency_score,
                "attention_clarity": 1.0 - attention_entropy,
                "complexity": len(architecture.layers) / 10.0
            }
        
        # Run NAS with transparency-aware evaluation
        result = self.nas_system.search_optimal_architecture(
            evaluation_function=transparency_aware_evaluation,
            max_evaluations=5,
            population_size=3
        )
        
        # Verify that transparency metrics are included
        self.assertIsNotNone(result.best_architecture)
        self.assertGreater(len(self.transparency_system.transparency_logs), 0)
        
        # Check that some architectures have transparency-related metrics
        transparency_aware_archs = [
            arch for arch in self.nas_system.search_history 
            if "transparency_score" in arch.performance_metrics
        ]
        self.assertGreater(len(transparency_aware_archs), 0)
    
    def test_comprehensive_phase4_workflow(self):
        """Test comprehensive Phase 4 workflow"""
        # 1. Set up analytics dashboard with network topology
        self.dashboard.add_network_node("main_agent", "agent", (0, 0, 0), activity_level=0.9)
        self.dashboard.add_network_node("optimizer", "decision_hub", (1, 0, 0), activity_level=0.7)
        self.dashboard.add_network_edge("optimization_link", "main_agent", "optimizer", "dependency", strength=0.8)
        
        # 2. Add performance monitoring
        self.dashboard.add_performance_metric("system_efficiency", 0.75, 0.8, 0.9)
        self.dashboard.update_energy_distribution("main_agent", 0.6, 0.8)
        
        # 3. Run architecture search with comprehensive evaluation
        def comprehensive_evaluation(architecture: NetworkArchitecture) -> Dict[str, Any]:
            # Simulate comprehensive evaluation including all Phase 4 metrics
            
            # Basic performance metrics
            base_accuracy = np.random.uniform(0.6, 0.9)
            base_efficiency = np.random.uniform(0.5, 0.9)
            
            # Architecture complexity
            num_params = sum(layer.get("size", 32) for layer in architecture.layers)
            complexity = num_params / 10000.0
            
            # Decision transparency analysis
            decision_data = {
                "decision_id": f"comprehensive_{architecture.architecture_id}",
                "decision": "architecture_validation",
                "confidence": base_accuracy,
                "inputs": {
                    "accuracy": base_accuracy,
                    "efficiency": base_efficiency,
                    "complexity": complexity
                },
                "context": {"comprehensive_evaluation": True}
            }
            
            transparency_analysis = self.transparency_system.analyze_decision_transparency(
                decision_data["decision_id"], decision_data
            )
            
            return {
                "accuracy": base_accuracy,
                "efficiency": base_efficiency,
                "complexity": complexity,
                "transparency_score": transparency_analysis["transparency_metrics"]["overall_transparency_score"],
                "readability": transparency_analysis["transparency_metrics"]["explanation_readability"],
                "energy_efficiency": 1.0 - (complexity * 0.5)  # Lower complexity = higher energy efficiency
            }
        
        # 4. Run optimization
        nas_result = self.nas_system.search_optimal_architecture(
            evaluation_function=comprehensive_evaluation,
            max_evaluations=8,
            population_size=4
        )
        
        # 5. Generate comprehensive analytics
        dashboard = self.dashboard.get_comprehensive_dashboard()
        transparency_dashboard = self.transparency_system.get_transparency_dashboard()
        nas_analytics = self.nas_system.get_search_analytics()
        
        # 6. Verify comprehensive workflow results
        self.assertIsNotNone(nas_result.best_architecture)
        self.assertIn("network_topology", dashboard)
        self.assertIn("energy_heatmap", dashboard)
        self.assertIn("summary_metrics", transparency_dashboard)
        self.assertIn("search_summary", nas_analytics)
        
        # Check integration of metrics
        best_arch = nas_result.best_architecture
        self.assertIn("transparency_score", best_arch.performance_metrics)
        self.assertIn("energy_efficiency", best_arch.performance_metrics)
        
        # Verify data flow between systems
        self.assertGreater(len(self.dashboard.network_nodes), 0)
        self.assertGreater(len(self.transparency_system.transparency_logs), 0)
        self.assertGreater(len(self.nas_system.search_history), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)