#!/usr/bin/env python3
"""
Phase 4: Explainability & Advanced Analytics Demonstration

This script demonstrates the key features implemented in Phase 4:
1. Advanced Analytics Dashboard with real-time network topology visualization
2. Decision Transparency with attention mechanisms and counterfactual reasoning
3. Neural Architecture Search with multi-objective optimization
4. Automated feature engineering and self-debugging systems

Usage:
    python demo_phase4_showcase.py
"""

import time
import json
import logging
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Phase 4 modules
from core.advanced_analytics import AdvancedAnalyticsDashboard
from core.decision_transparency import DecisionTransparencySystem
from core.neural_architecture_search import (
    NeuralArchitectureSearchSystem, SearchSpace, MultiObjective,
    NetworkArchitecture
)


def print_section_header(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_json_summary(data: Dict[str, Any], title: str, max_items: int = 5) -> None:
    """Print a formatted summary of JSON data"""
    print(f"\n{title}:")
    print("-" * 40)
    
    if isinstance(data, dict):
        for i, (key, value) in enumerate(data.items()):
            if i >= max_items:
                print(f"  ... and {len(data) - max_items} more items")
                break
            
            if isinstance(value, (dict, list)):
                if isinstance(value, list):
                    print(f"  {key}: [{len(value)} items]")
                else:
                    print(f"  {key}: {{{len(value)} keys}}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  {str(data)[:200]}...")


def demo_advanced_analytics_dashboard():
    """Demonstrate Advanced Analytics Dashboard features"""
    print_section_header("4.1.1 ADVANCED ANALYTICS DASHBOARD")
    
    print("Creating Advanced Analytics Dashboard...")
    dashboard = AdvancedAnalyticsDashboard()
    
    print("\n1. Setting up Network Topology...")
    
    # Add network nodes representing different system components
    nodes_data = [
        ("central_ai", "decision_hub", (0, 0, 0), 0.9, 0.85, 0.4),
        ("trust_agent_1", "agent", (-2, 1, 0), 0.8, 0.9, 0.3),
        ("trust_agent_2", "agent", (2, 1, 0), 0.7, 0.8, 0.35),
        ("memory_bank", "memory_system", (0, -2, 0), 0.6, 0.95, 0.6),
        ("optimizer", "optimization_node", (0, 2, 0), 0.85, 0.7, 0.5),
        ("monitor", "monitoring_system", (1, -1, 1), 0.9, 0.8, 0.2)
    ]
    
    for node_id, node_type, position, activity, trust, energy in nodes_data:
        dashboard.add_network_node(
            node_id=node_id,
            node_type=node_type,
            position=position,
            activity_level=activity,
            trust_score=trust,
            energy_consumption=energy
        )
        print(f"  Added {node_type}: {node_id} (activity: {activity:.2f}, trust: {trust:.2f})")
    
    print("\n2. Setting up Network Connections...")
    
    # Add network edges representing connections
    connections_data = [
        ("trust_flow_1", "central_ai", "trust_agent_1", "trust", 0.9, 0.4, 25),
        ("trust_flow_2", "central_ai", "trust_agent_2", "trust", 0.8, 0.3, 30),
        ("memory_access", "central_ai", "memory_bank", "dependency", 0.95, 0.8, 15),
        ("optimization_link", "central_ai", "optimizer", "communication", 0.7, 0.6, 40),
        ("monitoring_link", "monitor", "central_ai", "feedback", 0.85, 0.2, 10),
        ("agent_collaboration", "trust_agent_1", "trust_agent_2", "trust", 0.6, 0.1, 50)
    ]
    
    for edge_id, source, target, conn_type, strength, flow_rate, latency in connections_data:
        dashboard.add_network_edge(
            edge_id=edge_id,
            source_node=source,
            target_node=target,
            connection_type=conn_type,
            strength=strength,
            data_flow_rate=flow_rate,
            latency=latency
        )
        print(f"  Added {conn_type} connection: {source} -> {target}")
    
    print("\n3. Adding Performance Metrics...")
    
    # Add performance metrics for monitoring
    metrics_data = [
        ("system_throughput", 0.75, 0.8, 0.9),
        ("response_time", 0.3, 0.5, 0.8),
        ("trust_coherence", 0.85, 0.7, 0.9),
        ("energy_efficiency", 0.7, 0.6, 0.8),
        ("decision_quality", 0.9, 0.8, 0.95)
    ]
    
    for metric_name, current, warning, critical in metrics_data:
        dashboard.add_performance_metric(metric_name, current, warning, critical)
        print(f"  Added metric: {metric_name} = {current:.2f} (warn: {warning:.2f}, crit: {critical:.2f})")
    
    print("\n4. Setting up Energy Distribution...")
    
    # Update energy distribution for nodes
    for node_id, _, _, _, _, energy in nodes_data:
        efficiency = np.random.uniform(0.6, 0.9)
        dashboard.update_energy_distribution(
            node_id=node_id,
            energy_consumption=energy,
            energy_efficiency=efficiency,
            optimal_range=(0.2, 0.7)
        )
        print(f"  Updated energy for {node_id}: consumption={energy:.2f}, efficiency={efficiency:.2f}")
    
    print("\n5. Generating Real-time Topology Visualization...")
    topology_viz = dashboard.get_real_time_topology_visualization()
    print_json_summary(topology_viz["statistics"], "Network Statistics")
    
    print(f"\nTopology Overview:")
    print(f"  Nodes: {len(topology_viz['nodes'])}")
    print(f"  Edges: {len(topology_viz['edges'])}")
    print(f"  Network Density: {topology_viz['statistics']['network_density']:.3f}")
    print(f"  Average Activity: {topology_viz['statistics']['average_activity']:.3f}")
    print(f"  Average Trust: {topology_viz['statistics']['average_trust']:.3f}")
    
    print("\n6. Performance Degradation Early Warning System...")
    warnings = dashboard.get_performance_degradation_warnings()
    
    if warnings:
        print(f"⚠️  Found {len(warnings)} performance warnings:")
        for warning in warnings:
            print(f"  - {warning['metric_name']}: {warning['warning_level']} level")
            print(f"    Current: {warning['current_value']:.3f}, Trend: {warning['trend']}")
            if warning['recommended_actions']:
                print(f"    Actions: {', '.join(warning['recommended_actions'][:2])}")
    else:
        print("✅ No performance warnings detected")
    
    print("\n7. Trust Network Flow Analysis...")
    trust_analysis = dashboard.analyze_trust_network_flows()
    print_json_summary(trust_analysis["flow_metrics"], "Trust Flow Metrics")
    
    print(f"\nTrust Network Overview:")
    print(f"  Trust Clusters: {len(trust_analysis['trust_clusters'])}")
    print(f"  High Trust Connections: {trust_analysis['flow_metrics']['high_trust_connections']}")
    print(f"  Active Flows: {trust_analysis['flow_metrics']['active_flows']}")
    
    print("\n8. Energy Distribution Heat Map...")
    heatmap = dashboard.get_energy_distribution_heatmap()
    print_json_summary(heatmap["statistics"], "Energy Statistics")
    
    print(f"\nEnergy Overview:")
    print(f"  Total Nodes: {heatmap['statistics']['total_nodes']}")
    print(f"  Average Heat Level: {heatmap['statistics']['average_heat_level']:.3f}")
    print(f"  Overheating Nodes: {heatmap['statistics']['overheating_nodes']}")
    print(f"  Efficient Nodes: {heatmap['statistics']['efficient_nodes']}")
    
    return dashboard


def demo_decision_transparency():
    """Demonstrate Decision Transparency System features"""
    print_section_header("4.1.2 DECISION TRANSPARENCY SYSTEM")
    
    print("Creating Decision Transparency System...")
    transparency_system = DecisionTransparencySystem()
    
    print("\n1. Attention Mechanisms for Decision Tracking...")
    
    # Simulate complex decision scenario
    decision_data = {
        "decision_id": "network_optimization_001",
        "decision": "optimize_neural_pathways",
        "confidence": 0.82,
        "inputs": {
            "network_load": 0.75,
            "trust_level": 0.88,
            "energy_consumption": 0.65,
            "memory_utilization": 0.70,
            "processing_time": 0.45,
            "user_priority": 0.90,
            "system_stability": 0.85,
            "risk_assessment": 0.25
        },
        "context": {
            "urgency": "high",
            "stakeholders": ["user_agent", "system_monitor", "energy_manager"],
            "constraints": ["energy_limit", "time_limit", "trust_requirement"]
        },
        "reasoning_chain": [
            {
                "type": "observation",
                "description": "System detected high network load requiring optimization",
                "confidence": 0.9,
                "processing_time": 0.1
            },
            {
                "type": "analysis", 
                "description": "Analyzed trust relationships and energy constraints",
                "confidence": 0.85,
                "processing_time": 0.3
            },
            {
                "type": "evaluation",
                "description": "Evaluated optimization strategies against constraints",
                "confidence": 0.8,
                "processing_time": 0.4
            },
            {
                "type": "decision",
                "description": "Selected pathway optimization with energy conservation",
                "confidence": 0.82,
                "processing_time": 0.2
            }
        ],
        "ethical_factors": [
            {
                "factor_name": "fairness",
                "compliance_score": 0.9,
                "assessment": "All agents receive equitable processing",
                "law_reference": "Fairness Principle #3"
            },
            {
                "factor_name": "transparency",
                "compliance_score": 0.85,
                "assessment": "Decision process is traceable and explainable",
                "law_reference": "Transparency Law #7"
            }
        ],
        "overall_ethics_score": 0.87,
        "processing_duration": 1.0
    }
    
    print("  Analyzing attention patterns...")
    attention_analysis = transparency_system.analyze_decision_transparency(
        decision_data["decision_id"], decision_data
    )
    
    attention_metrics = attention_analysis["attention_analysis"]
    print(f"  Attention Entropy: {attention_metrics['attention_map']['attention_entropy']:.3f}")
    print(f"  Overall Confidence: {attention_metrics['attention_map']['overall_confidence']:.3f}")
    print(f"  Focus Regions: {len(attention_metrics['attention_map']['focus_regions'])}")
    
    # Show dominant attention factors
    attention_weights = attention_metrics["attention_map"]["attention_weights"]
    top_factors = sorted(attention_weights, key=lambda x: x["weight"], reverse=True)[:3]
    print("  Top Attention Factors:")
    for factor in top_factors:
        print(f"    - {factor['factor_name']}: {factor['weight']:.3f}")
    
    print("\n2. Counterfactual Reasoning...")
    
    counterfactual_analysis = attention_analysis["counterfactual_analysis"]
    scenarios = counterfactual_analysis["scenarios"]
    
    print(f"  Generated {len(scenarios)} counterfactual scenarios:")
    for i, scenario in enumerate(scenarios[:3]):  # Show first 3 scenarios
        print(f"    Scenario {i+1}: {scenario['explanation']}")
        print(f"      Confidence: {scenario['confidence']:.3f}, Probability: {scenario['probability']:.3f}")
        print(f"      Impact Score: {np.mean(list(scenario['impact_assessment'].values())):.3f}")
    
    impact_analysis = counterfactual_analysis["impact_analysis"]
    print(f"\n  Impact Analysis:")
    print(f"    Average Confidence: {impact_analysis['average_confidence']:.3f}")
    print(f"    High Impact Scenarios: {len(impact_analysis['high_impact_scenarios'])}")
    print(f"    Low Impact Scenarios: {len(impact_analysis['low_impact_scenarios'])}")
    
    print("\n3. Natural Language Explanations...")
    
    nl_explanations = attention_analysis["natural_language_explanations"]
    
    for explanation_type in ["summary", "detailed", "ethical"]:
        explanation = nl_explanations[explanation_type]
        print(f"\n  {explanation_type.title()} Explanation:")
        print(f"    {explanation['explanation_text']}")
        print(f"    Readability Score: {explanation['readability_score']:.3f}")
    
    print("\n4. Transparency Metrics Overview...")
    
    transparency_metrics = attention_analysis["transparency_metrics"]
    print(f"  Overall Transparency Score: {transparency_metrics['overall_transparency_score']:.3f}")
    print(f"  Attention Entropy: {transparency_metrics['attention_entropy']:.3f}")
    print(f"  Explanation Readability: {transparency_metrics['explanation_readability']:.3f}")
    print(f"  Counterfactual Diversity: {transparency_metrics['counterfactual_diversity']}")
    
    print("\n5. Decision Audit Trail...")
    
    dashboard = transparency_system.get_transparency_dashboard([decision_data["decision_id"]])
    summary_metrics = dashboard["summary_metrics"]
    
    print(f"  Total Decisions Analyzed: {summary_metrics['total_decisions_analyzed']}")
    print(f"  Average Transparency Score: {summary_metrics.get('average_transparency_score', 0):.3f}")
    print(f"  High Transparency Decisions: {summary_metrics.get('decisions_with_high_transparency', 0)}")
    
    return transparency_system


def demo_neural_architecture_search():
    """Demonstrate Neural Architecture Search features"""
    print_section_header("4.2.1 NEURAL ARCHITECTURE SEARCH & MULTI-OBJECTIVE OPTIMIZATION")
    
    print("Setting up Neural Architecture Search System...")
    
    # Define search space
    search_space = SearchSpace(
        layer_types=["dense", "conv1d", "lstm", "attention"],
        layer_size_range=(32, 512),
        max_layers=8,
        activation_functions=["relu", "tanh", "sigmoid", "swish", "gelu"],
        optimizer_options=["adam", "sgd", "rmsprop", "adagrad"],
        regularization_options={
            "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
            "l2_regularization": [0.0, 0.001, 0.01, 0.1]
        },
        connection_patterns=["sequential", "skip", "residual"]
    )
    
    # Define multi-objective optimization goals
    objectives = [
        MultiObjective(
            objective_id="accuracy",
            name="Model Accuracy",
            weight=0.35,
            maximize=True,
            target_range=(0.85, 1.0),
            evaluator=lambda data: data.get("accuracy", 0.5)
        ),
        MultiObjective(
            objective_id="efficiency",
            name="Energy Efficiency",
            weight=0.25,
            maximize=True,
            target_range=(0.7, 1.0),
            evaluator=lambda data: data.get("energy_efficiency", 0.5)
        ),
        MultiObjective(
            objective_id="speed",
            name="Training Speed",
            weight=0.20,
            maximize=True,
            target_range=(0.6, 1.0),
            evaluator=lambda data: 1.0 - min(1.0, data.get("training_time", 100) / 200)
        ),
        MultiObjective(
            objective_id="interpretability",
            name="Model Interpretability",
            weight=0.20,
            maximize=True,
            target_range=(0.5, 1.0),
            evaluator=lambda data: 1.0 - min(1.0, data.get("complexity", 0.5))
        )
    ]
    
    nas_system = NeuralArchitectureSearchSystem(search_space, objectives)
    
    print(f"  Search Space: {len(search_space.layer_types)} layer types, {search_space.max_layers} max layers")
    print(f"  Objectives: {len(objectives)} optimization goals")
    
    print("\n1. Automated Feature Engineering...")
    
    # Demonstrate feature engineering
    sample_data = {
        "base_accuracy": 0.78,
        "model_size": 50000,
        "training_epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32,
        "validation_loss": 0.35,
        "convergence_time": [45, 52, 58, 61, 63, 64, 64.5],  # Loss over epochs
        "memory_usage": 0.65,
        "energy_per_epoch": 0.15
    }
    
    feature_engineer = nas_system.feature_engineer
    engineered_features = feature_engineer.engineer_features(sample_data)
    
    original_features = len(sample_data)
    engineered_count = len(engineered_features)
    new_features = engineered_count - original_features
    
    print(f"  Original Features: {original_features}")
    print(f"  Engineered Features: {engineered_count} (+{new_features} new)")
    print("  New Feature Types:")
    
    # Show types of new features
    new_feature_types = set()
    for key in engineered_features.keys():
        if key not in sample_data:
            if "_squared" in key:
                new_feature_types.add("Polynomial")
            elif "_x_" in key or "_div_" in key:
                new_feature_types.add("Interaction")
            elif "_mean" in key or "_std" in key:
                new_feature_types.add("Statistical")
            elif "_fft_" in key:
                new_feature_types.add("Frequency")
    
    for feature_type in new_feature_types:
        print(f"    - {feature_type} features")
    
    print("\n2. Self-Debugging System...")
    
    # Demonstrate self-debugging with a problematic architecture
    problematic_arch = NetworkArchitecture(
        architecture_id="problematic_demo",
        layers=[
            {"layer_id": 0, "type": "dense", "size": 100000},  # Too large
            {"layer_id": 1, "type": "dense", "size": -10},     # Invalid size
            {"layer_id": 2, "type": "dense", "size": 64}
        ],
        connections=[(0, 1), (1, 2), (0, 5)],  # Invalid connection
        activation_functions=["relu", "sigmoid"],  # Missing activation
        optimizer_config={"name": "sgd", "learning_rate": 100.0},  # Too high
        regularization={},
        performance_metrics={"accuracy": float('nan')},  # NaN value
        complexity_metrics={},
        energy_metrics={},
        timestamp=time.time()
    )
    
    debug_metrics = {
        "loss": 50.0,  # Very high loss
        "loss_history": [10.0, 10.1, 10.2, 10.15, 10.18],  # Not converging
        "memory_usage": 0.95  # High memory usage
    }
    
    debug_system = nas_system.debug_system
    corrected_arch, corrections = debug_system.detect_and_correct_errors(
        problematic_arch, debug_metrics, {"memory_usage": 0.95}
    )
    
    print(f"  Detected Issues: {len(corrections)}")
    for i, correction in enumerate(corrections):
        print(f"    {i+1}. {correction}")
    
    print(f"  Original Architecture ID: {problematic_arch.architecture_id}")
    print(f"  Corrected Architecture ID: {corrected_arch.architecture_id}")
    print(f"  Layer Size Corrections: {[layer['size'] for layer in corrected_arch.layers]}")
    print(f"  Learning Rate Corrected: {corrected_arch.optimizer_config['learning_rate']}")
    
    print("\n3. Multi-Objective Architecture Search...")
    
    def comprehensive_evaluation(architecture: NetworkArchitecture) -> Dict[str, Any]:
        """Comprehensive evaluation function for demonstration"""
        
        # Simulate realistic evaluation metrics
        num_params = sum(layer.get("size", 32) for layer in architecture.layers)
        complexity_score = min(1.0, num_params / 100000.0)
        
        # Simulate accuracy based on architecture properties
        layer_diversity = len(set(layer["type"] for layer in architecture.layers))
        base_accuracy = 0.6 + (layer_diversity * 0.08) + np.random.uniform(0, 0.2)
        base_accuracy = min(0.95, base_accuracy)
        
        # Energy efficiency inversely related to size but also depends on architecture
        energy_efficiency = max(0.3, 1.0 - (complexity_score * 0.6) + np.random.uniform(0, 0.2))
        
        # Training time increases with complexity
        training_time = 20 + (complexity_score * 80) + np.random.uniform(0, 20)
        
        # Add some engineered features
        engineered_data = feature_engineer.engineer_features({
            "accuracy": base_accuracy,
            "complexity": complexity_score,
            "energy_efficiency": energy_efficiency,
            "training_time": training_time,
            "num_layers": len(architecture.layers)
        })
        
        return engineered_data
    
    print("  Running architecture search with 12 evaluations...")
    
    start_time = time.time()
    optimization_result = nas_system.search_optimal_architecture(
        evaluation_function=comprehensive_evaluation,
        max_evaluations=12,
        population_size=6
    )
    search_time = time.time() - start_time
    
    print(f"  Search completed in {search_time:.2f} seconds")
    print(f"  Total Evaluations: {optimization_result.total_evaluations}")
    print(f"  Pareto Front Size: {len(optimization_result.pareto_front)}")
    print(f"  Improvement over Baseline: {optimization_result.improvement_over_baseline:.2f}%")
    
    # Show best architecture
    best_arch = optimization_result.best_architecture
    if best_arch:
        print(f"\n  Best Architecture: {best_arch.architecture_id}")
        print(f"    Layers: {len(best_arch.layers)}")
        print(f"    Layer Types: {[layer['type'] for layer in best_arch.layers]}")
        print(f"    Optimizer: {best_arch.optimizer_config['name']}")
        print(f"    Performance Metrics:")
        for metric, value in best_arch.performance_metrics.items():
            if isinstance(value, (int, float)):
                print(f"      {metric}: {value:.3f}")
    
    print("\n4. Search Analytics...")
    
    analytics = nas_system.get_search_analytics()
    
    search_summary = analytics["search_summary"]
    print(f"  Architectures Evaluated: {search_summary['total_architectures_evaluated']}")
    print(f"  Pareto Front Size: {search_summary['pareto_front_size']}")
    print(f"  Dominated Solutions: {search_summary['dominated_solutions']}")
    
    diversity_metrics = analytics["diversity_metrics"]
    print(f"  Architecture Diversity: {diversity_metrics['layer_count_diversity']:.3f}")
    print(f"  Unique Activation Functions: {diversity_metrics['unique_activation_functions']}")
    
    if "feature_engineering_impact" in analytics:
        fe_impact = analytics["feature_engineering_impact"]
        print(f"  Average Features Added: {fe_impact.get('average_features_added', 0):.1f}")
        print(f"  Transformation Success Rate: {fe_impact.get('transformation_success_rate', 0):.3f}")
    
    return nas_system, optimization_result


def demo_integration_showcase():
    """Demonstrate integration between all Phase 4 systems"""
    print_section_header("4.3 INTEGRATED PHASE 4 SHOWCASE")
    
    print("Demonstrating integrated workflow combining all Phase 4 features...")
    
    # 1. Set up systems
    print("\n1. Initializing integrated systems...")
    dashboard = AdvancedAnalyticsDashboard()
    transparency_system = DecisionTransparencySystem()
    
    # Simple NAS setup for integration demo
    search_space = SearchSpace(
        layer_types=["dense", "lstm"],
        layer_size_range=(32, 128),
        max_layers=4,
        activation_functions=["relu", "tanh"],
        optimizer_options=["adam"],
        regularization_options={"dropout": [0.0, 0.2]},
        connection_patterns=["sequential"]
    )
    
    objectives = [
        MultiObjective("performance", "Performance", 0.6, True, None, lambda x: x.get("performance", 0.5)),
        MultiObjective("transparency", "Transparency", 0.4, True, None, lambda x: x.get("transparency", 0.5))
    ]
    
    nas_system = NeuralArchitectureSearchSystem(search_space, objectives)
    
    print("  ✅ Advanced Analytics Dashboard")
    print("  ✅ Decision Transparency System")
    print("  ✅ Neural Architecture Search")
    
    # 2. Set up monitoring infrastructure
    print("\n2. Setting up integrated monitoring...")
    
    # Add network components for monitoring
    dashboard.add_network_node("nas_optimizer", "optimization_node", (0, 0, 0), activity_level=0.9)
    dashboard.add_network_node("transparency_engine", "decision_hub", (1, 0, 0), activity_level=0.8)
    dashboard.add_network_node("performance_monitor", "monitoring_system", (0.5, 1, 0), activity_level=0.7)
    
    dashboard.add_network_edge("opt_transparency", "nas_optimizer", "transparency_engine", "communication", 0.9)
    dashboard.add_network_edge("monitor_feedback", "performance_monitor", "nas_optimizer", "feedback", 0.8)
    
    # Add performance metrics
    dashboard.add_performance_metric("system_integration", 0.85, 0.8, 0.95)
    dashboard.add_performance_metric("transparency_quality", 0.82, 0.75, 0.9)
    
    print("  ✅ Network topology established")
    print("  ✅ Performance monitoring active")
    
    # 3. Run integrated optimization with transparency analysis
    print("\n3. Running transparency-aware architecture optimization...")
    
    def integrated_evaluation(architecture: NetworkArchitecture) -> Dict[str, Any]:
        """Evaluation function that integrates transparency analysis"""
        
        # Basic performance simulation
        num_layers = len(architecture.layers)
        layer_sizes = [layer.get("size", 32) for layer in architecture.layers]
        avg_size = np.mean(layer_sizes)
        
        base_performance = 0.7 + (num_layers * 0.05) - (avg_size / 1000.0)
        base_performance = max(0.4, min(0.95, base_performance))
        
        # Create decision context for transparency analysis
        decision_data = {
            "decision_id": f"arch_eval_{architecture.architecture_id}",
            "decision": "architecture_evaluation",
            "confidence": base_performance,
            "inputs": {
                "num_layers": num_layers,
                "avg_layer_size": avg_size,
                "architecture_complexity": num_layers * avg_size / 1000.0,
                "optimizer_type": architecture.optimizer_config.get("name", "adam")
            },
            "context": {"evaluation_phase": "integrated_demo"},
            "reasoning_chain": [
                {"type": "analysis", "description": f"Analyzed {num_layers}-layer architecture", "confidence": 0.8}
            ],
            "ethical_factors": [
                {"factor_name": "efficiency", "compliance_score": 0.8, "assessment": "resource-efficient"}
            ],
            "overall_ethics_score": 0.8,
            "processing_duration": 0.5
        }
        
        # Get transparency analysis
        transparency_analysis = transparency_system.analyze_decision_transparency(
            decision_data["decision_id"], decision_data
        )
        
        transparency_score = transparency_analysis["transparency_metrics"]["overall_transparency_score"]
        
        # Update dashboard with current evaluation
        dashboard.update_energy_distribution(
            "nas_optimizer", 
            energy_consumption=0.3 + (num_layers * 0.1),
            energy_efficiency=base_performance
        )
        
        return {
            "performance": base_performance,
            "transparency": transparency_score,
            "architecture_complexity": num_layers * avg_size / 1000.0,
            "energy_efficiency": base_performance * 0.8,
            "interpretability": transparency_score * 0.9
        }
    
    # Run integrated search
    result = nas_system.search_optimal_architecture(
        evaluation_function=integrated_evaluation,
        max_evaluations=8,
        population_size=4
    )
    
    print(f"  ✅ Optimization completed: {result.total_evaluations} evaluations")
    print(f"  ✅ Best performance improvement: {result.improvement_over_baseline:.2f}%")
    
    # 4. Generate comprehensive analytics
    print("\n4. Generating integrated analytics dashboard...")
    
    # Dashboard analytics
    comprehensive_dashboard = dashboard.get_comprehensive_dashboard()
    system_health = comprehensive_dashboard["system_health"]["overall_status"]
    
    # Transparency analytics
    transparency_dashboard = transparency_system.get_transparency_dashboard()
    avg_transparency = transparency_dashboard["summary_metrics"].get("average_transparency_score", 0)
    
    # NAS analytics
    nas_analytics = nas_system.get_search_analytics()
    architecture_diversity = nas_analytics["diversity_metrics"]["layer_count_diversity"]
    
    print(f"  System Health: {system_health}")
    print(f"  Average Transparency Score: {avg_transparency:.3f}")
    print(f"  Architecture Diversity: {architecture_diversity:.3f}")
    print(f"  Total Decisions Analyzed: {transparency_dashboard['summary_metrics']['total_decisions_analyzed']}")
    
    # 5. Show integrated insights
    print("\n5. Integrated Phase 4 Insights:")
    
    if result.best_architecture:
        best_arch = result.best_architecture
        print(f"  🏆 Best Architecture:")
        print(f"     ID: {best_arch.architecture_id}")
        print(f"     Layers: {len(best_arch.layers)}")
        print(f"     Performance Score: {best_arch.performance_metrics.get('performance', 0):.3f}")
        print(f"     Transparency Score: {best_arch.performance_metrics.get('transparency', 0):.3f}")
    
    print(f"  📊 Network Analytics:")
    network_stats = comprehensive_dashboard["network_topology"]["statistics"]
    print(f"     Monitored Nodes: {network_stats['node_count']}")
    print(f"     Network Density: {network_stats['network_density']:.3f}")
    print(f"     Total Energy: {network_stats['total_energy']:.3f}")
    
    print(f"  🔍 Transparency Insights:")
    print(f"     Decisions with High Transparency: {transparency_dashboard['summary_metrics'].get('decisions_with_high_transparency', 0)}")
    print(f"     Average Explanation Readability: {transparency_dashboard['summary_metrics'].get('average_readability', 0):.3f}")
    
    print(f"  🧠 Architecture Search Results:")
    print(f"     Pareto Front Size: {len(result.pareto_front)}")
    print(f"     Search Efficiency: {result.total_evaluations / result.optimization_time:.1f} evals/sec")
    
    return {
        "dashboard": dashboard,
        "transparency_system": transparency_system, 
        "nas_system": nas_system,
        "optimization_result": result
    }


def main():
    """Main demonstration function"""
    print("🚀 PHASE 4: EXPLAINABILITY & ADVANCED ANALYTICS DEMONSTRATION")
    print("================================================================")
    print("\nThis demonstration showcases the key capabilities implemented in Phase 4:")
    print("• Advanced Analytics Dashboard with real-time network visualization")
    print("• Decision Transparency with attention mechanisms & counterfactual reasoning")
    print("• Neural Architecture Search with multi-objective optimization")
    print("• Automated feature engineering and self-debugging systems")
    print("• Integrated workflow combining all Phase 4 features")
    
    try:
        # Run individual demonstrations
        print("\n" + "🔄 Starting Phase 4 Feature Demonstrations..." + "\n")
        
        dashboard = demo_advanced_analytics_dashboard()
        transparency_system = demo_decision_transparency()
        nas_system, optimization_result = demo_neural_architecture_search()
        
        # Run integration showcase
        integration_results = demo_integration_showcase()
        
        # Final summary
        print_section_header("PHASE 4 DEMONSTRATION COMPLETE")
        
        print("✅ Successfully demonstrated all Phase 4 features:")
        print("   • Advanced Analytics Dashboard - Real-time monitoring & visualization")
        print("   • Decision Transparency - Attention, counterfactuals, & natural language")
        print("   • Neural Architecture Search - Multi-objective optimization")
        print("   • Automated Systems - Feature engineering & self-debugging")
        print("   • Integrated Workflow - All systems working together")
        
        print(f"\n📈 Demonstration Statistics:")
        print(f"   • Network Nodes Created: {len(dashboard.network_nodes)}")
        print(f"   • Decisions Analyzed: {len(transparency_system.transparency_logs)}")
        print(f"   • Architectures Evaluated: {len(nas_system.search_history)}")
        print(f"   • Performance Improvements: {optimization_result.improvement_over_baseline:.2f}%")
        
        print(f"\n🎯 Phase 4 Goals Achieved:")
        print("   ✅ Real-time network topology visualization")
        print("   ✅ Performance degradation early warning systems")
        print("   ✅ Trust network flow analysis")
        print("   ✅ Energy distribution heat maps")
        print("   ✅ Attention mechanisms for decision tracking")
        print("   ✅ Counterfactual reasoning capabilities")
        print("   ✅ Natural language explanations")
        print("   ✅ Neural architecture search for topology optimization")
        print("   ✅ Multi-objective hyperparameter optimization")
        print("   ✅ Automated feature engineering")
        print("   ✅ Self-debugging and error correction")
        
        print("\n🔬 Ready for Production Integration!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)