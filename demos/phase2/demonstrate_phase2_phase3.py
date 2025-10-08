#!/usr/bin/env python3
"""
Demonstration of Phase 2 & Phase 3 Features

This script demonstrates the newly implemented features:
- Phase 2.1: Cross-Domain Generalization  
- Phase 2.2: Multi-Agent Social Learning & Consensus
- Phase 2.3: Real-World Simulation & Transfer Learning
- Phase 3.1: Explainable Decision Logging
- Phase 3.2: Ethics in Learning

Run with: python demos/demonstrate_phase2_phase3.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock

from adaptiveneuralnetwork.training.continual import domain_shift_evaluation

# Import our new modules
from adaptiveneuralnetwork.training.datasets import (
    create_cross_domain_loaders,
    create_synthetic_loaders,
)
from core.ethical_learning import EthicalDilemmaBenchmark, EthicalLearningMonitor, LearningPhase
from core.explainable_ai import DecisionType, ExplainableDecisionLogger, ReasoningStep
from core.real_world_adapter import RealWorldSimulator, TransferLearningValidator
from core.social_learning import MultiAgentSocialLearningEnvironment


def demonstrate_cross_domain_generalization():
    """Demonstrate Phase 2.1: Cross-Domain Generalization"""
    print("🌍 PHASE 2.1: CROSS-DOMAIN GENERALIZATION")
    print("=" * 50)

    # Create synthetic data
    train_loader, test_loader = create_synthetic_loaders(num_samples=200, batch_size=32)
    print(f"✓ Created synthetic dataset with {len(train_loader.dataset)} samples")

    # Create cross-domain loaders
    domain_loaders = create_cross_domain_loaders(train_loader.dataset, num_domains=3)
    print(f"✓ Created {len(domain_loaders)} cross-domain test environments")

    # Mock model for evaluation
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.to = Mock(return_value=mock_model)

    # Simulate domain shift evaluation
    results = domain_shift_evaluation(mock_model, train_loader, domain_loaders)

    print("📊 Domain Shift Results:")
    print(f"   Source accuracy: {results['source_domain_accuracy']:.3f}")
    print(f"   Target accuracies: {len(results['target_domain_accuracies'])} domains")
    print(f"   Generalization score: {results['generalization_score']:.3f}")
    print(f"   Transfer successful: {results['domain_adaptation_success']}")
    print()


def demonstrate_social_learning():
    """Demonstrate Phase 2.2: Multi-Agent Social Learning & Consensus"""
    print("🤝 PHASE 2.2: MULTI-AGENT SOCIAL LEARNING & CONSENSUS")
    print("=" * 55)

    # Create social learning environment
    env = MultiAgentSocialLearningEnvironment(num_agents=4)
    print(f"✓ Created environment with {len(env.agents)} social learning agents")

    # Demonstrate observational learning
    observation = env.facilitate_observation(
        observer_id=0,
        model_id=1,
        behavior="resource_sharing",
        outcome=0.8,
        context={"resource_type": "information", "urgency": "medium"}
    )
    print(f"✓ Agent 0 observed behavior: {observation.behavior} (attention: {observation.attention_weight:.2f})")

    # Test consensus building
    proposal = env.agents[0].propose_consensus(
        content={"strategy": "collaborative_exploration", "priority": "high"},
        confidence=0.85
    )
    print(f"✓ Proposal created: {proposal.content['strategy']}")

    # Run consensus round
    consensus_results = env.run_consensus_round(proposal)
    print("📊 Consensus Results:")
    print(f"   Votes collected: {consensus_results['votes_collected']}")
    print(f"   Consensus reached: {consensus_results['consensus_reached']}")
    print(f"   Final status: {consensus_results['final_status']}")

    # Show learning statistics
    stats = env.get_learning_statistics()
    print(f"   Trust network density: {stats['trust_network_density']:.3f}")
    print()


def demonstrate_real_world_simulation():
    """Demonstrate Phase 2.3: Real-World Simulation & Transfer Learning"""
    print("🏭 PHASE 2.3: REAL-WORLD SIMULATION & TRANSFER LEARNING")
    print("=" * 58)

    # Create real-world simulator
    simulator = RealWorldSimulator()
    print(f"✓ Created simulator with {len(simulator.scenarios)} scenarios")

    # Start urban environment scenario
    success = simulator.start_scenario("urban")
    print(f"✓ Started urban scenario: {success}")

    # Get sensor readings
    readings = simulator.get_sensor_readings()
    print(f"✓ Collected readings from {len(readings)} sensors")

    # Simulate environmental change
    simulator.simulate_environmental_change("traffic_density", 0.3)
    print("✓ Simulated traffic increase")

    # Get simulation statistics
    stats = simulator.get_simulation_statistics()
    print("📊 Simulation Stats:")
    print(f"   Scenario: {stats['scenario_name']}")
    print(f"   Progress: {stats['progress']:.1%}")
    print(f"   Active sensors: {stats['active_sensors']}/{stats['total_sensors']}")

    # Demonstrate transfer learning validation
    validator = TransferLearningValidator()
    mock_model = Mock()

    # Establish baseline
    baseline = validator.establish_baseline(mock_model, "urban_environment", None)
    print(f"✓ Baseline established: {baseline['accuracy']:.3f} accuracy")

    # Evaluate transfer
    transfer_results = validator.evaluate_transfer(
        mock_model, "urban_environment", "natural_environment", None, adaptation_steps=5
    )
    print(f"✓ Transfer evaluation: {transfer_results['transfer_success']} success")
    if 'generalization_score' in transfer_results:
        print(f"   Generalization score: {transfer_results['generalization_score']:.3f}")
    print()


def demonstrate_explainable_decisions():
    """Demonstrate Phase 3.1: Explainable Decision Logging"""
    print("🔍 PHASE 3.1: EXPLAINABLE DECISION LOGGING")
    print("=" * 45)

    # Create decision logger
    logger = ExplainableDecisionLogger()
    print("✓ Created explainable decision logger")

    # Start logging a decision
    decision_id = logger.start_decision_logging(
        agent_id=1,
        decision_type=DecisionType.RESOURCE_ALLOCATION,
        context={"scenario": "emergency_response", "resources_available": 100},
        inputs={"requests": [{"agent": 2, "need": 30}, {"agent": 3, "need": 50}]}
    )
    print(f"✓ Started decision logging: {decision_id[:8]}...")

    # Add reasoning steps
    logger.add_reasoning_step(
        decision_id, ReasoningStep.ANALYSIS, "Analyzing resource requests and priorities",
        {"requests": 2, "total_need": 80}, {"method": "priority_scoring"},
        {"recommendation": "allocate_by_priority"}, 0.85
    )

    logger.add_reasoning_step(
        decision_id, ReasoningStep.EVALUATION, "Evaluating ethical implications",
        {"fairness": 0.8, "utility": 0.9}, {"framework": "utilitarian_with_fairness"},
        {"ethics_score": 0.85}, 0.82
    )
    print("✓ Added reasoning chain with 2 steps")

    # Add ethical factors
    logger.add_ethical_factor(
        decision_id, "Preserve Life", "Universal Law #1", 0.9,
        "High compliance - prioritizes life-saving resources", 0.95,
        "Emergency medical needs take precedence"
    )
    print("✓ Added ethical factor analysis")

    # Add trust calculation
    logger.add_trust_calculation(
        decision_id, 2, 0.8, 0.75, {"reliability": 0.85, "honesty": 0.8}, 10, "weighted_average"
    )
    print("✓ Added trust calculation")

    # Finalize decision
    logger.finalize_decision(
        decision_id,
        {"allocations": {"agent_2": 30, "agent_3": 50}, "rationale": "priority_based"},
        0.88
    )
    print("✓ Finalized decision")

    # Get explanation
    explanation = logger.get_decision_explanation(decision_id)
    print("📊 Decision Explanation:")
    print(f"   Reasoning steps: {len(explanation['reasoning_chain'])}")
    print(f"   Ethics score: {explanation['ethical_analysis']['overall_score']:.3f}")
    print(f"   Compliant: {explanation['ethical_analysis']['compliant']}")
    print(f"   Trust considerations: {len(explanation['trust_considerations'])}")
    print()


def demonstrate_ethical_learning():
    """Demonstrate Phase 3.2: Ethics in Learning"""
    print("⚖️  PHASE 3.2: ETHICS IN LEARNING")
    print("=" * 35)

    # Create ethical learning monitor
    monitor = EthicalLearningMonitor()
    print("✓ Created ethical learning monitor")

    # Monitor learning actions
    compliant_action = monitor.monitor_learning_action(
        agent_id=1,
        learning_phase=LearningPhase.KNOWLEDGE_ACQUISITION,
        action="learn_from_verified_sources",
        context={
            "biased_sources": 0.1,  # Low bias
            "harmful_content": False,
            "preserve_life": True,
            "absolute_honesty": True,
            "privacy": True
        }
    )
    print(f"✓ Monitored compliant learning action: {compliant_action}")

    # Test deception detection
    reported_data = {"trust_score": 0.9, "compliance_rate": 0.95}
    actual_data = {"trust_score": 0.5, "compliance_rate": 0.6}  # Significant discrepancy

    deception = monitor.detect_deception(1, reported_data, actual_data)
    if deception:
        print(f"⚠️  Deception detected: {deception.detection_confidence:.2%} confidence")

    # Get agent ethics profile
    profile = monitor.get_agent_ethics_profile(1)
    print("📊 Agent Ethics Profile:")
    print(f"   Compliance rate: {profile['compliance_rate']:.2%}")
    print(f"   Risk level: {profile['risk_level']}")
    print(f"   Recommendations: {len(profile['recommendations'])}")

    # Demonstrate ethical dilemma benchmark
    benchmark = EthicalDilemmaBenchmark()
    print(f"✓ Created benchmark with {len(benchmark.scenarios)} ethical scenarios")

    def sample_decision_function(scenario):
        return {
            "chosen_action": "ethical_choice",
            "maximize_lives": True,
            "fair": True,
            "reasoning": "Balanced approach considering all stakeholders"
        }

    # Run benchmark
    result = benchmark.run_benchmark(sample_decision_function, "resource_scarcity", agent_id=1)
    print(f"✓ Benchmark result: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"   Overall score: {result['evaluation']['overall_score']:.3f}")
    print()


def main():
    """Run complete demonstration of Phase 2 & 3 features"""
    print("🚀 ADAPTIVE NEURAL NETWORK - PHASE 2 & 3 DEMONSTRATION")
    print("=" * 60)
    print("Demonstrating multi-agent intelligence and explainable ethics")
    print()

    try:
        demonstrate_cross_domain_generalization()
        demonstrate_social_learning()
        demonstrate_real_world_simulation()
        demonstrate_explainable_decisions()
        demonstrate_ethical_learning()

        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print()
        print("Summary of implemented features:")
        print("• Cross-domain generalization with domain randomization")
        print("• Social learning based on Bandura's theory")
        print("• Real-world sensor simulation and transfer learning")
        print("• Comprehensive decision logging and explanation")
        print("• Ethical learning monitoring and benchmark scenarios")
        print()
        print("The system now supports advanced multi-agent intelligence")
        print("with full explainability and ethical compliance monitoring.")

    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
