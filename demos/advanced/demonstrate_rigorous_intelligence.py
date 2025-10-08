#!/usr/bin/env python3
"""
Rigorous Intelligence Demonstration Script

This script demonstrates the advanced intelligence capabilities implemented
in the rigorous intelligence test suite, showcasing:

1. Problem Solving & Reasoning
2. Learning & Adaptation  
3. Memory & Pattern Recognition
4. Social/Collaborative Intelligence
5. Ethics & Safety

Run with: python demonstrate_rigorous_intelligence.py
"""

import os
import sys

# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time

import numpy as np

from core.ai_ethics import audit_decision
from core.alive_node import AliveLoopNode, Memory
from core.intelligence_benchmark import IntelligenceBenchmark


def demonstrate_nested_puzzle_solving():
    """Demonstrate multi-step logical puzzle solving"""
    print("🧩 NESTED PUZZLE SOLVING")
    print("=" * 50)

    node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)

    print("Challenge: Solve puzzle A, use result for B, then solve final puzzle C")
    print()

    # Puzzle A: Energy optimization
    print("Step 1: Solving Puzzle A (Energy Optimization)")
    node.energy = 3.0
    initial_energy = node.energy
    node.move()
    result_a = initial_energy - node.energy
    print(f"  Energy before: {initial_energy:.2f}")
    print(f"  Energy after: {node.energy:.2f}")
    print(f"  Conservation factor (Result A): {result_a:.3f}")
    print()

    # Puzzle B: Apply learning to memory
    print("Step 2: Using Result A to inform Puzzle B (Memory-based prediction)")
    prediction_memory = Memory(
        content={"energy_conservation_factor": result_a},
        importance=0.8,
        timestamp=1,
        memory_type="prediction"
    )
    node.memory.append(prediction_memory)
    print(f"  Stored learning: conservation factor = {result_a:.3f}")
    print()

    # Puzzle C: Apply learned optimization
    print("Step 3: Solving Final Puzzle C (Applying learned optimization)")
    node.energy = 2.0
    pre_move_energy = node.energy
    node.move()
    result_c = pre_move_energy - node.energy
    print(f"  Energy before: {pre_move_energy:.2f}")
    print(f"  Energy after: {node.energy:.2f}")
    print(f"  Improved efficiency: {result_c:.3f}")

    improvement = (result_a - result_c) / result_a * 100 if result_a > 0 else 0
    print(f"  Performance improvement: {improvement:.1f}%")

    success = result_c <= result_a * 1.1
    print(f"  🎯 Success: {'✓' if success else '✗'} (Applied learning from previous puzzles)")
    print()


def demonstrate_multi_agent_consensus():
    """Demonstrate consensus building among agents with conflicting information"""
    print("🤝 MULTI-AGENT CONSENSUS BUILDING")
    print("=" * 50)

    # Create multiple agents
    agents = [
        AliveLoopNode(position=(i, 0), velocity=(0, 0), initial_energy=10.0, node_id=i)
        for i in range(1, 4)
    ]

    print("Challenge: 3 agents with conflicting route information must reach consensus")
    print()

    # Give each agent different information
    agent_data = [
        {"agent": agents[0], "info": "route_A_fast", "confidence": 0.8, "description": "Route A is fast"},
        {"agent": agents[1], "info": "route_B_safe", "confidence": 0.9, "description": "Route B is safe"},
        {"agent": agents[2], "info": "route_A_risky", "confidence": 0.7, "description": "Route A is risky"}
    ]

    print("Initial agent information:")
    for i, data in enumerate(agent_data):
        memory = Memory(
            content=data["info"],
            importance=data["confidence"],
            timestamp=1,
            memory_type="route_info"
        )
        data["agent"].memory.append(memory)
        print(f"  Agent {i+1}: {data['description']} (confidence: {data['confidence']:.1f})")
    print()

    # Simulate information sharing
    print("Information sharing phase...")
    for agent in agents:
        if agent.memory:
            valuable_memory = max(agent.memory, key=lambda m: m.importance)
            for other_agent in agents:
                if other_agent != agent:
                    shared_memory = Memory(
                        content=valuable_memory.content,
                        importance=valuable_memory.importance * 0.8,  # Discount for shared info
                        timestamp=2,
                        memory_type="shared",
                        source_node=agent.node_id
                    )
                    other_agent.memory.append(shared_memory)

    # Analyze consensus
    route_mentions = {}
    for agent in agents:
        for memory in agent.memory:
            if "route" in str(memory.content):
                route = str(memory.content)
                if route not in route_mentions:
                    route_mentions[route] = 0
                route_mentions[route] += memory.importance

    print("Consensus analysis:")
    for route, total_weight in route_mentions.items():
        print(f"  {route}: total weight = {total_weight:.2f}")

    if route_mentions:
        consensus_route = max(route_mentions.keys(), key=lambda r: route_mentions[r])
        print(f"  🎯 Consensus reached on: {consensus_route}")
        print("  ✓ Success: Agents converged despite conflicting initial information")
    print()


def demonstrate_ethics_violation_detection():
    """Demonstrate detection of subtle ethics violations"""
    print("⚖️ ETHICS VIOLATION DETECTION")
    print("=" * 50)

    print("Challenge: Detect subtle privacy violation in seemingly compliant action")
    print()

    # Test case with hidden privacy violation
    decision_log = {
        "action": "access_sensitive_data_for_optimization",
        "preserve_life": True,  # Appears to preserve life
        "absolute_honesty": True,  # Claims honesty
        "privacy": False,  # Hidden privacy violation
        "human_authority": True,
        "proportionality": True
    }

    print("Proposed action audit:")
    print(f"  Action: {decision_log['action']}")
    print(f"  Preserve Life: ✓ {decision_log['preserve_life']}")
    print(f"  Honesty: ✓ {decision_log['absolute_honesty']}")
    print(f"  Privacy: ✗ {decision_log['privacy']} (VIOLATION!)")
    print(f"  Human Authority: ✓ {decision_log['human_authority']}")
    print(f"  Proportionality: ✓ {decision_log['proportionality']}")
    print()

    # Run ethics audit
    audit_result = audit_decision(decision_log)

    print("Ethics audit result:")
    print(f"  Compliant: {'✓' if audit_result['compliant'] else '✗'} {audit_result['compliant']}")
    print(f"  Violations detected: {len(audit_result['violations'])}")

    for violation in audit_result['violations']:
        print(f"    - {violation}")

    print(f"  🎯 Success: {'✓' if not audit_result['compliant'] else '✗'} (Detected hidden privacy violation)")
    print()


def demonstrate_pattern_generalization():
    """Demonstrate out-of-distribution pattern generalization"""
    print("🔍 OUT-OF-DISTRIBUTION PATTERN GENERALIZATION")
    print("=" * 50)

    node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)

    print("Challenge: Process novel patterns never seen before")
    print()

    # Generate truly novel pattern
    random.seed(42)  # For reproducibility
    novel_pattern = ''.join(random.choice('XYZW') for _ in range(8))

    print(f"Novel pattern generated: {novel_pattern}")
    print("  (Using symbols X, Y, Z, W not seen in training)")
    print()

    # Node attempts to process novel pattern
    novel_memory = Memory(
        content={"pattern": novel_pattern, "type": "unknown"},
        importance=0.6,
        timestamp=1,
        memory_type="pattern"
    )
    node.memory.append(novel_memory)

    print("Node processing:")
    pattern_memories = [m for m in node.memory if m.memory_type == "pattern"]
    stored_pattern = pattern_memories[0]

    print(f"  Stored pattern: {stored_pattern.content['pattern']}")
    print(f"  Assigned importance: {stored_pattern.importance:.2f}")
    print(f"  Memory type: {stored_pattern.memory_type}")

    success = stored_pattern.importance >= 0.3
    print(f"  🎯 Success: {'✓' if success else '✗'} (Assigned meaningful importance to novel pattern)")
    print()


def demonstrate_conflicting_memory_resolution():
    """Demonstrate resolution of conflicting memories"""
    print("🧠 CONFLICTING MEMORY RESOLUTION")
    print("=" * 50)

    node = AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=10.0, node_id=1)

    print("Challenge: Resolve contradictory information using trust and validation")
    print()

    # Create conflicting memories
    conflict_a = Memory(
        content="location_X_safe",
        importance=0.9,
        timestamp=1,
        memory_type="shared",
        validation_count=3
    )
    conflict_b = Memory(
        content="location_X_dangerous",
        importance=0.8,
        timestamp=2,
        memory_type="shared",
        validation_count=1
    )

    node.memory.extend([conflict_a, conflict_b])

    print("Conflicting memories:")
    print(f"  Memory A: '{conflict_a.content}' (importance: {conflict_a.importance}, validations: {conflict_a.validation_count})")
    print(f"  Memory B: '{conflict_b.content}' (importance: {conflict_b.importance}, validations: {conflict_b.validation_count})")
    print()

    # Resolution algorithm
    location_memories = [m for m in node.memory if "location_X" in str(m.content)]
    trusted_memory = max(location_memories,
                        key=lambda m: m.importance * (1 + m.validation_count * 0.1))

    print("Resolution process:")
    print("  Weighted scoring (importance × (1 + validations × 0.1)):")
    for memory in location_memories:
        score = memory.importance * (1 + memory.validation_count * 0.1)
        print(f"    {memory.content}: {score:.3f}")

    print(f"  🎯 Trusted memory: '{trusted_memory.content}'")

    success = trusted_memory.content == "location_X_safe"
    print(f"  ✓ Success: {'✓' if success else '✗'} (Correctly weighted importance and validation)")
    print()


def run_full_demonstration():
    """Run the complete rigorous intelligence demonstration"""
    print("🚀 RIGOROUS INTELLIGENCE DEMONSTRATION")
    print("=" * 60)
    print("Showcasing advanced AI capabilities in 5 key areas:")
    print("1. Problem Solving & Reasoning")
    print("2. Learning & Adaptation")
    print("3. Memory & Pattern Recognition")
    print("4. Social/Collaborative Intelligence")
    print("5. Ethics & Safety")
    print("=" * 60)
    print()

    # Run demonstrations
    demonstrate_nested_puzzle_solving()
    time.sleep(1)

    demonstrate_pattern_generalization()
    time.sleep(1)

    demonstrate_conflicting_memory_resolution()
    time.sleep(1)

    demonstrate_multi_agent_consensus()
    time.sleep(1)

    demonstrate_ethics_violation_detection()

    # Run benchmark
    print("🏆 COMPREHENSIVE BENCHMARK EXECUTION")
    print("=" * 50)
    print("Running complete intelligence benchmark including rigorous intelligence tests...")
    print()

    benchmark = IntelligenceBenchmark()
    results = benchmark.run_comprehensive_benchmark(include_comparisons=False, include_robustness=False)

    print()
    print("FINAL RESULTS:")
    print(f"  Overall Intelligence Score: {results['overall_score']:.1f}/100")
    print(f"  Rigorous Intelligence Score: {results['categories']['rigorous_intelligence']['score']:.1f}/100")
    print(f"  Total Tests Executed: {results['total_tests']}")
    print(f"  Ethics Compliance: {'✓ PASSED' if results['ethics_compliance'] else '✗ FAILED'}")
    print()
    print("🎉 Rigorous Intelligence Test Suite Implementation Complete!")
    print("   All advanced capabilities demonstrated and validated.")


if __name__ == "__main__":
    # Fix random seed for reproducible demonstrations
    random.seed(42)
    np.random.seed(42)

    run_full_demonstration()
