#!/usr/bin/env python
"""
Quick demonstration of Phase 1: Adaptive Learning & Continual Improvement capabilities.

This script provides a concise overview of all four major Phase 1 components:
1. Self-supervised learning
2. Curriculum learning
3. Enhanced memory systems
4. Dynamic benchmarking
"""

import torch
import torch.nn as nn

from adaptiveneuralnetwork.applications import (  # Curriculum learning; Enhanced memory; Self-supervised learning
    DynamicBenchmarkConfig,
    EnhancedMemoryConfig,
    EventDrivenLearningSystem,
    SelfSupervisedConfig,
    SelfSupervisedLearningSystem,
    create_curriculum_system,
    create_dynamic_benchmark,
)


class SimpleModel(nn.Module):
    def __init__(self, input_dim=128, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.fc(x)

    def train_step(self, data, labels):
        pred = self(data)
        loss = nn.CrossEntropyLoss()(pred, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def get_feature_size(self):
        return self.input_dim


def main():
    print("🚀 Phase 1: Adaptive Learning & Continual Improvement - Quick Demo")
    print("=" * 70)

    # 1. Self-Supervised Learning
    print("\n1️⃣  Self-Supervised Learning")
    print("-" * 40)

    ss_config = SelfSupervisedConfig(prediction_horizon=5, context_window=20)
    ss_system = SelfSupervisedLearningSystem(ss_config, input_dim=100)

    # Generate temporal signals
    signals = torch.randn(4, 30, 100)  # batch, seq, features
    labels = torch.randint(0, 5, (4,))

    losses = ss_system.train_step(signals, labels)
    representations = ss_system.get_representations(signals)

    print(f"✅ Signal prediction loss: {losses['prediction_loss']:.3f}")
    print(f"✅ Contrastive loss: {losses['contrastive_loss']:.3f}")
    print(f"✅ Extracted representations shape: {representations.shape}")

    # 2. Curriculum Learning
    print("\n2️⃣  Curriculum Learning")
    print("-" * 40)

    model = SimpleModel(input_dim=128, output_dim=5)
    curriculum_system = create_curriculum_system(model, 128, 5)

    # Train a few episodes
    for episode in range(10):
        metrics = curriculum_system.train_episode()
        if episode % 3 == 0:
            print(
                f"Episode {episode}: difficulty={metrics['difficulty']:.2f}, "
                f"performance={metrics['performance']:.3f}"
            )

    print(f"✅ Final difficulty: {curriculum_system.difficulty_controller.get_difficulty():.2f}")

    # 3. Enhanced Memory Systems
    print("\n3️⃣  Enhanced Memory Systems")
    print("-" * 40)

    memory_config = EnhancedMemoryConfig(memory_size=100)
    memory_model = SimpleModel(input_dim=64, output_dim=8)
    memory_system = EventDrivenLearningSystem(memory_config, memory_model)

    # Process experiences
    for batch in range(5):
        features = torch.randn(8, 64)
        labels = torch.randint(0, 8, (8,))
        loss_values = torch.rand(8) * (2.0 - batch * 0.3)  # Decreasing loss over time

        memory_system.process_experience(features, labels, task_id=batch, loss_values=loss_values)

    stats = memory_system.get_memory_statistics()
    print(f"✅ Memory samples stored: {stats.get('stored_samples', 0)}")
    print(f"✅ Average importance: {stats.get('avg_importance', 0):.3f}")
    print(f"✅ Learning triggers: {stats.get('learning_triggers', 0)}")

    # 4. Dynamic Benchmarking
    print("\n4️⃣  Dynamic Benchmarking")
    print("-" * 40)

    benchmark_model = SimpleModel(input_dim=784, output_dim=10)
    benchmark_config = DynamicBenchmarkConfig(max_score_threshold=70.0)  # Lower for demo
    benchmark_system = create_dynamic_benchmark(benchmark_model, config=benchmark_config)

    # Run evaluations
    initial_difficulty = benchmark_system.current_difficulty

    for eval_step in range(10):
        results = benchmark_system.evaluate_model(include_adversarial=True)

        if eval_step % 3 == 0:
            print(
                f"Eval {eval_step}: standard={results['standard_score']:.1f}, "
                f"adversarial={results['adversarial_score']:.1f}, "
                f"difficulty={results['difficulty']:.2f}"
            )

    final_difficulty = benchmark_system.current_difficulty
    difficulty_changes = len(benchmark_system.difficulty_adjustments)

    print(f"✅ Difficulty changes: {difficulty_changes}")
    print(f"✅ Difficulty progression: {initial_difficulty:.2f} → {final_difficulty:.2f}")

    # Summary
    print("\n🎉 Phase 1 Demo Complete!")
    print("=" * 70)
    print("All four major components are working:")
    print("✅ Self-supervised learning with signal prediction and representation learning")
    print("✅ Curriculum learning with adaptive difficulty adjustment")
    print("✅ Enhanced memory systems with dynamic prioritization and event detection")
    print("✅ Dynamic benchmarking with adversarial testing and automatic scaling")
    print("\nThese capabilities provide a solid foundation for adaptive learning")
    print("and continual improvement in neural networks! 🧠🚀")


if __name__ == "__main__":
    main()
