#!/usr/bin/env python
"""
Quick test of Phase 1 implementations.
"""

import torch
import torch.nn as nn
from adaptiveneuralnetwork.applications import (
    SelfSupervisedConfig, SelfSupervisedLearningSystem,
    CurriculumConfig, create_curriculum_system,
    EnhancedMemoryConfig, EventDrivenLearningSystem,
    DynamicBenchmarkConfig, create_dynamic_benchmark
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
        return {'loss': loss.item()}
    
    def get_feature_size(self):
        return self.input_dim  # Return input dimension, not some other fixed value

def test_self_supervised():
    print("Testing Self-Supervised Learning...")
    config = SelfSupervisedConfig(prediction_horizon=5, context_window=20)
    system = SelfSupervisedLearningSystem(config, input_dim=50)
    
    # Test training
    signals = torch.randn(4, 30, 50)  # batch, seq, features
    labels = torch.randint(0, 3, (4,))
    losses = system.train_step(signals, labels)
    
    print(f"✓ Self-supervised training successful: {losses}")
    return True

def test_curriculum_learning():
    print("Testing Curriculum Learning...")
    model = SimpleModel(input_dim=100, output_dim=5)
    system = create_curriculum_system(model, 100, 5)
    
    # Test training episode
    metrics = system.train_episode()
    
    print(f"✓ Curriculum learning successful: difficulty={metrics['difficulty']:.2f}")
    return True

def test_enhanced_memory():
    print("Testing Enhanced Memory Systems...")
    config = EnhancedMemoryConfig(memory_size=100)
    model = SimpleModel(input_dim=64, output_dim=8)
    system = EventDrivenLearningSystem(config, model)
    
    # Test experience processing with correct feature size
    features = torch.randn(8, 64)  # Match model input_dim
    labels = torch.randint(0, 8, (8,))
    system.process_experience(features, labels, task_id=1)
    
    stats = system.get_memory_statistics()
    print(f"✓ Enhanced memory successful: stored={stats.get('stored_samples', 0)}")
    return True

def test_dynamic_benchmarking():
    print("Testing Dynamic Benchmarking...")
    model = SimpleModel(input_dim=784, output_dim=10)
    system = create_dynamic_benchmark(model)
    
    # Test evaluation
    results = system.evaluate_model(include_adversarial=True)
    
    print(f"✓ Dynamic benchmarking successful: score={results['standard_score']:.2f}")
    return True

def main():
    print("="*50)
    print("PHASE 1 QUICK FUNCTIONALITY TEST")
    print("="*50)
    
    tests = [
        test_self_supervised,
        test_curriculum_learning, 
        test_enhanced_memory,
        test_dynamic_benchmarking
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Phase 1 implementations working correctly!")
    else:
        print("⚠️  Some tests failed, check implementation")

if __name__ == "__main__":
    main()