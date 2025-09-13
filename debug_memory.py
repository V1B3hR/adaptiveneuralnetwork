import torch
import torch.nn as nn
from adaptiveneuralnetwork.applications.enhanced_memory_systems import (
    EnhancedMemoryConfig, DynamicPriorityBuffer, EventDrivenLearningSystem
)

class SimpleModel(nn.Module):
    def __init__(self, input_dim=64, output_dim=8):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.parameters())
        
    def forward(self, x):
        return self.fc(x)

# Test DynamicPriorityBuffer directly
config = EnhancedMemoryConfig(memory_size=100)
buffer = DynamicPriorityBuffer(config, feature_size=64)

print(f"Buffer initialized with feature_size=64")
print(f"Buffer memory features shape: {buffer.features.shape}")

# Test storing
features = torch.randn(8, 64)
labels = torch.randint(0, 8, (8,))
print(f"Storing features shape: {features.shape}")

try:
    buffer.store(features, labels, task_id=1)
    print("✓ Store successful")
except Exception as e:
    print(f"✗ Store failed: {e}")
    import traceback
    traceback.print_exc()

# Test EventDrivenLearningSystem
model = SimpleModel(input_dim=64, output_dim=8)
print(f"Model fc.in_features: {model.fc.in_features}")

try:
    system = EventDrivenLearningSystem(config, model)
    print("✓ EventDrivenLearningSystem initialized")
    
    system.process_experience(features, labels, task_id=1)
    print("✓ Experience processing successful")
    
except Exception as e:
    print(f"✗ EventDrivenLearningSystem failed: {e}")
    import traceback
    traceback.print_exc()