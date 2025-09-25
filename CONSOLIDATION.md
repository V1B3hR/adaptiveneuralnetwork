# Unified Consolidation System

The Unified Consolidation System consolidates multiple memory consolidation mechanisms that were previously scattered across the codebase into a single, coherent framework.

## Overview

The consolidation system addresses memory stabilization and transfer across different time scales and neural processing phases:

- **Phase-Based Consolidation**: Memory strengthening during sleep phases
- **Synaptic Consolidation**: Weight protection against catastrophic forgetting (EWC)
- **Memory Consolidation**: Transfer from episodic to semantic memory

## Architecture

### Core Components

1. **`UnifiedConsolidationManager`**: Central orchestrator that coordinates all consolidation mechanisms
2. **`ConsolidationMechanism`**: Abstract base class for all consolidation types
3. **Specific Mechanisms**: Implementations for different consolidation strategies

### Consolidation Types

```python
class ConsolidationType(Enum):
    PHASE_BASED = "phase_based"      # Sleep-phase consolidation
    SYNAPTIC = "synaptic"           # EWC-based weight protection
    MEMORY = "memory"               # Episodic to semantic transfer
```

## Usage

### Basic Usage

```python
from adaptiveneuralnetwork.core.consolidation import create_default_consolidation_manager

# Create manager with all default mechanisms
model = MyNeuralNetwork()
manager = create_default_consolidation_manager(model=model, memory_dim=128)

# Run all consolidation mechanisms
results = manager.consolidate_all(
    node_state=node_state,
    phase_scheduler=phase_scheduler,
    data_loader=training_data,
    episodic_memories=episodic_mem,
    semantic_memories=semantic_mem
)
```

### Advanced Usage

```python
from adaptiveneuralnetwork.core.consolidation import (
    UnifiedConsolidationManager,
    PhaseBasedConsolidation,
    SynapticConsolidation
)

# Create custom manager
manager = UnifiedConsolidationManager()

# Add specific mechanisms
phase_consolidation = PhaseBasedConsolidation(
    memory_decay=0.05,
    stability_boost=1.5
)
manager.register_mechanism(phase_consolidation)

# Control activation
manager.deactivate_mechanism("phase_consolidation")
manager.activate_mechanism("phase_consolidation")
```

## Integration with Existing Systems

### Plugin System Integration

The consolidation system integrates seamlessly with the existing plugin system:

```python
from adaptiveneuralnetwork.core.plugin_system import ConsolidationPhase

# The ConsolidationPhase plugin now uses the unified system internally
plugin = ConsolidationPhase()
plugin.consolidation_manager  # Access to unified system
```

### Continual Learning Integration

Existing continual learning code maintains full backward compatibility:

```python
from adaptiveneuralnetwork.applications.continual_learning import SynapticConsolidation

# Legacy interface still works
consolidation = SynapticConsolidation(model)
loss = consolidation.consolidation_loss()  # Still works as before

# But uses unified system internally
consolidation.consolidation_manager  # Access to new unified system
```

## Key Benefits

1. **Unified Interface**: Single API for all consolidation mechanisms
2. **Coordinated Operation**: Mechanisms can work together effectively  
3. **Backward Compatibility**: Existing code continues to work unchanged
4. **Extensibility**: Easy to add new consolidation mechanisms
5. **Monitoring**: Comprehensive tracking of consolidation operations

## Mechanisms Details

### Phase-Based Consolidation

Operates during sleep phases to strengthen important memories:

- Reduces activity in sleep nodes (memory decay)
- Boosts energy for important nodes (high pre-sleep activity)
- Configurable decay and boost parameters

### Synaptic Consolidation (EWC)

Protects important network weights from catastrophic forgetting:

- Estimates Fisher Information Matrix for parameter importance
- Computes consolidation loss to regularize learning
- Maintains optimal parameters for previous tasks

### Memory Consolidation

Transfers important episodic memories to semantic storage:

- Filters memories by importance threshold
- Combines episodic and semantic context
- Uses neural network for consolidation transformation

## Configuration

Each mechanism supports configuration:

```python
# Phase-based consolidation config
phase_config = {
    "memory_decay": 0.1,           # Activity reduction in sleep
    "stability_boost": 1.2,        # Energy boost for important nodes
    "consolidation_strength": 0.8   # Overall strength
}

# Synaptic consolidation config  
synaptic_config = {
    "consolidation_strength": 1.0,  # EWC regularization strength
    "fisher_samples": 1000,         # Samples for Fisher estimation
    "importance_threshold": 1e-6    # Parameter importance threshold
}

# Memory consolidation config
memory_config = {
    "consolidation_threshold": 0.7, # Importance threshold for consolidation
    "memory_decay": 0.05,           # Episodic memory decay
    "semantic_boost": 1.1           # Semantic memory enhancement
}
```

## Monitoring and Analysis

The system provides comprehensive monitoring:

```python
# Get consolidation information
info = manager.get_consolidation_info()
print(f"Active mechanisms: {info['active_mechanisms']}")
print(f"Total strength: {info['total_consolidation_strength']}")

# Access consolidation history
history = manager.consolidation_history
for step in history:
    print(f"Step modifications: {step['summary']['total_modifications']}")
```

## Testing

Comprehensive tests ensure system reliability:

```bash
# Run consolidation tests
python -m pytest test_unified_consolidation.py -v

# Run integration tests
python demo_unified_consolidation.py
```

## Migration Guide

For existing code using separate consolidation mechanisms:

1. **Plugin System**: No changes needed - plugins automatically use unified system
2. **Continual Learning**: No changes needed - backward compatibility maintained  
3. **Custom Consolidation**: Migrate to new `ConsolidationMechanism` interface

## Future Extensions

The unified system supports easy extension:

1. **New Mechanisms**: Implement `ConsolidationMechanism` interface
2. **Custom Scheduling**: Add time-based consolidation triggers
3. **Multi-scale Coordination**: Cross-mechanism communication protocols
4. **Performance Optimization**: Batch processing and parallel execution

The unified consolidation system provides a robust foundation for memory consolidation across all time scales in adaptive neural networks.