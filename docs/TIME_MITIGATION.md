# Time Mitigation Solution

## Problem Statement

The original issue was: **"Clock-time--> real. Mitigate to real world time."**

The system had mixed two different types of time:
1. **Simulation time** - Discrete integer steps used for game logic (circadian cycles, phases)
2. **Real time** - Wall-clock time used for timestamps and performance measurement

This mixing created several problems:
- Inconsistent time representation across the system
- Difficulty running simulations at different speeds
- Performance measurement intertwined with simulation logic
- Hard to reproduce timing-dependent behaviors

## Solution Overview

We implemented a centralized `TimeManager` class that cleanly separates simulation time from real time while maintaining backward compatibility.

### Key Components

#### 1. TimeManager Class (`core/time_manager.py`)

The `TimeManager` provides:
- **Simulation time tracking**: Discrete integer steps for game logic
- **Real time tracking**: Performance measurement and timestamps
- **Circadian cycle management**: Automatic calculation of time-of-day (0-23 hours)
- **Time scaling**: Configurable simulation speed relative to real time
- **Performance metrics**: Automatic tracking of execution times

#### 2. Updated Components

**alive_node.py**:
- Uses centralized time for circadian cycles via `step_phase()`
- Timestamps use centralized `get_timestamp()` function
- Maintains backward compatibility with explicit `current_time` parameter

**network.py**:
- Network ticks use centralized performance measurement
- `calculate_performance_and_stability()` uses centralized timing
- Evolution engine uses clean time tracking

**time_series_tracker.py**:
- All timestamps use centralized time management
- Consistent time representation for data tracking

## Usage Examples

### Basic Usage

```python
from core.time_manager import TimeManager, set_time_manager

# Create and configure time manager
tm = TimeManager()
set_time_manager(tm)

# Advance simulation time rapidly
tm.advance_simulation(100)  # 100 simulation steps instantly
print(f"Circadian hour: {tm.circadian_time}")  # Hour of day (0-23)

# Measure performance
tm.start_performance_measurement()
# ... do work ...
duration = tm.end_performance_measurement()
```

### Node Integration

```python
from core.alive_node import AliveLoopNode

node = AliveLoopNode((0, 0), (0, 0))

# New way - uses centralized time
node.step_phase()  # Automatically uses TimeManager

# Old way - still works for backward compatibility
node.step_phase(current_time=15)
```

### Network Integration

```python
from core.network import AdaptiveClockNetwork

network = AdaptiveClockNetwork(genome)

# Network automatically uses centralized time
network.network_tick(stimuli)

# Get performance metrics
metrics = network.calculate_performance_and_stability()
```

### Time Scaling

```python
from core.time_manager import TimeConfig, TimeManager

# Configure time scaling
config = TimeConfig(simulation_time_scale=2.0)  # 2x speed
tm = TimeManager(config)
set_time_manager(tm)
```

## API Reference

### TimeManager

**Properties:**
- `simulation_step: int` - Current simulation step
- `circadian_time: int` - Current hour of day (0-23)
- `real_time: float` - Current wall-clock time

**Methods:**
- `advance_simulation(steps: int)` - Advance simulation time
- `start_performance_measurement()` - Begin timing
- `end_performance_measurement() -> float` - End timing, return duration
- `network_tick()` - Called during network operations
- `get_statistics() -> Dict` - Get timing statistics
- `reset()` - Reset all timing

### Global Functions

- `get_time_manager() -> TimeManager` - Get global instance
- `set_time_manager(tm: TimeManager)` - Set global instance
- `get_simulation_time() -> int` - Get current simulation step
- `get_circadian_time() -> int` - Get current circadian hour
- `get_timestamp() -> float` - Get real timestamp

## Benefits

1. **Clean Separation**: Simulation logic uses simulation time, performance measurement uses real time
2. **Configurable Speed**: Run simulations at any speed relative to real time
3. **Reproducible**: Simulation time is deterministic and controllable
4. **Performance Tracking**: Automatic measurement of real execution times
5. **Backward Compatible**: Existing code continues to work unchanged
6. **Centralized**: Single source of truth for all time management

## Testing

The solution includes comprehensive tests:
- `tests/test_time_manager.py` - Core TimeManager functionality
- `tests/test_time_mitigation_integration.py` - End-to-end integration tests
- Updated existing tests to use centralized time

Run the demonstration:
```bash
python demo_time_mitigation.py
```

## Migration Guide

### For Existing Code

Most existing code requires no changes due to backward compatibility:

```python
# This still works exactly as before
node.step_phase(current_time=15)
```

### For New Code

Use centralized time management:

```python
# Get current simulation time
current_time = get_simulation_time()

# Get circadian hour
hour = get_circadian_time() 

# Get real timestamp
timestamp = get_timestamp()

# Let nodes use centralized time
node.step_phase()  # No explicit time needed
```

## Performance Impact

The centralized time management has minimal performance overhead:
- Time calculations are simple integer arithmetic
- Real time measurement uses high-precision `time.perf_counter()`
- No significant impact on simulation speed
- Actual performance improvement due to cleaner code structure

This solution fully addresses the "Clock-time--> real. Mitigate to real world time." requirement with a robust, scalable, and backward-compatible implementation.