# Adaptive Neural Network Demos

This directory contains demonstration scripts and example data files for the Adaptive Neural Network project.

## Demo Scripts

### Core Features
- `demo_enhanced_features.py` - Comprehensive demonstration of enhanced network capabilities
- `demo_3d_capabilities.py` - 3D spatial dimension capabilities showcase
- `demo_continual_learning.py` - Continual learning features and MNIST benchmarks
- `demo_v3_neuromorphic.py` - Version 3 neuromorphic features
- `demo_v030_features.py` - Version 0.30 feature demonstrations

### Specialized Capabilities
- `demo_enhanced_trust_system.py` - Advanced trust network management
- `demo_emotional_protocols.py` - Emotional state handling and protocols
- `demo_extended_emotional_states.py` - Extended emotional state management
- `demo_anxiety_comparison.py` - Anxiety handling and safety protocols
- `demo_capacitor_improvements.py` - Capacitor system enhancements
- `demo_production_signal_processing.py` - Production-grade signal processing
- `demo_sql_security.py` - SQL security features demonstration
- `demo_time_mitigation.py` - Time management and mitigation features
- `demonstrate_rigorous_intelligence.py` - Rigorous intelligence implementation

## Demo Data Files

### Network Configurations
- `demo_full_network.json` - Complete network configuration example
- `demo_network_status.json` - Network status snapshot
- `example_baseline.json` - Baseline configuration template

### Time Series Data
- `demo_full_network_timeseries.json` - Full network time series data
- `demo_timeseries.json` - Basic time series data

### Results and Analysis
- `3d_demo_results.json` - 3D capabilities demonstration results
- `demo_anxiety_comparison.png` - Anxiety analysis visualization
- `demo_node_analysis.png` - Node behavior analysis chart

## Running Demos

All demo scripts can be run from this directory. They automatically adjust their import paths to work with the main project modules.

### Examples

```bash
# Run the comprehensive features demo
python demo_enhanced_features.py --nodes 10 --steps 100 --visualize

# Show 3D capabilities
python demo_3d_capabilities.py

# Test continual learning
python demo_continual_learning.py

# See help for any demo
python <demo_script.py> --help
```

### Requirements

Make sure the main project is installed:
```bash
pip install -e .
```

Some demos may require additional dependencies:
```bash
pip install matplotlib  # For visualizations
```

## Demo Development

When creating new demo scripts, include the following path adjustment at the top:

```python
import sys
import os
# Add parent directory to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This ensures the demo can import from the main project modules regardless of where it's run from.