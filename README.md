# Adaptive Neural Network

A biologically-inspired network of nodes (“AliveLoopNode”) interacting with capacitors and external data streams, featuring:
- Staged sleep (light, REM, deep)
- Mixed/overlapping phases (active, inspired, interactive)
- Anxiety and restorative behaviour
- Memory replay/sharing during sleep
- External signal adaptation
- **Multi-dimensional spatial awareness (2D, 3D, N-D support)**

## Project Structure

- `core/` — Main model classes (`alive_node.py`, `capacitor.py`, `network.py`)
- `experiments/` — Testing and demonstration scripts (`demo_test.py`)
- `run_simulation.py` — Entry point for running the demo
- `README.md` — Project overview and instructions

## Getting Started

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Choose your spatial dimension** (optional):
   ```bash
   # For 2D (default)
   cp config/examples/network_config_2d.yaml config/network_config.yaml
   
   # For 3D
   cp config/examples/network_config_3d.yaml config/network_config.yaml
   
   # For high-dimensional research
   cp config/examples/network_config_high_dimensional.yaml config/network_config.yaml
   ```

3. Run the demo simulation:
   ```bash
   python runsimulation.py
   ```

See [Spatial Dimensions Guide](docs/SPATIAL_DIMENSIONS_GUIDE.md) for detailed configuration options.

## Features

- Staged sleep (light, REM, deep)
- Mixed/overlapping phases (active, inspired, interactive)
- Anxiety tracking and relief
- Memory replay and sharing during sleep
- External signal absorption (human, AI, world)
- **Multi-dimensional spatial support (2D, 3D, N-D)**
- Demonstrative test function

## Extending

- Integrate real-world APIs for external signals
- Extend phase logic for nuanced transitions
- Experiment with network size, topology, and connectivity
- **Explore different spatial dimensions for your use case**
- Add custom spatial topologies and neighbor search algorithms

## Documentation

- [Spatial Dimensions Guide](docs/SPATIAL_DIMENSIONS_GUIDE.md) - Complete guide to 2D, 3D, and N-D configuration
- [Configuration Examples](config/examples/) - Ready-to-use configurations for different dimensions

---

Contributions welcome!

License: GNU GPL v3.0
