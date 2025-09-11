# Spatial Dimensions Guide

## Overview

The Adaptive Neural Network system supports arbitrary spatial dimensions, allowing you to run simulations in 2D, 3D, or higher dimensional spaces. This guide explains how to configure and use different spatial dimensions in your network.

## Configuration

### Basic Configuration

The spatial dimension is controlled by the `spatial_dims` parameter in your configuration file:

```yaml
# config/network_config.yaml
spatial_dims: 2  # Default 2D
nodes: 9
topology: grid
external_signal_interval: 5
api_endpoints:
  human: null
  ai: null
  world: null
```

### 2D Configuration (Default)

For traditional 2D networks:

```yaml
spatial_dims: 2
nodes: 9
topology: grid
```

### 3D Configuration

For 3D networks:

```yaml
spatial_dims: 3
nodes: 27  # Consider cubic arrangements
topology: grid
```

### Higher Dimensions

For research into higher-dimensional spaces:

```yaml
spatial_dims: 5
nodes: 50
topology: random  # Grid becomes impractical in high dimensions
```

## Usage Examples

### Creating Nodes with Spatial Awareness

The system automatically handles spatial dimensions based on your configuration:

```python
from config.network_config import load_network_config
from core.alive_node import AliveLoopNode

# Load configuration
config = load_network_config()
spatial_dims = config.get("spatial_dims", 2)

# Create a node with automatic dimension detection
node = AliveLoopNode(
    position=[1.0, 2.0, 3.0],  # 3D position
    velocity=[0.1, -0.1, 0.05],  # 3D velocity
    initial_energy=10.0,
    field_strength=1.0,
    node_id=0
)
# spatial_dims is automatically inferred from position length

# Or explicitly specify dimensions
node = AliveLoopNode(
    position=[1.0, 2.0, 3.0],
    velocity=[0.1, -0.1, 0.05],
    initial_energy=10.0,
    field_strength=1.0,
    node_id=0,
    spatial_dims=3  # Explicit specification
)
```

### Using Spatial Utilities

The `core.spatial_utils` module provides dimension-agnostic helper functions:

```python
from core.spatial_utils import (
    zero_vector, rand_vector, distance, 
    validate_spatial_dimensions, create_random_positions
)

# Create zero vector in any dimension
zero_2d = zero_vector(2)  # [0., 0.]
zero_3d = zero_vector(3)  # [0., 0., 0.]

# Create random vectors
random_2d = rand_vector(2, (-1, 1))  # Random in [-1,1] for both dimensions
random_3d = rand_vector(3, [(-1, 1), (-2, 2), (-0.5, 0.5)])  # Per-dimension ranges

# Calculate distances (works in any dimension)
dist = distance([1, 2, 3], [4, 5, 6])  # Euclidean distance

# Validate dimensions
validate_spatial_dimensions([position, velocity], 3)  # Ensures both are 3D

# Create multiple random positions
positions = create_random_positions(count=100, dim=3, bounds=(-10, 10))
```

### Running Experiments in Different Dimensions

The extreme scale test automatically uses your configured spatial dimensions:

```python
# This will use spatial_dims from config
python experiments/extreme_scale_test.py
```

For 3D experiments, update your config:

```yaml
spatial_dims: 3
```

Then run:

```bash
python experiments/extreme_scale_test.py
```

## Considerations for Different Dimensions

### 2D Networks
- **Advantages**: Simple visualization, well-understood behavior, fast computation
- **Use Cases**: Traditional neural networks, 2D spatial simulations, planar robotics
- **Communication Range**: Standard ranges work well

### 3D Networks  
- **Advantages**: More realistic for physical simulations, richer interaction patterns
- **Use Cases**: 3D robotics, volumetric data processing, realistic spatial AI
- **Communication Range**: Consider that volume scales as r³, so you may need to adjust `communication_range` in your config
- **Neighbor Density**: Much lower density at same distances compared to 2D

### Higher Dimensions (4D+)
- **Advantages**: Research into high-dimensional spaces, curse of dimensionality studies
- **Use Cases**: Research, theoretical exploration, high-dimensional data analysis
- **Considerations**: 
  - Distances become increasingly uniform (curse of dimensionality)
  - Visualization becomes difficult
  - Consider using sparse topologies rather than grid arrangements
  - May need specialized neighbor search algorithms for performance

## Performance Considerations

### Neighbor Search
- 2D/3D: Standard algorithms work well
- Higher dimensions: Consider KD-tree or other spatial indexing (future Phase 3 feature)

### Memory Usage
- Scales linearly with spatial dimensions
- Each node stores position, velocity, attention_focus vectors of size `spatial_dims`

### Computation
- Distance calculations scale with dimensions
- Most operations remain efficient up to moderate dimensions (≤10)

## Migration Guide

### From 2D to 3D

1. Update your configuration:
   ```yaml
   spatial_dims: 3
   ```

2. If you have hardcoded 2D positions in your code, update them:
   ```python
   # Old 2D
   position = [1.0, 2.0]
   
   # New 3D
   position = [1.0, 2.0, 0.0]  # Add Z coordinate
   ```

3. Consider adjusting simulation parameters:
   ```yaml
   communication_range: 3.0  # May need to increase for 3D
   ```

### From Hardcoded to Configuration-Based

Replace hardcoded dimension assumptions:

```python
# Old hardcoded approach
attention_focus = np.zeros(2)

# New configuration-based approach  
from core.spatial_utils import zero_vector
attention_focus = zero_vector(self.spatial_dims)
```

## Testing

The system includes comprehensive tests for spatial dimensions in `tests/test_spatial_dimensions.py`. Run them to verify your setup:

```bash
python -m unittest tests.test_spatial_dimensions -v
```

## Advanced Features

### Bounds Validation
The system validates that positions stay within configured bounds:

```python
from core.spatial_utils import validate_position_in_bounds

bounds = [(-10, 10), (-5, 5), (-1, 1)]  # 3D bounds
position = [2.0, 1.0, 0.5]
validate_position_in_bounds(position, bounds)  # Raises ValueError if outside
```

### Custom Topology Generation
For higher dimensions, consider custom topology functions rather than grid layouts.

## Troubleshooting

### Dimension Mismatch Errors
If you see errors like "spatial dimensions don't match":
1. Check that all position/velocity vectors have the same length
2. Verify your `spatial_dims` configuration matches your data
3. Use `validate_spatial_dimensions()` to catch mismatches early

### Performance Issues
For high-dimensional or large-scale networks:
1. Profile your specific use case
2. Consider reducing `communication_range` 
3. Use sparse node arrangements rather than dense grids
4. Monitor memory usage as it scales with nodes × dimensions

### Visualization Challenges
- 2D: Use matplotlib scatter plots
- 3D: Use matplotlib 3D plots or plotly
- Higher dimensions: Use dimensionality reduction (PCA, t-SNE) for visualization

## Future Enhancements

The system is designed to support future enhancements including:
- KD-tree based neighbor search for large N (Phase 3)
- GPU acceleration via CuPy backend (Phase 5)
- Advanced 3D visualization tools
- Automatic parameter tuning for different dimensions