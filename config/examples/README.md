# Configuration Examples

This directory contains example configurations for different spatial dimensions and use cases.

## Available Examples

### `network_config_2d.yaml`
- **Spatial Dimensions**: 2D
- **Use Case**: Standard planar networks, traditional neural networks, 2D robotics
- **Advantages**: Fast computation, easy visualization, well-understood behavior
- **Nodes**: 9 (3×3 grid)
- **Communication Range**: 2.0

### `network_config_3d.yaml`
- **Spatial Dimensions**: 3D
- **Use Case**: Realistic spatial simulations, 3D robotics, volumetric processing
- **Advantages**: More realistic physical modeling, richer interaction patterns
- **Nodes**: 27 (3×3×3 cube)
- **Communication Range**: 3.0 (increased for 3D volume scaling)

### `network_config_high_dimensional.yaml`
- **Spatial Dimensions**: 5D
- **Use Case**: Research into high-dimensional spaces, curse of dimensionality studies
- **Advantages**: Theoretical exploration, understanding dimensional effects
- **Nodes**: 50 (random topology - grid becomes impractical)
- **Communication Range**: 4.0 (accounts for distance uniformity in high dimensions)

## How to Use

1. **Copy an example**: Copy the desired example to your main config directory
   ```bash
   cp config/examples/network_config_3d.yaml config/network_config.yaml
   ```

2. **Customize**: Edit the copied configuration to match your specific needs

3. **Run**: Use your application normally - it will automatically detect the spatial dimensions

## Parameter Guidelines

### Communication Range by Dimension
- **2D**: 2.0 - 2.5 (good connectivity without overcrowding)
- **3D**: 3.0 - 4.0 (volume scales as r³, so increase range)
- **Higher**: 4.0+ (distances become more uniform)

### Node Count by Dimension
- **2D**: 9-25 for small tests, 100-1000 for experiments
- **3D**: 27-125 for small tests, 1000-10000 for experiments  
- **Higher**: 50-500 (dense grids become impractical)

### Topology Recommendations
- **2D/3D**: Grid topology works well for structured arrangements
- **4D+**: Random topology recommended (grids become sparse and inefficient)

## Performance Considerations

### Memory Usage
Each node stores vectors of size `spatial_dims`, so memory usage scales approximately as:
`memory ≈ nodes × spatial_dims × vector_count`

### Computation Speed
- Distance calculations: O(spatial_dims)
- Neighbor search: O(nodes × spatial_dims) for brute force
- Overall: Linear scaling with dimensions for most operations

### Recommended Limits
- **Interactive use**: ≤3D, ≤1000 nodes
- **Batch experiments**: ≤10D, ≤100000 nodes (depending on hardware)
- **Research**: No theoretical limits, but performance degrades

## Validation

All examples are validated and should work with the current system. Run tests to verify:

```bash
# Test 2D configuration
cp config/examples/network_config_2d.yaml config/network_config.yaml
python -m unittest tests.test_spatial_dimensions.TestSpatialDimensions.test_configuration_spatial_dims

# Test 3D configuration  
cp config/examples/network_config_3d.yaml config/network_config.yaml
python -m unittest tests.test_spatial_dimensions.TestSpatialDimensions.test_3d_smoke_simulation
```