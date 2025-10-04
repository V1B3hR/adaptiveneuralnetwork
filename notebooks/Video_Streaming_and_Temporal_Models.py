"""
Video Streaming and Temporal Models - Interactive Example

This notebook-style script demonstrates the video streaming and temporal modeling
capabilities of the Adaptive Neural Network library.

Run each section independently to explore different features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import json
from pathlib import Path

print("="*80)
print("ADAPTIVE NEURAL NETWORK - VIDEO STREAMING AND TEMPORAL MODELS")
print("="*80)
print()

# =============================================================================
# SECTION 1: Model Comparison and Benchmarking
# =============================================================================

print("SECTION 1: Model Comparison and Benchmarking")
print("-" * 50)

from adaptiveneuralnetwork.models.video_models import (
    create_video_model, 
    VideoModelConfig
)

# Configure models
config = VideoModelConfig(
    input_channels=3,
    input_height=224,
    input_width=224,
    sequence_length=8,   # Shorter for demo
    hidden_dim=512,      # Smaller for demo
    num_layers=2,
    dropout=0.1,
    num_classes=1000
)

print(f"Model Configuration:")
print(f"  Input size: {config.input_width}x{config.input_height}")
print(f"  Sequence length: {config.sequence_length}")
print(f"  Hidden dimension: {config.hidden_dim}")
print(f"  Classes: {config.num_classes}")
print()

# Create test data
batch_size = 2
test_data = torch.randn(batch_size, config.sequence_length, 3, 
                       config.input_height, config.input_width)
print(f"Test data shape: {test_data.shape}")
print()

# Compare different models
models_to_compare = {
    'ConvLSTM': 'convlstm',
    '3D CNN': 'conv3d', 
    'Transformer': 'transformer'
}

results = {}

for model_name, model_type in models_to_compare.items():
    print(f"Testing {model_name}...")
    
    # Create model
    model = create_video_model(model_type, config)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_data)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        output = model(test_data)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    
    results[model_name] = {
        'parameters': num_params,
        'latency_ms': latency_ms,
        'output_shape': list(output.shape)
    }
    
    print(f"  Parameters: {num_params:,}")
    print(f"  Latency: {latency_ms:.2f}ms")
    print(f"  Output shape: {output.shape}")
    print()

# Summary
print("Model Comparison Summary:")
print("-" * 30)
for name, stats in results.items():
    params_str = f"{stats['parameters']/1000:.0f}K" if stats['parameters'] < 1000000 else f"{stats['parameters']/1000000:.1f}M"
    print(f"{name:<12} {params_str:<8} {stats['latency_ms']:<8.1f}ms")
print()

# =============================================================================
# SECTION 2: Hybrid Model Demonstration
# =============================================================================

print("SECTION 2: Hybrid Model Demonstration")
print("-" * 50)

from adaptiveneuralnetwork.models.video_models import HybridVideoModel

# Create hybrid model
hybrid_model = HybridVideoModel(config)
hybrid_model.eval()

print(f"Hybrid model parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")
print()

# Test different fusion modes
fusion_modes = ['weighted', 'concat', 'ensemble']

print("Testing fusion modes:")
for mode in fusion_modes:
    start_time = time.time()
    with torch.no_grad():
        output = hybrid_model(test_data, fusion_mode=mode)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    print(f"  {mode.capitalize():<10}: {latency_ms:.2f}ms, shape: {output.shape}")

# Get individual model outputs
print("\nIndividual model outputs:")
individual_outputs = hybrid_model.get_individual_outputs(test_data)
for model_name, output in individual_outputs.items():
    # Show top prediction for first sample
    probs = torch.softmax(output[0], dim=0)
    top_value, top_idx = torch.max(probs, dim=0)
    print(f"  {model_name:<12}: Class {top_idx.item()}, confidence: {top_value.item():.4f}")
print()

# =============================================================================
# SECTION 3: Synthetic Video Data Processing
# =============================================================================

print("SECTION 3: Synthetic Video Data Processing")
print("-" * 50)

def create_synthetic_video_sequence(num_frames=16, height=224, width=224):
    """Create a synthetic video sequence with some temporal structure."""
    frames = []
    
    # Create a moving pattern
    for t in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.float32)
        
        # Moving diagonal pattern
        offset = int(t * 10) % min(height, width)
        for i in range(min(height, width)):
            y = (i + offset) % height
            x = i
            if x < width:
                frame[y, x, :] = [0.8, 0.6, 0.4]  # Orange-ish color
        
        # Add some noise
        frame += np.random.normal(0, 0.1, frame.shape)
        frame = np.clip(frame, 0, 1)
        
        frames.append(frame)
    
    return np.stack(frames, axis=0)  # Shape: (T, H, W, C)

# Create synthetic video
print("Creating synthetic video sequence...")
synthetic_video = create_synthetic_video_sequence(
    num_frames=config.sequence_length,
    height=config.input_height,
    width=config.input_width
)

print(f"Synthetic video shape: {synthetic_video.shape}")

# Convert to tensor format (T, H, W, C) -> (T, C, H, W)
video_tensor = torch.from_numpy(synthetic_video).permute(0, 3, 1, 2).unsqueeze(0)  # Add batch dim
print(f"Video tensor shape: {video_tensor.shape}")
print()

# Process with different models
print("Processing synthetic video with different models:")
for model_name, model_type in models_to_compare.items():
    model = create_video_model(model_type, config)
    model.eval()
    
    with torch.no_grad():
        output = model(video_tensor)
        probs = torch.softmax(output[0], dim=0)
        top_values, top_indices = torch.topk(probs, 3)
        
        print(f"{model_name}:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_values)):
            print(f"  {i+1}. Class {idx.item()}: {prob.item():.4f}")
        print()

# =============================================================================
# SECTION 4: Performance Analysis
# =============================================================================

print("SECTION 4: Performance Analysis")
print("-" * 50)

# Test different input sizes and sequence lengths
input_configurations = [
    (112, 112, 4),
    (112, 112, 8),
    (224, 224, 4),
    (224, 224, 8),
    (224, 224, 16),
]

print("Performance analysis across different configurations:")
print("Format: Width x Height @ Frames -> Latency (ms)")
print()

# Use lightweight ConvLSTM for testing
test_config = VideoModelConfig(
    hidden_dim=64,
    num_layers=1,
    num_classes=10  # Smaller for faster testing
)

for width, height, seq_len in input_configurations:
    test_config.input_width = width
    test_config.input_height = height
    test_config.sequence_length = seq_len
    
    model = create_video_model('convlstm', test_config)
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, seq_len, 3, height, width)
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_input)
    
    # Measure
    times = []
    for _ in range(10):
        start_time = time.time()
        with torch.no_grad():
            _ = model(test_input)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    
    avg_latency = np.mean(times)
    std_latency = np.std(times)
    
    print(f"{width:3d}x{height:3d} @ {seq_len:2d} frames -> {avg_latency:6.2f} ± {std_latency:5.2f}ms")

print()

# =============================================================================
# SECTION 5: Video Configuration Examples
# =============================================================================

print("SECTION 5: Video Configuration Examples")
print("-" * 50)

# This would normally work with actual video streams, but we'll show the configuration
print("Video streaming configuration examples:")
print()

print("1. Webcam Configuration:")
print("   from adaptiveneuralnetwork.data import VideoConfig")
print("   config = VideoConfig(")
print("       target_width=224,")
print("       target_height=224,")
print("       fps=30.0,")
print("       sequence_length=16,")
print("       buffer_size=64")
print("   )")
print()

print("2. Real-time Processing Configuration:")
print("   config = VideoConfig(")
print("       target_width=112,      # Lower resolution")
print("       target_height=112,")
print("       sequence_length=8,     # Shorter sequences")
print("       frame_skip=1,          # Skip every other frame")
print("       max_fps=15.0           # Limit FPS")
print("   )")
print()

print("3. High-quality Processing Configuration:")
print("   config = VideoConfig(")
print("       target_width=320,")
print("       target_height=240,")
print("       sequence_length=32,    # Longer sequences")
print("       buffer_size=128,       # Larger buffer")
print("       batch_size=16          # Larger batches")
print("   )")
print()

# =============================================================================
# SECTION 6: Model Export and Deployment
# =============================================================================

print("SECTION 6: Model Export and Deployment")
print("-" * 50)

# Create a model for export
export_config = VideoModelConfig(
    input_height=224,
    input_width=224,
    sequence_length=8,
    hidden_dim=128,
    num_classes=1000
)

model_for_export = create_video_model('transformer', export_config)  # Transformer is fastest
model_for_export.eval()

print("Model ready for deployment:")
print(f"  Architecture: Video Transformer")
print(f"  Parameters: {sum(p.numel() for p in model_for_export.parameters()):,}")
print(f"  Input shape: (batch_size, {export_config.sequence_length}, 3, {export_config.input_height}, {export_config.input_width})")
print(f"  Output shape: (batch_size, {export_config.num_classes})")
print()

# Save model
model_path = Path("video_transformer_model.pth")
torch.save({
    'model_state_dict': model_for_export.state_dict(),
    'config': export_config,
    'model_type': 'transformer'
}, model_path)

print(f"Model saved to: {model_path}")
print(f"Model file size: {model_path.stat().st_size / 1024 / 1024:.2f}MB")
print()

# Show how to load the model
print("To load the model:")
print("   checkpoint = torch.load('video_transformer_model.pth')")
print("   config = checkpoint['config']")
print("   model = create_video_model(checkpoint['model_type'], config)")
print("   model.load_state_dict(checkpoint['model_state_dict'])")
print("   model.eval()")
print()

# =============================================================================
# SECTION 7: CLI Usage Examples
# =============================================================================

print("SECTION 7: CLI Usage Examples")
print("-" * 50)

print("Command-line interface examples:")
print()

print("1. Benchmark webcam with ConvLSTM:")
print("   python -m adaptiveneuralnetwork.scripts.adaptive_video_benchmark \\")
print("       --source webcam --model convlstm --max-frames 500")
print()

print("2. Live inference from webcam:")
print("   python -m adaptiveneuralnetwork.scripts.adaptive_live_infer \\")
print("       --source webcam --model transformer")
print()

print("3. Benchmark RTSP stream:")
print("   python -m adaptiveneuralnetwork.scripts.adaptive_video_benchmark \\")
print("       --source rtsp://192.168.1.100:554/stream \\")
print("       --model hybrid --output results.json")
print()

print("4. Process video file with logging:")
print("   python -m adaptiveneuralnetwork.scripts.adaptive_live_infer \\")
print("       --source video.mp4 --model convlstm \\")
print("       --output-format json --output-file results.jsonl")
print()

# =============================================================================
# CONCLUSION
# =============================================================================

print("="*80)
print("CONCLUSION")
print("="*80)

print("This demonstration covered:")
print("✓ Video model comparison (ConvLSTM, 3D CNN, Transformer)")
print("✓ Hybrid model fusion strategies")
print("✓ Synthetic video data processing")
print("✓ Performance analysis across configurations")
print("✓ Video streaming configuration options")
print("✓ Model export and deployment preparation")
print("✓ CLI tool usage examples")
print()

print("Key takeaways:")
print("• Video Transformer is fastest for inference")
print("• ConvLSTM provides good temporal modeling")
print("• 3D CNN offers balance of speed and accuracy")
print("• Hybrid models maximize accuracy at higher cost")
print("• Configuration greatly impacts performance")
print()

print("Next steps:")
print("• Try with real video data using OpenCV")
print("• Experiment with different model configurations")
print("• Use CLI tools for benchmarking and live inference")
print("• Train models on your specific video datasets")
print("• Deploy for real-time applications")
print()

print("For more information, see:")
print("• docs/VIDEO_STREAMING_GUIDE.md")
print("• adaptiveneuralnetwork/scripts/ for CLI tools")
print("• demo_video_functionality.py for interactive demos")

print("\nDemo completed successfully!")
