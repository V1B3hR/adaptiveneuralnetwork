# Video Streaming and Temporal Models Guide

This guide covers the comprehensive video and live stream support in the Adaptive Neural Network library, including streaming data ingestion, temporal models, real-time inference, and CLI utilities.

## Table of Contents

1. [Overview](#overview)
2. [Video Streaming](#video-streaming)
3. [Temporal Models](#temporal-models)
4. [Real-Time Inference](#real-time-inference)
5. [CLI Tools](#cli-tools)
6. [Examples](#examples)
7. [Performance Guidelines](#performance-guidelines)

## Overview

The Adaptive Neural Network library provides full-featured video and live stream support with:

- **Functional streaming data loader** supporting local files, webcams, and network streams
- **Advanced temporal models** including ConvLSTM, 3D CNN, and Transformer approaches
- **Real-time inference pipeline** with adaptive computation and low latency
- **CLI utilities** for easy benchmarking and live inference
- **Hybrid model combinations** for optimal performance

## Video Streaming

### VideoConfig

Configure video processing parameters:

```python
from adaptiveneuralnetwork.data import VideoConfig

config = VideoConfig(
    target_width=224,           # Target frame width
    target_height=224,          # Target frame height
    fps=30.0,                   # Target FPS (None for source FPS)
    sequence_length=16,         # Frames per sequence
    frame_skip=0,               # Skip frames (0 = no skipping)
    buffer_size=64,             # Frame buffer size
    batch_size=8,               # Batch size for processing
    normalize=True,             # Apply normalization
    mean=(0.485, 0.456, 0.406), # ImageNet normalization
    std=(0.229, 0.224, 0.225)
)
```

### Video Sources

#### Webcam Stream

```python
from adaptiveneuralnetwork.data import create_webcam_stream

# Default webcam
stream = create_webcam_stream(camera_id=0, target_width=224, target_height=224)

# Start streaming
if stream.start_streaming():
    # Get frame sequences
    sequence = stream.get_sequence(timeout=1.0)  # Returns torch.Tensor
    batch = stream.get_batch(batch_size=4)       # Returns batched sequences
    
    # Stop when done
    stream.stop_streaming()
```

#### Video File Stream

```python
from adaptiveneuralnetwork.data import create_file_stream

stream = create_file_stream(
    "path/to/video.mp4",
    sequence_length=16,
    target_fps=15.0
)

stream.start_streaming()
# Process video...
stream.stop_streaming()
```

#### RTSP/Network Stream

```python
from adaptiveneuralnetwork.data import create_rtsp_stream

stream = create_rtsp_stream(
    "rtsp://192.168.1.100:554/stream",
    buffer_size=32,
    reconnect_attempts=3
)

stream.start_streaming()
# Process live stream...
stream.stop_streaming()
```

### Stream Information

```python
info = stream.get_info()
print(f"Source: {info['source']}")
print(f"Running: {info['is_running']}")
print(f"Buffer sizes: {info['buffer_sizes']}")
```

## Temporal Models

### Model Configuration

```python
from adaptiveneuralnetwork.models import VideoModelConfig

config = VideoModelConfig(
    input_channels=3,
    input_height=224,
    input_width=224,
    sequence_length=16,
    hidden_dim=256,
    num_layers=2,
    dropout=0.1,
    num_classes=1000
)
```

### ConvLSTM Model

Ideal for spatiotemporal modeling with memory:

```python
from adaptiveneuralnetwork.models import create_convlstm_model

model = create_convlstm_model(
    num_classes=1000,
    hidden_dim=256,
    sequence_length=16
)

# Input: (batch_size, sequence_length, channels, height, width)
video_tensor = torch.randn(4, 16, 3, 224, 224)
output = model(video_tensor)  # (4, 1000)
```

### 3D CNN Model

Best for spatiotemporal convolutions:

```python
from adaptiveneuralnetwork.models import create_conv3d_model

model = create_conv3d_model(
    num_classes=1000,
    hidden_dim=256
)

output = model(video_tensor)
```

### Video Transformer

Attention-based temporal modeling:

```python
from adaptiveneuralnetwork.models import create_video_transformer

model = create_video_transformer(
    num_classes=1000,
    hidden_dim=256,
    num_layers=4
)

output = model(video_tensor)
```

### Hybrid Model

Combines all three approaches:

```python
from adaptiveneuralnetwork.models import create_hybrid_model

model = create_hybrid_model(num_classes=1000)

# Different fusion modes
output_weighted = model(video_tensor, fusion_mode="weighted")
output_concat = model(video_tensor, fusion_mode="concat")
output_ensemble = model(video_tensor, fusion_mode="ensemble")

# Get individual model outputs
individual_outputs = model.get_individual_outputs(video_tensor)
```

## Real-Time Inference

### Inference Configuration

```python
from adaptiveneuralnetwork.core import InferenceConfig

inference_config = InferenceConfig(
    target_latency_ms=100.0,        # Target inference latency
    max_latency_ms=200.0,           # Maximum acceptable latency
    target_fps=15.0,                # Target inference FPS
    enable_adaptive_resolution=True, # Dynamic resolution scaling
    enable_frame_skipping=True,     # Skip frames under high load
    enable_batching=True,           # Batch processing
    max_batch_size=4
)
```

### Stream Inference Pipeline

```python
from adaptiveneuralnetwork.core import VideoStreamInference

# Create inference pipeline
pipeline = VideoStreamInference(
    model=model,
    video_config=video_config,
    inference_config=inference_config
)

# Add result callback
def handle_result(result):
    print(f"Frame {result.frame_id}: confidence={result.confidence:.3f}, "
          f"latency={result.latency_ms:.1f}ms")
    
    # Get top prediction
    predictions = torch.softmax(result.predictions, dim=0)
    top_value, top_idx = torch.max(predictions, dim=0)
    print(f"Top prediction: Class {top_idx.item()} ({top_value.item():.3f})")

pipeline.add_result_callback(handle_result)

# Start inference
pipeline.start_stream_inference("webcam")

# Let it run...
time.sleep(30)

# Stop inference
pipeline.stop_stream_inference()

# Get performance stats
stats = pipeline.get_stream_info()
print(f"Performance: {stats['inference']}")
```

### Performance Monitoring

```python
from adaptiveneuralnetwork.core import PerformanceMonitor

monitor = PerformanceMonitor(window_size=100)

# Record metrics during inference
# (automatically done by inference pipeline)

stats = monitor.get_stats()
print(f"Average latency: {stats['latency_ms']['mean']:.2f}ms")
print(f"Average FPS: {stats['fps']['mean']:.2f}")
```

## CLI Tools

### Video Benchmarking

Benchmark different models and sources:

```bash
# Benchmark webcam with ConvLSTM
python -m adaptiveneuralnetwork.scripts.adaptive_video_benchmark \
    --source webcam --model convlstm --max-frames 1000

# Benchmark RTSP stream with 3D CNN
python -m adaptiveneuralnetwork.scripts.adaptive_video_benchmark \
    --source rtsp://192.168.1.100:554/stream --model conv3d \
    --output results.json

# Benchmark video file with hybrid model
python -m adaptiveneuralnetwork.scripts.adaptive_video_benchmark \
    --source video.mp4 --model hybrid --duration 60 \
    --target-fps 20 --batch-size 4
```

### Live Inference

Real-time inference with visualization:

```bash
# Live inference from webcam
python -m adaptiveneuralnetwork.scripts.adaptive_live_infer \
    --source webcam --model convlstm

# Live inference with logging
python -m adaptiveneuralnetwork.scripts.adaptive_live_infer \
    --source webcam --model transformer \
    --output-format json --output-file results.jsonl

# Live inference with custom settings
python -m adaptiveneuralnetwork.scripts.adaptive_live_infer \
    --source rtsp://stream-url --model hybrid \
    --target-latency 50 --batch-size 2 \
    --report-file final_report.json
```

## Examples

### Basic Video Processing

```python
import torch
from adaptiveneuralnetwork.models import create_video_model, VideoModelConfig
from adaptiveneuralnetwork.data import create_file_stream, VideoConfig

# Create model
model_config = VideoModelConfig(num_classes=10, sequence_length=8)
model = create_video_model('convlstm', model_config)

# Create video stream
video_config = VideoConfig(sequence_length=8, target_width=224)
stream = create_file_stream("sample_video.mp4", **video_config.__dict__)

# Process video
stream.start_streaming()

while True:
    sequence = stream.get_sequence(timeout=1.0)
    if sequence is None:
        break
    
    with torch.no_grad():
        predictions = model(sequence.unsqueeze(0))  # Add batch dimension
        probs = torch.softmax(predictions, dim=1)
        top_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        
        print(f"Predicted class: {top_class.item()}, "
              f"confidence: {confidence.item():.3f}")

stream.stop_streaming()
```

### Real-Time Action Recognition

```python
import torch
import time
from adaptiveneuralnetwork.models import create_video_transformer
from adaptiveneuralnetwork.core import create_stream_inference
from adaptiveneuralnetwork.data import VideoConfig
from adaptiveneuralnetwork.core import InferenceConfig

# Load model (replace with your trained weights)
model = create_video_transformer(num_classes=400)  # Kinetics-400 classes
# model.load_state_dict(torch.load('kinetics_model.pth'))

# Configuration for real-time processing
video_config = VideoConfig(
    target_width=224,
    target_height=224,
    sequence_length=16,
    max_fps=30.0,
    buffer_size=32
)

inference_config = InferenceConfig(
    target_latency_ms=50.0,  # Very responsive
    enable_adaptive_resolution=True,
    enable_frame_skipping=True
)

# Action labels (example)
action_labels = [
    "walking", "running", "jumping", "sitting", "standing",
    # ... more actions
]

# Result handler
def process_action_recognition(result):
    predictions = torch.softmax(result.predictions, dim=0)
    top_value, top_idx = torch.max(predictions, dim=0)
    
    action = action_labels[top_idx.item()] if top_idx.item() < len(action_labels) else f"Action_{top_idx.item()}"
    
    print(f"Detected action: {action} (confidence: {top_value.item():.3f}, "
          f"latency: {result.latency_ms:.1f}ms)")

# Create and run inference
pipeline = create_stream_inference(
    model=model,
    source="webcam",
    video_config=video_config,
    inference_config=inference_config
)

pipeline.add_result_callback(process_action_recognition)
pipeline.start_stream_inference("webcam")

print("Action recognition running... Press Ctrl+C to stop")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pipeline.stop_stream_inference()
    print("Stopped action recognition")
```

## Performance Guidelines

### Model Selection

- **ConvLSTM**: Best for temporal sequences with memory requirements
- **3D CNN**: Fastest inference, good for short-term spatiotemporal patterns
- **Transformer**: Most accurate for long-range dependencies, moderate speed
- **Hybrid**: Best accuracy but highest computational cost

### Optimization Tips

1. **Input Resolution**: Lower resolution (112x112) for real-time applications
2. **Sequence Length**: Shorter sequences (8-16 frames) for better performance
3. **Batch Size**: Use batching for throughput, avoid for low latency
4. **Frame Skipping**: Enable for high frame rate sources
5. **Adaptive Resolution**: Enable for variable load conditions

### Hardware Recommendations

- **CPU**: Intel i7/AMD Ryzen 7 or better for real-time processing
- **GPU**: NVIDIA GTX 1060 or better for GPU acceleration
- **Memory**: 8GB+ RAM, 4GB+ VRAM for large models
- **Storage**: SSD for video file processing

### Troubleshooting

#### Common Issues

1. **"OpenCV not available"**: Install with `pip install opencv-python`
2. **High latency**: Reduce input resolution or enable frame skipping
3. **Memory errors**: Reduce batch size or sequence length
4. **Stream connection failed**: Check network connectivity and stream URL

#### Performance Monitoring

```python
# Get detailed performance statistics
stats = pipeline.get_stream_info()

print("Video Stream Stats:")
print(f"  Buffer sizes: {stats['video_stream']['buffer_sizes']}")

print("Inference Stats:")
print(f"  Average latency: {stats['inference']['latency_ms']['mean']:.2f}ms")
print(f"  Queue sizes: {stats['inference']['queue_sizes']}")
print(f"  Total frames: {stats['inference']['total_frames_processed']}")
```

## Advanced Features

### Custom Video Sources

```python
from adaptiveneuralnetwork.data.video_streaming import VideoStreamDataset, VideoConfig

# Custom video source function
def custom_frame_generator(frame_id):
    # Generate or load custom frames
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# Create custom stream
config = VideoConfig()
stream = VideoStreamDataset(custom_frame_generator, config)
```

### Model Ensembling

```python
# Use multiple models for robust predictions
models = [
    create_convlstm_model(num_classes=1000),
    create_conv3d_model(num_classes=1000),
    create_video_transformer(num_classes=1000)
]

def ensemble_inference(video_input):
    predictions = []
    for model in models:
        with torch.no_grad():
            pred = model(video_input)
            predictions.append(torch.softmax(pred, dim=1))
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred
```

This guide provides comprehensive coverage of the video streaming and temporal modeling capabilities. For more examples and advanced usage, see the demo scripts and CLI tools in the repository.