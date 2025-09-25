# Video and Stream Processing Features

The Adaptive Neural Network library now includes comprehensive video and live stream support with state-of-the-art temporal models and real-time inference capabilities.

## 🎥 Key Features

### Video Streaming Infrastructure
- **Multi-source support**: Local files, webcams, RTSP/RTMP streams
- **Real-time processing**: Frame buffering, batching, and adaptive computation
- **OpenCV integration**: Robust video capture and preprocessing
- **Configuration flexibility**: Customizable resolution, FPS, sequence length

### Temporal Models
- **ConvLSTM**: Spatiotemporal modeling with memory (2.7M params, ~485ms)
- **3D CNN**: Fast spatiotemporal convolutions (2.1M params, ~210ms)
- **Video Transformer**: Attention-based modeling (938K params, ~107ms)
- **Hybrid Models**: Combine all approaches with multiple fusion strategies

### Real-time Inference
- **Low-latency pipeline**: Target <100ms inference latency
- **Adaptive features**: Dynamic resolution, frame skipping, batch sizing
- **Performance monitoring**: Real-time statistics and optimization
- **Streaming outputs**: Continuous result processing

## 🚀 Quick Start

### Install Dependencies
```bash
pip install opencv-python  # For video processing (optional)
pip install ffmpeg-python  # For network streams (optional)
```

### Basic Usage
```python
import torch
from adaptiveneuralnetwork.models import create_convlstm_model
from adaptiveneuralnetwork.data import create_webcam_stream

# Create model
model = create_convlstm_model(num_classes=1000)

# Create video stream
stream = create_webcam_stream(camera_id=0, target_width=224)
stream.start_streaming()

# Process video
sequence = stream.get_sequence()  # Get frame sequence tensor
with torch.no_grad():
    predictions = model(sequence.unsqueeze(0))  # Add batch dimension

stream.stop_streaming()
```

### CLI Tools
```bash
# Benchmark webcam with ConvLSTM
python -m adaptiveneuralnetwork.scripts.adaptive_video_benchmark \
    --source webcam --model convlstm --max-frames 500

# Live inference from webcam
python -m adaptiveneuralnetwork.scripts.adaptive_live_infer \
    --source webcam --model transformer

# Benchmark RTSP stream
python -m adaptiveneuralnetwork.scripts.adaptive_video_benchmark \
    --source rtsp://192.168.1.100:554/stream --model hybrid \
    --output results.json
```

## 📊 Performance Benchmarks

| Model | Parameters | Avg Latency | Best Use Case |
|-------|------------|-------------|---------------|
| **Video Transformer** | 938K | ~107ms | Real-time applications |
| **3D CNN** | 2.1M | ~210ms | Balanced speed/accuracy |
| **ConvLSTM** | 2.7M | ~485ms | Temporal memory tasks |
| **Hybrid** | 6.3M | ~800ms | Maximum accuracy |

*Benchmarks on CPU with 224x224 input, 8-frame sequences*

## 🛠️ Advanced Features

### Hybrid Model Fusion
```python
from adaptiveneuralnetwork.models import create_hybrid_model

model = create_hybrid_model(num_classes=1000)

# Different fusion strategies
output_weighted = model(video_input, fusion_mode="weighted")
output_concat = model(video_input, fusion_mode="concat") 
output_ensemble = model(video_input, fusion_mode="ensemble")
```

### Real-time Inference Pipeline
```python
from adaptiveneuralnetwork.core import VideoStreamInference, InferenceConfig

# Configure for low latency
inference_config = InferenceConfig(
    target_latency_ms=50.0,
    enable_adaptive_resolution=True,
    enable_frame_skipping=True
)

# Create pipeline
pipeline = VideoStreamInference(model, inference_config=inference_config)

# Add result handler
def handle_result(result):
    print(f"Frame {result.frame_id}: {result.confidence:.3f} confidence")

pipeline.add_result_callback(handle_result)
pipeline.start_stream_inference("webcam")
```

### Custom Video Processing
```python
from adaptiveneuralnetwork.data import VideoConfig, VideoStreamDataset

config = VideoConfig(
    target_width=224,
    target_height=224,
    sequence_length=16,
    frame_skip=1,       # Skip every other frame
    max_fps=15.0,       # Limit FPS
    buffer_size=32      # Smaller buffer for low latency
)

stream = VideoStreamDataset("path/to/video.mp4", config)
```

## 📚 Documentation and Examples

- **[Complete Guide](docs/VIDEO_STREAMING_GUIDE.md)**: Comprehensive documentation
- **[Interactive Notebook](notebooks/Video_Streaming_and_Temporal_Models.py)**: Step-by-step examples
- **[Demo Script](demo_video_functionality.py)**: Full functionality demonstration

## 🎯 Use Cases

### Action Recognition
```python
# Real-time action recognition from webcam
model = create_video_transformer(num_classes=400)  # Kinetics-400 actions
pipeline = VideoStreamInference(model)
pipeline.start_stream_inference("webcam")
```

### Video Surveillance
```python
# Monitor RTSP security camera
config = VideoConfig(frame_skip=2, sequence_length=8)  # Optimized for surveillance
stream = create_rtsp_stream("rtsp://camera-ip:554/stream", **config.__dict__)
```

### Content Analysis
```python
# Process video files for content classification
stream = create_file_stream("content_video.mp4")
model = create_hybrid_model(num_classes=1000)  # Maximum accuracy
```

## 🔧 Configuration Options

### Video Processing
- **Resolution**: 112x112 (fast) to 320x240 (high quality)
- **Sequence Length**: 4-32 frames for temporal context
- **Frame Rate**: Adaptive FPS control and frame skipping
- **Preprocessing**: Normalization, augmentation, batching

### Model Selection
- **ConvLSTM**: For tasks requiring temporal memory
- **3D CNN**: Best balance of speed and spatial-temporal modeling
- **Transformer**: Fastest inference, good for real-time applications
- **Hybrid**: Combine multiple approaches for maximum accuracy

### Inference Optimization
- **Adaptive Resolution**: Dynamic scaling based on performance
- **Frame Skipping**: Skip frames under high load
- **Batch Processing**: Group frames for throughput optimization
- **Performance Monitoring**: Real-time latency and FPS tracking

## 🐛 Troubleshooting

### Common Issues
1. **"OpenCV not available"**: Install with `pip install opencv-python`
2. **High latency**: Reduce input resolution or enable frame skipping
3. **Memory errors**: Reduce batch size or sequence length
4. **Stream connection failed**: Check network connectivity and stream URL

### Performance Tips
1. Use Video Transformer for real-time applications (fastest)
2. Enable adaptive features for variable load conditions
3. Lower resolution (112x112) for better performance
4. Use GPU acceleration when available
5. Optimize sequence length for your specific use case

## 📈 Future Enhancements

- [ ] Additional video datasets integration (UCF101, Kinetics)
- [ ] Pre-trained model weights for common tasks
- [ ] Mobile-optimized models for edge deployment
- [ ] Video augmentation and data preprocessing pipelines
- [ ] Integration with popular video frameworks

## 🤝 Contributing

Video functionality contributions are welcome! Areas for improvement:
- New temporal model architectures
- Additional video source integrations
- Performance optimizations
- Dataset-specific preprocessing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

The video and streaming capabilities transform the Adaptive Neural Network library into a comprehensive solution for real-time video analysis, suitable for applications ranging from action recognition to surveillance and content analysis.