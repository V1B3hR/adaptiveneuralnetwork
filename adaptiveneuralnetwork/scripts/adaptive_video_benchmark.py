#!/usr/bin/env python3
"""
CLI tool for video benchmarking with Adaptive Neural Network.

This script provides command-line interface for benchmarking video models with:
- Multiple video sources (files, webcam, network streams)
- Various temporal models (ConvLSTM, 3D CNN, Transformer, Hybrid)
- Real-time performance analysis
- Comprehensive reporting
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.video_streaming import VideoConfig, create_video_stream
from models.video_models import create_video_model, VideoModelConfig
from core.video_inference import RealTimeInferenceEngine, InferenceConfig, InferenceResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoBenchmark:
    """Video benchmarking system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create model
        self.model = self._create_model()
        
        # Create video stream
        self.video_stream = None
        
        # Results storage
        self.results = []
        self.start_time = None
        
    def _create_model(self) -> torch.nn.Module:
        """Create video model based on configuration."""
        model_type = self.config.get('model', 'convlstm')
        num_classes = self.config.get('num_classes', 1000)
        
        # Model configuration
        model_config = VideoModelConfig(
            input_channels=3,
            input_height=self.config.get('height', 224),
            input_width=self.config.get('width', 224),
            sequence_length=self.config.get('sequence_length', 16),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.1),
            num_classes=num_classes
        )
        
        model = create_video_model(model_type, model_config)
        logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def _create_video_stream(self, source: str):
        """Create video stream from source."""
        # Video configuration
        video_config = VideoConfig(
            target_width=self.config.get('width', 224),
            target_height=self.config.get('height', 224),
            fps=self.config.get('fps'),
            max_fps=self.config.get('max_fps', 30.0),
            sequence_length=self.config.get('sequence_length', 16),
            frame_skip=self.config.get('frame_skip', 0),
            buffer_size=self.config.get('buffer_size', 64),
            batch_size=self.config.get('batch_size', 8)
        )
        
        # Handle different source types
        if source == 'webcam':
            video_source = 0  # Default webcam
        elif source.startswith('rtsp://') or source.startswith('rtmp://'):
            video_source = source
        elif Path(source).exists():
            video_source = source
        else:
            # Try to parse as integer (camera index)
            try:
                video_source = int(source)
            except ValueError:
                video_source = source
        
        self.video_stream = create_video_stream(video_source, video_config)
        logger.info(f"Created video stream for source: {source}")
    
    def run_benchmark(self, source: str, duration: Optional[float] = None) -> Dict[str, Any]:
        """Run benchmark on video source."""
        logger.info(f"Starting benchmark on {source}")
        
        # Setup
        self._create_video_stream(source)
        
        # Inference configuration
        inference_config = InferenceConfig(
            target_latency_ms=self.config.get('target_latency_ms', 100.0),
            max_latency_ms=self.config.get('max_latency_ms', 200.0),
            target_fps=self.config.get('target_fps', 15.0),
            enable_adaptive_resolution=self.config.get('adaptive_resolution', True),
            enable_frame_skipping=self.config.get('frame_skipping', True),
            enable_batching=self.config.get('batching', True),
            max_batch_size=self.config.get('batch_size', 4)
        )
        
        # Create inference engine
        inference_engine = RealTimeInferenceEngine(
            self.model, inference_config, self.device
        )
        
        try:
            # Start streaming
            if not self.video_stream.start_streaming():
                raise RuntimeError(f"Failed to start video stream from {source}")
            
            # Start inference
            inference_engine.start_inference()
            
            # Benchmark loop
            self.start_time = time.time()
            self.results = []
            
            # Result callback
            def result_callback(result: InferenceResult):
                self.results.append({
                    'frame_id': result.frame_id,
                    'timestamp': result.timestamp,
                    'confidence': result.confidence,
                    'latency_ms': result.latency_ms,
                    'model_type': result.model_type,
                    'metadata': result.metadata
                })
            
            # Process for specified duration
            target_frames = self.config.get('max_frames', 1000)
            processed_frames = 0
            
            logger.info(f"Processing up to {target_frames} frames...")
            
            while processed_frames < target_frames:
                # Get frame sequence
                sequence = self.video_stream.get_sequence(timeout=1.0)
                if sequence is None:
                    logger.warning("No sequence available, continuing...")
                    time.sleep(0.1)
                    continue
                
                # Submit for inference
                inference_engine.submit_frame_sequence(sequence)
                
                # Get result
                result = inference_engine.get_result(timeout=1.0)
                if result is not None:
                    result_callback(result)
                    processed_frames += 1
                    
                    # Progress logging
                    if processed_frames % 50 == 0:
                        elapsed = time.time() - self.start_time
                        fps = processed_frames / elapsed
                        logger.info(f"Processed {processed_frames} frames, {fps:.1f} FPS")
                
                # Check duration limit
                if duration is not None:
                    if time.time() - self.start_time >= duration:
                        break
            
            # Generate final statistics
            benchmark_stats = self._generate_stats(inference_engine)
            
            logger.info("Benchmark completed successfully")
            return benchmark_stats
            
        finally:
            # Cleanup
            if self.video_stream:
                self.video_stream.stop_streaming()
            inference_engine.stop_inference()
    
    def _generate_stats(self, inference_engine: RealTimeInferenceEngine) -> Dict[str, Any]:
        """Generate comprehensive benchmark statistics."""
        total_time = time.time() - self.start_time
        
        # Basic statistics
        stats = {
            'benchmark_info': {
                'total_time_seconds': total_time,
                'total_frames': len(self.results),
                'average_fps': len(self.results) / total_time if total_time > 0 else 0,
                'model_type': self.config.get('model', 'unknown'),
                'source': self.config.get('source', 'unknown'),
                'device': str(self.device)
            }
        }
        
        # Performance statistics from inference engine
        engine_stats = inference_engine.get_performance_stats()
        stats['performance'] = engine_stats
        
        # Result analysis
        if self.results:
            latencies = [r['latency_ms'] for r in self.results]
            confidences = [r['confidence'] for r in self.results]
            
            stats['inference_analysis'] = {
                'latency_stats': {
                    'mean': np.mean(latencies),
                    'median': np.median(latencies),
                    'std': np.std(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99)
                },
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'median': np.median(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
            }
        
        # Configuration used
        stats['configuration'] = self.config
        
        return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Video benchmarking tool for Adaptive Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark webcam with ConvLSTM
  python adaptive_video_benchmark.py --source webcam --model convlstm --epochs 5

  # Benchmark RTSP stream with 3D CNN
  python adaptive_video_benchmark.py --source rtsp://192.168.1.100:554/stream --model conv3d --output results.json

  # Benchmark video file with Transformer
  python adaptive_video_benchmark.py --source video.mp4 --model transformer --max-frames 500

  # Benchmark with hybrid model and custom settings
  python adaptive_video_benchmark.py --source webcam --model hybrid --batch-size 4 --target-fps 20
        """
    )
    
    # Source and model
    parser.add_argument(
        '--source',
        required=True,
        help='Video source: "webcam", file path, or RTSP/RTMP URL'
    )
    
    parser.add_argument(
        '--model',
        choices=['convlstm', 'conv3d', 'transformer', 'hybrid'],
        default='convlstm',
        help='Model type to benchmark (default: convlstm)'
    )
    
    # Performance settings
    parser.add_argument(
        '--max-frames',
        type=int,
        default=1000,
        help='Maximum frames to process (default: 1000)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        help='Maximum duration in seconds (overrides --max-frames)'
    )
    
    parser.add_argument(
        '--target-fps',
        type=float,
        default=15.0,
        help='Target inference FPS (default: 15.0)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for inference (default: 4)'
    )
    
    # Model configuration
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1000,
        help='Number of output classes (default: 1000)'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=16,
        help='Frame sequence length (default: 16)'
    )
    
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden dimension size (default: 256)'
    )
    
    # Video processing
    parser.add_argument(
        '--width',
        type=int,
        default=224,
        help='Frame width (default: 224)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=224,
        help='Frame height (default: 224)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        help='Target video FPS (default: source FPS)'
    )
    
    # Adaptive features
    parser.add_argument(
        '--no-adaptive-resolution',
        action='store_true',
        help='Disable adaptive resolution'
    )
    
    parser.add_argument(
        '--no-frame-skipping',
        action='store_true',
        help='Disable frame skipping'
    )
    
    parser.add_argument(
        '--no-batching',
        action='store_true',
        help='Disable batching'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce logging output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Build configuration
    config = {
        'source': args.source,
        'model': args.model,
        'max_frames': args.max_frames,
        'target_fps': args.target_fps,
        'batch_size': args.batch_size,
        'num_classes': args.num_classes,
        'sequence_length': args.sequence_length,
        'hidden_dim': args.hidden_dim,
        'width': args.width,
        'height': args.height,
        'adaptive_resolution': not args.no_adaptive_resolution,
        'frame_skipping': not args.no_frame_skipping,
        'batching': not args.no_batching
    }
    
    if args.fps:
        config['fps'] = args.fps
    
    try:
        # Run benchmark
        benchmark = VideoBenchmark(config)
        results = benchmark.run_benchmark(args.source, args.duration)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_path}")
        else:
            # Print key statistics
            benchmark_info = results['benchmark_info']
            performance = results.get('performance', {})
            inference_analysis = results.get('inference_analysis', {})
            
            print("\n" + "="*60)
            print("VIDEO BENCHMARK RESULTS")
            print("="*60)
            print(f"Model: {benchmark_info['model_type']}")
            print(f"Source: {benchmark_info['source']}")
            print(f"Device: {benchmark_info['device']}")
            print(f"Total Time: {benchmark_info['total_time_seconds']:.2f}s")
            print(f"Total Frames: {benchmark_info['total_frames']}")
            print(f"Average FPS: {benchmark_info['average_fps']:.2f}")
            
            if 'latency_ms' in performance:
                latency_stats = performance['latency_ms']
                print(f"\nLatency Statistics:")
                print(f"  Mean: {latency_stats['mean']:.2f}ms")
                print(f"  Median: {latency_stats['median']:.2f}ms")
                print(f"  P95: {latency_stats.get('p95', 'N/A')}")
                print(f"  P99: {latency_stats.get('p99', 'N/A')}")
            
            if 'fps' in performance:
                fps_stats = performance['fps']
                print(f"\nThroughput:")
                print(f"  Mean FPS: {fps_stats['mean']:.2f}")
                print(f"  Median FPS: {fps_stats['median']:.2f}")
            
            if 'confidence' in performance:
                conf_stats = performance['confidence']
                print(f"\nConfidence:")
                print(f"  Mean: {conf_stats['mean']:.3f}")
                print(f"  Min: {conf_stats['min']:.3f}")
                print(f"  Max: {conf_stats['max']:.3f}")
        
        print("\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()