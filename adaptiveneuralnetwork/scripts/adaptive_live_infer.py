#!/usr/bin/env python3
"""
CLI tool for live video inference with Adaptive Neural Network.

This script provides real-time video inference with:
- Live video sources (webcam, network streams)
- Real-time prediction display
- Performance monitoring
- Adaptive computation
- Output streaming and recording
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.video_streaming import VideoConfig
from models.video_models import VideoModelConfig, create_video_model

from core.video_inference import InferenceConfig, InferenceResult, VideoStreamInference

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LiveInferenceApp:
    """Live video inference application."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create model
        self.model = self._create_model()

        # Inference pipeline
        self.inference_pipeline = None

        # State management
        self.is_running = False
        self.results_history = []
        self.stats_history = []

        # Output handling
        self.output_handlers = []

        # Class labels (if provided)
        self.class_labels = self._load_class_labels()

    def _create_model(self) -> torch.nn.Module:
        """Create video model based on configuration."""
        model_type = self.config.get("model", "convlstm")
        num_classes = self.config.get("num_classes", 1000)

        # Model configuration
        model_config = VideoModelConfig(
            input_channels=3,
            input_height=self.config.get("height", 224),
            input_width=self.config.get("width", 224),
            sequence_length=self.config.get("sequence_length", 16),
            hidden_dim=self.config.get("hidden_dim", 256),
            num_layers=self.config.get("num_layers", 2),
            dropout=self.config.get("dropout", 0.1),
            num_classes=num_classes,
        )

        model = create_video_model(model_type, model_config)
        logger.info(
            f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters"
        )

        # Load pretrained weights if specified
        if self.config.get("weights"):
            try:
                checkpoint = torch.load(self.config["weights"], map_location=self.device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded weights from {self.config['weights']}")
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}")

        return model

    def _load_class_labels(self) -> Optional[List[str]]:
        """Load class labels if provided."""
        labels_path = self.config.get("labels")
        if not labels_path:
            return None

        try:
            labels_path = Path(labels_path)
            if labels_path.suffix == ".json":
                with open(labels_path) as f:
                    labels_data = json.load(f)
                    if isinstance(labels_data, dict):
                        # Assume format: {"0": "label0", "1": "label1", ...}
                        return [
                            labels_data[str(i)] for i in sorted(int(k) for k in labels_data.keys())
                        ]
                    else:
                        return labels_data
            else:
                # Text file, one label per line
                with open(labels_path) as f:
                    return [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.warning(f"Failed to load class labels: {e}")
            return None

    def _setup_inference_pipeline(self, source: str):
        """Setup video inference pipeline."""
        # Video configuration
        video_config = VideoConfig(
            target_width=self.config.get("width", 224),
            target_height=self.config.get("height", 224),
            fps=self.config.get("fps"),
            max_fps=self.config.get("max_fps", 30.0),
            sequence_length=self.config.get("sequence_length", 16),
            frame_skip=self.config.get("frame_skip", 0),
            buffer_size=self.config.get("buffer_size", 64),
            batch_size=self.config.get("batch_size", 8),
        )

        # Inference configuration
        inference_config = InferenceConfig(
            target_latency_ms=self.config.get("target_latency_ms", 100.0),
            max_latency_ms=self.config.get("max_latency_ms", 200.0),
            target_fps=self.config.get("target_fps", 15.0),
            enable_adaptive_resolution=self.config.get("adaptive_resolution", True),
            enable_frame_skipping=self.config.get("frame_skipping", True),
            enable_batching=self.config.get("batching", True),
            max_batch_size=self.config.get("batch_size", 4),
        )

        # Create inference pipeline
        self.inference_pipeline = VideoStreamInference(
            self.model, video_config, inference_config, self.device
        )

        # Add result callback
        self.inference_pipeline.add_result_callback(self._handle_inference_result)

        logger.info("Setup inference pipeline")

    def _handle_inference_result(self, result: InferenceResult):
        """Handle inference result."""
        # Store in history
        self.results_history.append(result)

        # Limit history size
        if len(self.results_history) > 1000:
            self.results_history = self.results_history[-1000:]

        # Process result
        self._process_result(result)

        # Call output handlers
        for handler in self.output_handlers:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Error in output handler: {e}")

    def _process_result(self, result: InferenceResult):
        """Process and display inference result."""
        # Get top predictions
        predictions = torch.softmax(result.predictions, dim=0)
        top_k = min(5, len(predictions))
        top_values, top_indices = torch.topk(predictions, top_k)

        # Create display text
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(result.timestamp))

        print(f"\n[{timestamp_str}] Frame {result.frame_id}")
        print(f"Confidence: {result.confidence:.3f}, Latency: {result.latency_ms:.1f}ms")

        # Show top predictions
        print("Top predictions:")
        for i, (idx, value) in enumerate(zip(top_indices, top_values)):
            class_idx = idx.item()
            confidence = value.item()

            if self.class_labels and class_idx < len(self.class_labels):
                class_name = self.class_labels[class_idx]
            else:
                class_name = f"Class_{class_idx}"

            print(f"  {i + 1}. {class_name}: {confidence:.3f}")

    def add_output_handler(self, handler):
        """Add custom output handler."""
        self.output_handlers.append(handler)

    def start_live_inference(self, source: str) -> bool:
        """Start live inference."""
        logger.info(f"Starting live inference from {source}")

        try:
            # Setup pipeline
            self._setup_inference_pipeline(source)

            # Handle different source types
            if source == "webcam":
                video_source = 0  # Default webcam
            elif source.startswith("rtsp://") or source.startswith("rtmp://"):
                video_source = source
            elif Path(source).exists():
                video_source = source
            else:
                try:
                    video_source = int(source)
                except ValueError:
                    video_source = source

            # Start inference
            if not self.inference_pipeline.start_stream_inference(video_source):
                logger.error("Failed to start stream inference")
                return False

            self.is_running = True

            # Start statistics monitoring
            if not self.config.get("quiet", False):
                stats_thread = threading.Thread(target=self._stats_monitor, daemon=True)
                stats_thread.start()

            logger.info("Live inference started. Press Ctrl+C to stop.")
            return True

        except Exception as e:
            logger.error(f"Failed to start live inference: {e}")
            return False

    def stop_live_inference(self):
        """Stop live inference."""
        self.is_running = False

        if self.inference_pipeline:
            self.inference_pipeline.stop_stream_inference()

        logger.info("Live inference stopped")

    def _stats_monitor(self):
        """Monitor and display statistics."""
        last_stats_time = time.time()

        while self.is_running:
            time.sleep(5)  # Update every 5 seconds

            if self.inference_pipeline:
                try:
                    stats = self.inference_pipeline.get_stream_info()
                    current_time = time.time()

                    # Store stats
                    stats["timestamp"] = current_time
                    self.stats_history.append(stats)

                    # Limit history
                    if len(self.stats_history) > 100:
                        self.stats_history = self.stats_history[-100:]

                    # Display stats
                    if not self.config.get("quiet", False):
                        self._display_stats(stats)

                    last_stats_time = current_time

                except Exception as e:
                    logger.error(f"Error in stats monitor: {e}")

    def _display_stats(self, stats: Dict[str, Any]):
        """Display performance statistics."""
        print("\n" + "=" * 50)
        print("PERFORMANCE STATISTICS")
        print("=" * 50)

        # Video stream stats
        if "video_stream" in stats:
            vs = stats["video_stream"]
            if "buffer_sizes" in vs:
                buf = vs["buffer_sizes"]
                print(
                    f"Buffer sizes - Frames: {buf.get('frames', 0)}, Sequences: {buf.get('sequences', 0)}"
                )

        # Inference stats
        if "inference" in stats:
            inf = stats["inference"]

            if "latency_ms" in inf:
                lat = inf["latency_ms"]
                print(f"Latency - Mean: {lat['mean']:.1f}ms, P95: {lat.get('max', 'N/A')}")

            if "fps" in inf:
                fps = inf["fps"]
                print(f"FPS - Mean: {fps['mean']:.1f}")

            if "confidence" in inf:
                conf = inf["confidence"]
                print(f"Confidence - Mean: {conf['mean']:.3f}")

            if "queue_sizes" in inf:
                qs = inf["queue_sizes"]
                print(f"Queue sizes - Input: {qs.get('input', 0)}, Output: {qs.get('output', 0)}")

            print(f"Total frames: {inf.get('total_frames_processed', 0)}")
            print(f"Current resolution: {inf.get('current_resolution', 'N/A')}")

    def get_final_report(self) -> Dict[str, Any]:
        """Generate final performance report."""
        if not self.results_history:
            return {"error": "No inference results available"}

        # Aggregate statistics
        latencies = [r.latency_ms for r in self.results_history]
        confidences = [r.confidence for r in self.results_history]

        report = {
            "summary": {
                "total_frames": len(self.results_history),
                "total_time_seconds": self.results_history[-1].timestamp
                - self.results_history[0].timestamp,
                "model_type": self.config.get("model", "unknown"),
                "device": str(self.device),
            },
            "latency_stats": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "std": np.std(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
            },
            "confidence_stats": {
                "mean": np.mean(confidences),
                "median": np.median(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
            },
            "configuration": self.config,
        }

        # Calculate average FPS
        if len(self.results_history) > 1:
            total_time = self.results_history[-1].timestamp - self.results_history[0].timestamp
            report["summary"]["average_fps"] = len(self.results_history) / total_time

        return report


def create_output_handler(output_type: str, config: Dict[str, Any]):
    """Create output handler based on type."""
    if output_type == "json_log":
        output_file = config.get("output_file", "inference_log.jsonl")

        def json_handler(result: InferenceResult):
            log_entry = {
                "timestamp": result.timestamp,
                "frame_id": result.frame_id,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "predictions": result.predictions.tolist(),
            }

            with open(output_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return json_handler

    elif output_type == "csv_log":
        output_file = config.get("output_file", "inference_log.csv")

        # Write header if file doesn't exist
        if not Path(output_file).exists():
            with open(output_file, "w") as f:
                f.write("timestamp,frame_id,confidence,latency_ms,top_class,top_confidence\n")

        def csv_handler(result: InferenceResult):
            predictions = torch.softmax(result.predictions, dim=0)
            top_value, top_idx = torch.max(predictions, dim=0)

            with open(output_file, "a") as f:
                f.write(
                    f"{result.timestamp},{result.frame_id},{result.confidence},"
                    f"{result.latency_ms},{top_idx.item()},{top_value.item()}\n"
                )

        return csv_handler

    else:
        raise ValueError(f"Unknown output type: {output_type}")


def signal_handler(signum, frame):
    """Handle interrupt signals."""
    print("\nReceived interrupt signal. Stopping inference...")
    sys.exit(0)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Live video inference tool for Adaptive Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live inference from webcam
  python adaptive_live_infer.py --source webcam --model convlstm

  # Live inference from RTSP stream
  python adaptive_live_infer.py --source rtsp://192.168.1.100:554/stream --model hybrid

  # Live inference with custom settings and logging
  python adaptive_live_infer.py --source webcam --model transformer --output-format json --output-file results.jsonl

  # Live inference with pretrained weights and class labels
  python adaptive_live_infer.py --source webcam --weights model.pth --labels imagenet_labels.txt
        """,
    )

    # Source and model
    parser.add_argument(
        "--source", required=True, help='Video source: "webcam", file path, or RTSP/RTMP URL'
    )

    parser.add_argument(
        "--model",
        choices=["convlstm", "conv3d", "transformer", "hybrid"],
        default="convlstm",
        help="Model type (default: convlstm)",
    )

    parser.add_argument("--weights", type=str, help="Path to pretrained model weights")

    parser.add_argument(
        "--labels", type=str, help="Path to class labels file (JSON or text format)"
    )

    # Model configuration
    parser.add_argument(
        "--num-classes", type=int, default=1000, help="Number of output classes (default: 1000)"
    )

    parser.add_argument(
        "--sequence-length", type=int, default=16, help="Frame sequence length (default: 16)"
    )

    parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension size (default: 256)"
    )

    # Video processing
    parser.add_argument("--width", type=int, default=224, help="Frame width (default: 224)")

    parser.add_argument("--height", type=int, default=224, help="Frame height (default: 224)")

    parser.add_argument("--fps", type=float, help="Target video FPS (default: source FPS)")

    # Performance settings
    parser.add_argument(
        "--target-fps", type=float, default=15.0, help="Target inference FPS (default: 15.0)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for inference (default: 4)"
    )

    parser.add_argument(
        "--target-latency",
        type=float,
        default=100.0,
        help="Target latency in milliseconds (default: 100.0)",
    )

    # Adaptive features
    parser.add_argument(
        "--no-adaptive-resolution", action="store_true", help="Disable adaptive resolution"
    )

    parser.add_argument("--no-frame-skipping", action="store_true", help="Disable frame skipping")

    parser.add_argument("--no-batching", action="store_true", help="Disable batching")

    # Output options
    parser.add_argument(
        "--output-format", choices=["json", "csv"], help="Output format for logging results"
    )

    parser.add_argument(
        "--output-file", type=str, help="Output file path (default: inference_log.{format})"
    )

    parser.add_argument("--report-file", type=str, help="Final report output file (JSON format)")

    # Display options
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce output (no statistics display)"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Build configuration
    config = {
        "source": args.source,
        "model": args.model,
        "num_classes": args.num_classes,
        "sequence_length": args.sequence_length,
        "hidden_dim": args.hidden_dim,
        "width": args.width,
        "height": args.height,
        "target_fps": args.target_fps,
        "batch_size": args.batch_size,
        "target_latency_ms": args.target_latency,
        "adaptive_resolution": not args.no_adaptive_resolution,
        "frame_skipping": not args.no_frame_skipping,
        "batching": not args.no_batching,
        "quiet": args.quiet,
    }

    if args.fps:
        config["fps"] = args.fps
    if args.weights:
        config["weights"] = args.weights
    if args.labels:
        config["labels"] = args.labels

    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Create application
        app = LiveInferenceApp(config)

        # Setup output handler if specified
        if args.output_format:
            output_config = {
                "output_file": args.output_file or f"inference_log.{args.output_format}l"
            }
            handler = create_output_handler(f"{args.output_format}_log", output_config)
            app.add_output_handler(handler)

        # Start inference
        if not app.start_live_inference(args.source):
            sys.exit(1)

        # Keep running until interrupted
        try:
            while app.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        # Stop inference
        app.stop_live_inference()

        # Generate final report
        if args.report_file:
            report = app.get_final_report()
            with open(args.report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Final report saved to {args.report_file}")

        print("\nLive inference completed!")

    except Exception as e:
        logger.error(f"Live inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
