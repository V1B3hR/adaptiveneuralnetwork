"""
Real-time video inference pipeline for Adaptive Neural Network.

This module provides low-latency, real-time inference capabilities with:
- Live feedback and adaptive computation
- Frame skipping and dynamic resolution
- Streaming output handling
- Performance monitoring and optimization
"""

import logging
import queue
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..data.video_streaming import FrameInfo, VideoConfig, VideoStreamDataset

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for real-time inference."""

    # Performance targets
    target_latency_ms: float = 100.0  # Target inference latency
    max_latency_ms: float = 200.0  # Maximum acceptable latency
    target_fps: float = 15.0  # Target inference FPS

    # Adaptive computation
    enable_adaptive_resolution: bool = True
    min_resolution: int = 112  # Minimum resolution for adaptive scaling
    max_resolution: int = 224  # Maximum resolution
    adaptive_threshold: float = 0.8  # Confidence threshold for resolution adaptation

    # Frame processing
    enable_frame_skipping: bool = True
    max_skip_frames: int = 3  # Maximum frames to skip
    enable_batching: bool = True
    max_batch_size: int = 4

    # Output streaming
    output_buffer_size: int = 32
    enable_output_streaming: bool = True

    # Monitoring
    stats_window_size: int = 100  # Rolling window for statistics


@dataclass
class InferenceResult:
    """Result from video inference."""

    frame_id: int
    timestamp: float
    predictions: torch.Tensor
    confidence: float
    latency_ms: float
    model_type: str
    frame_info: Optional[FrameInfo] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """Monitor and track inference performance metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = []
        self.throughput_times = []
        self.confidence_scores = []
        self.frame_skips = []
        self.resolution_changes = []

    def record_inference(self, result: InferenceResult):
        """Record inference metrics."""
        self.latencies.append(result.latency_ms)
        self.confidence_scores.append(result.confidence)

        # Keep only recent measurements
        if len(self.latencies) > self.window_size:
            self.latencies = self.latencies[-self.window_size :]
            self.confidence_scores = self.confidence_scores[-self.window_size :]

    def record_throughput(self, processing_time: float):
        """Record throughput timing."""
        self.throughput_times.append(processing_time)
        if len(self.throughput_times) > self.window_size:
            self.throughput_times = self.throughput_times[-self.window_size :]

    def record_frame_skip(self, num_skipped: int):
        """Record frame skipping."""
        self.frame_skips.append(num_skipped)
        if len(self.frame_skips) > self.window_size:
            self.frame_skips = self.frame_skips[-self.window_size :]

    def record_resolution_change(self, old_res: int, new_res: int):
        """Record resolution adaptation."""
        self.resolution_changes.append((old_res, new_res, time.time()))

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}

        if self.latencies:
            stats["latency_ms"] = {
                "mean": statistics.mean(self.latencies),
                "median": statistics.median(self.latencies),
                "min": min(self.latencies),
                "max": max(self.latencies),
                "std": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
            }

        if self.throughput_times:
            fps = [1.0 / t for t in self.throughput_times if t > 0]
            if fps:
                stats["fps"] = {"mean": statistics.mean(fps), "median": statistics.median(fps)}

        if self.confidence_scores:
            stats["confidence"] = {
                "mean": statistics.mean(self.confidence_scores),
                "min": min(self.confidence_scores),
                "max": max(self.confidence_scores),
            }

        if self.frame_skips:
            stats["frame_skips"] = {
                "total": sum(self.frame_skips),
                "avg_per_inference": statistics.mean(self.frame_skips),
            }

        stats["resolution_changes"] = len(self.resolution_changes)

        return stats


class AdaptiveProcessor:
    """Adaptive processing for dynamic optimization."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.current_resolution = config.max_resolution
        self.consecutive_low_confidence = 0
        self.consecutive_high_latency = 0

    def should_skip_frame(self, current_latency: float, queue_size: int) -> int:
        """Determine how many frames to skip based on performance."""
        if not self.config.enable_frame_skipping:
            return 0

        skip_frames = 0

        # Skip frames if latency is too high
        if current_latency > self.config.max_latency_ms:
            skip_frames = min(
                self.config.max_skip_frames, int(current_latency / self.config.target_latency_ms)
            )

        # Skip frames if queue is backing up
        if queue_size > 5:
            skip_frames = max(skip_frames, min(2, queue_size // 3))

        return skip_frames

    def adapt_resolution(self, confidence: float, latency: float) -> int:
        """Adapt resolution based on confidence and latency."""
        if not self.config.enable_adaptive_resolution:
            return self.current_resolution

        # Track performance patterns
        if confidence < self.config.adaptive_threshold:
            self.consecutive_low_confidence += 1
        else:
            self.consecutive_low_confidence = 0

        if latency > self.config.target_latency_ms:
            self.consecutive_high_latency += 1
        else:
            self.consecutive_high_latency = 0

        # Decrease resolution if consistently poor performance
        if self.consecutive_high_latency >= 3 or (
            self.consecutive_low_confidence >= 5 and latency > self.config.target_latency_ms * 0.8
        ):
            new_resolution = max(self.config.min_resolution, self.current_resolution - 32)
            if new_resolution != self.current_resolution:
                logger.info(f"Decreasing resolution: {self.current_resolution} -> {new_resolution}")
                self.current_resolution = new_resolution
                self.consecutive_high_latency = 0

        # Increase resolution if consistently good performance
        elif (
            self.consecutive_high_latency == 0
            and self.consecutive_low_confidence == 0
            and latency < self.config.target_latency_ms * 0.5
        ):
            new_resolution = min(self.config.max_resolution, self.current_resolution + 32)
            if new_resolution != self.current_resolution:
                logger.info(f"Increasing resolution: {self.current_resolution} -> {new_resolution}")
                self.current_resolution = new_resolution

        return self.current_resolution

    def get_batch_size(self, current_latency: float) -> int:
        """Determine optimal batch size."""
        if not self.config.enable_batching:
            return 1

        # Reduce batch size if latency is high
        if current_latency > self.config.target_latency_ms:
            return 1
        elif current_latency > self.config.target_latency_ms * 0.8:
            return min(2, self.config.max_batch_size)
        else:
            return self.config.max_batch_size


class RealTimeInferenceEngine:
    """Real-time inference engine for video streams."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.config = config or InferenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Components
        self.performance_monitor = PerformanceMonitor(self.config.stats_window_size)
        self.adaptive_processor = AdaptiveProcessor(self.config)

        # Threading and queues
        self.input_queue = queue.Queue(maxsize=16)
        self.output_queue = queue.Queue(maxsize=self.config.output_buffer_size)
        self.inference_thread = None
        self.is_running = False

        # State
        self.frame_count = 0
        self.last_inference_time = time.time()

    def start_inference(self):
        """Start real-time inference in background thread."""
        if self.is_running:
            return

        self.is_running = True
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()
        logger.info("Started real-time inference engine")

    def stop_inference(self):
        """Stop real-time inference."""
        self.is_running = False

        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
            self.inference_thread = None

        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Stopped real-time inference engine")

    def submit_frame_sequence(self, sequence: torch.Tensor, frame_info: Optional[FrameInfo] = None):
        """Submit frame sequence for inference."""
        try:
            self.input_queue.put_nowait((sequence, frame_info, time.time()))
        except queue.Full:
            # Drop oldest frame if queue is full
            try:
                self.input_queue.get_nowait()
                self.input_queue.put_nowait((sequence, frame_info, time.time()))
            except queue.Empty:
                pass

    def get_result(self, timeout: float = 0.1) -> Optional[InferenceResult]:
        """Get inference result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _inference_worker(self):
        """Background worker for inference processing."""
        batch_sequences = []
        batch_infos = []
        batch_timestamps = []

        while self.is_running:
            try:
                # Collect batch
                target_batch_size = self.adaptive_processor.get_batch_size(
                    self.performance_monitor.latencies[-1]
                    if self.performance_monitor.latencies
                    else 0
                )

                # Get sequences for batch
                timeout = 0.1
                start_batch_time = time.time()

                while (
                    len(batch_sequences) < target_batch_size
                    and (time.time() - start_batch_time) < 0.05
                ):
                    try:
                        sequence, frame_info, submit_time = self.input_queue.get(timeout=timeout)
                        batch_sequences.append(sequence)
                        batch_infos.append(frame_info)
                        batch_timestamps.append(submit_time)
                    except queue.Empty:
                        break

                if not batch_sequences:
                    continue

                # Run inference
                inference_start = time.time()
                results = self._run_batch_inference(batch_sequences, batch_infos, batch_timestamps)
                inference_time = (time.time() - inference_start) * 1000  # Convert to ms

                # Submit results
                for result in results:
                    try:
                        self.output_queue.put_nowait(result)
                    except queue.Full:
                        # Remove oldest result
                        try:
                            self.output_queue.get_nowait()
                            self.output_queue.put_nowait(result)
                        except queue.Empty:
                            pass

                # Record performance
                if results:
                    avg_latency = sum(r.latency_ms for r in results) / len(results)
                    self.performance_monitor.record_throughput(time.time() - inference_start)

                # Clear batch
                batch_sequences.clear()
                batch_infos.clear()
                batch_timestamps.clear()

            except Exception as e:
                logger.error(f"Error in inference worker: {e}")
                time.sleep(0.01)

    def _run_batch_inference(
        self,
        sequences: List[torch.Tensor],
        frame_infos: List[Optional[FrameInfo]],
        timestamps: List[float],
    ) -> List[InferenceResult]:
        """Run inference on a batch of sequences."""
        results = []

        try:
            # Stack sequences into batch
            if len(sequences) == 1:
                batch_tensor = sequences[0].unsqueeze(0).to(self.device)
            else:
                # Ensure all sequences have same shape
                target_shape = sequences[0].shape
                valid_sequences = [seq for seq in sequences if seq.shape == target_shape]
                if not valid_sequences:
                    return results

                batch_tensor = torch.stack(valid_sequences, dim=0).to(self.device)

            # Run inference
            inference_start = time.time()
            with torch.no_grad():
                predictions = self.model(batch_tensor)
            inference_end = time.time()

            latency_ms = (inference_end - inference_start) * 1000

            # Process results
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)

            for i in range(predictions.size(0)):
                if i < len(frame_infos):
                    frame_info = frame_infos[i]
                    submit_time = timestamps[i]

                    # Calculate confidence (max probability)
                    pred_probs = torch.softmax(predictions[i], dim=0)
                    confidence = torch.max(pred_probs).item()

                    # Calculate total latency (including queue time)
                    total_latency = (inference_end - submit_time) * 1000

                    result = InferenceResult(
                        frame_id=self.frame_count,
                        timestamp=inference_end,
                        predictions=predictions[i].cpu(),
                        confidence=confidence,
                        latency_ms=total_latency,
                        model_type=self.model.__class__.__name__,
                        frame_info=frame_info,
                        metadata={
                            "inference_latency_ms": latency_ms,
                            "queue_latency_ms": total_latency - latency_ms,
                            "batch_size": predictions.size(0),
                        },
                    )

                    results.append(result)
                    self.performance_monitor.record_inference(result)
                    self.frame_count += 1

                    # Adaptive processing
                    new_resolution = self.adaptive_processor.adapt_resolution(
                        confidence, total_latency
                    )
                    skip_frames = self.adaptive_processor.should_skip_frame(
                        total_latency, self.input_queue.qsize()
                    )

                    if skip_frames > 0:
                        self.performance_monitor.record_frame_skip(skip_frames)

        except Exception as e:
            logger.error(f"Error during batch inference: {e}")

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_monitor.get_stats()
        stats.update(
            {
                "queue_sizes": {
                    "input": self.input_queue.qsize(),
                    "output": self.output_queue.qsize(),
                },
                "current_resolution": self.adaptive_processor.current_resolution,
                "total_frames_processed": self.frame_count,
                "is_running": self.is_running,
            }
        )
        return stats


class VideoStreamInference:
    """Complete video stream inference pipeline."""

    def __init__(
        self,
        model: nn.Module,
        video_config: Optional[VideoConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.video_config = video_config or VideoConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Components
        self.inference_engine = RealTimeInferenceEngine(model, self.inference_config, self.device)
        self.video_stream = None

        # Results handling
        self.result_callbacks = []
        self.results_thread = None
        self.is_streaming = False

    def add_result_callback(self, callback: Callable[[InferenceResult], None]):
        """Add callback for handling inference results."""
        self.result_callbacks.append(callback)

    def start_stream_inference(self, source: Union[str, int]) -> bool:
        """Start inference on video stream."""
        if self.is_streaming:
            return True

        # Create video stream
        self.video_stream = VideoStreamDataset(source, self.video_config)

        if not self.video_stream.start_streaming():
            logger.error(f"Failed to start video stream from {source}")
            return False

        # Start inference engine
        self.inference_engine.start_inference()

        # Start processing threads
        self.is_streaming = True
        self.results_thread = threading.Thread(target=self._results_handler, daemon=True)
        self.results_thread.start()

        # Start stream processing
        threading.Thread(target=self._stream_processor, daemon=True).start()

        logger.info(f"Started stream inference from {source}")
        return True

    def stop_stream_inference(self):
        """Stop stream inference."""
        self.is_streaming = False

        if self.video_stream:
            self.video_stream.stop_streaming()
            self.video_stream = None

        if self.inference_engine:
            self.inference_engine.stop_inference()

        if self.results_thread:
            self.results_thread.join(timeout=2.0)
            self.results_thread = None

        logger.info("Stopped stream inference")

    def _stream_processor(self):
        """Process video stream and submit for inference."""
        while self.is_streaming and self.video_stream:
            try:
                # Get frame sequence
                sequence = self.video_stream.get_sequence(timeout=0.1)
                if sequence is not None:
                    # Submit for inference
                    self.inference_engine.submit_frame_sequence(sequence)

            except Exception as e:
                logger.error(f"Error in stream processor: {e}")
                time.sleep(0.01)

    def _results_handler(self):
        """Handle inference results."""
        while self.is_streaming:
            try:
                result = self.inference_engine.get_result(timeout=0.1)
                if result is not None:
                    # Call registered callbacks
                    for callback in self.result_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Error in result callback: {e}")

            except Exception as e:
                logger.error(f"Error in results handler: {e}")
                time.sleep(0.01)

    def get_stream_info(self) -> Dict[str, Any]:
        """Get comprehensive stream and inference information."""
        info = {}

        if self.video_stream:
            info["video_stream"] = self.video_stream.get_info()

        if self.inference_engine:
            info["inference"] = self.inference_engine.get_performance_stats()

        info["is_streaming"] = self.is_streaming
        info["num_callbacks"] = len(self.result_callbacks)

        return info


def create_stream_inference(
    model: nn.Module,
    source: Union[str, int],
    video_config: Optional[VideoConfig] = None,
    inference_config: Optional[InferenceConfig] = None,
    device: Optional[torch.device] = None,
) -> VideoStreamInference:
    """Factory function to create video stream inference pipeline."""
    return VideoStreamInference(model, video_config, inference_config, device)
