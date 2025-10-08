"""
Video streaming and frame loading infrastructure for Adaptive Neural Network.

This module provides comprehensive video frame loading capabilities supporting:
- Local video files
- Webcam capture  
- Network streams (RTSP/RTMP)
- Real-time preprocessing and batching
- Frame skipping and temporal sampling
"""

import logging
import queue
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torchvision import transforms

# Try importing video processing libraries
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False
    warnings.warn("OpenCV not available. Video streaming will be limited.", stacklevel=2)

try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    ffmpeg = None
    HAS_FFMPEG = False
    warnings.warn("ffmpeg-python not available. Network streaming may be limited.", stacklevel=2)

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Configuration for video streaming and processing."""
    # Frame processing
    target_width: int = 224
    target_height: int = 224
    fps: float | None = None  # Target FPS, None for source FPS
    max_fps: float = 30.0  # Maximum FPS to prevent overload

    # Temporal sampling
    frame_skip: int = 0  # Skip frames (0 = no skipping)
    sequence_length: int = 16  # Number of frames per sequence
    temporal_stride: int = 1  # Temporal stride for sampling

    # Buffer and performance
    buffer_size: int = 64  # Frame buffer size
    batch_size: int = 8  # Batch size for processing
    num_workers: int = 2  # Background processing threads

    # Preprocessing
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Stream specific
    reconnect_attempts: int = 3
    connection_timeout: float = 10.0
    rtsp_transport: str = "tcp"  # tcp or udp


@dataclass
class FrameInfo:
    """Information about a video frame."""
    frame_id: int
    timestamp: float
    source_fps: float
    width: int
    height: int
    channels: int
    source: str


class VideoStreamLoader(ABC):
    """Abstract base class for video stream loaders."""

    def __init__(self, config: VideoConfig):
        self.config = config
        self.is_active = False
        self._frame_count = 0
        self._start_time = None

    @abstractmethod
    def open(self) -> bool:
        """Open the video stream."""
        pass

    @abstractmethod
    def read_frame(self) -> tuple[np.ndarray | None, FrameInfo | None]:
        """Read a single frame. Returns (frame, info) or (None, None) if failed."""
        pass

    @abstractmethod
    def close(self):
        """Close the video stream."""
        pass

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_stream_info(self) -> dict[str, Any]:
        """Get stream information."""
        return {
            "is_active": self.is_active,
            "frame_count": self._frame_count,
            "config": self.config
        }


class OpenCVVideoLoader(VideoStreamLoader):
    """OpenCV-based video loader supporting files, webcams, and network streams."""

    def __init__(self, source: str | int, config: VideoConfig):
        super().__init__(config)
        self.source = source
        self.cap = None
        self.source_fps = None
        self.source_info = None

    def open(self) -> bool:
        """Open video source using OpenCV."""
        if not HAS_OPENCV:
            logger.error("OpenCV not available")
            return False

        try:
            # Create VideoCapture object
            self.cap = cv2.VideoCapture(self.source)

            # Configure for network streams
            if isinstance(self.source, str) and (
                self.source.startswith('rtsp://') or
                self.source.startswith('rtmp://') or
                self.source.startswith('http://')
            ):
                # Set buffer size for network streams
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Set timeout
                if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                               int(self.config.connection_timeout * 1000))

            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False

            # Get source properties
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.source_info = {
                "fps": self.source_fps,
                "width": width,
                "height": height,
                "source": str(self.source)
            }

            self.is_active = True
            self._start_time = time.time()

            logger.info(f"Opened video source: {self.source} ({width}x{height} @ {self.source_fps} FPS)")
            return True

        except Exception as e:
            logger.error(f"Error opening video source {self.source}: {e}")
            return False

    def read_frame(self) -> tuple[np.ndarray | None, FrameInfo | None]:
        """Read frame from OpenCV VideoCapture."""
        if not self.is_active or not self.cap:
            return None, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None, None

            # Create frame info
            frame_info = FrameInfo(
                frame_id=self._frame_count,
                timestamp=time.time(),
                source_fps=self.source_fps or 30.0,
                width=frame.shape[1],
                height=frame.shape[0],
                channels=frame.shape[2] if len(frame.shape) > 2 else 1,
                source=str(self.source)
            )

            self._frame_count += 1
            return frame, frame_info

        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None, None

    def close(self):
        """Close OpenCV VideoCapture."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_active = False
        logger.info(f"Closed video source: {self.source}")


class VideoFrameProcessor:
    """Process and transform video frames."""

    def __init__(self, config: VideoConfig):
        self.config = config
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build torchvision transform pipeline."""
        transform_list = []

        # Resize if specified
        if self.config.target_width and self.config.target_height:
            transform_list.append(
                transforms.Resize((self.config.target_height, self.config.target_width))
            )

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize if specified
        if self.config.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            )

        return transforms.Compose(transform_list)

    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Process a single frame."""
        # Convert BGR to RGB (OpenCV uses BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if HAS_OPENCV:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Fallback: assume RGB and swap channels manually
                frame = frame[:, :, ::-1]

        # Convert to PIL Image for transforms
        from PIL import Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame)

        # Apply transforms
        tensor = self.transform(pil_image)
        return tensor

    def process_sequence(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Process a sequence of frames into a tensor."""
        processed_frames = []
        for frame in frames:
            processed_frames.append(self.process_frame(frame))

        # Stack frames: (T, C, H, W)
        sequence_tensor = torch.stack(processed_frames, dim=0)
        return sequence_tensor


class VideoStreamDataset:
    """Dataset-like interface for video streams with batching and buffering."""

    def __init__(self, source: str | int, config: VideoConfig | None = None):
        self.source = source
        self.config = config or VideoConfig()
        self.loader = None
        self.processor = VideoFrameProcessor(self.config)

        # Buffering
        self.frame_buffer = queue.Queue(maxsize=self.config.buffer_size)
        self.sequence_buffer = queue.Queue(maxsize=self.config.batch_size)
        self.worker_thread = None
        self.is_running = False

    def start_streaming(self) -> bool:
        """Start video streaming in background thread."""
        if self.is_running:
            return True

        # Create appropriate loader
        self.loader = OpenCVVideoLoader(self.source, self.config)

        if not self.loader.open():
            return False

        # Start background processing
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

        logger.info(f"Started video streaming from {self.source}")
        return True

    def stop_streaming(self):
        """Stop video streaming."""
        self.is_running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

        if self.loader:
            self.loader.close()
            self.loader = None

        # Clear buffers
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break

        while not self.sequence_buffer.empty():
            try:
                self.sequence_buffer.get_nowait()
            except queue.Empty:
                break

        logger.info("Stopped video streaming")

    def _stream_worker(self):
        """Background worker for frame capture and processing."""
        frame_sequence = []
        last_frame_time = time.time()
        target_frame_interval = 1.0 / min(self.config.max_fps, 30.0)

        while self.is_running and self.loader:
            try:
                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - last_frame_time
                if time_since_last < target_frame_interval:
                    time.sleep(target_frame_interval - time_since_last)
                    continue

                # Read frame
                frame, frame_info = self.loader.read_frame()
                if frame is None:
                    logger.warning("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue

                last_frame_time = time.time()

                # Skip frames if configured
                if self.config.frame_skip > 0:
                    if frame_info.frame_id % (self.config.frame_skip + 1) != 0:
                        continue

                # Add to frame buffer (non-blocking)
                try:
                    self.frame_buffer.put_nowait((frame, frame_info))
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait((frame, frame_info))
                    except queue.Empty:
                        pass

                # Build sequences
                frame_sequence.append(frame)
                if len(frame_sequence) >= self.config.sequence_length:
                    # Process sequence
                    try:
                        sequence_tensor = self.processor.process_sequence(frame_sequence)
                        self.sequence_buffer.put_nowait(sequence_tensor)
                    except queue.Full:
                        # Remove oldest sequence
                        try:
                            self.sequence_buffer.get_nowait()
                            self.sequence_buffer.put_nowait(sequence_tensor)
                        except queue.Empty:
                            pass

                    # Slide window with stride
                    frame_sequence = frame_sequence[self.config.temporal_stride:]

            except Exception as e:
                logger.error(f"Error in stream worker: {e}")
                time.sleep(0.1)

    def get_frame(self, timeout: float = 1.0) -> tuple[np.ndarray, FrameInfo] | None:
        """Get a single processed frame."""
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_sequence(self, timeout: float = 1.0) -> torch.Tensor | None:
        """Get a processed frame sequence tensor."""
        try:
            return self.sequence_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_batch(self, batch_size: int | None = None, timeout: float = 2.0) -> torch.Tensor | None:
        """Get a batch of sequences."""
        batch_size = batch_size or self.config.batch_size
        sequences = []

        start_time = time.time()
        while len(sequences) < batch_size and (time.time() - start_time) < timeout:
            sequence = self.get_sequence(timeout=0.1)
            if sequence is not None:
                sequences.append(sequence)

        if not sequences:
            return None

        # Pad batch if needed
        while len(sequences) < batch_size:
            if sequences:
                sequences.append(sequences[-1].clone())  # Repeat last sequence
            else:
                break

        # Stack into batch: (B, T, C, H, W)
        batch_tensor = torch.stack(sequences, dim=0)
        return batch_tensor

    def get_info(self) -> dict[str, Any]:
        """Get stream information."""
        info = {
            "source": self.source,
            "is_running": self.is_running,
            "config": self.config,
            "buffer_sizes": {
                "frames": self.frame_buffer.qsize(),
                "sequences": self.sequence_buffer.qsize()
            }
        }

        if self.loader:
            info.update(self.loader.get_stream_info())

        return info


def create_video_stream(source: str | int, config: VideoConfig | None = None) -> VideoStreamDataset:
    """Factory function to create video stream dataset."""
    return VideoStreamDataset(source, config)


# Pre-configured stream creators
def create_webcam_stream(camera_id: int = 0, **kwargs) -> VideoStreamDataset:
    """Create webcam stream."""
    config = VideoConfig(**kwargs)
    return create_video_stream(camera_id, config)


def create_file_stream(file_path: str, **kwargs) -> VideoStreamDataset:
    """Create file-based video stream."""
    config = VideoConfig(**kwargs)
    return create_video_stream(file_path, config)


def create_rtsp_stream(rtsp_url: str, **kwargs) -> VideoStreamDataset:
    """Create RTSP network stream."""
    config = VideoConfig(**kwargs)
    return create_video_stream(rtsp_url, config)
