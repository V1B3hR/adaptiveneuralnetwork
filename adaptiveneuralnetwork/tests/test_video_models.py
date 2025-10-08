"""
Tests for video models and streaming infrastructure.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ..core.video_inference import AdaptiveProcessor, InferenceConfig, PerformanceMonitor
from ..data.video_streaming import FrameInfo, VideoConfig, VideoFrameProcessor
from ..models.video_models import (
    Conv3D,
    ConvLSTM,
    HybridVideoModel,
    VideoModelConfig,
    VideoTransformer,
    create_video_model,
)


class TestVideoModels(unittest.TestCase):
    """Test video model implementations."""

    def setUp(self):
        self.config = VideoModelConfig(
            input_channels=3,
            input_height=224,
            input_width=224,
            sequence_length=8,  # Smaller for testing
            hidden_dim=64,      # Smaller for testing
            num_layers=1,       # Smaller for testing
            num_classes=10
        )

        # Create test input: (batch_size, seq_len, channels, height, width)
        self.test_input = torch.randn(2, 8, 3, 224, 224)

    def test_convlstm_model(self):
        """Test ConvLSTM model."""
        model = ConvLSTM(self.config)

        # Test forward pass
        output = model(self.test_input)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))  # (batch_size, num_classes)

        # Test that output is valid
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_conv3d_model(self):
        """Test 3D CNN model."""
        model = Conv3D(self.config)

        # Test forward pass
        output = model(self.test_input)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))

        # Test that output is valid
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_video_transformer(self):
        """Test Video Transformer model."""
        model = VideoTransformer(self.config)

        # Test forward pass
        output = model(self.test_input)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))

        # Test that output is valid
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_hybrid_model(self):
        """Test Hybrid model."""
        model = HybridVideoModel(self.config)

        # Test different fusion modes
        for fusion_mode in ["weighted", "concat", "ensemble"]:
            output = model(self.test_input, fusion_mode=fusion_mode)

            # Check output shape
            self.assertEqual(output.shape, (2, 10))

            # Test that output is valid
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_create_video_model(self):
        """Test model factory function."""
        models = ['convlstm', 'conv3d', 'transformer', 'hybrid']

        for model_type in models:
            model = create_video_model(model_type, self.config)
            self.assertIsNotNone(model)

            # Test forward pass
            output = model(self.test_input)
            self.assertEqual(output.shape, (2, 10))


class TestVideoStreaming(unittest.TestCase):
    """Test video streaming components."""

    def setUp(self):
        self.video_config = VideoConfig(
            target_width=224,
            target_height=224,
            sequence_length=8,
            batch_size=4
        )

    def test_video_config(self):
        """Test video configuration."""
        config = VideoConfig()

        # Test default values
        self.assertEqual(config.target_width, 224)
        self.assertEqual(config.target_height, 224)
        self.assertEqual(config.sequence_length, 16)

        # Test custom values
        custom_config = VideoConfig(target_width=128, sequence_length=10)
        self.assertEqual(custom_config.target_width, 128)
        self.assertEqual(custom_config.sequence_length, 10)

    def test_frame_info(self):
        """Test FrameInfo dataclass."""
        frame_info = FrameInfo(
            frame_id=42,
            timestamp=1234567890.0,
            source_fps=30.0,
            width=640,
            height=480,
            channels=3,
            source="test_source"
        )

        self.assertEqual(frame_info.frame_id, 42)
        self.assertEqual(frame_info.timestamp, 1234567890.0)
        self.assertEqual(frame_info.source_fps, 30.0)

    @patch('PIL.Image.fromarray')
    def test_video_frame_processor(self, mock_pil):
        """Test video frame processor without OpenCV dependency."""
        # Mock PIL Image
        mock_image = Mock()
        mock_pil.return_value = mock_image

        processor = VideoFrameProcessor(self.video_config)

        # Test processing a single frame
        test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Mock the transform pipeline
        with patch.object(processor, 'transform') as mock_transform:
            mock_transform.return_value = torch.randn(3, 224, 224)

            processed = processor.process_frame(test_frame)

            # Check that transform was called
            mock_transform.assert_called_once()
            self.assertEqual(processed.shape, (3, 224, 224))


class TestInferenceComponents(unittest.TestCase):
    """Test inference pipeline components."""

    def setUp(self):
        self.inference_config = InferenceConfig(
            target_latency_ms=100.0,
            max_latency_ms=200.0,
            target_fps=15.0
        )

    def test_inference_config(self):
        """Test inference configuration."""
        config = InferenceConfig()

        # Test default values
        self.assertEqual(config.target_latency_ms, 100.0)
        self.assertEqual(config.max_latency_ms, 200.0)
        self.assertEqual(config.target_fps, 15.0)
        self.assertTrue(config.enable_adaptive_resolution)

    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor(window_size=10)

        # Create mock inference results
        from ..core.video_inference import InferenceResult

        for i in range(5):
            result = InferenceResult(
                frame_id=i,
                timestamp=1234567890.0 + i,
                predictions=torch.randn(10),
                confidence=0.8 + i * 0.02,
                latency_ms=100.0 + i * 10,
                model_type="test_model"
            )
            monitor.record_inference(result)

        # Test statistics
        stats = monitor.get_stats()

        self.assertIn('latency_ms', stats)
        self.assertIn('confidence', stats)

        # Check latency stats
        latency_stats = stats['latency_ms']
        self.assertAlmostEqual(latency_stats['mean'], 120.0, places=1)
        self.assertEqual(latency_stats['min'], 100.0)
        self.assertEqual(latency_stats['max'], 140.0)

    def test_adaptive_processor(self):
        """Test adaptive processing."""
        processor = AdaptiveProcessor(self.inference_config)

        # Test frame skipping logic
        skip_frames = processor.should_skip_frame(150.0, 3)  # High latency
        self.assertGreater(skip_frames, 0)

        skip_frames = processor.should_skip_frame(50.0, 1)   # Low latency
        self.assertEqual(skip_frames, 0)

        # Test resolution adaptation
        initial_resolution = processor.current_resolution

        # Simulate poor performance
        for _ in range(5):
            new_resolution = processor.adapt_resolution(0.5, 150.0)  # Low confidence, high latency

        # Should decrease resolution
        self.assertLessEqual(processor.current_resolution, initial_resolution)


class TestVideoIntegration(unittest.TestCase):
    """Integration tests for video functionality."""

    def test_end_to_end_processing(self):
        """Test end-to-end video processing pipeline."""
        # Create a simple model
        config = VideoModelConfig(
            input_channels=3,
            input_height=112,
            input_width=112,
            sequence_length=4,
            hidden_dim=32,
            num_layers=1,
            num_classes=5
        )

        model = create_video_model('convlstm', config)
        model.eval()

        # Create test video sequence
        test_sequence = torch.randn(1, 4, 3, 112, 112)

        # Run inference
        with torch.no_grad():
            output = model(test_sequence)

        # Check output
        self.assertEqual(output.shape, (1, 5))
        self.assertFalse(torch.isnan(output).any())

        # Test that we can get probabilities
        probabilities = torch.softmax(output, dim=1)
        self.assertAlmostEqual(probabilities.sum().item(), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
