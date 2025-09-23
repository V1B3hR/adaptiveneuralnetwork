"""
IoT sensors integration and edge computing optimization for adaptive neural networks.

This module implements real-world application integration including:
- IoT sensor data processing and real-time streams
- Edge computing optimization for mobile/embedded deployment
- Production-ready API endpoints
- Real-time inference optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of IoT sensors supported."""
    
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    GPS = "gps"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    LIGHT_SENSOR = "light"
    PROXIMITY = "proximity"


class EdgeDevice(Enum):
    """Supported edge computing devices."""
    
    MOBILE_PHONE = "mobile_phone"
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    CORAL_TPU = "coral_tpu"
    ARDUINO = "arduino"
    ESP32 = "esp32"
    EDGE_TPU = "edge_tpu"


@dataclass
class SensorData:
    """Sensor data structure."""
    
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Union[float, List[float], np.ndarray, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0


@dataclass
class EdgeDeploymentConfig:
    """Configuration for edge deployment."""
    
    # Device specifications
    device_type: EdgeDevice = EdgeDevice.MOBILE_PHONE
    max_memory_mb: int = 512
    max_compute_ops_per_sec: int = 1000000
    battery_aware: bool = True
    
    # Model optimization
    use_quantization: bool = True
    quantization_bits: int = 8
    use_pruning: bool = True
    pruning_sparsity: float = 0.5
    use_knowledge_distillation: bool = True
    
    # Inference optimization
    max_batch_size: int = 1
    target_latency_ms: float = 100.0
    enable_dynamic_batching: bool = False
    use_model_caching: bool = True
    
    # Real-time streaming
    max_queue_size: int = 100
    processing_threads: int = 2
    sensor_sampling_rate_hz: float = 10.0


class SensorDataProcessor(nn.Module):
    """Real-time IoT sensor data processor."""
    
    def __init__(self, sensor_types: List[SensorType], processing_dim: int = 128):
        super().__init__()
        self.sensor_types = sensor_types
        self.processing_dim = processing_dim
        
        # Sensor-specific preprocessors
        self.sensor_preprocessors = nn.ModuleDict()
        for sensor_type in sensor_types:
            input_dim = self._get_sensor_input_dim(sensor_type)
            self.sensor_preprocessors[sensor_type.value] = nn.Sequential(
                nn.Linear(input_dim, processing_dim),
                nn.ReLU(),
                nn.BatchNorm1d(processing_dim),
                nn.Dropout(0.1)
            )
        
        # Temporal fusion for sensor streams
        self.temporal_fusion = nn.LSTM(
            input_size=processing_dim,
            hidden_size=processing_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False  # Causal for real-time
        )
        
        # Multi-sensor fusion
        self.sensor_attention = nn.MultiheadAttention(
            embed_dim=processing_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Quality assessment
        self.quality_estimator = nn.Sequential(
            nn.Linear(processing_dim, processing_dim // 2),
            nn.ReLU(),
            nn.Linear(processing_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _get_sensor_input_dim(self, sensor_type: SensorType) -> int:
        """Get input dimension for each sensor type."""
        sensor_dims = {
            SensorType.TEMPERATURE: 1,
            SensorType.HUMIDITY: 1,
            SensorType.PRESSURE: 1,
            SensorType.ACCELEROMETER: 3,
            SensorType.GYROSCOPE: 3,
            SensorType.MAGNETOMETER: 3,
            SensorType.GPS: 2,  # lat, lon
            SensorType.CAMERA: 512,  # Pre-extracted features
            SensorType.MICROPHONE: 128,  # Pre-extracted features
            SensorType.LIDAR: 64,  # Pre-processed point cloud features
            SensorType.ULTRASONIC: 1,
            SensorType.LIGHT_SENSOR: 1,
            SensorType.PROXIMITY: 1
        }
        return sensor_dims.get(sensor_type, 1)
    
    def preprocess_sensor_data(self, sensor_data: SensorData) -> torch.Tensor:
        """Preprocess individual sensor data."""
        data = sensor_data.data
        
        # Convert to tensor if needed
        if isinstance(data, (list, np.ndarray)):
            data = torch.tensor(data, dtype=torch.float32)
        elif isinstance(data, (int, float)):
            data = torch.tensor([data], dtype=torch.float32)
        
        # Add batch dimension if needed
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        # Normalize based on sensor type
        data = self._normalize_sensor_data(data, sensor_data.sensor_type)
        
        return data
    
    def _normalize_sensor_data(self, data: torch.Tensor, sensor_type: SensorType) -> torch.Tensor:
        """Normalize sensor data based on type."""
        # Sensor-specific normalization ranges
        normalization_ranges = {
            SensorType.TEMPERATURE: (-40.0, 85.0),  # Celsius
            SensorType.HUMIDITY: (0.0, 100.0),  # Percentage
            SensorType.PRESSURE: (300.0, 1100.0),  # hPa
            SensorType.ACCELEROMETER: (-16.0, 16.0),  # g
            SensorType.GYROSCOPE: (-2000.0, 2000.0),  # deg/s
            SensorType.MAGNETOMETER: (-4800.0, 4800.0),  # µT
            SensorType.GPS: ((-180.0, 180.0), (-90.0, 90.0)),  # lon, lat
            SensorType.LIGHT_SENSOR: (0.0, 100000.0),  # lux
            SensorType.PROXIMITY: (0.0, 255.0),  # distance units
        }
        
        if sensor_type in normalization_ranges:
            range_vals = normalization_ranges[sensor_type]
            if isinstance(range_vals, tuple) and len(range_vals) == 2:
                min_val, max_val = range_vals
                data = (data - min_val) / (max_val - min_val)
            # Handle special cases like GPS with different ranges per dimension
        
        return data
    
    def forward(self, sensor_batch: List[SensorData]) -> Dict[str, torch.Tensor]:
        """
        Process batch of sensor data from multiple sensors.
        
        Args:
            sensor_batch: List of sensor data from different sensors/timestamps
            
        Returns:
            Dictionary with processed sensor features
        """
        # Group by sensor type
        sensor_groups = {}
        for sensor_data in sensor_batch:
            sensor_type = sensor_data.sensor_type.value
            if sensor_type not in sensor_groups:
                sensor_groups[sensor_type] = []
            sensor_groups[sensor_type].append(sensor_data)
        
        # Process each sensor type
        processed_features = []
        quality_scores = []
        
        for sensor_type, sensor_list in sensor_groups.items():
            # Preprocess sensor data
            sensor_tensors = []
            for sensor_data in sensor_list:
                preprocessed = self.preprocess_sensor_data(sensor_data)
                sensor_tensors.append(preprocessed)
            
            if sensor_tensors:
                # Stack temporal data
                stacked_data = torch.cat(sensor_tensors, dim=0)  # (T, D)
                
                # Apply sensor-specific preprocessing
                if sensor_type in self.sensor_preprocessors:
                    processed = self.sensor_preprocessors[sensor_type](stacked_data)
                    processed_features.append(processed.unsqueeze(0))  # (1, T, D)
                    
                    # Estimate quality
                    quality = self.quality_estimator(processed.mean(dim=0, keepdim=True))
                    quality_scores.append(quality)
        
        if not processed_features:
            return {'fused_features': torch.zeros(1, 1, self.processing_dim)}
        
        # Concatenate all sensor features
        all_features = torch.cat(processed_features, dim=0)  # (N_sensors, T, D)
        
        # Apply temporal fusion across all sensors
        B, T, D = all_features.shape
        all_features_flat = all_features.view(-1, T, D)
        temporal_output, _ = self.temporal_fusion(all_features_flat)
        
        # Multi-sensor attention fusion
        fused_features, attention_weights = self.sensor_attention(
            temporal_output, temporal_output, temporal_output
        )
        
        # Global pooling
        final_features = fused_features.mean(dim=1)  # (N_sensors, D)
        global_features = final_features.mean(dim=0, keepdim=True)  # (1, D)
        
        return {
            'fused_features': global_features,
            'individual_features': final_features,
            'attention_weights': attention_weights,
            'quality_scores': torch.cat(quality_scores) if quality_scores else torch.tensor([1.0])
        }


class EdgeOptimizedModel(nn.Module):
    """Edge-optimized neural network for mobile/embedded deployment."""
    
    def __init__(self, input_dim: int, output_dim: int, config: EdgeDeploymentConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Efficient architecture for edge devices
        hidden_dim = min(128, config.max_memory_mb // 4)  # Memory-aware sizing
        
        self.backbone = nn.Sequential(
            # Depthwise separable convolutions for efficiency
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU6(inplace=True),  # ReLU6 is more efficient on mobile
            nn.BatchNorm1d(hidden_dim),
            
            # Bottleneck layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            
            # Output layer
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Quantization if enabled
        if config.use_quantization:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
        
        # Model pruning if enabled
        if config.use_pruning:
            self._apply_pruning()
    
    def _apply_pruning(self):
        """Apply structured pruning to the model."""
        import torch.nn.utils.prune as prune
        
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.config.pruning_sparsity)
                prune.remove(module, 'weight')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass optimized for edge devices."""
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)
        return x
    
    def optimize_for_edge(self):
        """Apply edge optimizations."""
        # Model quantization
        if self.config.use_quantization:
            self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self, inplace=True)
            # Note: In practice, you would run calibration data here
            torch.quantization.convert(self, inplace=True)
        
        # Set to evaluation mode for inference
        self.eval()
        
        # Enable TorchScript compilation
        example_input = torch.randn(1, self.input_dim)
        self.traced_model = torch.jit.trace(self, example_input)
        
        return self.traced_model


class RealTimeInferenceEngine:
    """Real-time inference engine for streaming sensor data."""
    
    def __init__(self, model: nn.Module, config: EdgeDeploymentConfig):
        self.model = model
        self.config = config
        
        # Streaming infrastructure
        self.input_queue = queue.Queue(maxsize=config.max_queue_size)
        self.output_queue = queue.Queue(maxsize=config.max_queue_size)
        self.processing_active = False
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.processing_threads)
        
        # Performance monitoring
        self.inference_times = []
        self.throughput_counter = 0
        self.last_throughput_time = time.time()
        
        # Model caching for repeated patterns
        if config.use_model_caching:
            self.inference_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
    
    def start_processing(self):
        """Start real-time processing."""
        self.processing_active = True
        
        # Start processing threads
        for _ in range(self.config.processing_threads):
            self.executor.submit(self._processing_loop)
        
        logger.info(f"Started real-time inference engine with {self.config.processing_threads} threads")
    
    def stop_processing(self):
        """Stop real-time processing."""
        self.processing_active = False
        self.executor.shutdown(wait=True)
        logger.info("Stopped real-time inference engine")
    
    def _processing_loop(self):
        """Main processing loop for inference thread."""
        while self.processing_active:
            try:
                # Get input from queue with timeout
                sensor_data = self.input_queue.get(timeout=0.1)
                
                # Process the data
                start_time = time.time()
                result = self._process_sensor_data(sensor_data)
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Monitor performance
                self.inference_times.append(inference_time)
                if len(self.inference_times) > 1000:
                    self.inference_times.pop(0)  # Keep last 1000 measurements
                
                # Put result in output queue
                self.output_queue.put({
                    'result': result,
                    'inference_time_ms': inference_time,
                    'timestamp': time.time()
                })
                
                self.throughput_counter += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    def _process_sensor_data(self, sensor_data: List[SensorData]) -> torch.Tensor:
        """Process sensor data through the model."""
        # Check cache if enabled
        if self.config.use_model_caching:
            cache_key = self._compute_cache_key(sensor_data)
            if cache_key in self.inference_cache:
                self.cache_hits += 1
                return self.inference_cache[cache_key]
            else:
                self.cache_misses += 1
        
        # Run inference
        with torch.no_grad():
            # Convert sensor data to model input
            sensor_processor = SensorDataProcessor([data.sensor_type for data in sensor_data])
            processed = sensor_processor(sensor_data)
            model_input = processed['fused_features']
            
            # Run model inference
            output = self.model(model_input)
            
            # Cache result if enabled
            if self.config.use_model_caching and len(self.inference_cache) < 1000:
                self.inference_cache[cache_key] = output
        
        return output
    
    def _compute_cache_key(self, sensor_data: List[SensorData]) -> str:
        """Compute cache key for sensor data."""
        # Simple cache key based on sensor values (rounded for similar patterns)
        key_parts = []
        for data in sensor_data:
            if isinstance(data.data, (int, float)):
                rounded_val = round(data.data, 2)
                key_parts.append(f"{data.sensor_type.value}:{rounded_val}")
        return "|".join(sorted(key_parts))
    
    def add_sensor_data(self, sensor_data: List[SensorData]) -> bool:
        """Add sensor data to processing queue."""
        try:
            self.input_queue.put(sensor_data, block=False)
            return True
        except queue.Full:
            logger.warning("Input queue full, dropping sensor data")
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get processed result from output queue."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        current_time = time.time()
        time_diff = current_time - self.last_throughput_time
        
        stats = {
            'avg_inference_time_ms': np.mean(self.inference_times) if self.inference_times else 0.0,
            'max_inference_time_ms': np.max(self.inference_times) if self.inference_times else 0.0,
            'throughput_per_sec': self.throughput_counter / max(time_diff, 0.001),
            'queue_utilization': self.input_queue.qsize() / self.config.max_queue_size,
        }
        
        if self.config.use_model_caching:
            total_requests = self.cache_hits + self.cache_misses
            stats['cache_hit_rate'] = self.cache_hits / max(total_requests, 1)
        
        # Reset throughput counter periodically
        if time_diff > 10.0:  # Reset every 10 seconds
            self.throughput_counter = 0
            self.last_throughput_time = current_time
        
        return stats


class ProductionAPIServer:
    """Production-ready API server for real-time inference."""
    
    def __init__(self, inference_engine: RealTimeInferenceEngine):
        self.inference_engine = inference_engine
        self.request_count = 0
        self.error_count = 0
        
    async def process_sensor_request(self, sensor_data_json: str) -> Dict[str, Any]:
        """Process incoming sensor data request."""
        try:
            self.request_count += 1
            
            # Parse sensor data
            sensor_data_list = self._parse_sensor_data(sensor_data_json)
            
            # Add to processing queue
            success = self.inference_engine.add_sensor_data(sensor_data_list)
            if not success:
                return {
                    'error': 'Processing queue full',
                    'status': 'queue_full',
                    'retry_after_ms': 100
                }
            
            # Wait for result
            result = self.inference_engine.get_result(timeout=self.inference_engine.config.target_latency_ms / 1000)
            
            if result is None:
                return {
                    'error': 'Processing timeout',
                    'status': 'timeout',
                    'retry_after_ms': 50
                }
            
            return {
                'prediction': result['result'].tolist(),
                'inference_time_ms': result['inference_time_ms'],
                'timestamp': result['timestamp'],
                'status': 'success'
            }
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"API request error: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def _parse_sensor_data(self, sensor_data_json: str) -> List[SensorData]:
        """Parse JSON sensor data into SensorData objects."""
        data = json.loads(sensor_data_json)
        sensor_list = []
        
        for item in data.get('sensors', []):
            sensor_data = SensorData(
                sensor_id=item['sensor_id'],
                sensor_type=SensorType(item['sensor_type']),
                timestamp=item.get('timestamp', time.time()),
                data=item['data'],
                metadata=item.get('metadata', {}),
                quality_score=item.get('quality_score', 1.0)
            )
            sensor_list.append(sensor_data)
        
        return sensor_list
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        performance_stats = self.inference_engine.get_performance_stats()
        
        return {
            'status': 'healthy',
            'uptime_seconds': time.time() - getattr(self, 'start_time', time.time()),
            'requests_processed': self.request_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'performance': performance_stats
        }


class IoTEdgeManager:
    """Main manager for IoT edge computing integration."""
    
    def __init__(self, model: nn.Module, config: EdgeDeploymentConfig):
        self.config = config
        
        # Optimize model for edge deployment
        self.edge_model = EdgeOptimizedModel(
            input_dim=128,  # From sensor processor
            output_dim=model.classifier[-1].out_features if hasattr(model, 'classifier') else 10,
            config=config
        )
        
        # Copy weights from original model if compatible
        self._transfer_weights(model, self.edge_model)
        
        # Optimize for edge
        self.optimized_model = self.edge_model.optimize_for_edge()
        
        # Create inference engine
        self.inference_engine = RealTimeInferenceEngine(self.optimized_model, config)
        
        # Create API server
        self.api_server = ProductionAPIServer(self.inference_engine)
        
        # Device monitoring
        self.device_stats = {
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0,
            'battery_level_percent': 100,
            'temperature_celsius': 25
        }
    
    def _transfer_weights(self, source_model: nn.Module, target_model: nn.Module):
        """Transfer compatible weights from source to target model."""
        try:
            # Simple weight transfer for compatible layers
            source_dict = source_model.state_dict()
            target_dict = target_model.state_dict()
            
            # Transfer compatible layers
            transferred = 0
            for name, param in target_dict.items():
                if name in source_dict and source_dict[name].shape == param.shape:
                    target_dict[name].copy_(source_dict[name])
                    transferred += 1
            
            logger.info(f"Transferred {transferred} compatible layers to edge model")
            
        except Exception as e:
            logger.warning(f"Weight transfer failed: {e}")
    
    def start_service(self):
        """Start the IoT edge service."""
        self.inference_engine.start_processing()
        logger.info("IoT Edge Manager service started")
    
    def stop_service(self):
        """Stop the IoT edge service."""
        self.inference_engine.stop_processing()
        logger.info("IoT Edge Manager service stopped")
    
    def update_device_stats(self, stats: Dict[str, float]):
        """Update device performance statistics."""
        self.device_stats.update(stats)
        
        # Adjust processing based on device constraints
        if self.config.battery_aware and stats.get('battery_level_percent', 100) < 20:
            # Reduce processing frequency to save battery
            self.config.sensor_sampling_rate_hz *= 0.5
            logger.info("Reduced sampling rate due to low battery")
    
    async def process_iot_data(self, sensor_data_json: str) -> Dict[str, Any]:
        """Main entry point for processing IoT sensor data."""
        return await self.api_server.process_sensor_request(sensor_data_json)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'device_stats': self.device_stats,
            'inference_stats': self.inference_engine.get_performance_stats(),
            'api_stats': {
                'requests_processed': self.api_server.request_count,
                'error_rate': self.api_server.error_count / max(self.api_server.request_count, 1)
            },
            'config': {
                'device_type': self.config.device_type.value,
                'max_memory_mb': self.config.max_memory_mb,
                'target_latency_ms': self.config.target_latency_ms,
                'quantization_enabled': self.config.use_quantization,
                'pruning_enabled': self.config.use_pruning
            }
        }


# Factory functions for easy deployment
def create_mobile_deployment(model: nn.Module) -> IoTEdgeManager:
    """Create mobile-optimized deployment."""
    config = EdgeDeploymentConfig(
        device_type=EdgeDevice.MOBILE_PHONE,
        max_memory_mb=256,
        target_latency_ms=50.0,
        use_quantization=True,
        use_pruning=True,
        battery_aware=True
    )
    return IoTEdgeManager(model, config)


def create_raspberry_pi_deployment(model: nn.Module) -> IoTEdgeManager:
    """Create Raspberry Pi deployment."""
    config = EdgeDeploymentConfig(
        device_type=EdgeDevice.RASPBERRY_PI,
        max_memory_mb=1024,
        target_latency_ms=100.0,
        use_quantization=True,
        processing_threads=4
    )
    return IoTEdgeManager(model, config)


def create_jetson_nano_deployment(model: nn.Module) -> IoTEdgeManager:
    """Create Jetson Nano deployment with GPU acceleration."""
    config = EdgeDeploymentConfig(
        device_type=EdgeDevice.JETSON_NANO,
        max_memory_mb=2048,
        target_latency_ms=30.0,
        max_batch_size=4,
        processing_threads=8
    )
    return IoTEdgeManager(model, config)