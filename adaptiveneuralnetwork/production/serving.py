"""
Real-time model serving with FastAPI for sub-100ms latency.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Optional FastAPI dependencies
try:
    import uvicorn
    from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

from ..api.config import AdaptiveConfig
from ..api.model import AdaptiveModel


@dataclass
class ServingConfig:
    """Configuration for model serving."""
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    max_workers: int = 4
    batch_size: int = 8
    max_batch_delay_ms: int = 10
    enable_batching: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    enable_monitoring: bool = True
    log_level: str = "INFO"


class InferenceRequest(BaseModel):
    """Request model for inference."""
    data: list[list[float]] = Field(..., description="Input data as list of features")
    model_name: str | None = Field(None, description="Optional model name")
    options: dict[str, Any] | None = Field(None, description="Additional options")

    class Config:
        schema_extra = {
            "example": {
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "model_name": "default",
                "options": {"temperature": 0.7}
            }
        }


class InferenceResponse(BaseModel):
    """Response model for inference."""
    predictions: list[list[float]] = Field(..., description="Model predictions")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_name: str = Field(..., description="Model used for inference")
    batch_size: int = Field(..., description="Batch size processed")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    gpu_available: bool = Field(..., description="Whether GPU is available")


class ModelServer:
    """High-performance model server with batching and caching."""

    def __init__(self, config: ServingConfig):
        self.config = config
        self.model = None
        self.model_config = None
        self.start_time = time.time()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.batch_queue = []
        self.batch_futures = []
        self.cache = {} if config.enable_caching else None
        self.request_count = 0
        self.total_latency = 0.0

        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path: str) -> None:
        """Load the adaptive neural network model."""
        try:
            model_path = Path(model_path)

            # Load model configuration
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                self.model_config = AdaptiveConfig.from_dict(config_dict)
            else:
                # Use default configuration
                self.model_config = AdaptiveConfig()

            # Load model weights
            self.model = AdaptiveModel(self.model_config)

            weights_path = model_path / "model.pth"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict)

            self.model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                self.logger.info("Model loaded on GPU")
            else:
                self.logger.info("Model loaded on CPU")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _get_cache_key(self, data: list[list[float]]) -> str:
        """Generate cache key for input data."""
        if not self.config.enable_caching:
            return None

        # Simple hash of input data
        data_str = str(sorted([sorted(row) for row in data]))
        return str(hash(data_str))

    def _preprocess_input(self, data: list[list[float]]) -> torch.Tensor:
        """Preprocess input data for the model."""
        try:
            # Convert to tensor
            tensor = torch.tensor(data, dtype=torch.float32)

            # Move to GPU if model is on GPU
            if next(self.model.parameters()).is_cuda:
                tensor = tensor.cuda()

            return tensor
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

    def _postprocess_output(self, output: torch.Tensor) -> list[list[float]]:
        """Postprocess model output."""
        try:
            # Move to CPU and convert to list
            if output.is_cuda:
                output = output.cpu()

            return output.detach().numpy().tolist()
        except Exception as e:
            self.logger.error(f"Postprocessing failed: {e}")
            raise

    async def _predict_batch(self, batch_data: list[list[list[float]]]) -> list[list[list[float]]]:
        """Process a batch of predictions."""
        if not self.model:
            raise ValueError("Model not loaded")

        try:
            # Combine all batch data
            combined_data = []
            batch_sizes = []

            for data in batch_data:
                combined_data.extend(data)
                batch_sizes.append(len(data))

            # Preprocess
            input_tensor = self._preprocess_input(combined_data)

            # Inference
            with torch.no_grad():
                output = self.model(input_tensor)

            # Postprocess
            predictions = self._postprocess_output(output)

            # Split back into original batches
            results = []
            start_idx = 0
            for batch_size in batch_sizes:
                end_idx = start_idx + batch_size
                results.append(predictions[start_idx:end_idx])
                start_idx = end_idx

            return results

        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise

    async def predict(self, data: list[list[float]], options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a prediction with the loaded model."""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._get_cache_key(data)
            if cache_key and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                cached_result["latency_ms"] = (time.time() - start_time) * 1000
                return cached_result

            if self.config.enable_batching:
                # Add to batch queue
                future = asyncio.Future()
                self.batch_queue.append((data, future, start_time))
                self.batch_futures.append(future)

                # Process batch if conditions are met
                if (len(self.batch_queue) >= self.config.batch_size or
                    len(self.batch_queue) > 0 and
                    (time.time() - self.batch_queue[0][2]) * 1000 > self.config.max_batch_delay_ms):

                    await self._process_batch()

                # Wait for result
                predictions = await future
            else:
                # Direct prediction
                input_tensor = self._preprocess_input(data)

                with torch.no_grad():
                    output = self.model(input_tensor)

                predictions = self._postprocess_output(output)

            latency_ms = (time.time() - start_time) * 1000

            result = {
                "predictions": predictions,
                "latency_ms": latency_ms,
                "model_name": "adaptive_neural_network",
                "batch_size": len(data),
                "metadata": {
                    "timestamp": time.time(),
                    "request_id": self.request_count
                }
            }

            # Cache result
            if cache_key and self.cache is not None:
                if len(self.cache) >= self.config.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[cache_key] = result.copy()

            # Update metrics
            self.request_count += 1
            self.total_latency += latency_ms

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    async def _process_batch(self):
        """Process accumulated batch queue."""
        if not self.batch_queue:
            return

        try:
            # Extract batch data and futures
            batch_items = self.batch_queue.copy()
            self.batch_queue.clear()

            batch_data = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]

            # Process batch
            results = await self._predict_batch(batch_data)

            # Set results for each future
            for future, result in zip(futures, results, strict=False):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # Set exception for all futures
            for _, future, _ in self.batch_queue:
                if not future.done():
                    future.set_exception(e)
            self.batch_queue.clear()

    def get_health_status(self) -> dict[str, Any]:
        """Get server health status."""
        uptime = time.time() - self.start_time

        # Get memory usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        return {
            "status": "healthy" if self.model else "unhealthy",
            "model_loaded": self.model is not None,
            "uptime_seconds": uptime,
            "memory_usage_mb": memory_mb,
            "gpu_available": torch.cuda.is_available(),
            "requests_processed": self.request_count,
            "average_latency_ms": self.total_latency / max(self.request_count, 1),
            "cache_size": len(self.cache) if self.cache else 0
        }


class FastAPIServer:
    """FastAPI-based REST server for the model."""

    def __init__(self, config: ServingConfig):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI dependencies not available. Install with: pip install fastapi uvicorn")

        self.config = config
        self.model_server = ModelServer(config)
        self.app = FastAPI(
            title="Adaptive Neural Network API",
            description="Production API for adaptive neural network inference",
            version="1.0.0"
        )
        self.security = HTTPBearer(auto_error=False)

        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            health_data = self.model_server.get_health_status()
            return HealthResponse(**health_data)

        @self.app.post("/predict", response_model=InferenceResponse)
        async def predict(
            request: InferenceRequest,
            credentials: HTTPAuthorizationCredentials | None = Depends(self.security)
        ):
            """Make predictions with the model."""
            try:
                # Optional authentication check would go here

                result = await self.model_server.predict(
                    request.data,
                    request.options
                )

                return InferenceResponse(**result)

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/models")
        async def list_models():
            """List available models."""
            return {
                "models": [{
                    "name": "adaptive_neural_network",
                    "status": "loaded" if self.model_server.model else "not_loaded",
                    "config": self.model_server.model_config.__dict__ if self.model_server.model_config else None
                }]
            }

        @self.app.post("/models/{model_name}/load")
        async def load_model(model_name: str, background_tasks: BackgroundTasks):
            """Load a model."""
            try:
                model_path = Path(self.config.model_path) / model_name
                self.model_server.load_model(str(model_path))
                return {"status": "success", "message": f"Model {model_name} loaded"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @self.app.get("/metrics")
        async def get_metrics():
            """Get server metrics."""
            return self.model_server.get_health_status()

    def load_model(self, model_path: str):
        """Load model into the server."""
        self.model_server.load_model(model_path)

    def run(self, **kwargs):
        """Run the FastAPI server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            **kwargs
        )
