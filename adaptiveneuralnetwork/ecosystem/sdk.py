"""
Comprehensive SDK for adaptive neural network development.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union, Callable, AsyncIterable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time

# Optional HTTP client dependencies
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    # Dummy class to avoid import errors
    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def aclose(self):
            pass

import torch
import numpy as np

from ..api.model import AdaptiveModel
from ..api.config import AdaptiveConfig
from ..production.serving import ServingConfig
from ..production.database import DatabaseConfig, HybridDatabaseManager
from ..production.messaging import MessagingConfig, HybridMessageQueue


@dataclass
class SDKConfig:
    """Configuration for the SDK."""
    # Server connection
    server_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    
    # Local model settings
    model_path: Optional[str] = None
    cache_dir: str = "~/.adaptive_nn_cache"
    
    # Database settings
    database_config: Optional[DatabaseConfig] = None
    
    # Messaging settings
    messaging_config: Optional[MessagingConfig] = None
    
    # Logging
    log_level: str = "INFO"


class SDKClient:
    """Client for interacting with adaptive neural network services."""
    
    def __init__(self, config: SDKConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # HTTP client setup
        if HTTPX_AVAILABLE:
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            
            self.http_client = httpx.AsyncClient(
                base_url=config.server_url,
                headers=headers,
                timeout=config.timeout
            )
        else:
            self.http_client = None
            self.logger.warning("httpx not available, remote API calls disabled")
        
        # Local model
        self.local_model = None
        
        # Database and messaging
        self.db_manager = None
        self.message_queue = None
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize database and messaging components."""
        if self.config.database_config:
            try:
                self.db_manager = HybridDatabaseManager(self.config.database_config)
                self.logger.info("Database manager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize database: {e}")
        
        if self.config.messaging_config:
            try:
                self.message_queue = HybridMessageQueue(self.config.messaging_config)
                self.logger.info("Message queue initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize messaging: {e}")
    
    async def load_local_model(self, model_path: Optional[str] = None) -> bool:
        """Load a model locally."""
        path = Path(model_path or self.config.model_path)
        
        if not path.exists():
            self.logger.error(f"Model path does not exist: {path}")
            return False
        
        try:
            # Load configuration
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                model_config = AdaptiveConfig.from_dict(config_dict)
            else:
                model_config = AdaptiveConfig()
            
            # Create and load model
            self.local_model = AdaptiveModel(model_config)
            
            weights_path = path / "model.pth"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location="cpu")
                self.local_model.load_state_dict(state_dict)
            
            self.local_model.eval()
            self.logger.info(f"Local model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
            return False
    
    async def predict(self, data: List[List[float]], 
                     model_name: Optional[str] = None,
                     use_local: bool = False,
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make predictions using local or remote model."""
        if use_local and self.local_model:
            return await self._predict_local(data, options)
        elif self.http_client:
            return await self._predict_remote(data, model_name, options)
        else:
            raise RuntimeError("No prediction method available")
    
    async def _predict_local(self, data: List[List[float]], 
                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make predictions using local model."""
        start_time = time.time()
        
        try:
            # Convert to tensor
            input_tensor = torch.tensor(data, dtype=torch.float32)
            
            # Inference
            with torch.no_grad():
                output = self.local_model(input_tensor)
            
            # Convert to list
            predictions = output.detach().numpy().tolist()
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = {
                "predictions": predictions,
                "latency_ms": latency_ms,
                "model_name": "local_model",
                "batch_size": len(data),
                "metadata": {
                    "timestamp": time.time(),
                    "source": "local"
                }
            }
            
            # Store in database if available
            if self.db_manager:
                await self.db_manager.store_prediction({
                    **result,
                    "input_data": data
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Local prediction failed: {e}")
            raise
    
    async def _predict_remote(self, data: List[List[float]], 
                            model_name: Optional[str] = None,
                            options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make predictions using remote API."""
        if not self.http_client:
            raise RuntimeError("HTTP client not available")
        
        try:
            payload = {
                "data": data,
                "model_name": model_name,
                "options": options
            }
            
            response = await self.http_client.post("/predict", json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Store in database if available
            if self.db_manager:
                await self.db_manager.store_prediction({
                    **result,
                    "input_data": data
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Remote prediction failed: {e}")
            raise
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about available models."""
        if not self.http_client:
            if self.local_model:
                return {
                    "models": [{
                        "name": "local_model",
                        "status": "loaded",
                        "config": self.local_model.config.__dict__
                    }]
                }
            else:
                return {"models": []}
        
        try:
            response = await self.http_client.get("/models")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            raise
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status."""
        if not self.http_client:
            return {
                "status": "local_only",
                "local_model_loaded": self.local_model is not None,
                "database_available": self.db_manager is not None,
                "messaging_available": self.message_queue is not None
            }
        
        try:
            response = await self.http_client.get("/health")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        if not self.http_client:
            return {"error": "Remote metrics not available"}
        
        try:
            response = await self.http_client.get("/metrics")
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            raise
    
    async def stream_predictions(self, 
                               data_stream: AsyncIterable[List[List[float]]],
                               callback: Callable[[Dict[str, Any]], None],
                               batch_size: int = 1) -> None:
        """Stream predictions for real-time processing."""
        batch = []
        
        async for data in data_stream:
            batch.extend(data)
            
            if len(batch) >= batch_size:
                try:
                    result = await self.predict(batch)
                    if callback:
                        callback(result)
                except Exception as e:
                    self.logger.error(f"Streaming prediction failed: {e}")
                
                batch = []
        
        # Process remaining batch
        if batch:
            try:
                result = await self.predict(batch)
                if callback:
                    callback(result)
            except Exception as e:
                self.logger.error(f"Final batch prediction failed: {e}")
    
    async def send_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """Send a message to the message queue."""
        if not self.message_queue:
            self.logger.error("Message queue not available")
            return False
        
        try:
            await self.message_queue.send_message(topic, message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def subscribe_to_topic(self, topic: str, 
                               handler: Callable[[Dict[str, Any]], None]) -> bool:
        """Subscribe to a message queue topic."""
        if not self.message_queue:
            self.logger.error("Message queue not available")
            return False
        
        try:
            await self.message_queue.consume_messages(topic, handler)
            return True
        except Exception as e:
            self.logger.error(f"Failed to subscribe to topic: {e}")
            return False
    
    async def get_prediction_history(self, 
                                   filters: Optional[Dict[str, Any]] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get prediction history from database."""
        if not self.db_manager:
            self.logger.error("Database not available")
            return []
        
        try:
            return await self.db_manager.get_predictions(filters, limit)
        except Exception as e:
            self.logger.error(f"Failed to get prediction history: {e}")
            return []
    
    async def close(self):
        """Close the SDK client."""
        if self.http_client:
            await self.http_client.aclose()
        
        if self.message_queue:
            await self.message_queue.disconnect()


class AdaptiveNeuralNetworkSDK:
    """Main SDK class providing high-level interfaces."""
    
    def __init__(self, config: SDKConfig):
        self.config = config
        self.client = SDKClient(config)
        self.logger = logging.getLogger(__name__)
        
        # Convenience methods
        self.predict = self.client.predict
        self.get_health = self.client.get_health
        self.get_metrics = self.client.get_metrics
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.close()
    
    def create_batch_predictor(self, batch_size: int = 32, 
                              max_delay_ms: int = 100) -> 'BatchPredictor':
        """Create a batch predictor for efficient processing."""
        return BatchPredictor(self.client, batch_size, max_delay_ms)
    
    def create_model_trainer(self, config: AdaptiveConfig) -> 'ModelTrainer':
        """Create a model trainer for local training."""
        return ModelTrainer(self.client, config)
    
    @classmethod
    def create_simple(cls, server_url: str = "http://localhost:8000", 
                     api_key: Optional[str] = None) -> 'AdaptiveNeuralNetworkSDK':
        """Create a simple SDK instance with minimal configuration."""
        config = SDKConfig(server_url=server_url, api_key=api_key)
        return cls(config)


class BatchPredictor:
    """Batch predictor for efficient processing."""
    
    def __init__(self, client: SDKClient, batch_size: int = 32, max_delay_ms: int = 100):
        self.client = client
        self.batch_size = batch_size
        self.max_delay_ms = max_delay_ms
        self.batch_queue = []
        self.batch_futures = []
        
    async def predict(self, data: List[List[float]]) -> Dict[str, Any]:
        """Add prediction to batch queue."""
        future = asyncio.Future()
        self.batch_queue.append((data, future))
        self.batch_futures.append(future)
        
        # Process batch if conditions are met
        if (len(self.batch_queue) >= self.batch_size or
            len(self.batch_queue) > 0):
            await self._process_batch()
        
        return await future
    
    async def _process_batch(self):
        """Process the current batch."""
        if not self.batch_queue:
            return
        
        try:
            # Combine all batch data
            combined_data = []
            futures = []
            batch_sizes = []
            
            for data, future in self.batch_queue:
                combined_data.extend(data)
                futures.append(future)
                batch_sizes.append(len(data))
            
            # Clear queue
            self.batch_queue.clear()
            
            # Make prediction
            result = await self.client.predict(combined_data)
            predictions = result["predictions"]
            
            # Split results back to individual requests
            start_idx = 0
            for i, (future, batch_size) in enumerate(zip(futures, batch_sizes)):
                end_idx = start_idx + batch_size
                individual_result = {
                    **result,
                    "predictions": predictions[start_idx:end_idx],
                    "batch_size": batch_size
                }
                
                if not future.done():
                    future.set_result(individual_result)
                
                start_idx = end_idx
                
        except Exception as e:
            # Set exception for all futures
            for _, future in self.batch_queue:
                if not future.done():
                    future.set_exception(e)
            self.batch_queue.clear()


class ModelTrainer:
    """Model trainer for local training workflows."""
    
    def __init__(self, client: SDKClient, config: AdaptiveConfig):
        self.client = client
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def create_model(self) -> AdaptiveModel:
        """Create a new model instance."""
        self.model = AdaptiveModel(self.config)
        return self.model
    
    async def train(self, train_data: torch.utils.data.DataLoader,
                   val_data: Optional[torch.utils.data.DataLoader] = None,
                   epochs: int = 10,
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the model."""
        if not self.model:
            self.create_model()
        
        # Simple training loop (extend as needed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_data):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_data)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_data:
                self.model.eval()
                epoch_val_loss = 0.0
                
                with torch.no_grad():
                    for data, target in val_data:
                        output = self.model(data)
                        loss = criterion(output, target)
                        epoch_val_loss += loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_data)
                val_losses.append(avg_val_loss)
                
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Save model if path provided
        if save_path:
            await self.save_model(save_path)
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None
        }
    
    async def save_model(self, save_path: str) -> bool:
        """Save the trained model."""
        if not self.model:
            self.logger.error("No model to save")
            return False
        
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model weights
            torch.save(self.model.state_dict(), save_path / "model.pth")
            
            # Save configuration
            with open(save_path / "config.json", 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            self.logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False