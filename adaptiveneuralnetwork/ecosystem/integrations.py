"""
Integration with popular ML frameworks (PyTorch, TensorFlow, JAX).
"""

from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import logging

import torch
import numpy as np

from ..api.model import AdaptiveModel
from ..api.config import AdaptiveConfig

# Optional framework dependencies
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Dummy classes to avoid import errors
    class tf:
        class keras:
            class Model:
                pass
            class Input:
                pass
            class layers:
                class Dense:
                    pass
            class models:
                @staticmethod
                def model_from_json(json_str):
                    return None

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from flax import struct
    import optax
    JAX_AVAILABLE = True
except ImportError:  
    JAX_AVAILABLE = False
    # Dummy classes to avoid import errors
    class jax:
        class nn:
            @staticmethod
            def relu(x):
                return x
        class random:
            @staticmethod
            def PRNGKey(x):
                return x
    class jnp:
        @staticmethod
        def ones(shape):
            return None
        @staticmethod
        def array(x):
            return x
    class nn:
        class Module:
            pass


class FrameworkIntegration(ABC):
    """Base class for framework integrations."""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def convert_from_adaptive(self, model: AdaptiveModel) -> Any:
        """Convert AdaptiveModel to framework-specific model."""
        pass
    
    @abstractmethod
    def convert_to_adaptive(self, framework_model: Any) -> AdaptiveModel:
        """Convert framework-specific model to AdaptiveModel."""
        pass
    
    @abstractmethod
    def export_weights(self, model: AdaptiveModel) -> Dict[str, Any]:
        """Export model weights in framework format."""
        pass
    
    @abstractmethod
    def import_weights(self, weights: Dict[str, Any]) -> AdaptiveModel:
        """Import weights from framework format."""
        pass


class PyTorchIntegration(FrameworkIntegration):
    """Integration with PyTorch framework."""
    
    def __init__(self, config: AdaptiveConfig):
        super().__init__(config)
        self.logger.info("PyTorch integration initialized")
    
    def convert_from_adaptive(self, model: AdaptiveModel) -> torch.nn.Module:
        """Convert AdaptiveModel to PyTorch Module."""
        # AdaptiveModel is already a PyTorch model, so return as-is
        return model
    
    def convert_to_adaptive(self, pytorch_model: torch.nn.Module) -> AdaptiveModel:
        """Convert PyTorch Module to AdaptiveModel."""
        if isinstance(pytorch_model, AdaptiveModel):
            return pytorch_model
        
        # Create new AdaptiveModel and try to transfer weights
        adaptive_model = AdaptiveModel(self.config)
        
        # Try to match layers and transfer weights
        adaptive_state = adaptive_model.state_dict()
        pytorch_state = pytorch_model.state_dict()
        
        # Simple name matching (extend as needed)
        for adaptive_name, adaptive_param in adaptive_state.items():
            if adaptive_name in pytorch_state:
                pytorch_param = pytorch_state[adaptive_name]
                if adaptive_param.shape == pytorch_param.shape:
                    adaptive_state[adaptive_name] = pytorch_param
                    self.logger.debug(f"Transferred parameter: {adaptive_name}")
        
        adaptive_model.load_state_dict(adaptive_state)
        return adaptive_model
    
    def export_weights(self, model: AdaptiveModel) -> Dict[str, Any]:
        """Export model weights in PyTorch format."""
        state_dict = model.state_dict()
        
        # Convert tensors to numpy for serialization
        numpy_weights = {}
        for name, param in state_dict.items():
            numpy_weights[name] = param.detach().cpu().numpy()
        
        return {
            "framework": "pytorch",
            "weights": numpy_weights,
            "config": self.config.__dict__,
            "architecture": self._extract_architecture_info(model)
        }
    
    def import_weights(self, weights: Dict[str, Any]) -> AdaptiveModel:
        """Import weights from PyTorch format."""
        if weights.get("framework") != "pytorch":
            raise ValueError("Weights are not in PyTorch format")
        
        # Create model
        model = AdaptiveModel(self.config)
        
        # Load weights
        state_dict = {}
        for name, numpy_weight in weights["weights"].items():
            state_dict[name] = torch.from_numpy(numpy_weight)
        
        model.load_state_dict(state_dict)
        return model
    
    def _extract_architecture_info(self, model: AdaptiveModel) -> Dict[str, Any]:
        """Extract architecture information from the model."""
        return {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        }
    
    def create_pytorch_trainer(self, model: AdaptiveModel) -> 'PyTorchTrainer':
        """Create a PyTorch-specific trainer."""
        return PyTorchTrainer(model, self.config)
    
    def create_data_loader(self, data: np.ndarray, labels: np.ndarray, 
                          batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(data).float(),
            torch.from_numpy(labels).float()
        )
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )


if TENSORFLOW_AVAILABLE:
    class TensorFlowIntegration(FrameworkIntegration):
        """Integration with TensorFlow framework."""
        
        def __init__(self, config: AdaptiveConfig):
            super().__init__(config)
            self.logger.info("TensorFlow integration initialized")
        
        def convert_from_adaptive(self, model: AdaptiveModel) -> tf.keras.Model:
            """Convert AdaptiveModel to TensorFlow Keras model."""
            # Implementation details...
            pass
        
        def convert_to_adaptive(self, keras_model: tf.keras.Model) -> AdaptiveModel:
            """Convert TensorFlow Keras model to AdaptiveModel."""
            # Implementation details...
            pass
        
        def export_weights(self, model: AdaptiveModel) -> Dict[str, Any]:
            """Export model weights in TensorFlow format."""
            return {"framework": "tensorflow", "message": "TensorFlow not available"}
        
        def import_weights(self, weights: Dict[str, Any]) -> AdaptiveModel:
            """Import weights from TensorFlow format."""
            raise NotImplementedError("TensorFlow not available")
else:
    class TensorFlowIntegration(FrameworkIntegration):
        """Dummy TensorFlow integration when TensorFlow is not available."""
        
        def __init__(self, config: AdaptiveConfig):
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        def convert_from_adaptive(self, model: AdaptiveModel):
            raise ImportError("TensorFlow not available")
        
        def convert_to_adaptive(self, keras_model):
            raise ImportError("TensorFlow not available")
        
        def export_weights(self, model: AdaptiveModel) -> Dict[str, Any]:
            raise ImportError("TensorFlow not available")
        
        def import_weights(self, weights: Dict[str, Any]):
            raise ImportError("TensorFlow not available")
        """Convert AdaptiveModel to TensorFlow Keras model."""
        # Extract architecture information
        state_dict = model.state_dict()
        
        # Create equivalent Keras model (simplified)
        inputs = tf.keras.Input(shape=(self.config.input_dim,))
        x = inputs
        
        # Add layers based on adaptive model structure
        for i in range(self.config.num_layers):
            layer_name = f"dense_{i}"
            if f"layers.{i}.weight" in state_dict:
                weight = state_dict[f"layers.{i}.weight"].detach().cpu().numpy()
                bias = state_dict[f"layers.{i}.bias"].detach().cpu().numpy() if f"layers.{i}.bias" in state_dict else None
                
                x = tf.keras.layers.Dense(
                    weight.shape[0],
                    activation='relu' if i < self.config.num_layers - 1 else None,
                    name=layer_name
                )(x)
        
        keras_model = tf.keras.Model(inputs, x)
        
        # Transfer weights
        self._transfer_weights_to_keras(model, keras_model)
        
        return keras_model
    
    def convert_to_adaptive(self, keras_model: tf.keras.Model) -> AdaptiveModel:
        """Convert TensorFlow Keras model to AdaptiveModel."""
        # Create new AdaptiveModel
        adaptive_model = AdaptiveModel(self.config)
        
        # Transfer weights from Keras model
        self._transfer_weights_from_keras(keras_model, adaptive_model)
        
        return adaptive_model
    
    def _transfer_weights_to_keras(self, adaptive_model: AdaptiveModel, keras_model: tf.keras.Model):
        """Transfer weights from AdaptiveModel to Keras model."""
        adaptive_state = adaptive_model.state_dict()
        
        for i, layer in enumerate(keras_model.layers):
            if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
                weight_name = f"layers.{i}.weight"
                bias_name = f"layers.{i}.bias"
                
                if weight_name in adaptive_state:
                    # PyTorch uses (out_features, in_features), TF uses (in_features, out_features)
                    weight = adaptive_state[weight_name].detach().cpu().numpy().T
                    layer.kernel.assign(weight)
                
                if bias_name in adaptive_state:
                    bias = adaptive_state[bias_name].detach().cpu().numpy()
                    layer.bias.assign(bias)
    
    def _transfer_weights_from_keras(self, keras_model: tf.keras.Model, adaptive_model: AdaptiveModel):
        """Transfer weights from Keras model to AdaptiveModel."""
        adaptive_state = adaptive_model.state_dict()
        
        for i, layer in enumerate(keras_model.layers):
            if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
                weight_name = f"layers.{i}.weight"
                bias_name = f"layers.{i}.bias"
                
                if weight_name in adaptive_state:
                    # Convert from TF format to PyTorch format
                    weight = torch.from_numpy(layer.kernel.numpy().T)
                    adaptive_state[weight_name] = weight
                
                if bias_name in adaptive_state and layer.bias is not None:
                    bias = torch.from_numpy(layer.bias.numpy())
                    adaptive_state[bias_name] = bias
        
        adaptive_model.load_state_dict(adaptive_state)
    
    def export_weights(self, model: AdaptiveModel) -> Dict[str, Any]:
        """Export model weights in TensorFlow format."""
        keras_model = self.convert_from_adaptive(model)
        
        # Save weights in TensorFlow format
        weights_data = {}
        for layer in keras_model.layers:
            if hasattr(layer, 'get_weights'):
                layer_weights = layer.get_weights()
                if layer_weights:
                    weights_data[layer.name] = [w.tolist() for w in layer_weights]
        
        return {
            "framework": "tensorflow",
            "weights": weights_data,
            "config": self.config.__dict__,
            "architecture": keras_model.to_json()
        }
    
    def import_weights(self, weights: Dict[str, Any]) -> AdaptiveModel:
        """Import weights from TensorFlow format."""
        if weights.get("framework") != "tensorflow":
            raise ValueError("Weights are not in TensorFlow format")
        
        # Create Keras model from architecture
        keras_model = tf.keras.models.model_from_json(weights["architecture"])
        
        # Load weights
        for layer_name, layer_weights in weights["weights"].items():
            layer = keras_model.get_layer(layer_name)
            layer.set_weights([np.array(w) for w in layer_weights])
        
        # Convert to AdaptiveModel
        return self.convert_to_adaptive(keras_model)
    
    def create_tensorflow_trainer(self, model: AdaptiveModel) -> 'TensorFlowTrainer':
        """Create a TensorFlow-specific trainer."""
        return TensorFlowTrainer(model, self.config)


if JAX_AVAILABLE:
    class JAXIntegration(FrameworkIntegration):
        """Integration with JAX framework."""
        
        def __init__(self, config: AdaptiveConfig):
            super().__init__(config)
            self.logger.info("JAX integration initialized")
        
        def convert_from_adaptive(self, model: AdaptiveModel) -> 'JAXModel':
            """Convert AdaptiveModel to JAX/Flax model."""
            return JAXModel(self.config, model)
        
        def convert_to_adaptive(self, jax_model: 'JAXModel') -> AdaptiveModel:
            """Convert JAX model to AdaptiveModel."""
            return jax_model.to_adaptive_model()
        
        def export_weights(self, model: AdaptiveModel) -> Dict[str, Any]:
            """Export model weights in JAX format."""
            jax_model = self.convert_from_adaptive(model)
            
            return {
                "framework": "jax",
                "weights": jax_model.get_weights_dict(),
                "config": self.config.__dict__,
                "architecture": jax_model.get_architecture_info()
            }
        
        def import_weights(self, weights: Dict[str, Any]) -> AdaptiveModel:
            """Import weights from JAX format."""
            if weights.get("framework") != "jax":
                raise ValueError("Weights are not in JAX format")
            
            jax_model = JAXModel(self.config)
            jax_model.load_weights_dict(weights["weights"])
            
            return jax_model.to_adaptive_model()
else:
    class JAXIntegration(FrameworkIntegration):
        """Dummy JAX integration when JAX is not available."""
        
        def __init__(self, config: AdaptiveConfig):
            raise ImportError("JAX not available. Install with: pip install jax jaxlib flax optax")
        
        def convert_from_adaptive(self, model: AdaptiveModel):
            raise ImportError("JAX not available")
        
        def convert_to_adaptive(self, jax_model):
            raise ImportError("JAX not available")
        
        def export_weights(self, model: AdaptiveModel) -> Dict[str, Any]:
            raise ImportError("JAX not available")
        
        def import_weights(self, weights: Dict[str, Any]):
            raise ImportError("JAX not available")


if JAX_AVAILABLE:
    class JAXModel(nn.Module):
        """JAX/Flax implementation of AdaptiveModel."""
        
        config: AdaptiveConfig
        
        def setup(self):
            self.layers = [
                nn.Dense(
                    features=self.config.hidden_dim if i < self.config.num_layers - 1 else self.config.output_dim,
                    name=f"dense_{i}"
                )
                for i in range(self.config.num_layers)
            ]
        
        def __call__(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # No activation on last layer
                    x = jax.nn.relu(x)
            return x
        
        @classmethod
        def create_from_adaptive(cls, config: AdaptiveConfig, adaptive_model: AdaptiveModel):
            """Create JAX model from AdaptiveModel."""
            jax_model = cls(config)
            
            # Initialize with dummy input to get parameters
            key = jax.random.PRNGKey(0)
            dummy_input = jnp.ones((1, config.input_dim))
            params = jax_model.init(key, dummy_input)
            
            # Transfer weights from PyTorch model
            adaptive_state = adaptive_model.state_dict()
            
            for i in range(config.num_layers):
                layer_params = params['params'][f'dense_{i}']
                
                weight_name = f"layers.{i}.weight"
                bias_name = f"layers.{i}.bias"
                
                if weight_name in adaptive_state:
                    # PyTorch: (out_features, in_features), JAX: (in_features, out_features)
                    pytorch_weight = adaptive_state[weight_name].detach().cpu().numpy()
                    layer_params['kernel'] = jnp.array(pytorch_weight.T)
                
                if bias_name in adaptive_state:
                    pytorch_bias = adaptive_state[bias_name].detach().cpu().numpy()
                    layer_params['bias'] = jnp.array(pytorch_bias)
            
            return jax_model, params
        
        def to_adaptive_model(self, params: Dict[str, Any]) -> AdaptiveModel:
            """Convert JAX model to AdaptiveModel."""
            adaptive_model = AdaptiveModel(self.config)
            adaptive_state = adaptive_model.state_dict()
            
            for i in range(self.config.num_layers):
                layer_params = params['params'][f'dense_{i}']
                
                weight_name = f"layers.{i}.weight"
                bias_name = f"layers.{i}.bias"
                
                if 'kernel' in layer_params:
                    # Convert JAX format to PyTorch format
                    jax_weight = np.array(layer_params['kernel'])
                    adaptive_state[weight_name] = torch.from_numpy(jax_weight.T)
                
                if 'bias' in layer_params:
                    jax_bias = np.array(layer_params['bias'])
                    adaptive_state[bias_name] = torch.from_numpy(jax_bias)
            
            adaptive_model.load_state_dict(adaptive_state)
            return adaptive_model
        
        def get_weights_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
            """Get weights as serializable dictionary."""
            weights = {}
            for layer_name, layer_params in params['params'].items():
                weights[layer_name] = {}
                for param_name, param_value in layer_params.items():
                    weights[layer_name][param_name] = np.array(param_value).tolist()
            return weights
        
        def load_weights_dict(self, weights: Dict[str, Any]) -> Dict[str, Any]:
            """Load weights from dictionary."""
            params = {'params': {}}
            for layer_name, layer_weights in weights.items():
                params['params'][layer_name] = {}
                for param_name, param_value in layer_weights.items():
                    params['params'][layer_name][param_name] = jnp.array(param_value)
            return params
        
        def get_architecture_info(self) -> Dict[str, Any]:
            """Get architecture information."""
            return {
                "framework": "jax",
                "model_type": "JAXModel",
                "layers": [
                    {
                        "name": f"dense_{i}",
                        "type": "Dense",
                        "features": self.config.hidden_dim if i < self.config.num_layers - 1 else self.config.output_dim
                    }
                    for i in range(self.config.num_layers)
                ]
            }


# Training wrapper classes
class PyTorchTrainer:
    """PyTorch-specific trainer for AdaptiveModel."""
    
    def __init__(self, model: AdaptiveModel, config: AdaptiveConfig):
        self.model = model
        self.config = config
        self.optimizer = None
        self.criterion = None
        self.logger = logging.getLogger(__name__)
    
    def setup_training(self, optimizer_name: str = "adam", learning_rate: float = 0.001,
                      loss_function: str = "mse"):
        """Setup optimizer and loss function."""
        # Setup optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup loss function
        if loss_function.lower() == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss_function.lower() == "crossentropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch."""
        if not self.optimizer or not self.criterion:
            raise RuntimeError("Training not setup. Call setup_training() first.")
        
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)


if TENSORFLOW_AVAILABLE:
    class TensorFlowTrainer:
        """TensorFlow-specific trainer for AdaptiveModel."""
        
        def __init__(self, model: AdaptiveModel, config: AdaptiveConfig):
            self.adaptive_model = model
            self.config = config
            self.keras_model = None
            self.logger = logging.getLogger(__name__)
        
        def setup_training(self, optimizer_name: str = "adam", learning_rate: float = 0.001,
                          loss_function: str = "mse"):
            """Setup Keras model for training."""
            # Convert to Keras model
            integration = TensorFlowIntegration(self.config)
            self.keras_model = integration.convert_from_adaptive(self.adaptive_model)
            
            # Setup optimizer
            if optimizer_name.lower() == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name.lower() == "sgd":
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
            # Setup loss function
            if loss_function.lower() == "mse":
                loss = "mse"
            elif loss_function.lower() == "crossentropy":
                loss = "sparse_categorical_crossentropy"
            else:
                raise ValueError(f"Unsupported loss function: {loss_function}")
            
            self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        def train(self, x_train: np.ndarray, y_train: np.ndarray,
                 validation_data=None, epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
            """Train the model."""
            if not self.keras_model:
                raise RuntimeError("Training not setup. Call setup_training() first.")
            
            history = self.keras_model.fit(
                x_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Transfer weights back to AdaptiveModel
            integration = TensorFlowIntegration(self.config)
            integration._transfer_weights_from_keras(self.keras_model, self.adaptive_model)
            
            return history.history