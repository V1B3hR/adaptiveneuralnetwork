"""
JAX backend implementation for adaptive neural networks.

This module provides JAX-based implementations of the core adaptive neural network
components for advanced acceleration and functional programming benefits.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    from jax.scipy.special import sigmoid
    import flax.linen as nn
    from flax.training import train_state
    import optax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


if not JAX_AVAILABLE:
    logger.warning("JAX not available. JAX backend will not function.")
    # Create dummy types when JAX is not available
    class _DummyPRNGKey:
        pass
    
    # Mock jax.random for type annotations
    class _MockJax:
        class random:
            PRNGKey = _DummyPRNGKey
    
    if jax is None:
        jax = _MockJax()


@dataclass
class JAXNodeConfig:
    """Configuration for JAX-based adaptive nodes."""
    num_nodes: int = 100
    hidden_dim: int = 64
    energy_decay: float = 0.95
    activity_threshold: float = 0.5
    adaptation_rate: float = 0.01
    phase_sensitivity: float = 0.1


if JAX_AVAILABLE:
    class JAXNodeState:
        """
        JAX-based node state management with functional programming paradigm.
        
        This replaces the PyTorch-based NodeState with JAX arrays and pure functions.
        """
        
        def __init__(self, config: JAXNodeConfig, key: Optional[jax.random.PRNGKey] = None):
            if not JAX_AVAILABLE:
                raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")
            
            self.config = config
            self.key = key or random.PRNGKey(42)
            
            # Initialize state arrays
            self.state = self._initialize_state(self.key)
        
        def _initialize_state(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
            """Initialize node state arrays."""
            keys = random.split(key, 4)
            
            return {
                'hidden_state': random.normal(keys[0], (1, self.config.num_nodes, self.config.hidden_dim)),
                'energy': jnp.ones((1, self.config.num_nodes, 1)),
                'activity': jnp.zeros((1, self.config.num_nodes, 1)), 
                'phase_state': jnp.zeros((1, self.config.num_nodes, 1))
            }
        
        def expand_batch(self, state: Dict[str, jnp.ndarray], batch_size: int) -> Dict[str, jnp.ndarray]:
            """Expand state arrays to accommodate new batch size."""
            current_batch = state['hidden_state'].shape[0]
            
            if batch_size == current_batch:
                return state
            
            expanded_state = {}
            for key, array in state.items():
                if batch_size > current_batch:
                    # Expand by repeating the first batch
                    repeats = batch_size - current_batch
                    extra = jnp.repeat(array[:1], repeats, axis=0)
                    expanded_state[key] = jnp.concatenate([array, extra], axis=0)
                else:
                    # Truncate to desired batch size
                    expanded_state[key] = array[:batch_size]
            
            return expanded_state
        
        def get_batch_size(self, state: Dict[str, jnp.ndarray]) -> int:
            """Get current batch size from state."""
            return state['hidden_state'].shape[0]
        
        def reset_state(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
            """Reset state to initial values."""
            return self._initialize_state(key)
else:
    # Dummy class when JAX is not available
    class JAXNodeState:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")


if JAX_AVAILABLE:
    @jax.jit
    def update_energy(energy: jnp.ndarray, activity: jnp.ndarray, decay: float) -> jnp.ndarray:
        """Update energy levels based on activity and decay."""
        # Energy decays over time and is replenished by activity
        new_energy = energy * decay + activity * (1 - decay)
        return jnp.clip(new_energy, 0.0, 1.0)


    @jax.jit 
    def update_activity(hidden_state: jnp.ndarray, threshold: float) -> jnp.ndarray:
        """Update activity based on hidden state."""
        # Activity is based on how much the hidden state exceeds threshold
        state_magnitude = jnp.linalg.norm(hidden_state, axis=-1, keepdims=True)
        activity = sigmoid((state_magnitude - threshold) * 4.0)
        return activity


    @jax.jit
    def compute_phase_influence(phase_state: jnp.ndarray, phase_weights: jnp.ndarray) -> jnp.ndarray:
        """Compute phase influence on node dynamics."""
        # Phase influence modulates node behavior
        return jnp.dot(phase_state, phase_weights)
else:
    # Dummy functions when JAX is not available
    def update_energy(*args, **kwargs):
        raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")
    
    def update_activity(*args, **kwargs):
        raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")
    
    def compute_phase_influence(*args, **kwargs):
        raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")


if JAX_AVAILABLE:
    class JAXAdaptiveDynamics(nn.Module):
        """
        JAX/Flax implementation of adaptive dynamics.
        
        This replaces the PyTorch nn.Module with a Flax module for JAX compatibility.
        """
        
        hidden_dim: int
        config: JAXNodeConfig
        
        def setup(self):
            self.state_update = nn.Dense(self.hidden_dim)
            self.energy_update = nn.Dense(1)
            self.activity_update = nn.Dense(1)
        
        def __call__(self, hidden_state: jnp.ndarray, energy: jnp.ndarray, 
                     phase_state: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Forward pass of adaptive dynamics.
            
            Args:
                hidden_state: Current hidden state [batch, nodes, hidden_dim]
                energy: Current energy levels [batch, nodes, 1]
                phase_state: Current phase state [batch, nodes, 1]
                training: Whether in training mode
            
            Returns:
                Updated (hidden_state, energy, activity)
            """
            batch_size, num_nodes, _ = hidden_state.shape
            
            # Flatten for processing
            flat_hidden = hidden_state.reshape(batch_size * num_nodes, -1)
            flat_energy = energy.reshape(batch_size * num_nodes, -1)
            flat_phase = phase_state.reshape(batch_size * num_nodes, -1)
            
            # Concatenate features
            features = jnp.concatenate([flat_hidden, flat_energy], axis=-1)
            
            # Update hidden state
            new_hidden = self.state_update(features)
            new_hidden = jax.nn.tanh(new_hidden)  # Bounded activation
            
            # Update energy 
            energy_features = jnp.concatenate([new_hidden, flat_phase], axis=-1)
            energy_delta = self.energy_update(energy_features)
            new_energy = flat_energy + energy_delta * self.config.adaptation_rate
            new_energy = jnp.clip(new_energy, 0.0, 1.0)
            
            # Update activity
            activity_features = jnp.concatenate([new_hidden[:, :2], flat_energy], axis=-1)  # Use first 2 dims of hidden
            new_activity = self.activity_update(activity_features)
            new_activity = sigmoid(new_activity)
            
            # Reshape back
            new_hidden = new_hidden.reshape(batch_size, num_nodes, -1)
            new_energy = new_energy.reshape(batch_size, num_nodes, -1)
            new_activity = new_activity.reshape(batch_size, num_nodes, -1)
            
            return new_hidden, new_energy, new_activity


    class JAXAdaptiveModel(nn.Module):
        """
        JAX/Flax implementation of the complete adaptive neural network model.
        """
        
        config: JAXNodeConfig
        input_dim: int
        output_dim: int
        
        def setup(self):
            self.input_projection = nn.Dense(self.config.hidden_dim)
            self.dynamics = JAXAdaptiveDynamics(
                hidden_dim=self.config.hidden_dim,
                config=self.config
            )
            self.output_projection = nn.Dense(self.output_dim)
        
        def __call__(self, x: jnp.ndarray, state: Dict[str, jnp.ndarray], 
                     training: bool = True) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            """
            Forward pass of the adaptive model.
            
            Args:
                x: Input data [batch, input_dim]
                state: Current node state dictionary
                training: Whether in training mode
            
            Returns:
                (output, updated_state)
            """
            batch_size = x.shape[0]
            
            # Expand state if needed
            if batch_size != state['hidden_state'].shape[0]:
                # Use JAX-compatible expansion
                state = self._expand_state_jax(state, batch_size)
            
            # Project input to hidden dimension
            input_proj = self.input_projection(x)  # [batch, hidden_dim]
            input_proj = jnp.expand_dims(input_proj, axis=1)  # [batch, 1, hidden_dim]
            
            # Add input to first node
            hidden_state = state['hidden_state'].at[:, 0, :].add(input_proj.squeeze(1))
            
            # Run dynamics
            new_hidden, new_energy, new_activity = self.dynamics(
                hidden_state, state['energy'], state['phase_state'], training=training
            )
            
            # Aggregate hidden state for output (mean pooling)
            aggregated = jnp.mean(new_hidden, axis=1)  # [batch, hidden_dim]
            
            # Project to output
            output = self.output_projection(aggregated)
            
            # Update state
            new_state = {
                'hidden_state': new_hidden,
                'energy': new_energy,
                'activity': new_activity,
                'phase_state': state['phase_state']  # Phase updated separately
            }
            
            return output, new_state
        
        def _expand_state_jax(self, state: Dict[str, jnp.ndarray], batch_size: int) -> Dict[str, jnp.ndarray]:
            """JAX-compatible state expansion."""
            current_batch = state['hidden_state'].shape[0]
            
            if batch_size == current_batch:
                return state
            
            new_state = {}
            for key, array in state.items():
                if batch_size > current_batch:
                    # Tile to expand
                    repeats = batch_size // current_batch + 1
                    tiled = jnp.tile(array, (repeats, 1, 1))
                    new_state[key] = tiled[:batch_size]
                else:
                    # Truncate
                    new_state[key] = array[:batch_size]
            
            return new_state


    class JAXTrainingState:
        """
        JAX training state management using Optax optimizers.
        """
        
        def __init__(
            self,
            model: JAXAdaptiveModel,
            learning_rate: float = 0.001,
            optimizer: str = 'adam'
        ):
            if not JAX_AVAILABLE:
                raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")
            
            self.model = model
            self.learning_rate = learning_rate
            
            # Setup optimizer
            if optimizer == 'adam':
                self.tx = optax.adam(learning_rate)
            elif optimizer == 'sgd':
                self.tx = optax.sgd(learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        def create_train_state(self, key: jax.random.PRNGKey, 
                              input_shape: Tuple[int, ...]) -> train_state.TrainState:
            """Create initial training state."""
            # Initialize model parameters
            dummy_input = jnp.ones(input_shape)
            dummy_state = {
                'hidden_state': jnp.ones((1, self.model.config.num_nodes, self.model.config.hidden_dim)),
                'energy': jnp.ones((1, self.model.config.num_nodes, 1)),
                'activity': jnp.zeros((1, self.model.config.num_nodes, 1)),
                'phase_state': jnp.zeros((1, self.model.config.num_nodes, 1))
            }
            
            params = self.model.init(key, dummy_input, dummy_state)
            
            return train_state.TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.tx
            )
else:
    # Dummy classes when JAX is not available
    class JAXAdaptiveDynamics:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")
    
    class JAXAdaptiveModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")
    
    class JAXTrainingState:
        def __init__(self, *args, **kwargs):
            raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")


if JAX_AVAILABLE:
    @jax.jit
    def train_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray], 
                   node_state: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, float, Dict[str, jnp.ndarray]]:
        """
        JAX JIT-compiled training step.
        
        Args:
            state: Training state
            batch: (inputs, targets) batch
            node_state: Current node state
        
        Returns:
            (updated_state, loss, updated_node_state)
        """
        def loss_fn(params):
            inputs, targets = batch
            outputs, new_node_state = state.apply_fn(params, inputs, node_state)
            loss = optax.softmax_cross_entropy_with_integer_labels(outputs, targets).mean()
            return loss, (outputs, new_node_state)
        
        (loss, (outputs, new_node_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss, new_node_state
else:
    def train_step(*args, **kwargs):
        raise ImportError("JAX is not available. Please install JAX to use the JAX backend.")


def convert_pytorch_to_jax_config(pytorch_config) -> JAXNodeConfig:
    """
    Convert PyTorch AdaptiveConfig to JAX configuration.
    
    Args:
        pytorch_config: PyTorch AdaptiveConfig object
    
    Returns:
        JAX-compatible configuration
    """
    return JAXNodeConfig(
        num_nodes=getattr(pytorch_config, 'num_nodes', 100),
        hidden_dim=getattr(pytorch_config, 'hidden_dim', 64),
        energy_decay=getattr(pytorch_config, 'energy_decay', 0.95),
        activity_threshold=getattr(pytorch_config, 'activity_threshold', 0.5),
        adaptation_rate=getattr(pytorch_config, 'adaptation_rate', 0.01),
        phase_sensitivity=getattr(pytorch_config, 'phase_sensitivity', 0.1)
    )


def is_jax_available() -> bool:
    """Check if JAX backend is available."""
    return JAX_AVAILABLE


if __name__ == "__main__":
    # Example usage
    if JAX_AVAILABLE:
        print("JAX backend test")
        
        # Create config
        config = JAXNodeConfig(num_nodes=50, hidden_dim=32)
        
        # Initialize model
        model = JAXAdaptiveModel(
            config=config,
            input_dim=784,  # MNIST
            output_dim=10
        )
        
        # Create training state
        trainer = JAXTrainingState(model, learning_rate=0.001)
        key = random.PRNGKey(42)
        train_state = trainer.create_train_state(key, (32, 784))  # batch_size=32
        
        print(f"Model parameters: {sum(x.size for x in jax.tree_leaves(train_state.params))}")
        print("JAX backend initialized successfully!")
    else:
        print("JAX backend not available - please install JAX")