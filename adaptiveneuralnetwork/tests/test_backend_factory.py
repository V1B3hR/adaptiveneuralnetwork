"""
Tests for backend factory and multi-backend support.
"""

from unittest.mock import MagicMock, patch

import pytest

from adaptiveneuralnetwork.api.backend_factory import (
    BackendFactory,
    create_adaptive_model,
    get_backend_info,
    list_available_backends,
)
from adaptiveneuralnetwork.api.config import AdaptiveConfig
from adaptiveneuralnetwork.api.model import AdaptiveModel


class TestBackendFactory:
    """Test cases for BackendFactory."""

    def test_pytorch_backend_creation(self):
        """Test creating PyTorch backend model."""
        config = AdaptiveConfig(backend="pytorch", num_nodes=32)
        model = BackendFactory.create_model(config)

        assert isinstance(model, AdaptiveModel)

    def test_invalid_backend_error(self):
        """Test error handling for invalid backend."""
        config = AdaptiveConfig(backend="invalid_backend")

        with pytest.raises(ValueError, match="Unsupported backend"):
            BackendFactory.create_model(config)

    @patch('adaptiveneuralnetwork.core.jax_backend.is_jax_available')
    def test_jax_backend_unavailable_error(self, mock_jax_available):
        """Test error when JAX backend requested but not available."""
        mock_jax_available.return_value = False

        config = AdaptiveConfig(backend="jax")

        with pytest.raises(ImportError, match="JAX backend requested but JAX is not available"):
            BackendFactory.create_model(config)

    @patch('adaptiveneuralnetwork.core.jax_backend.is_jax_available')
    @patch('adaptiveneuralnetwork.core.jax_backend.JAXAdaptiveModel')
    def test_jax_backend_creation_when_available(self, mock_jax_model, mock_jax_available):
        """Test creating JAX backend when available."""
        mock_jax_available.return_value = True
        mock_instance = MagicMock()
        mock_jax_model.return_value = mock_instance

        config = AdaptiveConfig(backend="jax", num_nodes=32)

        with patch('adaptiveneuralnetwork.api.backend_factory.convert_pytorch_to_jax_config') as mock_convert:
            mock_convert.return_value = MagicMock()

            model = BackendFactory.create_model(config)

            assert model == mock_instance
            mock_jax_model.assert_called_once()

    def test_neuromorphic_backend_creation(self):
        """Test creating neuromorphic backend model."""
        config = AdaptiveConfig(backend="neuromorphic", num_nodes=32)

        with patch('adaptiveneuralnetwork.core.neuromorphic.NeuromorphicAdaptiveModel') as mock_neuro_model:
            mock_instance = MagicMock()
            mock_neuro_model.return_value = mock_instance

            model = BackendFactory.create_model(config)

            assert model == mock_instance
            mock_neuro_model.assert_called_once()


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_create_adaptive_model_default(self):
        """Test creating model with default configuration."""
        model = create_adaptive_model()

        assert isinstance(model, AdaptiveModel)

    def test_create_adaptive_model_with_backend(self):
        """Test creating model with specific backend."""
        model = create_adaptive_model(backend="pytorch", num_nodes=64)

        assert isinstance(model, AdaptiveModel)

    def test_create_adaptive_model_with_config(self):
        """Test creating model with provided configuration."""
        config = AdaptiveConfig(backend="pytorch", num_nodes=128)
        model = create_adaptive_model(config=config)

        assert isinstance(model, AdaptiveModel)
        assert model.config.num_nodes == 128

    def test_create_adaptive_model_config_override(self):
        """Test overriding config parameters."""
        config = AdaptiveConfig(backend="pytorch", num_nodes=64)

        model = create_adaptive_model(config=config, num_nodes=128)

        assert isinstance(model, AdaptiveModel)
        assert model.config.num_nodes == 128  # Should be overridden

    @patch('adaptiveneuralnetwork.core.jax_backend.is_jax_available')
    def test_list_available_backends(self, mock_jax_available):
        """Test listing available backends."""
        mock_jax_available.return_value = True

        backends = list_available_backends()

        assert isinstance(backends, dict)
        assert "pytorch" in backends
        assert "jax" in backends
        assert "neuromorphic" in backends

        assert backends["pytorch"] is True  # Always available
        assert backends["jax"] is True  # Mocked as available
        assert backends["neuromorphic"] is True  # Always available

    @patch('adaptiveneuralnetwork.core.jax_backend.is_jax_available')
    def test_list_available_backends_jax_unavailable(self, mock_jax_available):
        """Test listing backends when JAX is unavailable."""
        mock_jax_available.return_value = False

        backends = list_available_backends()

        assert backends["pytorch"] is True
        assert backends["jax"] is False
        assert backends["neuromorphic"] is True

    @patch('adaptiveneuralnetwork.core.jax_backend.is_jax_available')
    def test_get_backend_info(self, mock_jax_available):
        """Test getting detailed backend information."""
        mock_jax_available.return_value = True

        info = get_backend_info()

        assert isinstance(info, dict)

        # Check PyTorch info
        pytorch_info = info["pytorch"]
        assert pytorch_info["available"] is True
        assert "description" in pytorch_info
        assert "features" in pytorch_info
        assert "dependencies" in pytorch_info

        # Check JAX info
        jax_info = info["jax"]
        assert jax_info["available"] is True  # Mocked
        assert "JIT compilation" in jax_info["features"]

        # Check neuromorphic info
        neuro_info = info["neuromorphic"]
        assert neuro_info["available"] is True
        assert "Spike-based computation" in neuro_info["features"]


class TestBackendCompatibility:
    """Test cases for backend compatibility and integration."""

    def test_config_backend_field(self):
        """Test that config properly handles backend field."""
        config = AdaptiveConfig()
        assert hasattr(config, 'backend')
        assert config.backend == "pytorch"  # Default

        config_jax = AdaptiveConfig(backend="jax")
        assert config_jax.backend == "jax"

    def test_backend_case_insensitive(self):
        """Test that backend names are case insensitive."""
        config_upper = AdaptiveConfig(backend="PYTORCH")
        config_mixed = AdaptiveConfig(backend="PyTorch")

        # Both should work (case handling in factory)
        model1 = create_adaptive_model(config=config_upper)
        model2 = create_adaptive_model(config=config_mixed)

        assert isinstance(model1, AdaptiveModel)
        assert isinstance(model2, AdaptiveModel)

    def test_multiple_model_creation(self):
        """Test creating multiple models with different backends."""
        # Create multiple PyTorch models
        model1 = create_adaptive_model(backend="pytorch", num_nodes=32)
        model2 = create_adaptive_model(backend="pytorch", num_nodes=64)

        assert isinstance(model1, AdaptiveModel)
        assert isinstance(model2, AdaptiveModel)
        assert model1.config.num_nodes != model2.config.num_nodes

    def test_backend_parameter_validation(self):
        """Test that backend-specific parameters are properly handled."""
        # Test that PyTorch-specific parameters work
        config = AdaptiveConfig(
            backend="pytorch",
            device="cpu",
            dtype="float32"
        )

        model = BackendFactory.create_model(config)
        assert isinstance(model, AdaptiveModel)


class TestErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_none_config_handling(self):
        """Test handling of None configuration."""
        model = create_adaptive_model(config=None, backend="pytorch")
        assert isinstance(model, AdaptiveModel)

    def test_empty_kwargs_handling(self):
        """Test handling of empty kwargs."""
        model = create_adaptive_model(**{})
        assert isinstance(model, AdaptiveModel)

    def test_invalid_config_parameter(self):
        """Test handling of invalid configuration parameters."""
        # This should not raise an error, just ignore invalid params
        model = create_adaptive_model(
            backend="pytorch",
            invalid_param=123,
            num_nodes=32  # Valid param
        )
        assert isinstance(model, AdaptiveModel)
        assert model.config.num_nodes == 32

    def test_conflicting_backend_specification(self):
        """Test handling of conflicting backend specifications."""
        config = AdaptiveConfig(backend="pytorch")

        # Backend parameter should override config backend
        model = create_adaptive_model(config=config, backend="pytorch")
        assert isinstance(model, AdaptiveModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
