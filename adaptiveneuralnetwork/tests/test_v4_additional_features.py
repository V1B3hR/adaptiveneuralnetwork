"""
Tests for additional 0.4.0 features: distributed training, streaming datasets, graph/spatial reasoning.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ..api.config import AdaptiveConfig
from ..api.model import AdaptiveModel
from ..data.streaming_datasets import (
    MultiDatasetWrapper,
    StreamingConfig,
    StreamingDatasetWrapper,
    UnifiedDatasetManager,
    quick_stream_dataset,
)
from ..training.distributed import (
    DistributedConfig,
    DistributedTrainer,
    create_pytorch_distributed_config,
)


class TestDistributedTraining:
    """Test distributed training capabilities."""

    def test_distributed_config_creation(self):
        """Test creating distributed training configuration."""
        config = DistributedConfig(
            backend="gloo",
            world_size=2,
            rank=0,
            master_addr="localhost",
            master_port="12355"
        )

        assert config.backend == "gloo"
        assert config.world_size == 2
        assert config.rank == 0
        assert not config.use_ray

    def test_distributed_trainer_creation(self):
        """Test creating distributed trainer (single process)."""
        model_config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(model_config)

        dist_config = DistributedConfig(world_size=1, rank=0)  # Single process
        trainer = DistributedTrainer(model, dist_config)

        assert trainer.is_distributed == False  # Single process
        assert trainer.is_main_process == True
        assert trainer.ddp_model is None  # No DDP for single process

    def test_distributed_dataloader_creation(self):
        """Test creating distributed dataloaders."""
        model_config = AdaptiveConfig(num_nodes=4, hidden_dim=3, input_dim=6, output_dim=2)
        model = AdaptiveModel(model_config)

        dist_config = DistributedConfig(world_size=1, rank=0)
        trainer = DistributedTrainer(model, dist_config)

        # Create synthetic dataset
        data = torch.randn(100, 6)
        labels = torch.randint(0, 2, (100,))
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(data, labels)

        # Create distributed dataloader
        dataloader = trainer.create_distributed_dataloader(
            dataset, batch_size=16, shuffle=True
        )

        assert dataloader.batch_size == 16
        assert len(dataloader) > 0

    def test_pytorch_distributed_config_helper(self):
        """Test PyTorch distributed config helper."""
        config = create_pytorch_distributed_config(
            world_size=4,
            rank=1,
            local_rank=1,
            backend="nccl"
        )

        assert config.world_size == 4
        assert config.rank == 1
        assert config.local_rank == 1
        assert config.backend == "nccl"
        assert config.use_ray == False


class TestStreamingDatasets:
    """Test streaming dataset capabilities."""

    def test_streaming_config(self):
        """Test streaming configuration."""
        config = StreamingConfig(
            buffer_size=2000,
            batch_size=64,
            shuffle_buffer_size=5000
        )

        assert config.buffer_size == 2000
        assert config.batch_size == 64
        assert config.shuffle_buffer_size == 5000

    def test_streaming_dataset_wrapper(self):
        """Test streaming dataset wrapper with synthetic data."""
        config = StreamingConfig(buffer_size=100, cache_size_mb=10)

        # Use callable data source for testing
        def data_generator(index):
            return {
                'x': np.random.randn(32, 32, 3).astype(np.float32),
                'y': index % 5,
                'metadata': {'index': index}
            }

        dataset = StreamingDatasetWrapper(data_generator, config)

        assert len(dataset) > 0

        # Test getting items
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2  # x, y

        # Test batch getting
        batch = dataset.get_batch([0, 1, 2])
        assert isinstance(batch, tuple)
        assert len(batch) == 2

    def test_streaming_dataset_with_directory(self):
        """Test streaming dataset with directory source."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            for i in range(5):
                data = {
                    'x': np.random.randn(10, 10).tolist(),
                    'y': i % 3,
                    'index': i
                }
                with open(Path(temp_dir) / f"{i:06d}.json", 'w') as f:
                    json.dump(data, f)

            # Create info file
            info = {
                "name": "test_dataset",
                "description": "Test streaming dataset",
                "num_samples": 5
            }
            with open(Path(temp_dir) / "info.json", 'w') as f:
                json.dump(info, f)

            config = StreamingConfig()
            dataset = StreamingDatasetWrapper(temp_dir, config)

            assert len(dataset) >= 5

            # Test streaming
            stream_items = list(dataset.stream(shuffle=False))
            assert len(stream_items) >= 5

    def test_unified_dataset_manager(self):
        """Test unified dataset manager."""
        manager = UnifiedDatasetManager()

        # Create streaming dataset
        def simple_data_source(index):
            return np.random.randn(10), index % 3

        dataset = manager.create_streaming_dataset(simple_data_source)
        assert dataset is not None

        # Register dataset
        manager.register_dataset("test_dataset", dataset)
        assert "test_dataset" in manager.list_datasets()

        # Get dataset info
        info = manager.get_dataset_info("test_dataset")
        assert "name" in info

    def test_multi_dataset_wrapper(self):
        """Test multi-dataset wrapper."""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            def data_source(index, dataset_id=i):
                return np.random.randn(5), (index + dataset_id) % 4

            config = StreamingConfig()
            dataset = StreamingDatasetWrapper(data_source, config)
            datasets.append(dataset)

        # Create multi-dataset wrapper
        multi_dataset = MultiDatasetWrapper(datasets, sampling_strategy="round_robin")

        assert len(multi_dataset) == sum(len(d) for d in datasets)

        # Test streaming
        stream_items = []
        for i, item in enumerate(multi_dataset.stream(shuffle=False)):
            stream_items.append(item)
            if i >= 10:  # Limit test size
                break

        assert len(stream_items) > 0

    def test_quick_stream_dataset(self):
        """Test quick streaming dataset creation."""
        def data_source(index):
            return torch.randn(8), index % 3

        dataloader = quick_stream_dataset(
            data_source,
            batch_size=16,
            shuffle=True
        )

        assert dataloader.batch_size == 16

        # Test getting a batch
        for batch in dataloader:
            x, y = batch
            assert x.shape[0] <= 16  # Batch size
            assert x.shape[1] == 8   # Feature size
            break  # Just test one batch

    def test_unified_dataloader_creation(self):
        """Test unified dataloader creation."""
        manager = UnifiedDatasetManager()

        def data_source(index):
            return torch.randn(12), index % 4

        dataset = manager.create_streaming_dataset(data_source)
        dataloader = manager.create_unified_dataloader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )

        assert dataloader.batch_size == 8

        # Test iteration
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break

        assert batch_count > 0


class TestGraphSpatialIntegration:
    """Test graph and spatial reasoning capabilities."""

    @pytest.fixture(autouse=True)
    def setup_torch_geometric(self):
        """Check if torch_geometric is available, skip tests if not."""
        try:
            import torch_geometric
            from torch_geometric.data import Data
            self.torch_geometric_available = True
        except ImportError:
            pytest.skip("torch_geometric not available")

    def test_graph_config_creation(self):
        """Test graph configuration creation."""
        if not hasattr(self, 'torch_geometric_available'):
            pytest.skip("torch_geometric not available")

        from ..models.graph_spatial import GraphConfig

        config = GraphConfig(
            node_dim=32,
            edge_dim=16,
            hidden_dim=64,
            num_layers=2,
            spatial_dimensions=3
        )

        assert config.node_dim == 32
        assert config.edge_dim == 16
        assert config.hidden_dim == 64
        assert config.num_layers == 2
        assert config.spatial_dimensions == 3

    def test_adaptive_message_passing(self):
        """Test adaptive message passing layer."""
        if not hasattr(self, 'torch_geometric_available'):
            pytest.skip("torch_geometric not available")

        from torch_geometric.data import Data

        from ..models.graph_spatial import AdaptiveMessagePassing

        # Create simple test data
        num_nodes = 5
        in_channels = 8
        out_channels = 16

        layer = AdaptiveMessagePassing(in_channels, out_channels)

        # Create node features and edge index
        x = torch.randn(num_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        # Forward pass
        output = layer(x, edge_index)

        assert output.shape == (num_nodes, out_channels)
        assert torch.all(torch.isfinite(output))

    def test_spatial_reasoning_layer(self):
        """Test spatial reasoning layer."""
        if not hasattr(self, 'torch_geometric_available'):
            pytest.skip("torch_geometric not available")

        from ..models.graph_spatial import SpatialReasoningLayer

        input_dim = 16
        spatial_dim = 2
        num_nodes = 6

        layer = SpatialReasoningLayer(
            input_dim=input_dim,
            spatial_dim=spatial_dim,
            hidden_dim=32
        )

        # Create test data
        x = torch.randn(num_nodes, input_dim)
        positions = torch.randn(num_nodes, spatial_dim)

        # Forward pass
        output, attention_weights = layer(x, positions)

        assert output.shape == (num_nodes, input_dim)
        assert attention_weights.shape[0] == num_nodes
        assert torch.all(torch.isfinite(output))

    def test_spatial_relations_computation(self):
        """Test spatial relations computation."""
        if not hasattr(self, 'torch_geometric_available'):
            pytest.skip("torch_geometric not available")

        from ..models.graph_spatial import SpatialReasoningLayer

        layer = SpatialReasoningLayer(input_dim=8, spatial_dim=2)

        # Create positions
        positions = torch.tensor([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

        # Compute spatial relations
        edge_index, edge_relations = layer.compute_spatial_relations(
            positions, threshold=0.8
        )

        assert edge_index.shape[0] == 2  # [source, target]
        assert edge_index.shape[1] > 0   # Some edges should exist
        assert edge_relations.shape[0] == edge_index.shape[1]

    def test_graph_spatial_integration_creation(self):
        """Test creating graph-spatial integration."""
        if not hasattr(self, 'torch_geometric_available'):
            pytest.skip("torch_geometric not available")

        from ..models.graph_spatial import GraphConfig, GraphSpatialIntegration

        # Create adaptive model
        adaptive_config = AdaptiveConfig(num_nodes=6, hidden_dim=8, input_dim=10, output_dim=3)
        adaptive_model = AdaptiveModel(adaptive_config)

        # Create graph config
        graph_config = GraphConfig(
            node_dim=6,
            hidden_dim=8,
            num_layers=2,
            spatial_dimensions=2
        )

        # Create integration
        integration = GraphSpatialIntegration(
            adaptive_model=adaptive_model,
            graph_config=graph_config,
            enable_spatial=True,
            enable_graph=False  # Disable graph for simpler test
        )

        assert integration.enable_spatial == True
        assert integration.enable_graph == False
        assert hasattr(integration, 'spatial_layer')

    def test_integrated_forward_pass(self):
        """Test forward pass with graph-spatial integration."""
        if not hasattr(self, 'torch_geometric_available'):
            pytest.skip("torch_geometric not available")

        from ..models.graph_spatial import GraphConfig, create_graph_spatial_model

        # Create configurations
        adaptive_config = AdaptiveConfig(num_nodes=4, hidden_dim=6, input_dim=8, output_dim=2)
        graph_config = GraphConfig(node_dim=4, hidden_dim=6, spatial_dimensions=2)

        # Create integrated model
        model = create_graph_spatial_model(
            adaptive_config=adaptive_config,
            graph_config=graph_config,
            enable_spatial=True,
            enable_graph=False  # Disable for simpler test
        )

        # Test forward pass
        x = torch.randn(2, 8)  # batch_size=2, input_dim=8

        output, reasoning_info = model(x)

        assert output.shape == (2, 2)  # batch_size=2, output_dim=2
        assert torch.all(torch.isfinite(output))
        assert isinstance(reasoning_info, dict)
        assert 'spatial_processing' in reasoning_info

    def test_synthetic_graph_data_creation(self):
        """Test synthetic graph data creation utility."""
        if not hasattr(self, 'torch_geometric_available'):
            pytest.skip("torch_geometric not available")

        from ..models.graph_spatial import create_synthetic_graph_data

        graph_data = create_synthetic_graph_data(
            num_nodes=8,
            num_edges=12,
            feature_dim=16,
            spatial_dim=2
        )

        assert graph_data.x.shape == (8, 16)
        assert graph_data.edge_index.shape == (2, 12)
        assert graph_data.pos.shape == (8, 2)


class TestIntegrationFeatures:
    """Test integration between different 0.4.0 features."""

    def test_distributed_with_streaming(self):
        """Test distributed training with streaming datasets."""
        # Create streaming dataset
        def data_source(index):
            return torch.randn(10), index % 5

        config = StreamingConfig(buffer_size=50)
        dataset = StreamingDatasetWrapper(data_source, config)

        # Create distributed trainer (single process)
        model_config = AdaptiveConfig(num_nodes=8, hidden_dim=6, input_dim=10, output_dim=5)
        model = AdaptiveModel(model_config)

        dist_config = DistributedConfig(world_size=1, rank=0)
        trainer = DistributedTrainer(model, dist_config)

        # Create distributed dataloader from streaming dataset
        dataloader = trainer.create_distributed_dataloader(
            dataset, batch_size=8, shuffle=True
        )

        assert dataloader.batch_size == 8

        # Test one batch
        for batch in dataloader:
            x, y = batch
            assert x.shape[1] == 10  # input_dim
            break

    def test_energy_optimizers_with_streaming(self):
        """Test energy-aware optimizers with streaming data."""
        from ..training.energy_optimizers import create_energy_aware_optimizer

        # Create model
        config = AdaptiveConfig(num_nodes=6, hidden_dim=4, input_dim=8, output_dim=3)
        model = AdaptiveModel(config)

        # Create energy-aware optimizer
        optimizer = create_energy_aware_optimizer(
            'adam', model.parameters(), model.node_state, lr=0.01
        )

        # Create streaming dataset
        def data_source(index):
            return torch.randn(8), index % 3

        dataset_config = StreamingConfig(buffer_size=20)
        dataset = StreamingDatasetWrapper(data_source, dataset_config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

        # Test training step
        for batch in dataloader:
            x, y = batch

            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check that adaptation history is updated
            assert len(optimizer.adaptation_history) > 0
            break

    def test_reproducibility_with_features(self):
        """Test reproducibility harness with new features."""
        from ..utils.reproducibility import ReproducibilityHarness

        harness = ReproducibilityHarness(master_seed=42, strict_mode=False)

        def test_streaming_reproducibility():
            harness.set_seed(42)

            # Create streaming dataset
            def data_source(index):
                return torch.randn(5), index % 2

            config = StreamingConfig(buffer_size=10)
            dataset = StreamingDatasetWrapper(data_source, config)

            # Get first item
            item = dataset[0]
            return item[0].sum().item()  # Sum of features

        # Test determinism
        report = harness.verify_determinism(
            test_streaming_reproducibility,
            "streaming_reproducibility",
            run_count=3
        )

        # Should be deterministic due to seed setting
        assert report.is_deterministic
        assert report.unique_outputs == 1


if __name__ == "__main__":
    pytest.main([__file__])
