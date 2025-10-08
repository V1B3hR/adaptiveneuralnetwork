"""
Optional Graph and Spatial reasoning integration module.

This module provides graph neural network capabilities and spatial reasoning
extensions that can be integrated with the adaptive neural network architecture.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ..api.config import AdaptiveConfig
from ..api.model import AdaptiveModel
from ..core.phases import PhaseScheduler


class SpatialRelationType(Enum):
    """Types of spatial relationships."""
    DISTANCE = "distance"
    DIRECTION = "direction"
    CONTAINMENT = "containment"
    ADJACENCY = "adjacency"
    OVERLAP = "overlap"


@dataclass
class GraphConfig:
    """Configuration for graph neural network components."""
    node_dim: int = 64
    edge_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    aggregation: str = "mean"  # "mean", "max", "sum", "attention"
    message_passing_type: str = "gat"  # "gcn", "gat", "gin", "sage"
    spatial_dimensions: int = 2
    enable_edge_features: bool = True
    enable_global_context: bool = True


class AdaptiveMessagePassing(MessagePassing):
    """Adaptive message passing layer that integrates with node phases."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        phase_scheduler: PhaseScheduler | None = None,
        **kwargs
    ):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.phase_scheduler = phase_scheduler

        # Message transformation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # Phase-aware attention
        self.phase_attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with adaptive message passing."""
        # Standard message passing
        out = self.propagate(edge_index, x=x, **kwargs)

        # Phase-aware attention if phase scheduler available
        if self.phase_scheduler is not None and hasattr(self.phase_scheduler, 'node_phases'):
            # Apply phase-aware attention
            out = self._apply_phase_attention(x, out)

        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute messages between nodes."""
        # Concatenate source and target node features
        message_input = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(message_input)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features."""
        # Combine aggregated messages with node features
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)

    def _apply_phase_attention(self, x: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Apply phase-aware attention to messages."""
        # Get phase information
        phases = self.phase_scheduler.node_phases

        # Create phase embeddings
        phase_emb = F.one_hot(phases, num_classes=4).float()  # 4 basic phases

        # Apply attention based on phases
        # Nodes in similar phases should attend more to each other
        query = messages.unsqueeze(0)  # Add batch dimension
        key = value = messages.unsqueeze(0)

        attended, _ = self.phase_attention(query, key, value)
        return attended.squeeze(0)


class SpatialReasoningLayer(nn.Module):
    """Layer for spatial reasoning and relationship modeling."""

    def __init__(
        self,
        input_dim: int,
        spatial_dim: int = 2,
        num_relation_types: int = 5,
        hidden_dim: int = 64
    ):
        super().__init__()

        self.input_dim = input_dim
        self.spatial_dim = spatial_dim
        self.num_relation_types = num_relation_types
        self.hidden_dim = hidden_dim

        # Spatial position encoding
        self.position_encoder = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Relation type embedding
        self.relation_embeddings = nn.Embedding(num_relation_types, hidden_dim)

        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=input_dim + hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(input_dim + hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        relations: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for spatial reasoning.
        
        Args:
            x: Node features [num_nodes, input_dim]
            positions: Spatial positions [num_nodes, spatial_dim]
            relations: Relation types [num_edges] (optional)
            
        Returns:
            Enhanced node features with spatial reasoning
        """
        # Encode spatial positions
        pos_encoding = self.position_encoder(positions)

        # Combine features with position encoding
        enhanced_features = torch.cat([x, pos_encoding], dim=-1)

        # Apply spatial attention
        enhanced_features = enhanced_features.unsqueeze(0)  # Add batch dim
        attended, attention_weights = self.spatial_attention(
            enhanced_features, enhanced_features, enhanced_features
        )
        attended = attended.squeeze(0)  # Remove batch dim

        # Project back to original dimension
        output = self.output_projection(attended)

        return output, attention_weights.squeeze(0)

    def compute_spatial_relations(
        self,
        positions: torch.Tensor,
        threshold: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spatial relationships between nodes.
        
        Args:
            positions: Node positions [num_nodes, spatial_dim]
            threshold: Distance threshold for adjacency
            
        Returns:
            edge_index: Edge connectivity [2, num_edges]
            edge_relations: Relation types [num_edges]
        """
        num_nodes = positions.shape[0]

        # Compute pairwise distances
        distances = torch.cdist(positions, positions)

        # Find adjacent nodes (within threshold)
        adjacency_mask = (distances < threshold) & (distances > 0)
        edge_index = adjacency_mask.nonzero().t()

        # Compute relation types based on spatial properties
        edge_relations = self._classify_spatial_relations(
            positions, edge_index, distances
        )

        return edge_index, edge_relations

    def _classify_spatial_relations(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """Classify spatial relations between connected nodes."""
        source_nodes, target_nodes = edge_index

        # Get positions of connected nodes
        source_pos = positions[source_nodes]
        target_pos = positions[target_nodes]

        # Compute relation features
        edge_distances = distances[source_nodes, target_nodes]

        # Simple relation classification based on distance
        relations = torch.zeros(edge_index.shape[1], dtype=torch.long)

        # Distance-based relations
        relations[edge_distances < 0.3] = SpatialRelationType.ADJACENCY.value[0]  # Close
        relations[edge_distances >= 0.3] = SpatialRelationType.DISTANCE.value[0]  # Far

        return relations


class GraphSpatialIntegration(nn.Module):
    """Integration module that combines graph and spatial reasoning with adaptive networks."""

    def __init__(
        self,
        adaptive_model: AdaptiveModel,
        graph_config: GraphConfig,
        enable_spatial: bool = True,
        enable_graph: bool = True
    ):
        super().__init__()

        self.adaptive_model = adaptive_model
        self.config = graph_config
        self.enable_spatial = enable_spatial
        self.enable_graph = enable_graph

        # Get dimensions from adaptive model
        self.node_dim = adaptive_model.config.num_nodes
        self.hidden_dim = adaptive_model.config.hidden_dim

        # Graph neural network layers
        if enable_graph:
            self.graph_layers = nn.ModuleList()
            for _ in range(graph_config.num_layers):
                layer = AdaptiveMessagePassing(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    phase_scheduler=adaptive_model.phase_scheduler
                )
                self.graph_layers.append(layer)

        # Spatial reasoning layer
        if enable_spatial:
            self.spatial_layer = SpatialReasoningLayer(
                input_dim=self.hidden_dim,
                spatial_dim=graph_config.spatial_dimensions,
                hidden_dim=graph_config.hidden_dim
            )

        # Feature fusion
        fusion_input_dim = self.hidden_dim
        if enable_spatial:
            fusion_input_dim += self.hidden_dim

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # Graph-to-node mapping
        self.graph_to_node = nn.Linear(self.hidden_dim, self.node_dim)

        # Initialize spatial positions if needed
        if enable_spatial:
            self.register_buffer(
                'node_positions',
                torch.randn(self.node_dim, graph_config.spatial_dimensions)
            )

    def forward(
        self,
        x: torch.Tensor,
        graph_data: Data | None = None,
        update_positions: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Forward pass with graph and spatial reasoning.
        
        Args:
            x: Input features [batch_size, input_dim]
            graph_data: Optional graph data structure
            update_positions: Whether to update spatial positions
            
        Returns:
            Enhanced output and reasoning information
        """
        batch_size = x.shape[0]

        # Get standard adaptive model output
        adaptive_output = self.adaptive_model(x)

        # Extract node states for graph/spatial processing
        node_features = self.adaptive_model.node_state.hidden_state
        if node_features.dim() == 3:
            node_features = node_features.mean(dim=0)  # Average over batch

        reasoning_info = {}
        enhanced_features = node_features

        # Graph reasoning
        if self.enable_graph and graph_data is not None:
            graph_enhanced = self._apply_graph_reasoning(
                enhanced_features, graph_data
            )
            enhanced_features = graph_enhanced
            reasoning_info['graph_processing'] = True

        # Spatial reasoning
        if self.enable_spatial:
            spatial_enhanced, attention_weights = self._apply_spatial_reasoning(
                enhanced_features, update_positions
            )
            enhanced_features = torch.cat([enhanced_features, spatial_enhanced], dim=-1)
            reasoning_info['spatial_attention'] = attention_weights
            reasoning_info['spatial_processing'] = True

        # Feature fusion
        if self.enable_spatial or self.enable_graph:
            enhanced_features = self.feature_fusion(enhanced_features)

        # Map back to output space
        graph_spatial_output = self.graph_to_node(enhanced_features)

        # Combine with adaptive output
        if graph_spatial_output.shape != adaptive_output.shape:
            # Project to match output dimensions
            output_projection = nn.Linear(
                graph_spatial_output.shape[-1],
                adaptive_output.shape[-1]
            ).to(x.device)
            graph_spatial_output = output_projection(graph_spatial_output)

        # Expand to match batch size
        if graph_spatial_output.dim() == 1:
            graph_spatial_output = graph_spatial_output.unsqueeze(0).expand(
                batch_size, -1
            )
        elif graph_spatial_output.dim() == 2 and graph_spatial_output.shape[0] != batch_size:
            graph_spatial_output = graph_spatial_output.mean(dim=0).unsqueeze(0).expand(
                batch_size, -1
            )

        # Combine outputs (weighted sum)
        alpha = 0.7  # Weight for adaptive output
        combined_output = alpha * adaptive_output + (1 - alpha) * graph_spatial_output

        reasoning_info['combination_weight'] = alpha
        reasoning_info['adaptive_output_shape'] = adaptive_output.shape
        reasoning_info['graph_spatial_output_shape'] = graph_spatial_output.shape

        return combined_output, reasoning_info

    def _apply_graph_reasoning(
        self,
        node_features: torch.Tensor,
        graph_data: Data
    ) -> torch.Tensor:
        """Apply graph neural network reasoning."""
        x = node_features
        edge_index = graph_data.edge_index

        # Apply graph layers
        for layer in self.graph_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        return x

    def _apply_spatial_reasoning(
        self,
        node_features: torch.Tensor,
        update_positions: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial reasoning."""
        # Update positions based on node energy/activity if requested
        if update_positions:
            self._update_spatial_positions()

        # Apply spatial reasoning layer
        enhanced_features, attention_weights = self.spatial_layer(
            node_features, self.node_positions
        )

        return enhanced_features, attention_weights

    def _update_spatial_positions(self):
        """Update spatial positions based on node dynamics."""
        if hasattr(self.adaptive_model, 'node_state'):
            # Use node energy and activity to influence positions
            energy = self.adaptive_model.node_state.energy.flatten()
            activity = self.adaptive_model.node_state.activity.flatten()

            # Small movements based on energy/activity
            energy_movement = (energy - energy.mean()) * 0.01
            activity_movement = (activity - activity.mean()) * 0.01

            # Update positions (simple 2D case)
            if self.node_positions.shape[1] >= 2:
                self.node_positions[:, 0] += energy_movement[:self.node_positions.shape[0]]
                self.node_positions[:, 1] += activity_movement[:self.node_positions.shape[0]]

                # Keep positions bounded
                self.node_positions = torch.clamp(self.node_positions, -5.0, 5.0)

    def create_graph_from_nodes(self, connectivity_threshold: float = 0.5) -> Data:
        """Create graph structure from adaptive network nodes."""
        # Use node similarities to create edges
        node_features = self.adaptive_model.node_state.hidden_state
        if node_features.dim() == 3:
            node_features = node_features.mean(dim=0)

        # Compute similarity matrix
        similarities = torch.mm(node_features, node_features.t())
        similarities = F.softmax(similarities, dim=-1)

        # Create edges based on similarity threshold
        edge_mask = similarities > connectivity_threshold
        edge_index = edge_mask.nonzero().t()

        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            pos=self.node_positions if hasattr(self, 'node_positions') else None
        )

        return graph_data

    def get_spatial_relationships(self) -> dict[str, Any]:
        """Get information about spatial relationships between nodes."""
        if not self.enable_spatial:
            return {}

        # Compute spatial relations
        edge_index, edge_relations = self.spatial_layer.compute_spatial_relations(
            self.node_positions
        )

        return {
            'node_positions': self.node_positions.detach().cpu().numpy(),
            'edge_index': edge_index.detach().cpu().numpy(),
            'edge_relations': edge_relations.detach().cpu().numpy(),
            'num_spatial_edges': edge_index.shape[1],
        }


def create_graph_spatial_model(
    adaptive_config: AdaptiveConfig,
    graph_config: GraphConfig | None = None,
    enable_spatial: bool = True,
    enable_graph: bool = True
) -> GraphSpatialIntegration:
    """
    Create integrated graph-spatial-adaptive model.
    
    Args:
        adaptive_config: Configuration for adaptive model
        graph_config: Configuration for graph/spatial components
        enable_spatial: Enable spatial reasoning
        enable_graph: Enable graph neural networks
        
    Returns:
        Integrated model with graph and spatial capabilities
    """
    # Create base adaptive model
    adaptive_model = AdaptiveModel(adaptive_config)

    # Create graph config if not provided
    if graph_config is None:
        graph_config = GraphConfig(
            node_dim=adaptive_config.num_nodes,
            hidden_dim=adaptive_config.hidden_dim
        )

    # Create integrated model
    integrated_model = GraphSpatialIntegration(
        adaptive_model=adaptive_model,
        graph_config=graph_config,
        enable_spatial=enable_spatial,
        enable_graph=enable_graph
    )

    return integrated_model


# Utility functions for graph/spatial data creation
def create_synthetic_graph_data(
    num_nodes: int = 10,
    num_edges: int = 20,
    feature_dim: int = 64,
    spatial_dim: int = 2
) -> Data:
    """Create synthetic graph data for testing."""
    # Random node features
    x = torch.randn(num_nodes, feature_dim)

    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Random positions
    pos = torch.randn(num_nodes, spatial_dim)

    return Data(x=x, edge_index=edge_index, pos=pos)


def visualize_spatial_layout(
    model: GraphSpatialIntegration,
    save_path: str | None = None
) -> None:
    """Visualize spatial layout of nodes (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    if not model.enable_spatial:
        print("Spatial reasoning not enabled")
        return

    positions = model.node_positions.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.scatter(positions[:, 0], positions[:, 1], s=100, alpha=0.7)

    # Add node labels
    for i, (x, y) in enumerate(positions):
        plt.annotate(f'N{i}', (x, y), xytext=(5, 5), textcoords='offset points')

    plt.title('Adaptive Network Spatial Layout')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial layout saved to {save_path}")
    else:
        plt.show()
