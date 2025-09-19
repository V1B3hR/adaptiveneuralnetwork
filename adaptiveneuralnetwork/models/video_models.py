"""
Temporal models for video analysis in Adaptive Neural Network.

This module implements various temporal models for sequential frame analysis:
- ConvLSTM for spatiotemporal modeling
- 3D CNN for spatiotemporal convolutions
- Video Transformers for attention-based modeling
- Hybrid models combining multiple approaches
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class VideoModelConfig:
    """Configuration for video models."""
    # Input dimensions
    input_channels: int = 3
    input_height: int = 224
    input_width: int = 224
    sequence_length: int = 16
    
    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    
    # Output
    num_classes: int = 1000
    
    # Model-specific parameters
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM Cell for spatiotemporal modeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutional layers for input-to-state and state-to-state transitions
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, g, o gates
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
    
    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of ConvLSTM cell.
        
        Args:
            input_tensor: (B, C, H, W)
            cur_state: (h_cur, c_cur) where each is (B, hidden_dim, H, W)
            
        Returns:
            (h_next, c_next): Next hidden and cell states
        """
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Compute all gates in one convolution
        combined_conv = self.conv_gates(combined)
        
        # Split into individual gates
        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gate functions
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        g = torch.tanh(cc_g)     # New cell content
        o = torch.sigmoid(cc_o)  # Output gate
        
        # Update cell state
        c_next = f * c_cur + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states."""
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, dtype=torch.float, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, dtype=torch.float, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """Multi-layer Convolutional LSTM for video sequence modeling."""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__()
        
        self.config = config
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        
        # Feature extraction (spatial)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(config.input_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, config.hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # ConvLSTM layers
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(
                input_dim=config.hidden_dim if i == 0 else config.hidden_dim,
                hidden_dim=config.hidden_dim,
                kernel_size=config.kernel_size
            )
            for i in range(self.num_layers)
        ])
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ConvLSTM.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states for all layers
        device = x.device
        hidden_states = []
        for layer in self.convlstm_layers:
            # Calculate feature map size after spatial feature extraction
            with torch.no_grad():
                dummy_input = torch.zeros(1, channels, height, width, device=device)
                dummy_features = self.feature_extractor(dummy_input)
                feat_h, feat_w = dummy_features.shape[-2:]
            
            h, c = layer.init_hidden(batch_size, (feat_h, feat_w), device)
            hidden_states.append((h, c))
        
        # Process sequence
        last_output = None
        for t in range(seq_len):
            # Extract spatial features
            frame = x[:, t]  # (B, C, H, W)
            features = self.feature_extractor(frame)  # (B, hidden_dim, H', W')
            
            # Pass through ConvLSTM layers
            layer_input = features
            for layer_idx, layer in enumerate(self.convlstm_layers):
                h, c = hidden_states[layer_idx]
                h, c = layer(layer_input, (h, c))
                hidden_states[layer_idx] = (h, c)
                layer_input = h
            
            last_output = h
        
        # Global pooling and classification
        output = self.global_pool(last_output)  # (B, hidden_dim, 1, 1)
        output = output.view(batch_size, -1)    # (B, hidden_dim)
        output = self.dropout(output)
        output = self.classifier(output)        # (B, num_classes)
        
        return output


class Conv3D(nn.Module):
    """3D CNN for spatiotemporal video analysis."""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__()
        
        self.config = config
        
        # 3D Convolutional layers
        self.conv3d_layers = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(config.input_channels, 64, kernel_size=(3, 7, 7), 
                     stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Second 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Fourth 3D conv block
            nn.Conv3d(256, config.hidden_dim, kernel_size=(3, 3, 3), 
                     stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 3D CNN.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes)
        """
        # Rearrange to (B, C, T, H, W) for 3D convolution
        x = x.permute(0, 2, 1, 3, 4)
        
        # Apply 3D convolutions
        x = self.conv3d_layers(x)
        
        # Global pooling
        x = self.global_pool(x)  # (B, hidden_dim, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, hidden_dim)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class VideoTransformer(nn.Module):
    """Transformer model for video sequence analysis."""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.hidden_dim
        
        # Spatial feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(config.input_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Spatial pooling and projection to d_model
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(256, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=config.sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Classification head
        self.classifier = nn.Linear(self.d_model, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Video Transformer.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Extract spatial features for each frame
        frame_features = []
        for t in range(seq_len):
            frame = x[:, t]  # (B, C, H, W)
            features = self.feature_extractor(frame)  # (B, 256, H', W')
            features = self.spatial_pool(features)    # (B, 256, 1, 1)
            features = features.view(batch_size, -1)  # (B, 256)
            features = self.projection(features)      # (B, d_model)
            frame_features.append(features)
        
        # Stack frame features: (B, T, d_model)
        sequence_features = torch.stack(frame_features, dim=1)
        
        # Add positional encoding
        # Transpose for positional encoding: (T, B, d_model)
        sequence_features = sequence_features.transpose(0, 1)
        sequence_features = self.pos_encoding(sequence_features)
        # Transpose back: (B, T, d_model)
        sequence_features = sequence_features.transpose(0, 1)
        
        # Apply transformer encoder
        encoded_features = self.transformer(sequence_features)  # (B, T, d_model)
        
        # Global temporal pooling (mean over time)
        pooled_features = encoded_features.mean(dim=1)  # (B, d_model)
        
        # Classification
        output = self.dropout(pooled_features)
        output = self.classifier(output)
        
        return output


class HybridVideoModel(nn.Module):
    """Hybrid model combining ConvLSTM, 3D CNN, and Transformer approaches."""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__()
        
        self.config = config
        
        # Individual model components
        self.convlstm = ConvLSTM(config)
        self.conv3d = Conv3D(config)
        self.transformer = VideoTransformer(config)
        
        # Fusion layer
        fusion_dim = config.num_classes * 3  # Concatenate outputs
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Learnable combination weights
        self.combination_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, x: torch.Tensor, fusion_mode: str = "weighted") -> torch.Tensor:
        """
        Forward pass of hybrid model.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            fusion_mode: "weighted", "concat", or "ensemble"
            
        Returns:
            Output tensor of shape (B, num_classes)
        """
        # Get outputs from all models
        convlstm_out = self.convlstm(x)
        conv3d_out = self.conv3d(x)
        transformer_out = self.transformer(x)
        
        if fusion_mode == "weighted":
            # Weighted combination
            weights = F.softmax(self.combination_weights, dim=0)
            output = (weights[0] * convlstm_out + 
                     weights[1] * conv3d_out + 
                     weights[2] * transformer_out)
        elif fusion_mode == "concat":
            # Concatenate and fuse
            combined = torch.cat([convlstm_out, conv3d_out, transformer_out], dim=1)
            output = self.fusion(combined)
        elif fusion_mode == "ensemble":
            # Simple ensemble (average)
            output = (convlstm_out + conv3d_out + transformer_out) / 3
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        
        return output
    
    def get_individual_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get outputs from individual models for analysis."""
        return {
            "convlstm": self.convlstm(x),
            "conv3d": self.conv3d(x),
            "transformer": self.transformer(x)
        }


def create_video_model(model_type: str, config: Optional[VideoModelConfig] = None, **kwargs) -> nn.Module:
    """Factory function to create video models."""
    
    if config is None:
        config = VideoModelConfig(**kwargs)
    
    if model_type.lower() == "convlstm":
        return ConvLSTM(config)
    elif model_type.lower() == "conv3d":
        return Conv3D(config)
    elif model_type.lower() == "transformer":
        return VideoTransformer(config)
    elif model_type.lower() == "hybrid":
        return HybridVideoModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Pre-configured model creators
def create_convlstm_model(num_classes: int = 1000, **kwargs) -> ConvLSTM:
    """Create ConvLSTM model."""
    config = VideoModelConfig(num_classes=num_classes, **kwargs)
    return ConvLSTM(config)


def create_conv3d_model(num_classes: int = 1000, **kwargs) -> Conv3D:
    """Create 3D CNN model.""" 
    config = VideoModelConfig(num_classes=num_classes, **kwargs)
    return Conv3D(config)


def create_video_transformer(num_classes: int = 1000, **kwargs) -> VideoTransformer:
    """Create Video Transformer model."""
    config = VideoModelConfig(num_classes=num_classes, **kwargs)
    return VideoTransformer(config)


def create_hybrid_model(num_classes: int = 1000, **kwargs) -> HybridVideoModel:
    """Create hybrid video model."""
    config = VideoModelConfig(num_classes=num_classes, **kwargs)
    return HybridVideoModel(config)