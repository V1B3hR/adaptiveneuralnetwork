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

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, input_tensor: torch.Tensor, cur_state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
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

    def init_hidden(self, batch_size: int, image_size: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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

    def get_individual_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get outputs from individual models for analysis."""
        return {
            "convlstm": self.convlstm(x),
            "conv3d": self.conv3d(x),
            "transformer": self.transformer(x)
        }


class AdvancedTemporalReasoning(nn.Module):
    """Advanced temporal reasoning module for video sequences."""

    def __init__(self, feature_dim: int, num_frames: int, reasoning_depth: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.reasoning_depth = reasoning_depth

        # Multi-scale temporal convolutions for different time scales
        self.temporal_conv_1 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.temporal_conv_2 = nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2)
        self.temporal_conv_3 = nn.Conv1d(feature_dim, feature_dim, kernel_size=7, padding=3)

        # Causal reasoning layers
        self.causal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(reasoning_depth)
        ])

        # Temporal relationship modeling
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

        # Future prediction head
        self.future_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, sequence_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Advanced temporal reasoning across video sequences.
        
        Args:
            sequence_features: (B, T, D) - Batch, Time, Feature dimension
            
        Returns:
            Dictionary with temporal analysis results
        """
        B, T, D = sequence_features.shape

        # Multi-scale temporal convolutions
        features_1d = sequence_features.transpose(1, 2)  # (B, D, T)
        temp_conv_1 = F.relu(self.temporal_conv_1(features_1d))
        temp_conv_2 = F.relu(self.temporal_conv_2(features_1d))
        temp_conv_3 = F.relu(self.temporal_conv_3(features_1d))

        # Combine multi-scale features
        multi_scale = (temp_conv_1 + temp_conv_2 + temp_conv_3) / 3
        multi_scale = multi_scale.transpose(1, 2)  # (B, T, D)

        # Apply causal reasoning
        reasoned_features = multi_scale
        for layer in self.causal_layers:
            reasoned_features = layer(reasoned_features)

        # Temporal relationship attention
        relations, _ = self.relation_attention(
            reasoned_features, reasoned_features, reasoned_features
        )

        # Future prediction
        current_state = reasoned_features[:, -1, :]  # Use last frame
        predicted_future = self.future_predictor(current_state)

        return {
            'temporal_features': reasoned_features,
            'temporal_relations': relations,
            'future_prediction': predicted_future,
            'multi_scale_features': multi_scale
        }


class ActionRecognitionHead(nn.Module):
    """Advanced action recognition and prediction head."""

    def __init__(self, feature_dim: int, num_actions: int, temporal_window: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.temporal_window = temporal_window

        # Action classification
        self.action_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_actions)
        )

        # Action prediction (next action)
        self.action_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_actions)
        )

        # Action confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, temporal_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Recognize current action and predict next action.
        
        Args:
            temporal_features: (B, T, D) temporal features
            
        Returns:
            Dictionary with action recognition results
        """
        # Use global temporal pooling for current action
        current_action_features = temporal_features.mean(dim=1)  # (B, D)

        # Current action classification
        current_action_logits = self.action_classifier(current_action_features)

        # Next action prediction using last few frames
        last_frames = temporal_features[:, -self.temporal_window//4:, :].mean(dim=1)
        next_action_logits = self.action_predictor(last_frames)

        # Confidence estimation
        confidence = self.confidence_estimator(current_action_features)

        return {
            'current_action_logits': current_action_logits,
            'next_action_logits': next_action_logits,
            'action_confidence': confidence
        }


class VideoTextAudioFusion(nn.Module):
    """Multimodal fusion for video, text, and audio."""

    def __init__(self, video_dim: int, text_dim: int, audio_dim: int, fusion_dim: int):
        super().__init__()
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.fusion_dim = fusion_dim

        # Individual modality projections
        self.video_proj = nn.Linear(video_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)

        # Cross-modal attention
        self.video_text_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, batch_first=True
        )
        self.video_audio_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, batch_first=True
        )
        self.text_audio_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, batch_first=True
        )

        # Trimodal fusion
        self.trimodal_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )

    def forward(self, video_features: torch.Tensor, text_features: torch.Tensor,
                audio_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Fuse video, text, and audio modalities.
        
        Args:
            video_features: (B, T, D_v) video features
            text_features: (B, L, D_t) text features
            audio_features: (B, A, D_a) audio features
            
        Returns:
            Dictionary with fused features
        """
        B = video_features.shape[0]

        # Project to common dimension
        video_proj = self.video_proj(video_features)  # (B, T, fusion_dim)
        text_proj = self.text_proj(text_features)     # (B, L, fusion_dim)
        audio_proj = self.audio_proj(audio_features)  # (B, A, fusion_dim)

        # Cross-modal attention
        video_text_fused, _ = self.video_text_attention(
            video_proj.mean(dim=1, keepdim=True),  # (B, 1, fusion_dim)
            text_proj, text_proj
        )

        video_audio_fused, _ = self.video_audio_attention(
            video_proj.mean(dim=1, keepdim=True),
            audio_proj, audio_proj
        )

        text_audio_fused, _ = self.text_audio_attention(
            text_proj.mean(dim=1, keepdim=True),
            audio_proj, audio_proj
        )

        # Combine all modalities
        combined = torch.cat([
            video_text_fused.squeeze(1),    # (B, fusion_dim)
            video_audio_fused.squeeze(1),   # (B, fusion_dim)
            text_audio_fused.squeeze(1)     # (B, fusion_dim)
        ], dim=1)  # (B, fusion_dim * 3)

        # Final trimodal fusion
        fused_features = self.trimodal_fusion(combined)

        return {
            'fused_features': fused_features,
            'video_text_attention': video_text_fused.squeeze(1),
            'video_audio_attention': video_audio_fused.squeeze(1),
            'text_audio_attention': text_audio_fused.squeeze(1)
        }


class AdvancedVideoTransformer(VideoTransformer):
    """Enhanced Video Transformer with advanced temporal reasoning and action recognition."""

    def __init__(self, config: VideoModelConfig):
        super().__init__(config)

        # Advanced temporal reasoning
        self.temporal_reasoning = AdvancedTemporalReasoning(
            feature_dim=config.hidden_dim,
            num_frames=config.sequence_length
        )

        # Action recognition head
        self.action_head = ActionRecognitionHead(
            feature_dim=config.hidden_dim,
            num_actions=config.num_classes
        )

        # Replace simple classifier with more advanced prediction head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor, return_detailed: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Enhanced forward pass with advanced temporal reasoning.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            return_detailed: Whether to return detailed analysis
            
        Returns:
            Either classification logits or detailed analysis dictionary
        """
        batch_size, seq_len, channels, height, width = x.shape

        # Process each frame through CNN
        frame_features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # (B, C, H, W)
            features = self.feature_extractor(frame)  # (B, 256, H', W')
            features = self.spatial_pool(features)    # (B, 256, 1, 1)
            features = features.view(batch_size, -1)  # (B, 256)
            features = self.projection(features)      # (B, d_model)
            frame_features.append(features)

        # Stack frame features: (B, T, d_model)
        sequence_features = torch.stack(frame_features, dim=1)

        # Add positional encoding
        sequence_features = sequence_features.transpose(0, 1)
        sequence_features = self.pos_encoding(sequence_features)
        sequence_features = sequence_features.transpose(0, 1)

        # Apply transformer encoder
        encoded_features = self.transformer(sequence_features)  # (B, T, d_model)

        # Advanced temporal reasoning
        temporal_analysis = self.temporal_reasoning(encoded_features)

        # Action recognition
        action_analysis = self.action_head(temporal_analysis['temporal_features'])

        # Final classification using reasoned features
        pooled_features = temporal_analysis['temporal_features'].mean(dim=1)
        classification_output = self.classifier(pooled_features)

        if return_detailed:
            return {
                'classification_logits': classification_output,
                'action_recognition': action_analysis,
                'temporal_reasoning': temporal_analysis,
                'sequence_features': encoded_features
            }
        else:
            return classification_output


def create_video_model(model_type: str, config: VideoModelConfig | None = None, **kwargs) -> nn.Module:
    """Factory function to create video models."""

    if config is None:
        config = VideoModelConfig(**kwargs)

    if model_type.lower() == "convlstm":
        return ConvLSTM(config)
    elif model_type.lower() == "conv3d":
        return Conv3D(config)
    elif model_type.lower() == "transformer":
        return VideoTransformer(config)
    elif model_type.lower() == "advanced_transformer":
        return AdvancedVideoTransformer(config)
    elif model_type.lower() == "hybrid":
        return HybridVideoModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Pre-configured model creators
def create_convlstm_model(num_classes: int = 1000, **kwargs) -> ConvLSTM:
    """Create ConvLSTM model."""
    config = VideoModelConfig(num_classes=num_classes, **kwargs)
    return ConvLSTM(config)

def create_advanced_video_transformer(num_classes: int = 1000, **kwargs) -> AdvancedVideoTransformer:
    """Create advanced video transformer with temporal reasoning."""
    config = VideoModelConfig(num_classes=num_classes, **kwargs)
    return AdvancedVideoTransformer(config)


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
