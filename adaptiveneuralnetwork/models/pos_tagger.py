"""
Part-of-Speech Tagging models with BiLSTM and optional Transformer architectures.
Supports variable-length sequences with padding and masking.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class POSTaggerConfig:
    """Configuration for POS tagger models."""
    vocab_size: int
    num_tags: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    model_type: str = "bilstm"  # "bilstm" or "transformer"
    max_length: int = 512
    pad_token_id: int = 0

    # Transformer-specific configs
    num_heads: int = 8
    feedforward_dim: int = 512
    layer_norm: bool = True


class POSTagger(nn.Module):
    """
    Sequence labeling model for Part-of-Speech tagging.
    
    Supports two architectures:
    - BiLSTM (default): Embedding -> BiLSTM -> Linear
    - Transformer: Embedding -> Transformer Encoder -> Linear
    """

    def __init__(self, config: POSTaggerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Model architecture
        if config.model_type.lower() == "transformer":
            self.encoder = self._build_transformer_encoder(config)
            encoder_output_dim = config.embedding_dim
        else:  # BiLSTM
            self.encoder = self._build_bilstm_encoder(config)
            encoder_output_dim = config.hidden_dim * 2  # Bidirectional

        # Classification head
        self.classifier = nn.Linear(encoder_output_dim, config.num_tags)

        # Initialize weights
        self._init_weights()

    def _build_bilstm_encoder(self, config: POSTaggerConfig) -> nn.Module:
        """Build BiLSTM encoder."""
        return nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

    def _build_transformer_encoder(self, config: POSTaggerConfig) -> nn.Module:
        """Build Transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=config.layer_norm
        )

        return nn.TransformerEncoder(encoder_layer, config.num_layers)

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] token ids
            attention_mask: [batch_size, seq_len] attention mask (1 for real tokens, 0 for padding)
            lengths: [batch_size] actual sequence lengths (for BiLSTM packing)
            
        Returns:
            Dict with:
                - logits: [batch_size, seq_len, num_tags] token predictions
                - hidden_states: [batch_size, seq_len, hidden_dim] encoder outputs
        """
        batch_size, seq_len = input_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        # Token embeddings
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embeddings = self.dropout(embeddings)

        # Encoder
        if self.config.model_type.lower() == "transformer":
            # For Transformer, create key padding mask
            key_padding_mask = (input_ids == self.config.pad_token_id)  # True for padding
            hidden_states = self.encoder(
                embeddings,
                src_key_padding_mask=key_padding_mask
            )
        else:
            # BiLSTM with packing for efficiency
            if lengths is not None:
                # Pack sequences for efficient processing
                packed_embeddings = pack_padded_sequence(
                    embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_outputs, _ = self.encoder(packed_embeddings)
                hidden_states, _ = pad_packed_sequence(
                    packed_outputs, batch_first=True, total_length=seq_len
                )
            else:
                hidden_states, _ = self.encoder(embeddings)

        hidden_states = self.dropout(hidden_states)

        # Token classification
        logits = self.classifier(hidden_states)  # [batch_size, seq_len, num_tags]

        return {
            "logits": logits,
            "hidden_states": hidden_states
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Generate predictions for input tokens.
        
        Returns:
            predictions: [batch_size, seq_len] predicted tag ids
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, lengths)
            predictions = torch.argmax(outputs["logits"], dim=-1)
            return predictions


def compute_sequence_lengths(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Compute actual sequence lengths for each batch item.
    
    Args:
        input_ids: [batch_size, seq_len] 
        pad_token_id: ID of padding token
        
    Returns:
        lengths: [batch_size] actual lengths
    """
    mask = (input_ids != pad_token_id)
    lengths = mask.sum(dim=1)
    return lengths


def create_pos_tagger(
    vocab_size: int,
    num_tags: int,
    model_type: str = "bilstm",
    **kwargs
) -> POSTagger:
    """
    Factory function to create POS tagger with common configurations.
    
    Args:
        vocab_size: Size of vocabulary
        num_tags: Number of POS tags
        model_type: "bilstm" or "transformer"
        **kwargs: Additional config parameters
        
    Returns:
        POSTagger model
    """
    config = POSTaggerConfig(
        vocab_size=vocab_size,
        num_tags=num_tags,
        model_type=model_type,
        **kwargs
    )
    return POSTagger(config)


# Utility functions for training
def compute_token_accuracy(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute token-level accuracy ignoring padding tokens.
    
    Args:
        predictions: [batch_size, seq_len] predicted tag ids
        labels: [batch_size, seq_len] true tag ids  
        mask: [batch_size, seq_len] mask where 1=real token, 0=padding
        
    Returns:
        Token accuracy as float
    """
    correct = (predictions == labels) & mask.bool()
    total = mask.sum()
    if total == 0:
        return 0.0
    return (correct.sum().float() / total.float()).item()


def masked_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss ignoring padding tokens.
    
    Args:
        logits: [batch_size, seq_len, num_tags]
        labels: [batch_size, seq_len] 
        mask: [batch_size, seq_len] mask where 1=real token, 0=padding
        
    Returns:
        Masked cross-entropy loss
    """
    # Flatten for loss computation
    logits_flat = logits.view(-1, logits.size(-1))  # [batch_size*seq_len, num_tags]
    labels_flat = labels.view(-1)  # [batch_size*seq_len]
    mask_flat = mask.view(-1).bool()  # [batch_size*seq_len]

    # Compute loss only for non-padding tokens
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = F.cross_entropy(logits_flat[mask_flat], labels_flat[mask_flat])
    return loss
