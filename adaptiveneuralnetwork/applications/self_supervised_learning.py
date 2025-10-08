"""
Self-supervised learning modules for adaptive neural networks.

This module implements self-supervised signal prediction and representation learning
as part of Phase 1: Adaptive Learning & Continual Improvement.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SelfSupervisedConfig:
    """Configuration for self-supervised learning."""
    # Signal prediction parameters
    prediction_horizon: int = 10  # Steps ahead to predict
    context_window: int = 50     # Historical context length
    hidden_dim: int = 128        # Hidden layer dimensions

    # Representation learning
    embedding_dim: int = 64      # Representation dimension
    contrastive_margin: float = 1.0  # Contrastive loss margin

    # Training parameters
    learning_rate: float = 0.001
    temperature: float = 0.1     # Temperature for contrastive learning


class SignalPredictor(nn.Module):
    """Self-supervised signal prediction module."""

    def __init__(self, config: SelfSupervisedConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # Encoder for historical context
        self.context_encoder = nn.LSTM(
            input_dim,
            config.hidden_dim,
            batch_first=True
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, input_dim * config.prediction_horizon)
        )

        logger.debug(f"Initialized SignalPredictor with {input_dim} input dims")

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict future signals from historical context.
        
        Args:
            context: Historical signal context [batch, seq_len, input_dim]
            
        Returns:
            predictions: Future signal predictions [batch, horizon, input_dim]
        """
        # Encode context
        _, (hidden, _) = self.context_encoder(context)

        # Predict future signals
        predictions = self.predictor(hidden.squeeze(0))
        predictions = predictions.view(
            -1,
            self.config.prediction_horizon,
            self.input_dim
        )

        return predictions

    def compute_prediction_loss(
        self,
        context: torch.Tensor,
        future: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction loss."""
        predictions = self.forward(context)
        return nn.MSELoss()(predictions, future)


class ContrastiveRepresentationLearner(nn.Module):
    """Contrastive learning for signal representations."""

    def __init__(self, config: SelfSupervisedConfig, input_dim: int):
        super().__init__()
        self.config = config

        # Representation encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )

        logger.debug("Initialized ContrastiveRepresentationLearner")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute signal representations.
        
        Args:
            x: Input signals [batch, input_dim]
            
        Returns:
            embeddings: Signal embeddings [batch, embedding_dim]
            projections: Projected embeddings for contrastive learning
        """
        embeddings = self.encoder(x)
        projections = self.projection_head(embeddings)
        return embeddings, projections

    def compute_contrastive_loss(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss using NT-Xent."""
        batch_size = projections.size(0)
        temperature = self.config.temperature

        # Normalize projections
        projections = nn.functional.normalize(projections, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / temperature

        # Create positive/negative masks
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Average over positive pairs
        loss = -(log_prob * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return loss.mean()


class SelfSupervisedLearningSystem(nn.Module):
    """Complete self-supervised learning system."""

    def __init__(self, config: SelfSupervisedConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim

        # Initialize components
        self.signal_predictor = SignalPredictor(config, input_dim)
        self.representation_learner = ContrastiveRepresentationLearner(config, input_dim)

        # Combined optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate
        )

        logger.info("Initialized SelfSupervisedLearningSystem")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility with other systems."""
        # Use representation learner as the main forward pass
        embeddings, _ = self.representation_learner(x)
        return embeddings

    def train_step(
        self,
        signals: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> dict[str, float]:
        """
        Perform one training step.
        
        Args:
            signals: Input signals [batch, seq_len, input_dim]
            labels: Optional labels for contrastive learning
            
        Returns:
            Dictionary of loss values
        """
        self.train()
        losses = {}

        # Split signals into context and future
        context_len = self.config.context_window
        if signals.size(1) > context_len + self.config.prediction_horizon:
            context = signals[:, :context_len]
            future = signals[:, context_len:context_len + self.config.prediction_horizon]

            # Prediction loss
            pred_loss = self.signal_predictor.compute_prediction_loss(context, future)
            losses['prediction_loss'] = pred_loss.item()
        else:
            pred_loss = torch.tensor(0.0, device=signals.device)
            losses['prediction_loss'] = 0.0

        # Representation learning loss
        if labels is not None:
            # Use last timestep for representation learning
            last_signals = signals[:, -1] if signals.dim() == 3 else signals
            _, projections = self.representation_learner(last_signals)
            contrastive_loss = self.representation_learner.compute_contrastive_loss(
                projections, labels
            )
            losses['contrastive_loss'] = contrastive_loss.item()
        else:
            contrastive_loss = torch.tensor(0.0, device=signals.device)
            losses['contrastive_loss'] = 0.0

        # Combined loss
        total_loss = pred_loss + contrastive_loss
        losses['total_loss'] = total_loss.item()

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return losses

    def get_representations(self, signals: torch.Tensor) -> torch.Tensor:
        """Extract learned representations from signals."""
        self.eval()
        with torch.no_grad():
            if signals.dim() == 3:
                signals = signals[:, -1]  # Use last timestep
            embeddings, _ = self.representation_learner(signals)
            return embeddings

    def predict_future(self, context: torch.Tensor) -> torch.Tensor:
        """Predict future signals from context."""
        self.eval()
        with torch.no_grad():
            predictions = self.signal_predictor(context)
            return predictions
