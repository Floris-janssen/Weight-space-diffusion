"""Vector Quantized Variational Autoencoder (VQ-VAE) for 1D weight vectors.

This module implements a discrete latent representation for neural network
weights / LoRA adapter vectors. It quantizes continuous latent embeddings
against a learned codebook using the straight-through estimator.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantization (VQ) layer with straight-through estimator.

    Args:
        num_embeddings: Size of the discrete codebook ($K$).
        embedding_dim: Dimensionality of each codebook vector ($D$).
        commitment_cost: Weighting factor ($\beta$) for the commitment loss.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings,
            1.0 / self.num_embeddings,
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent vectors.

        Args:
            inputs: Continuous latent tensor of shape (B, num_tokens, embedding_dim).

        Returns:
            quantized: Quantized tensor with straight-through gradients.
            loss: Combined VQ and commitment loss.
            perplexity: Codebook perplexity measuring code utilization.
            encoding_indices: Codebook indices of shape (B, num_tokens).
        """
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2.0 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1.0)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        indices_out = encoding_indices.view(input_shape[0], input_shape[1])

        return quantized, vq_loss, perplexity, indices_out

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Retrieve codebook vectors corresponding to discrete indices.

        Args:
            indices: Tensor of shape (B, num_tokens).

        Returns:
            Quantized vectors of shape (B, num_tokens, embedding_dim).
        """
        return self.embedding(indices)


class ResBlock1D(nn.Module):
    """1D residual layer for MLP / latent feature refinement."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class WeightVQVAE(nn.Module):
    """VQ-VAE for high-dimensional 1D weight and adapter vectors.

    Args:
        input_dim: Dimensionality of the flattened weight vector ($D_{in}$).
        num_tokens: Number of discrete latent tokens / chunks ($L$).
        embedding_dim: Dimension of each discrete codebook vector ($D$).
        num_embeddings: Number of discrete embeddings in the codebook ($K$).
        commitment_cost: Commitment loss weight ($\beta$).
    """

    def __init__(
        self,
        input_dim: int,
        num_tokens: int = 16,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.latent_total_dim = num_tokens * embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            ResBlock1D(2048),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            ResBlock1D(1024),
            nn.Linear(1024, self.latent_total_dim),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_total_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            ResBlock1D(1024),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            ResBlock1D(2048),
            nn.Linear(2048, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input weights to continuous latent tokens (B, L, D)."""
        z_e = self.encoder(x)
        return z_e.view(-1, self.num_tokens, self.embedding_dim)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latent tokens to reconstructed weights (B, D_in)."""
        flat_z_q = z_q.view(-1, self.latent_total_dim)
        return self.decoder(flat_z_q)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete token indices directly to weight space."""
        z_q = self.quantizer.decode_indices(indices)
        return self.decode(z_q)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VQ-VAE.

        Args:
            x: Input weight tensor of shape (B, input_dim).

        Returns:
            recon_x: Reconstructed weights of shape (B, input_dim).
            vq_loss: Vector quantization loss.
            perplexity: Codebook perplexity.
            indices: Discrete token indices of shape (B, num_tokens).
        """
        z_e = self.encode(x)
        z_q, vq_loss, perplexity, indices = self.quantizer(z_e)
        recon_x = self.decode(z_q)

        return recon_x, vq_loss, perplexity, indices


def vqvae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    vq_loss: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the combined VQ-VAE reconstruction and quantization loss.

    Args:
        recon_x: Reconstructed weights from decoder.
        x: Ground-truth input weights.
        vq_loss: VQ and commitment loss from VectorQuantizer.

    Returns:
        total_loss: recon_loss + vq_loss.
        recon_loss: MSE reconstruction loss.
        vq_loss: Vector quantization loss.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    total_loss = recon_loss + vq_loss

    return total_loss, recon_loss, vq_loss