"""Trainer for the weight-space VQ-VAE."""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader


class VQVAETrainer:
    """Trainer for discrete WeightVQVAE models.

    Args:
        vqvae_model: The WeightVQVAE model instance.
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay factor.
        max_grad_norm: Maximum norm for gradient clipping.
        device: Computational device ('cuda' or 'cpu').
        checkpoint_dir: Directory to save model checkpoints.
    """

    def __init__(
        self,
        vqvae_model: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/vqvae",
    ) -> None:
        self.model = vqvae_model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history: list[dict] = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> Dict[str, float]:
        """Run a single training epoch.

        Args:
            dataloader: DataLoader supplying batches of flattened weight vectors.
            loss_fn: Loss function (e.g. vqvae_loss_function).

        Returns:
            Dictionary containing average losses and codebook perplexity.
        """
        self.model.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        vq_loss_sum = 0.0
        perplexity_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)
            self.optimizer.zero_grad()

            recon_x, vq_loss, perplexity, _ = self.model(x)
            total_loss, recon_loss, vq_loss = loss_fn(recon_x, x, vq_loss)
            total_loss.backward()

            if self.max_grad_norm > 0:
                nn_utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm,
                )

            self.optimizer.step()
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            vq_loss_sum += vq_loss.item()
            perplexity_sum += perplexity.item()
            num_batches += 1

        if num_batches == 0:
            return {"total": 0.0, "recon": 0.0, "vq": 0.0, "perplexity": 0.0}

        return {
            "total": total_loss_sum / num_batches,
            "recon": recon_loss_sum / num_batches,
            "vq": vq_loss_sum / num_batches,
            "perplexity": perplexity_sum / num_batches,
        }

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        loss_fn: Callable = None,
        log_interval: int = 10,
        save_interval: int = 50,
    ) -> list[dict]:
        """Execute the full training loop over the specified epochs."""
        if loss_fn is None:
            from src.models.weight_vqvae import vqvae_loss_function

            loss_fn = vqvae_loss_function

        print(f"Training VQ-VAE: input_dim={self.model.input_dim}, tokens={self.model.num_tokens}, embedding_dim={self.model.embedding_dim}, codebook={self.model.quantizer.num_embeddings}, epochs={epochs}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(dataloader, loss_fn)
            self.history.append(metrics)

            if epoch % log_interval == 0 or epoch == epochs:
                elapsed = time.time() - start_time

                print(
                    f"Epoch {epoch:4d}/{epochs:4d} | "
                    f"Total Loss: {metrics['total']:.5f} | "
                    f"Recon MSE: {metrics['recon']:.5f} | "
                    f"VQ Loss: {metrics['vq']:.5f} | "
                    f"Perplexity: {metrics['perplexity']:.2f}/{self.model.quantizer.num_embeddings} | "
                    f"Time: {elapsed:.0f}s"
                )

            if epoch % save_interval == 0 or epoch == epochs:
                self.save_checkpoint(f"vqvae_epoch_{epoch}.pt")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save VQ-VAE weights and training history."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
                "model_config": {
                    "input_dim": self.model.input_dim,
                    "num_tokens": self.model.num_tokens,
                    "embedding_dim": self.model.embedding_dim,
                    "num_embeddings": self.model.quantizer.num_embeddings,
                },
            },
            path,
        )

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        self.history = ckpt.get("history", [])
        print(f"Loaded checkpoint: {path}")