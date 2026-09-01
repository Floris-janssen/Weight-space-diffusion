"""Trainer for a single WIRE or FINER SDF model."""

import os
import time
from collections import defaultdict

import torch
import torch.optim as optim

from ..models.wire_network import WIRE_SDF
from .losses import compute_total_loss
from .metrics import compute_sdf_metrics


class Phase1Trainer:
    """Train a single WIRE_SDF network for one shape.

    Args:
        model:          WIRE_SDF network
        sampler:        SDFSampler (provides batches of (pts, sdf))
        config:         training config dict (from phase1_base.yaml)
        device:         torch device
        checkpoint_dir: where to save model checkpoints
        shape_name:     string identifier for logging
    """

    def __init__(
        self,
        model: WIRE_SDF,
        sampler,
        config: dict,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/phase1",
        shape_name: str = "unknown",
    ):
        self.model = model.to(device)
        self.sampler = sampler
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.shape_name = shape_name

        t = config["training"]
        self.epochs = t["epochs"]
        self.lr = t["lr"]
        self.eikonal_weight = t["eikonal_weight"]
        self.num_points = t["num_points"]
        self.batch_size = t["batch_size"]

        l = config["logging"]
        self.log_interval = l["log_interval"]
        self.save_interval = l["save_interval"]

        self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser,
            T_max=self.epochs,
            eta_min=self.lr * 0.01,
        )

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.history: list[dict] = []

    def train_step(self, points: torch.Tensor, target_sdf: torch.Tensor) -> dict:
        """Single training step."""
        self.model.train()
        self.optimiser.zero_grad()

        losses = compute_total_loss(
            self.model,
            points,
            target_sdf,
            self.eikonal_weight,
        )
        losses["total"].backward()
        self.optimiser.step()

        return {k: v.item() for k, v in losses.items()}

    def train(self) -> dict:
        """Run the training loop."""
        print(f"Training {self.shape_name}")
        print(f"Params: {self.model.get_weight_count():,}")
        print(f"Epochs: {self.epochs}")
        print(f"LR: {self.lr}")
        print(f"Eikonal weight: {self.eikonal_weight}")

        start_time = time.time()

        for epoch in range(self.epochs):
            pts, sdf = self.sampler.sample_batch()
            epoch_losses = defaultdict(float)
            n_batches = 0

            for b_start in range(0, self.batch_size, self.batch_size):
                b_end = min(b_start + self.batch_size, pts.shape[0])
                batch_pts = pts[b_start:b_end].detach().clone().requires_grad_(True)
                batch_sdf = sdf[b_start:b_end]
                step_losses = self.train_step(batch_pts, batch_sdf)

                for k, v in step_losses.items():
                    epoch_losses[k] += v

                n_batches += 1

            epoch_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            self.scheduler.step()

            if epoch % self.log_interval == 0 or epoch == self.epochs - 1:
                elapsed = time.time() - start_time
                lr = self.scheduler.get_last_lr()[0]

                print(
                    f"[{self.shape_name}] epoch {epoch:4d}/{self.epochs} | "
                    f"loss {epoch_losses['total']:.4e} | "
                    f"recon {epoch_losses['recon']:.4e} | "
                    f"eik {epoch_losses['eikonal']:.4e} | "
                    f"|∇| {epoch_losses['grad_norm_mean']:.3f} | "
                    f"lr {lr:.2e} | time {elapsed:.0f}s"
                )

            if epoch % self.save_interval == 0 and epoch > 0:
                self.save_checkpoint(f"{self.shape_name}_epoch{epoch}.pt")

            self.history.append(epoch_losses)

        print(f"Training complete: {self.shape_name} ({time.time() - start_time:.0f}s)")
        metrics = compute_sdf_metrics(self.model, self.sampler, self.device)

        print(f"Final metrics: {metrics}")
        self.save_checkpoint(f"{self.shape_name}_final.pt")

        return metrics

    @torch.no_grad()
    def save_checkpoint(self, filename: str):
        """Save model state dict and training config."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config,
                "history": self.history,
            },
            path,
        )

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load a previously saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", [])

        print(f"Loaded checkpoint: {path}")