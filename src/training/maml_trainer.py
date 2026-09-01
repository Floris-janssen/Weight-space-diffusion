from __future__ import annotations

import copy
from typing import Any, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAMLLoRATrainer:
    """MAML-style trainer that optimizes base weights across SDF shapes with curvature guidance.

    Args:
        model: A model whose linear layers are MultiplicativeLoRALayer instances.
        inner_lr: Learning rate for the shape-specific adapter optimizer.
        outer_lr: Learning rate for the meta-learner that updates only w_base parameters.
        inner_steps: Number of adaptation steps to perform per shape.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 1e-3,
        outer_lr: float = 5e-5,
        inner_steps: int = 5,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        self.optimizer_outer = torch.optim.Adam(
            self._w_base_parameters(model),
            lr=outer_lr,
        )

    @staticmethod
    def _w_base_parameters(model: nn.Module) -> List[nn.Parameter]:
        """Return all ``w_base`` and ``bias`` parameters of ``model``."""
        return [
            p
            for name, p in model.named_parameters()
            if name.rsplit(".", 1)[-1] in ["w_base", "bias"]
        ]

    @staticmethod
    def _adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
        """Return all low-rank adapter parameters ``A`` and ``B``."""
        return [
            p
            for name, p in model.named_parameters()
            if name.rsplit(".", 1)[-1] in {"A", "B"}
        ]

    @classmethod
    def _clone_model(cls, model: nn.Module) -> nn.Module:
        """Clone model so that base weights remain differentiable with cloned tensors,
        while adapters are detached leaf parameters for the inner optimizer.
        """
        cloned = copy.deepcopy(model)

        for name, original_param in model.named_parameters():
            param_type = name.rsplit(".", 1)[-1]
            module: nn.Module = cloned
            parts = name.split(".")

            for part in parts[:-1]:
                module = module[int(part)] if part.isdigit() else getattr(module, part)

            delattr(module, parts[-1])

            if param_type in ["w_base", "bias"]:
                module.register_parameter(parts[-1], original_param.clone())
            else:
                module.register_parameter(
                    parts[-1],
                    nn.Parameter(original_param.detach().clone()),
                )

        return cloned

    @staticmethod
    def _unpack_shape(shape_data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(coords, sdf)`` from either a tuple or a dictionary."""
        if isinstance(shape_data, dict):
            return shape_data["coords"], shape_data["sdf"]

        coords, sdf = shape_data
        return coords, sdf

    def train_batch(self, batch: Iterable[Any]) -> torch.Tensor:
        """Perform one curvature-guided meta-learning step over a batch of SDF shapes."""
        self.optimizer_outer.zero_grad()
        device = next(self.model.parameters()).device
        total_loss = 0.0

        for shape_data in batch:
            coords, sdf = self._unpack_shape(shape_data)
            coords = coords.to(device)
            sdf = sdf.to(device)

            with torch.enable_grad():
                coords_curv = coords.clone().detach().requires_grad_(True)
                pred_base = self.model(coords_curv)
                grads_base = torch.autograd.grad(
                    outputs=pred_base,
                    inputs=coords_curv,
                    grad_outputs=torch.ones_like(pred_base),
                    create_graph=False,
                    retain_graph=False,
                )[0]
                eik_residual = (grads_base.norm(2, dim=-1) - 1.0).abs().detach()
                curv_weights = 1.0 + 4.0 * (
                    eik_residual / (eik_residual.max() + 1e-6)
                ).unsqueeze(-1)

            shape_model = self._clone_model(self.model)
            adapter_params = self._adapter_parameters(shape_model)
            optimizer_inner = torch.optim.Adam(adapter_params, lr=self.inner_lr)

            for p in self._w_base_parameters(shape_model):
                p.requires_grad = False

            for p in adapter_params:
                p.requires_grad = True

            for _ in range(self.inner_steps):
                optimizer_inner.zero_grad()
                pred = shape_model(coords)
                base_l1 = F.l1_loss(pred, sdf, reduction="none")
                loss = (base_l1 * curv_weights).mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                optimizer_inner.step()

            for p in self._w_base_parameters(shape_model):
                p.requires_grad = True

            coords_eik = coords.detach().requires_grad_(True)
            pred_final = shape_model(coords_eik)
            l1_final_raw = F.l1_loss(pred_final, sdf, reduction="none")
            l1_final = (l1_final_raw * curv_weights).mean()

            gradients = torch.autograd.grad(
                outputs=pred_final,
                inputs=coords_eik,
                grad_outputs=torch.ones_like(pred_final),
                create_graph=True,
                only_inputs=True,
            )[0]
            grad_norm = gradients.norm(2, dim=-1)

            eikonal_loss = F.mse_loss(grad_norm, torch.ones_like(grad_norm))
            shape_loss = l1_final + 0.1 * eikonal_loss
            scaled_loss = shape_loss / len(batch)
            scaled_loss.backward()
            total_loss += scaled_loss.item()
            del (
                shape_model,
                optimizer_inner,
                coords_curv,
                pred_base,
                grads_base,
                eik_residual,
                curv_weights,
                coords_eik,
                pred_final,
                gradients,
                loss,
                shape_loss,
                scaled_loss,
            )

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer_outer.step()

        return torch.tensor(total_loss, device=device)