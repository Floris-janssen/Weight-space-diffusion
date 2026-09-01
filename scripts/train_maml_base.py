import os
import random
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils.sdf_dataset import SDFSampler, ShapeDataManager
from src.models.finer_maml import FINER_SDF_MAML
from src.training.maml_trainer import MAMLLoRATrainer


class CurvatureMAMLLoRATrainer(MAMLLoRATrainer):
    def train_batch(self, batch):
        self.optimizer_outer.zero_grad()
        device = next(self.model.parameters()).device
        total_loss = 0.0

        for shape_data in batch:
            coords, sdf = self._unpack_shape(shape_data)
            coords, sdf = coords.to(device), sdf.to(device)

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
                curv_weights = 1.0 + 4.0 * (eik_residual / (eik_residual.max() + 1e-6)).unsqueeze(-1)

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
                loss = (F.l1_loss(pred, sdf, reduction="none") * curv_weights).mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
                optimizer_inner.step()

            for p in self._w_base_parameters(shape_model):
                p.requires_grad = True

            coords_eik = coords.detach().requires_grad_(True)
            pred_final = shape_model(coords_eik)
            l1_final = (F.l1_loss(pred_final, sdf, reduction="none") * curv_weights).mean()

            gradients = torch.autograd.grad(
                outputs=pred_final,
                inputs=coords_eik,
                grad_outputs=torch.ones_like(pred_final),
                create_graph=True,
                only_inputs=True,
            )[0]

            eikonal_loss = F.mse_loss(
                gradients.norm(2, dim=-1),
                torch.ones_like(gradients.norm(2, dim=-1)),
            )
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed_chairs")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "maml_lora")

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading shape data")
    data_manager = ShapeDataManager(data_dir, device=device)
    max_shapes = min(500, data_manager.num_shapes)
    shape_ids = data_manager.shape_ids[:max_shapes]

    all_shape_batches = []
    for sid in shape_ids:
        ds = data_manager.get_dataset(sid)
        sampler = SDFSampler(ds, num_points=2048, batch_size=2048, device=device)
        pts, sdf = sampler.sample_batch()
        all_shape_batches.append({"coords": pts, "sdf": sdf})

    model = FINER_SDF_MAML(hidden_features=512, num_layers=4, rank=16, omega_0=10.0).to(device)

    print("Pre-training base prior")
    base_params = [p for n, p in model.named_parameters() if "w_base" in n or "bias" in n]

    optimizer_prior = torch.optim.Adam(base_params, lr=5e-4)
    ds_prior = data_manager.get_dataset(shape_ids[0])
    sampler_prior = SDFSampler(ds_prior, num_points=4096, batch_size=4096, device=device)

    model.train()
    for step in range(1, 1501):
        optimizer_prior.zero_grad()
        pts, target_sdf = sampler_prior.sample_batch()
        pts.requires_grad_(True)

        pred = model(pts)
        l1 = F.l1_loss(pred, target_sdf)
        grads = torch.autograd.grad(
            outputs=pred,
            inputs=pts,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
        )[0]

        eikonal = F.mse_loss(grads.norm(2, dim=-1), torch.ones_like(grads.norm(2, dim=-1)))
        loss = l1 + 0.1 * eikonal
        loss.backward()
        optimizer_prior.step()

    print("Starting MAML training")
    trainer = CurvatureMAMLLoRATrainer(model=model, inner_lr=1e-3, outer_lr=5e-5)

    for epoch in range(250):
        trainer.inner_steps = 5 if epoch < 50 else (10 if epoch < 150 else 20)
        random.shuffle(all_shape_batches)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(all_shape_batches), 16):
            batch = all_shape_batches[i : i + 16]
            loss = trainer.train_batch(batch)
            epoch_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch + 1:03d}/250 | Loss: {epoch_loss / num_batches:.4f}")

        if (epoch + 1) % 50 == 0:
            torch.save({"model_state_dict": model.state_dict()}, os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt"))

    torch.save({"model_state_dict": model.state_dict()}, os.path.join(PROJECT_ROOT, "model_epoch_250.pt"))
    print("MAML training complete")


if __name__ == "__main__":
    main()