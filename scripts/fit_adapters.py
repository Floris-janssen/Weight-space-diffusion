import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils.sdf_dataset import SDFSampler, ShapeDataManager
from src.models.finer_maml import FINER_SDF_MAML
from src.utils.weight_space_utils import flatten_adapters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed_chairs")
    output_dir = os.path.join(PROJECT_ROOT, "data", "adapters_chairs")

    os.makedirs(output_dir, exist_ok=True)
    base_ckpt_path = os.path.join(PROJECT_ROOT, "model_epoch_250.pt")

    if not os.path.exists(base_ckpt_path):
        raise FileNotFoundError("Run train_maml_base.py first to generate model_epoch_250.pt.")

    print("Loading base MAML model")
    model = FINER_SDF_MAML(hidden_features=512, num_layers=4, rank=16, omega_0=10.0).to(device)

    ckpt = torch.load(base_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    data_manager = ShapeDataManager(data_dir, device=device)

    print(f"Fitting adapters for {data_manager.num_shapes} chairs")

    for sid in data_manager.shape_ids:
        out_path = os.path.join(output_dir, f"adapter_{sid}.pt")
        if os.path.exists(out_path):
            continue

        ds = data_manager.get_dataset(sid)
        sampler = SDFSampler(ds, num_points=4096, batch_size=4096, device=device)

        shape_model = FINER_SDF_MAML(hidden_features=512, num_layers=4, rank=16, omega_0=10.0).to(device)
        shape_model.load_state_dict(model.state_dict())
        shape_model.train()
        adapter_params = [p for n, p in shape_model.named_parameters() if "A" in n or "B" in n]

        for p in shape_model.parameters():
            p.requires_grad = False

        for p in adapter_params:
            p.requires_grad = True

        optimizer = torch.optim.Adam(adapter_params, lr=1e-3)

        for _ in range(150):
            optimizer.zero_grad()
            pts, sdf = sampler.sample_batch()
            pts.requires_grad_(True)

            pred = shape_model(pts)
            l1_loss = F.l1_loss(pred, sdf)

            grads = torch.autograd.grad(
                outputs=pred,
                inputs=pts,
                grad_outputs=torch.ones_like(pred),
                create_graph=True,
            )[0]

            eikonal_loss = F.mse_loss(
                grads.norm(2, dim=-1),
                torch.ones_like(grads.norm(2, dim=-1)),
            )
            loss = l1_loss + 0.1 * eikonal_loss
            loss.backward()
            optimizer.step()

        flat_adapter = flatten_adapters(shape_model).cpu()
        torch.save(flat_adapter, out_path)
        print(f"Saved adapter for shape {sid}")

    print("Adapter extraction complete")


if __name__ == "__main__":
    main()