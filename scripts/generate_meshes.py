import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.flexicubes import FlexiCubes
from src.models.finer_maml import FINER_SDF_MAML
from src.models.latent_dit import LatentDiT1D
from src.models.weight_vqvae import WeightVQVAE
from src.utils.weight_space_utils import unflatten_adapters


class DDIMSampler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device="cuda",
    ):
        self.device = device
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_train_timesteps,
            dtype=torch.float32,
            device=device,
        )
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)

    def sample_ddim(
        self,
        model,
        shape,
        ddim_steps=50,
        eta=0.0,
        seed=None,
        temperature=1.0,
    ):
        timesteps = torch.linspace(
            self.alphas_cumprod.shape[0] - 1,
            0,
            ddim_steps,
            dtype=torch.long,
            device=self.device,
        )

        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        x_t = torch.randn(shape, generator=gen, device=self.device) * temperature

        with torch.no_grad():
            for i in range(len(timesteps)):
                t = timesteps[i]
                prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=self.device)

                noise_pred = model(
                    x_t,
                    torch.full((shape[0],), t, device=self.device, dtype=torch.long),
                )
                alpha_prod_t = self.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.alphas_cumprod[prev_t]
                    if prev_t >= 0
                    else torch.tensor(1.0, device=self.device)
                )

                sigma_t = (
                    eta
                    * torch.sqrt(
                        torch.clamp(
                            (1.0 - alpha_prod_t_prev)
                            / (1.0 - alpha_prod_t)
                            * (1.0 - alpha_prod_t / alpha_prod_t_prev),
                            min=0.0,
                        )
                    )
                    if prev_t >= 0
                    else torch.tensor(0.0, device=self.device)
                )

                pred_x0 = (x_t - torch.sqrt(1.0 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
                pred_dir = torch.sqrt(torch.clamp(1.0 - alpha_prod_t_prev - sigma_t**2, min=0.0)) * noise_pred
                noise = torch.randn_like(x_t) if prev_t >= 0 and eta > 0.0 else torch.zeros_like(x_t)
                x_t = torch.sqrt(alpha_prod_t_prev) * pred_x0 + pred_dir + sigma_t * noise

        return x_t


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "generated_meshes")

    os.makedirs(output_dir, exist_ok=True)
    print("Loading models")

    maml = FINER_SDF_MAML(hidden_features=512, num_layers=4, rank=16, omega_0=10.0).to(device)
    ckpt = torch.load(os.path.join(PROJECT_ROOT, "model_epoch_250.pt"), map_location=device)
    maml.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    maml.eval()

    adapters_dir = os.path.join(PROJECT_ROOT, "data", "adapters_chairs")
    sample_adapter = torch.load(
        os.path.join(adapters_dir, os.listdir(adapters_dir)[0]),
        map_location="cpu",
    )

    vqvae = WeightVQVAE(
        input_dim=sample_adapter.shape[0],
        num_tokens=16,
        embedding_dim=8,
        num_embeddings=256,
    ).to(device)

    vq_ckpt = torch.load(os.path.join(PROJECT_ROOT, "outputs", "VQ_VAE", "vqvae_final.pt"), map_location=device)
    vqvae.load_state_dict(vq_ckpt["model_state"])
    vqvae.eval()

    dit = LatentDiT1D(input_dim=128, hidden_dim=384, depth=12, num_heads=12).to(device)
    dit.load_state_dict(
        torch.load(
            os.path.join(PROJECT_ROOT, "outputs", "diffusion", "dit_model_best.pt"),
            map_location=device,
        )
    )
    dit.eval()

    vqvae_dir = os.path.join(PROJECT_ROOT, "outputs", "VQ_VAE")
    latent_stats = torch.load(os.path.join(vqvae_dir, "dit_latent_stats.pt"))
    l_mean, l_std = latent_stats["mean"].to(device), latent_stats["std"].to(device)

    adapter_stats = torch.load(os.path.join(vqvae_dir, "adapter_normalizer_stats.pt"))
    a_mean, a_std = adapter_stats["mean"].to(device), adapter_stats["std"].to(device)

    fc = FlexiCubes(device=device)
    res = 256
    voxel_coords, cube_indices = fc.construct_voxel_grid(res)
    pts = voxel_coords * 2.0

    print("Generating chair meshes")
    sampler = DDIMSampler(device=device)

    with torch.no_grad():
        num_samples = 3
        scaled_latents = sampler.sample_ddim(
            dit,
            shape=(num_samples, 128),
            ddim_steps=50,
            temperature=1.0,
        )
        generated_latents = (scaled_latents * l_std) + l_mean

        for i in range(num_samples):
            z_flat = generated_latents[i].unsqueeze(0)
            z_seq = z_flat.view(1, 16, 8)
            z_snapped, _, _, _ = vqvae.quantizer(z_seq)
            norm_weights = vqvae.decode(z_snapped)
            raw_weights = (norm_weights * a_std) + a_mean
            unflatten_adapters(raw_weights[0], maml)
            sdf_vals = []

            for b in range(0, pts.shape[0], 8192):
                sdf_vals.append(maml(pts[b : b + 8192]).squeeze(-1))

            sdf_vals = torch.cat(sdf_vals, dim=0)
            verts, faces, _ = fc(
                x_nx3=voxel_coords,
                s_n=sdf_vals,
                cube_fx8=cube_indices,
                res=res,
            )
            verts = verts * 2.0
            out_path = os.path.join(output_dir, f"generated_chair_{i + 1}.obj")

            with open(out_path, "w") as f:
                for v in verts.cpu().numpy():
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for face in faces.cpu().numpy():
                    f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

            print(f"Saved mesh: {out_path}")

    print("Mesh generation complete")


if __name__ == "__main__":
    main()