import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.latent_dit import LatentDiT1D
from src.models.weight_vqvae import WeightVQVAE


class DDPM:
    def __init__(self, num_steps=1000, device="cuda"):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_a = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
        sqrt_one_minus_a = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1)
        return sqrt_a * x_0 + sqrt_one_minus_a * noise, noise


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "diffusion")
    os.makedirs(output_dir, exist_ok=True)

    adapters_dir = os.path.join(PROJECT_ROOT, "data", "adapters_chairs")
    vqvae_dir = os.path.join(PROJECT_ROOT, "outputs", "VQ_VAE")
    adapter_files = os.listdir(adapters_dir)

    tensors = [
        torch.load(os.path.join(adapters_dir, f), map_location="cpu")
        for f in adapter_files
        if f.endswith(".pt")
    ]
    raw_adapters = torch.stack(tensors)

    stats = torch.load(os.path.join(vqvae_dir, "adapter_normalizer_stats.pt"))
    normalizer_mean, normalizer_std = stats["mean"], stats["std"]
    normalized_data = (raw_adapters - normalizer_mean) / (normalizer_std + 1e-8)

    print("Loading VQ-VAE")
    vqvae = WeightVQVAE(
        input_dim=raw_adapters.shape[1],
        num_tokens=16,
        embedding_dim=8,
        num_embeddings=256,
    ).to(device)
    ckpt = torch.load(os.path.join(vqvae_dir, "vqvae_final.pt"), map_location=device)
    vqvae.load_state_dict(ckpt["model_state"])
    vqvae.eval()

    latent_vectors = []
    with torch.no_grad():
        for i in range(0, normalized_data.shape[0], 32):
            batch = normalized_data[i : i + 32].to(device)
            z_e = vqvae.encode(batch)
            latent_vectors.append(z_e.reshape(z_e.shape[0], -1).cpu())

    all_latents = torch.cat(latent_vectors, dim=0)
    latent_mean, latent_std = all_latents.mean(), all_latents.std()
    norm_latents = (all_latents - latent_mean) / (latent_std + 1e-8)
    torch.save({"mean": latent_mean, "std": latent_std}, os.path.join(vqvae_dir, "dit_latent_stats.pt"))

    train_loader = DataLoader(TensorDataset(norm_latents), batch_size=64, shuffle=True)
    print("Initializing LatentDiT1D")

    dit = LatentDiT1D(input_dim=128, hidden_dim=384, depth=12, num_heads=12).to(device)
    ddpm = DDPM(num_steps=1000, device=device)
    optimizer = torch.optim.AdamW(dit.parameters(), lr=1e-4)
    epochs = 10000

    print(f"Training DiT for {epochs} epochs")
    start_time = time.time()
    dit.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            x_0 = batch[0].to(device)
            t = torch.randint(0, ddpm.num_steps, (x_0.shape[0],), device=device)
            x_t, noise = ddpm.add_noise(x_0, t)
            optimizer.zero_grad()
            pred_noise = dit(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss / len(train_loader):.5f}")

    torch.save(dit.state_dict(), os.path.join(output_dir, "dit_model_best.pt"))
    print("DiT training complete")


if __name__ == "__main__":
    main()