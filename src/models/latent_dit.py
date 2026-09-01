import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Embeds continuous diffusion timesteps into a latent vector."""

    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.freq_emb_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.freq_emb_size // 2
        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        return self.mlp(emb)


class DiTBlock(nn.Module):
    """A 1D Transformer block with Adaptive LayerNorm (adaLN) conditioning."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm1 = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm2 = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class LatentDiT1D(nn.Module):
    """Diffusion Transformer designed for 16-token x 8-dim VQ-VAE latents."""

    def __init__(
        self,
        input_dim=128,
        num_tokens=16,
        token_dim=8,
        hidden_dim=384,
        depth=12,
        num_heads=12,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.x_embedder = nn.Linear(token_dim, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, hidden_dim),
            requires_grad=True,
        )
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, x, t):
        """
        x: [Batch, 128] continuous noisy latent
        t: [Batch] continuous timesteps
        """
        x_seq = x.view(x.shape[0], self.num_tokens, self.token_dim)
        x_seq = self.x_embedder(x_seq) + self.pos_embed
        c = self.t_embedder(t)

        for block in self.blocks:
            x_seq = block(x_seq, c)

        x_out = self.final_layer(x_seq)

        return x_out.view(x.shape[0], -1)