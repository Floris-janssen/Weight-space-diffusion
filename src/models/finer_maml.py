from __future__ import annotations

import torch
import torch.nn as nn

from src.models.lora_layer import MultiplicativeLoRALayer


class FINER_SDF_MAML(nn.Module):
    """A FINER-style SDF network using multiplicative LoRA linear layers.

    Args:
        hidden_features: Width of the hidden layers.
        num_layers: Number of hidden layers.
        rank: Rank of the low-rank adapters in each MultiplicativeLoRALayer.
        omega_0: Frequency scaling factor for the FINER activation.
    """

    def __init__(
        self,
        hidden_features: int,
        num_layers: int,
        rank: int,
        omega_0: float,
    ) -> None:
        super().__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.rank = rank
        self.omega_0 = omega_0

        self.input_layer = MultiplicativeLoRALayer(3, hidden_features, rank=rank)
        self.hidden_layers = nn.ModuleList(
            [
                MultiplicativeLoRALayer(hidden_features, hidden_features, rank=rank)
                for _ in range(num_layers - 1)
            ]
        )
        self.output_layer = MultiplicativeLoRALayer(hidden_features, 1, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FINER SDF network.

        Args:
            x: Input coordinates of shape ``(..., 3)``.
        Returns:
            Predicted SDF values of shape ``(..., 1)``.
        """
        x = torch.sin(self.omega_0 * self.input_layer(x))

        for layer in self.hidden_layers:
            x = torch.sin(self.omega_0 * layer(x))

        return self.output_layer(x)