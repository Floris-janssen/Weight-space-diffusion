"""SDF network built with WIRE or FINER activations."""

import numpy as np
import torch
import torch.nn as nn

from .activations import get_activation_layer


def _activation_out_multiplier(name: str) -> int:
    """How many output channels per unit this activation produces.

    "wire" (Gabor):  outputs complex → 2× (real + imag concatenated)
    "finer" (FINER): outputs real    → 1×
    """
    name = name.lower()
    multipliers = {"wire": 2, "finer": 1}

    if name not in multipliers:
        raise ValueError(
            f"Unknown activation: '{name}'. Known: {list(multipliers.keys())}"
        )

    return multipliers[name]


class WIRE_SDF(nn.Module):
    """WIRE or FINER SDF network.

    Automatically adjusts hidden-layer input widths based on activation type
    (complex → 2× channels for WIRE, real → 1× for FINER).

    Args:
        in_features:      3 (xyz input)
        hidden_features:  width of hidden layers (default 256)
        hidden_layers:    number of activation layers (default 4)
        out_features:     1 (scalar SDF)
        omega_0:          carrier frequency (10 for WIRE, 30 for FINER)
        scale:            Gaussian envelope scale (WIRE only)
        activation:       "wire" or "finer"
        bias_range:       FINER bias initialisation range (for "finer" only)
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        out_features: int = 1,
        omega_0: float = 10.0,
        scale: float = 10.0,
        activation: str = "wire",
        bias_range: tuple = (-10.0, 10.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_0 = omega_0
        self.scale = scale
        self.activation_name = activation
        self.bias_range = bias_range

        out_mult = _activation_out_multiplier(activation)
        layers = []
        first = get_activation_layer(
            activation,
            in_features,
            hidden_features,
            omega_0=omega_0,
            scale=scale,
            bias_range=bias_range,
            is_first=True,
        )
        layers.append(first)

        for _ in range(hidden_layers - 1):
            hidden = get_activation_layer(
                activation,
                out_mult * hidden_features,
                hidden_features,
                omega_0=omega_0,
                scale=scale,
                bias_range=bias_range,
                is_first=False,
            )
            layers.append(hidden)

        self.gabor_layers = nn.ModuleList(layers)
        final_in = out_mult * hidden_features
        self.final_linear = nn.Linear(final_in, out_features)

        with torch.no_grad():
            bound = np.sqrt(6.0 / final_in) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network on input points."""
        for layer in self.gabor_layers:
            x = layer(x)

        return self.final_linear(x)

    def get_weight_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_flat_weights(self) -> torch.Tensor:
        """Flatten all trainable parameters into a 1D tensor."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_weights(self, flat: torch.Tensor):
        """Load weights from a 1D vector (inverse of get_flat_weights)."""
        offset = 0

        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[offset : offset + n].view_as(p))
            offset += n

    def get_weight_shapes(self) -> list[tuple]:
        """Return list of (name, shape, numel) for each parameter."""
        return [(name, tuple(p.shape), p.numel()) for name, p in self.named_parameters()]

    def get_layer_weights(self) -> dict[str, torch.Tensor]:
        """Return linear-layer weights by name."""
        weights = {}

        for i, layer in enumerate(self.gabor_layers):
            weights[f"gabor_{i}"] = layer.linear.weight.data.clone()

        weights["final"] = self.final_linear.weight.data.clone()

        return weights


def make_wire_sdf(config: dict) -> WIRE_SDF:
    """Build a WIRE_SDF from a config dictionary."""
    m = config["model"]

    return WIRE_SDF(
        in_features=m.get("in_features", 3),
        hidden_features=m.get("hidden_features", 256),
        hidden_layers=m.get("hidden_layers", 4),
        out_features=m.get("out_features", 1),
        omega_0=m.get("omega_0", 10.0),
        scale=m.get("scale", 10.0),
        activation=m.get("activation", "wire"),
        bias_range=m.get("bias_range", (-10.0, 10.0)),
    )