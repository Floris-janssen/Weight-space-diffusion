"""Activation layers used by the implicit SDF models."""

import numpy as np
import torch
import torch.nn as nn


class GaborLayer(nn.Module):
    """Single WIRE layer with complex Gabor wavelet activation.

    Splits pre-activation into frequency and scale channels, applies:
        gaussian = exp(-(scale * z_scale)²)
        output   = cos(omega_0 * z_freq) * gaussian
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 10.0,
        scale: float = 10.0,
        is_first: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.scale = scale
        # *2: frequency and scale channels
        self.linear = nn.Linear(in_features, out_features * 2)

        with torch.no_grad():
            b = np.sqrt(6.0 / in_features) / (1.0 if is_first else omega_0)
            self.linear.weight.uniform_(-b, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        z_freq, z_scale = torch.chunk(z, 2, dim=-1)

        gaussian = torch.exp(-(self.scale * z_scale) ** 2)
        real_part = torch.cos(self.omega_0 * z_freq) * gaussian
        imag_part = torch.sin(self.omega_0 * z_freq) * gaussian

        return torch.cat([real_part, imag_part], dim=-1)


class FINERLayer(nn.Module):
    """FINER layer: variable-periodic sine activation with bias-controlled frequency.

    Drop-in replacement for GaborLayer. Instead of fixed ω₀, the bias determines the
    effective frequency for each neuron.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        bias_range: tuple = (-10.0, 10.0),
        is_first: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            b = np.sqrt(6.0 / in_features) / (1.0 if is_first else omega_0)
            self.linear.weight.uniform_(-b, b)
            nn.init.uniform_(self.linear.bias, bias_range[0], bias_range[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


def get_activation_layer(
    name: str,
    in_features: int,
    out_features: int,
    omega_0: float = 10.0,
    scale: float = 10.0,
    bias_range: tuple = (-10.0, 10.0),
    is_first: bool = False,
) -> nn.Module:
    """Factory for activation layers.

    Args:
        name:       "wire" | "finer"
        omega_0:    carrier frequency
        scale:      Gaussian envelope scale (WIRE only)
        bias_range: bias init range (FINER only)
        is_first:   first layer uses different weight init

    Returns:
        GaborLayer or FINERLayer
    """
    name = name.lower()

    if name == "wire":
        return GaborLayer(in_features, out_features, omega_0, scale, is_first)

    if name == "finer":
        return FINERLayer(in_features, out_features, omega_0, bias_range, is_first)

    raise ValueError(f"Unknown activation layer: '{name}' (use 'wire' or 'finer')")