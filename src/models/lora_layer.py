"""Multiplicative LoRA layer.

This module implements a lightweight adapter that multiplies a base weight
matrix by a low-rank correction factor of the form ``1 + B @ A``.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.w_base = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.A = nn.Parameter(torch.Tensor(rank, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        w_effective = self.w_base + torch.matmul(self.B, self.A)
        return nn.functional.linear(x, w_effective, self.bias)


class MultiplicativeLoRALayer(nn.Module):
    """A linear-like layer with a multiplicative low-rank adapter.

    The effective weight matrix is computed as::

        w_effective = w_base * (1.0 + B @ A)

    where ``w_base`` is a learnable base weight, and ``A`` and ``B`` are a
    low-rank adapter pair.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        rank: Rank of the low-rank adapter.
    """

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.w_base = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.w_base)

        self.bias = nn.Parameter(torch.zeros(out_features))
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the multiplicative LoRA linear transformation.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        w_effective = self.w_base * (1.0 + torch.matmul(self.B, self.A))

        return F.linear(x, w_effective, self.bias)