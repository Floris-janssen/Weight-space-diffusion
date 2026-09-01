"""Weight-space MixUp utilities for adapter parameters.

This module provides functions for blending low-rank adapter weights directly
in parameter space, which can be used to augment a dataset of learned adapters.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch

AdapterLike = Union[Dict[str, torch.Tensor], torch.Tensor]


def _flatten_adapter(adapter: AdapterLike) -> torch.Tensor:
    """Flatten an adapter into a single 1-D tensor.

    Args:
        adapter: Either a dictionary of parameter tensors or a tensor. For a
            dictionary, tensors are flattened in lexicographic key order and
            concatenated.

    Returns:
        A 1-D tensor containing all adapter parameters.
    """
    if isinstance(adapter, dict):
        if len(adapter) == 0:
            raise ValueError("Adapter dictionary must not be empty.")

        flattened_parts: List[torch.Tensor] = []

        for key in sorted(adapter):
            param = adapter[key]

            if not isinstance(param, torch.Tensor):
                raise TypeError(
                    f"Adapter value for key {key!r} is not a torch.Tensor."
                )

            flattened_parts.append(param.detach().reshape(-1))

        return torch.cat(flattened_parts)

    if not isinstance(adapter, torch.Tensor):
        raise TypeError(
            "Adapter must be a torch.Tensor or a dictionary of torch.Tensor."
        )

    return adapter.detach().reshape(-1)


def weight_space_mixup(
    adapter_a: AdapterLike,
    adapter_b: AdapterLike,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Return a convex combination of two adapters in weight space.

    A mixing coefficient ``lambda`` is drawn from
    ``Beta(alpha, alpha)`` and the interpolated weight vector is computed as::

        theta_mix = lambda * theta_a + (1 - lambda) * theta_b

    Args:
        adapter_a: First adapter, as a dictionary or a flattened tensor.
        adapter_b: Second adapter, as a dictionary or a flattened tensor.
        alpha: Concentration parameter for the Beta distribution used to draw
            the mixing coefficient.

    Returns:
        A flattened 1-D tensor containing the interpolated weights.
    """
    theta_a = _flatten_adapter(adapter_a)
    theta_b = _flatten_adapter(adapter_b)

    if theta_a.numel() != theta_b.numel():
        raise ValueError(
            "Adapters must have the same number of flattened parameters, "
            f"got {theta_a.numel()} and {theta_b.numel()}."
        )

    theta_b = theta_b.to(theta_a.device, dtype=theta_a.dtype)

    beta_distribution = torch.distributions.Beta(
        torch.tensor(float(alpha), device=theta_a.device),
        torch.tensor(float(alpha), device=theta_a.device),
    )
    lam = beta_distribution.sample()

    return lam * theta_a + (1.0 - lam) * theta_b


def augment_dataset_with_mixup(
    adapters_list: Sequence[AdapterLike],
    num_neighbors: int = 5,
    num_variants: int = 3,
) -> List[torch.Tensor]:
    """Amplify an adapter dataset with synthetic weight-space MixUp variants.

    For each original adapter, the function finds its nearest Euclidean
    neighbors among the other adapters in the list and generates
    ``num_variants`` mixed vectors for each neighbor. The returned dataset
    contains the original flattened adapters followed by all synthetic
    variants.

    Args:
        adapters_list: Original adapters, each represented as a flattened
            tensor or a dictionary of parameter tensors.
        num_neighbors: Maximum number of nearest neighbors to use per adapter.
        num_variants: Number of synthetic variants to generate for each
            ``(adapter, neighbor)`` pair.

    Returns:
        A list of flattened adapter tensors containing the original adapters
        first, followed by the generated MixUp variants.
    """
    if not adapters_list:
        return []

    flattened_adapters = [_flatten_adapter(adapter) for adapter in adapters_list]
    expected_length = flattened_adapters[0].numel()

    if any(adapter.numel() != expected_length for adapter in flattened_adapters):
        raise ValueError("All adapters in the list must have the same length.")

    device = flattened_adapters[0].device
    dtype = flattened_adapters[0].dtype
    augmented: List[torch.Tensor] = list(flattened_adapters)

    if num_variants <= 0 or len(flattened_adapters) < 2:
        return augmented

    matrix = torch.stack([adapter.to(device, dtype) for adapter in flattened_adapters])
    effective_neighbors = min(num_neighbors, len(flattened_adapters) - 1)

    for index, adapter in enumerate(flattened_adapters):
        distances = torch.cdist(matrix[index].unsqueeze(0), matrix).squeeze(0)
        distances[index] = float("inf")
        neighbor_indices = distances.topk(effective_neighbors, largest=False).indices

        for neighbor_index in neighbor_indices.tolist():
            neighbor = flattened_adapters[neighbor_index]

            for _ in range(num_variants):
                variant = weight_space_mixup(adapter, neighbor)
                augmented.append(variant)

    return augmented