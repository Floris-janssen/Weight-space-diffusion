import itertools

import torch


def generate_mixup_adapters(adapters_tensor):
    """Generate synthetic adapter vectors via pairwise linear interpolation.

    Given a tensor of shape (N, D), this function iterates over every unique
    pair of vectors using itertools.combinations and generates 3 new vectors
    per pair using beta values 0.25, 0.5, and 0.75.

    Args:
        adapters_tensor (torch.Tensor): Tensor of shape (N, D).

    Returns:
        torch.Tensor: Concatenation of the original and synthetic vectors.
    """
    synthetic_vectors = []
    lambdas = (0.25, 0.5, 0.75)

    for v1, v2 in itertools.combinations(adapters_tensor, 2):
        for lam in lambdas:
            mixed = lam * v1 + (1.0 - lam) * v2
            synthetic_vectors.append(mixed.unsqueeze(0))

    return torch.cat([adapters_tensor] + synthetic_vectors, dim=0)


class AdapterNormalizer:
    """Normalizes adapter vectors using global mean and standard deviation.

    The statistics are computed along dimension 0 (across the dataset).
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """Calculate and store the global mean and std of the dataset.

        Args:
            data (torch.Tensor): Dataset tensor of shape (N, D).
        """
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def normalize(self, data):
        """Normalize data using the stored mean and std.

        Args:
            data (torch.Tensor): Data to normalize.

        Returns:
            torch.Tensor: Normalized data.
        """
        return (data - self.mean) / (self.std + 1e-8)

    def denormalize(self, data):
        """Reverse the normalization operation.

        Args:
            data (torch.Tensor): Normalized data.

        Returns:
            torch.Tensor: Denormalized data.
        """
        return data * (self.std + 1e-8) + self.mean


def flatten_adapters(model):
    """Flatten all LoRA 'A' and 'B' parameters into a single 1D tensor.

    Iterates through the model's named parameters, selects those whose names
    contain 'A' or 'B', flattens them, and returns their concatenation.

    Args:
        model (torch.nn.Module): Model containing LoRA adapter parameters.

    Returns:
        torch.Tensor: Concatenated 1D tensor of all flattened adapter weights.
    """
    flattened_params = []

    for name, param in model.named_parameters():
        if "A" in name or "B" in name:
            flattened_params.append(param.data.flatten())

    return torch.cat(flattened_params)


def unflatten_adapters(vector, model):
    """Copy a 1D tensor back into the model's LoRA 'A' and 'B' parameters.

    Iterates through the model's named parameters in the same order as
    flatten_adapters, slices the 1D tensor according to each parameter's
    numel(), reshapes it to the parameter's original shape, and copies the
    values back using param.data.copy_().

    Args:
        vector (torch.Tensor): 1D tensor of flattened adapter weights.
        model (torch.nn.Module): Model to update in-place.
    """
    idx = 0

    for name, param in model.named_parameters():
        if "A" in name or "B" in name:
            numel = param.numel()
            sliced = vector[idx : idx + numel]
            param.data.copy_(sliced.reshape(param.shape))
            idx += numel
