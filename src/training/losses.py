"""Loss functions for SDF fitting and Weight-Space VQ-VAE functional validation.

L = L_recon + \lambda * L_eikonal
L_recon:   L1 loss between predicted and ground-truth SDF values
L_eikonal: Eikonal regularisation enforcing ||\nabla f|| = 1 everywhere
"""

import torch

from src.utils.weight_space_utils import unflatten_adapters


def compute_gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradient of y w.r.t. x via autograd.

    Args:
        y: [B, 1] scalar field values
        x: [B, 3] spatial coordinates (requires_grad=True)

    Returns:
        grad: [B, 3] gradient vectors
    """
    grad_outputs = torch.ones_like(y, device=y.device)

    grads = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    if grads is None:
        raise RuntimeError("gradient is None - did you set requires_grad on the input?")

    return grads


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss."""
    return torch.abs(pred - target).mean()


def eikonal_loss(
    gradient_norms: torch.Tensor,
    target_norm: float = 1.0,
) -> torch.Tensor:
    """Eikonal regularisation."""
    return ((gradient_norms - target_norm) ** 2).mean()


def compute_total_loss(
    model: torch.nn.Module,
    points: torch.Tensor,
    target_sdf: torch.Tensor,
    eikonal_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute the combined Phase 1 training loss.

    Args:
        model: WIRE_SDF / FINER network
        points: [B, 3] query points
        target_sdf: [B, 1] ground-truth SDF values
        eikonal_weight: Weight for Eikonal regularization

    Returns:
        dict with keys: 'total', 'recon', 'eikonal', 'grad_norm_mean'
    """
    points.requires_grad_(True)

    pred = model(points)
    recon = torch.abs(pred - target_sdf).mean()
    grad = compute_gradient(pred, points)

    grad_norm = torch.norm(grad, dim=-1)
    eik = ((grad_norm - 1.0) ** 2).mean()
    total = recon + eikonal_weight * eik

    return {
        "total": total,
        "recon": recon,
        "eikonal": eik,
        "grad_norm_mean": grad_norm.mean(),
    }


def functional_vqvae_loss(
    decoded_weights: torch.Tensor,
    original_weights: torch.Tensor,
    model: torch.nn.Module,
    probe_pts: torch.Tensor,
) -> float:
    """Evaluate geometric SDF difference between original and VQ-VAE decoded weights.

    Args:
        decoded_weights: [D] 1D weight tensor reconstructed by the VQ-VAE.
        original_weights: [D] 1D ground-truth weight tensor.
        model: Underlying INR/SDF model (e.g. FINER_SDF_MAML).
        probe_pts: [N, 3] query points sampled in [-1, 1]^3.

    Returns:
        Mean absolute difference in predicted SDF values.
    """
    model.eval()

    with torch.no_grad():
        unflatten_adapters(original_weights, model)
        orig_sdf = model(probe_pts)
        unflatten_adapters(decoded_weights, model)
        recon_sdf = model(probe_pts)

    return (recon_sdf - orig_sdf).abs().mean().item()