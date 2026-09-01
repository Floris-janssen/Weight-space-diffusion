"""Evaluation metrics for Phase 1 SDF fitting.
Gradient norm is computed numerically (central finite differences)
to avoid autograd graph issues after training.
"""

import torch

_FD_EPS = 1e-3
_FD_N_PTS = 1024


def compute_sdf_metrics(model, sampler, device="cuda"):
    model.eval()
    all_pts, all_sdf = [], []
    n_eval_points = 65536

    for _ in range(n_eval_points // sampler.batch_size):
        pts, sdf = sampler.sample_batch()
        all_pts.append(pts)
        all_sdf.append(sdf)

    pts = torch.cat(all_pts, dim=0)[:n_eval_points]
    target = torch.cat(all_sdf, dim=0)[:n_eval_points]

    with torch.no_grad():
        pred_full = model(pts)

        l1 = torch.abs(pred_full - target).mean().item()
        n_grad = min(_FD_N_PTS, pts.shape[0])

        idx = torch.randperm(pts.shape[0], device=pts.device)[:n_grad]
        pts_grad = pts[idx]
        grad = torch.zeros_like(pts_grad)

        for dim in range(3):
            shift = torch.zeros_like(pts_grad)
            shift[:, dim] = _FD_EPS

            sdf_plus = model(pts_grad + shift)
            sdf_minus = model(pts_grad - shift)
            grad[:, dim : dim + 1] = (sdf_plus - sdf_minus) / (2 * _FD_EPS)

        grad_norm = torch.norm(grad, dim=-1)
        eik_violation = torch.mean(torch.abs(grad_norm - 1.0)).item()
        grad_mean = grad_norm.mean().item()
        grad_std = grad_norm.std().item()
        surf_mask = (target.abs() < 0.05).squeeze()

        if surf_mask.sum() > 0:
            surf_accuracy = torch.abs(pred_full[surf_mask]).mean().item()
        else:
            surf_accuracy = float("nan")

    return {
        "recon_l1": l1,
        "eikonal_violation": eik_violation,
        "surface_accuracy": surf_accuracy,
        "grad_norm_mean": grad_mean,
        "grad_norm_std": grad_std,
    }


def compute_grid_iou(
    model,
    resolution: int = 128,
    box_min: float = -1.0,
    box_max: float = 1.0,
    device: str = "cuda",
) -> tuple:
    """Compute a signed-distance grid and a simple IoU surrogate."""
    model.eval()

    coords = torch.linspace(box_min, box_max, resolution, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")

    pts = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=-1)
    sdf_vals = []
    batch_size = 8192

    with torch.no_grad():
        for i in range(0, pts.shape[0], batch_size):
            batch = pts[i : i + batch_size]
            sdf_vals.append(model(batch).squeeze(-1))

    sdf_grid = torch.cat(sdf_vals, dim=0).reshape(resolution, resolution, resolution)
    volume_fraction = (sdf_grid < 0).float().mean().item()

    return sdf_grid, volume_fraction


def compute_chamfer_distance(
    pred_verts: torch.Tensor,
    target_verts: torch.Tensor,
    n_samples: int = 5000,
) -> float:
    """Approximate Chamfer distance between two point clouds."""
    if pred_verts.shape[0] == 0 or target_verts.shape[0] == 0:
        return float("inf")

    if pred_verts.shape[0] > n_samples:
        idx = torch.randperm(pred_verts.shape[0])[:n_samples]
        pred_verts = pred_verts[idx]

    if target_verts.shape[0] > n_samples:
        idx = torch.randperm(target_verts.shape[0])[:n_samples]
        target_verts = target_verts[idx]

    diff = pred_verts.unsqueeze(1) - target_verts.unsqueeze(0)
    dists = (diff ** 2).sum(-1)
    cd_ab = dists.min(dim=1)[0].mean().item()
    cd_ba = dists.min(dim=0)[0].mean().item()

    return cd_ab + cd_ba