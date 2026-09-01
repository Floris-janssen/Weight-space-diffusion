"""Utilities for SDF slice and metric plots."""

import os

import numpy as np
import torch


def render_sdf_slice(
    model,
    shape_name: str,
    save_dir: str,
    resolution: int = 256,
    axis: str = "z",
    slice_val: float = 0.0,
    box_min: float = -1.5,
    box_max: float = 1.5,
    device: str = "cuda",
):
    """Render a 2D slice through the SDF and save to disk.

    Produces two subplots:
        Left:  SDF heatmap (RdBu) with zero-level-set contour
        Right: |∇SDF| heatmap (inferno)

    Args:
        model:      WIRE_SDF network
        shape_name: identifier for filename
        save_dir:   directory for output PNG
        resolution: per-axis resolution
        axis:       slice axis ("x", "y", or "z")
        slice_val:  position along the slice axis
        box_min/max: spatial bounds
        device:     torch device
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    lin = torch.linspace(box_min, box_max, resolution, device=device)
    xx, yy = torch.meshgrid(lin, lin, indexing="ij")

    if axis == "z":
        pts = torch.stack(
            [
                xx.ravel(),
                yy.ravel(),
                torch.full_like(xx.ravel(), slice_val),
            ],
            dim=-1,
        )
        xlabel, ylabel = "x", "y"
    elif axis == "y":
        pts = torch.stack(
            [
                xx.ravel(),
                torch.full_like(xx.ravel(), slice_val),
                yy.ravel(),
            ],
            dim=-1,
        )
        xlabel, ylabel = "x", "z"
    else:
        pts = torch.stack(
            [
                torch.full_like(xx.ravel(), slice_val),
                xx.ravel(),
                yy.ravel(),
            ],
            dim=-1,
        )
        xlabel, ylabel = "y", "z"

    sdf_vals = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], 8192):
            batch = pts[i : i + 8192]
            sdf_vals.append(model(batch).squeeze(-1).cpu())

    sdf = torch.cat(sdf_vals, dim=0).reshape(resolution, resolution).numpy()
    cell_size = (box_max - box_min) / (resolution - 1)
    gx = np.gradient(sdf, axis=0) / cell_size
    gy = np.gradient(sdf, axis=1) / cell_size
    grad_norm = np.sqrt(gx**2 + gy**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(
        sdf.T,
        origin="lower",
        extent=[box_min, box_max, box_min, box_max],
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )
    ax1.contour(
        sdf.T,
        levels=[0],
        colors="black",
        linewidths=2,
        extent=[box_min, box_max, box_min, box_max],
    )
    ax1.set_title(f"{shape_name} — SDF ({axis}={slice_val})")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.colorbar(im1, ax=ax1, label="SDF value")

    im2 = ax2.imshow(
        grad_norm.T,
        origin="lower",
        extent=[box_min, box_max, box_min, box_max],
        cmap="inferno",
        vmin=0.0,
        vmax=2.0,
    )
    ax2.set_title(f"{shape_name} — |∇SDF| ({axis}={slice_val})")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    plt.colorbar(im2, ax=ax2, label="|∇SDF|")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"sdf_slice_{shape_name}_{axis}{slice_val}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved slice: {path}")


def render_all_slices(
    model,
    shape_name: str,
    save_dir: str,
    device: str = "cuda",
):
    """Render SDF slices on all three axes through the origin."""
    for axis in ["x", "y", "z"]:
        render_sdf_slice(
            model,
            shape_name,
            save_dir,
            axis=axis,
            slice_val=0.0,
            device=device,
        )


def plot_metrics_comparison(
    all_metrics: dict,
    save_path: str,
):
    """Plot a bar chart comparing metrics across all shapes.

    Args:
        all_metrics: dict mapping shape_name -> metrics dict
        save_path:   output PNG path
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(all_metrics.keys())
    recon = [all_metrics[n].get("recon_l1", 0) for n in names]
    eik = [all_metrics[n].get("eikonal_violation", 0) for n in names]
    surf = [all_metrics[n].get("surface_accuracy", 0) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(names, recon, color="steelblue", edgecolor="black")
    axes[0].set_title("Reconstruction L1 ↓")
    axes[0].set_ylabel("L1 loss")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(names, eik, color="coral", edgecolor="black")
    axes[1].set_title("Eikonal Violation ↓")
    axes[1].axhline(y=0.1, color="grey", linestyle="--", label="threshold 0.1")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(names, surf, color="seagreen", edgecolor="black")
    axes[2].set_title("Surface Accuracy ↓")
    axes[2].set_ylabel("Mean |SDF| at surface")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved metrics plot: {save_path}")