"""Mesh extraction utilities for SDF models."""

import os

import numpy as np
import torch


def extract_mesh_mc(
    model,
    resolution: int = 128,
    box_min: float = -1.0,
    box_max: float = 1.0,
    level: float = 0.0,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a triangle mesh via Marching Cubes."""
    from skimage.measure import marching_cubes

    coords = torch.linspace(box_min, box_max, resolution, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([xx.ravel(), yy.ravel(), zz.ravel()], dim=-1)
    sdf_vals = []
    batch_size = 8192
    model.eval()

    with torch.no_grad():
        for i in range(0, pts.shape[0], batch_size):
            batch = pts[i : i + batch_size]
            val = model(batch).squeeze(-1)
            sdf_vals.append(val.cpu())

    sdf_grid = (
        torch.cat(sdf_vals, dim=0)
        .reshape(resolution, resolution, resolution)
        .numpy()
    )
    spacing = (box_max - box_min) / (resolution - 1)
    verts, faces, _, _ = marching_cubes(
        sdf_grid,
        level=level,
        spacing=(spacing, spacing, spacing),
    )
    verts += box_min

    return verts.astype(np.float32), faces.astype(np.int32)


def save_mesh_obj(
    verts: np.ndarray,
    faces: np.ndarray,
    filepath: str,
    shape_name: str = "shape",
):
    """Save a triangle mesh as a Wavefront .obj file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with open(filepath, "w") as f:
        f.write(f"# Mesh: {shape_name}\n")
        f.write(f"# Vertices: {verts.shape[0]}, Faces: {faces.shape[0]}\n")

        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"Saved mesh: {filepath} ({verts.shape[0]} verts, {faces.shape[0]} faces)")

def load_mesh_verts(filepath: str) -> np.ndarray:
    """Load vertex positions from an OBJ file."""
    verts = []

    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(verts, dtype=np.float32)