import concurrent.futures
import gc
import os
import shutil

import numpy as np
import torch
import trimesh
from mesh_to_sdf import get_surface_point_cloud


BASE_DIR = r"C:\Users\janss\Desktop\research\3d-weight-diffusion\Final-pipeline\data"
RAW_DIR = os.path.join(BASE_DIR, "raw_meshes")
OUT_DIR = os.path.join(BASE_DIR, "processed_meshes")

MAX_SHAPES = 6778
MAX_WORKERS = 4
USE_GPU_SDF = False



def make_watertight(mesh: trimesh.Trimesh, voxel_res: int = 64):
    """Turn a polygon soup into a solid, watertight mesh when possible."""
    pitch = max(mesh.extents) / voxel_res

    try:
        voxel_grid = mesh.voxelized(pitch=pitch)
    except Exception:
        pitch = max(mesh.extents) / 32

        try:
            voxel_grid = mesh.voxelized(pitch=pitch)
        except Exception:
            return None

    try:
        voxel_grid = voxel_grid.fill()
    except Exception:
        pass

    try:
        return voxel_grid.marching_cubes
    except Exception:
        return None


def normalise_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_extent = np.linalg.norm(mesh.vertices, axis=1).max()
    scale = 0.85 / max_extent if max_extent > 0 else 1.0
    mesh.vertices *= scale
    return mesh




def process_single_mesh(mesh_path, out_path, n_points=200000):
    """Process one mesh."""
    try:
        mesh = trimesh.load(mesh_path, force="mesh")

        if isinstance(mesh, trimesh.Scene):
            geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geometries:
                return False
            mesh = trimesh.util.concatenate(geometries)

        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return False

        mesh = normalise_mesh(mesh)
        watertight_mesh = make_watertight(mesh, voxel_res=64)

        if (
            watertight_mesh is None
            or len(watertight_mesh.vertices) == 0
            or len(watertight_mesh.faces) == 0
        ):
            return False

        watertight_mesh = normalise_mesh(watertight_mesh)

        rng = np.random.RandomState(42)
        n_surface = int(n_points * 0.5)
        n_uniform = n_points - n_surface

        pts_uniform = rng.uniform(-1.0, 1.0, (n_uniform, 3)).astype(np.float32)
        pts_surf, _ = trimesh.sample.sample_surface(watertight_mesh, n_surface)
        pts_surf = pts_surf.astype(np.float32)
        pts_surf += rng.randn(*pts_surf.shape).astype(np.float32) * 0.01
        pts_surf = np.clip(pts_surf, -1.0, 1.0)

        all_pts = np.concatenate([pts_uniform, pts_surf], axis=0)

        cloud = get_surface_point_cloud(
            watertight_mesh,
            surface_point_method="sample",
            sample_point_count=50000,
            calculate_normals=True,
        )

        sdf = cloud.get_sdf_in_batches(all_pts, use_depth_buffer=USE_GPU_SDF)
        sdf = np.clip(sdf, -1.0, 1.0)

        idx = rng.permutation(len(all_pts))
        points_out = all_pts[idx]
        sdf_out = sdf[idx].reshape(-1, 1).astype(np.float32)

        torch.save(
            {
                "points": torch.from_numpy(points_out),
                "sdf": torch.from_numpy(sdf_out),
            },
            out_path,
        )

        del mesh, watertight_mesh, cloud, all_pts, sdf
        gc.collect()
        return True

    except Exception as e:
        print(f"Error: {os.path.basename(mesh_path)}: {e}")
        return False


if __name__ == "__main__":
    exts = (".off", ".obj", ".stl", ".ply")
    category = "chairs"

    print(f"Starting processing for {category}.")

    if os.path.exists(OUT_DIR):
        print(f"Clearing output directory: {OUT_DIR}")

    os.makedirs(OUT_DIR, exist_ok=True)

    raw_files = sorted(
        [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.lower().endswith(exts)]
    )

    print(f"Found {len(raw_files)} raw meshes; processing up to {MAX_SHAPES}.")
    successful_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_mesh = {
            executor.submit(
                process_single_mesh,
                mesh_path,
                os.path.join(OUT_DIR, f"{i + 1}.pt"),
            ): mesh_path
            for i, mesh_path in enumerate(raw_files)
        }

        for future in concurrent.futures.as_completed(future_to_mesh):
            if successful_count >= MAX_SHAPES:
                executor.shutdown(wait=False, cancel_futures=True)
                break

            success = future.result()
            if success:
                successful_count += 1
                print(f"Processed [{successful_count}/{MAX_SHAPES}]")

    print("Processing complete.")