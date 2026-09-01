"""Dataset and sampler for SDF point clouds."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

def load_pt_file(path: str) -> dict[str, torch.Tensor]:
    """Load a single .pt SDF data file."""

    data = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):

        return _normalise_dict(data)

    elif isinstance(data, (list, tuple)):
        points, sdf = data[0], data[1]

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)

        if not isinstance(sdf, torch.Tensor):
            sdf = torch.tensor(sdf, dtype=torch.float32)

        if sdf.ndim == 1:
            sdf = sdf.unsqueeze(-1)

        return {"points": points, "sdf": sdf}

    else:

        raise TypeError(f"Unrecognised data format in {path}: {type(data)}")

def _normalise_dict(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Ensure dict has 'points' and 'sdf' keys with correct shapes."""

    for old_key, new_key in [
        ("coords", "points"), ("xyz", "points"), ("pos", "points"),
        ("sdf_values", "sdf"), ("distance", "sdf"), ("d", "sdf"),
    ]:
        if old_key in data and new_key not in data:
            data[new_key] = data.pop(old_key)

    for key in ["points", "sdf"]:
        if key in data and data[key].dtype != torch.float32:
            data[key] = data[key].to(torch.float32)

    if "sdf" in data and data["sdf"].ndim == 1:
        data["sdf"] = data["sdf"].unsqueeze(-1)

    return data

class SDFDataset(Dataset):
    """Dataset that holds all pre-loaded SDF data for one shape."""

    def __init__(self, points: torch.Tensor, sdf: torch.Tensor):
        self.points = points
        self.sdf = sdf

        assert points.shape[0] == sdf.shape[0]

    def __len__(self):

        return self.points.shape[0]

    def __getitem__(self, idx):

        return self.points[idx], self.sdf[idx]

class SDFSampler:
    """Draw batches of (point, sdf) pairs with mixed uniform + surface sampling.

    Args:
        dataset:    SDFDataset for a specific shape
        num_points: total points to draw per epoch
        batch_size: GPU batch size for training
        surface_ratio: fraction of points drawn near surface (default 0.5)
        surface_sigma: std of gaussian noise for surface samples
        device:    target compute device
    """

    def __init__(
        self,
        dataset: SDFDataset,
        num_points: int = 16384,
        batch_size: int = 4096,
        surface_ratio: float = 0.5,
        surface_sigma: float = 0.01,
        device: str = "cuda",
    ):

        self.points = dataset.points.to(device)

        self.sdf = dataset.sdf.to(device)
        self.num_points = num_points
        self.batch_size = batch_size
        self.surface_ratio = surface_ratio
        self.surface_sigma = surface_sigma
        self.device = device
        self.surf_mask = (self.sdf.abs() < 0.1).squeeze()

        self.surf_indices = torch.where(self.surf_mask)[0]

        self.unif_indices = torch.arange(len(self.points), device=device)

        if self.surf_indices.numel() == 0:
            print("[SDFSampler] No near-surface points found; using uniform sampling.")
            self.surface_ratio = 0.0

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one batch of (points, sdf) with mixed sampling."""
        n_surf = int(self.batch_size * self.surface_ratio)
        n_unif = self.batch_size - n_surf

        unif_ix = torch.randint(0, len(self.points), (n_unif,), device=self.device)
        pts_u = self.points[unif_ix]
        sdf_u = self.sdf[unif_ix]

        if n_surf > 0 and self.surf_indices.numel() >= n_surf:
            surf_ix = torch.randint(
                0, self.surf_indices.numel(), (n_surf,), device=self.device
            )
            idx = self.surf_indices[surf_ix]

            pts_s = self.points[idx] + torch.randn_like(self.points[idx]) * self.surface_sigma
            sdf_s = self.sdf[idx]

            return torch.cat([pts_u, pts_s], dim=0), torch.cat([sdf_u, sdf_s], dim=0)
        return pts_u, sdf_u

    def __iter__(self):
        """Iterate over batches for one epoch."""
        n_batches = max(1, self.num_points // self.batch_size)

        for _ in range(n_batches):
            yield self.sample_batch()

class ShapeDataManager:
    """Loads and manages SDF data for all shapes.

    Args:
        data_dir:  path to processed_sdf/ directory containing {1..N}.pt
        device:    target compute device
    """

    def __init__(self, data_dir: str, device: str = "cuda"):
        self.device = device
        self.data_dir = data_dir
        self.datasets: dict[int, SDFDataset] = {}
        self._load_all()

    def _load_all(self):
        """Load all .pt files from the data directory."""
        pt_files = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith(".pt")
        )

        for fname in pt_files:
            shape_id = int(os.path.splitext(fname)[0])
            path = os.path.join(self.data_dir, fname)
            data = load_pt_file(path)
            self.datasets[shape_id] = SDFDataset(
                data["points"].to(self.device),
                data["sdf"].to(self.device),
            )

    def get_dataset(self, shape_id: int) -> SDFDataset:

        return self.datasets[shape_id]

    @property

    def num_shapes(self) -> int:

        return len(self.datasets)

    @property

    def shape_ids(self) -> list[int]:

        return sorted(self.datasets.keys())
