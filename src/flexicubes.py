import torch
from .tables import *

__all__ = [
    'FlexiCubes'
]

class FlexiCubes:
    """FlexiCubes mesh extraction for signed-distance fields."""

    def __init__(self, device="cuda", qef_reg_scale=1e-3, weight_scale=0.99):
        self.device = device

        self.dmc_table = torch.tensor(dmc_table, dtype=torch.long, device=device, requires_grad=False)

        self.num_vd_table = torch.tensor(num_vd_table,

                                         dtype=torch.long, device=device, requires_grad=False)

        self.check_table = torch.tensor(
            check_table,

            dtype=torch.long, device=device, requires_grad=False)

        self.tet_table = torch.tensor(tet_table, dtype=torch.long, device=device, requires_grad=False)

        self.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device, requires_grad=False)

        self.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device, requires_grad=False)

        self.quad_split_train = torch.tensor(

            [0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=device, requires_grad=False)

        self.cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [

                                         1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float, device=device)

        self.cube_corners_idx = torch.pow(2, torch.arange(8, requires_grad=False))

        self.cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,

                                       2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, device=device, requires_grad=False)

        self.edge_dir_table = torch.tensor([0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1],

                                           dtype=torch.long, device=device)

        self.dir_faces_table = torch.tensor([
            [[5, 4], [3, 2], [4, 5], [2, 3]],
            [[5, 4], [1, 0], [4, 5], [0, 1]],
            [[3, 2], [1, 0], [2, 3], [0, 1]]

        ], dtype=torch.long, device=device)

        self.adj_pairs = torch.tensor([0, 1, 1, 3, 3, 2, 2, 0], dtype=torch.long, device=device)
        self.qef_reg_scale = qef_reg_scale
        self.weight_scale = weight_scale

    def construct_voxel_grid(self, res):
        """Construct the voxel grid and cube indices for a given resolution."""

        base_cube_f = torch.arange(8).to(self.device)

        if isinstance(res, int):
            res = (res, res, res)

        voxel_grid_template = torch.ones(res, device=self.device)
        res_tensor = torch.tensor(res, dtype=torch.float, device=self.device)

        coords = torch.nonzero(voxel_grid_template).float() / res_tensor
        verts = (self.cube_corners.unsqueeze(0) / res_tensor + coords.unsqueeze(1)).reshape(-1, 3)
        cubes = (
            base_cube_f.unsqueeze(0)
            + torch.arange(coords.shape[0], device=self.device).unsqueeze(1) * 8
        ).reshape(-1)

        verts_rounded = torch.round(verts * 10**5) / (10**5)
        verts_unique, inverse_indices = torch.unique(verts_rounded, dim=0, return_inverse=True)
        cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 8)

        return verts_unique - 0.5, cubes

    def __call__(self, x_nx3, s_n, cube_fx8, res, beta_fx12=None, alpha_fx8=None,
                 gamma_f=None, training=False, output_tetmesh=False, grad_func=None):
        r"""Extract a mesh from a scalar field using FlexiCubes."""
        surf_cubes, occ_fx8 = self._identify_surf_cubes(s_n, cube_fx8)

        if surf_cubes.sum() == 0:

            return torch.zeros(
                (0, 3),

                device=self.device), torch.zeros(
                (0, 4),

                dtype=torch.long, device=self.device) if output_tetmesh else torch.zeros(
                (0, 3),

                dtype=torch.long, device=self.device), torch.zeros(
                (0),
                device=self.device)
        beta_fx12, alpha_fx8, gamma_f = self._normalize_weights(beta_fx12, alpha_fx8, gamma_f, surf_cubes)
        case_ids = self._get_case_id(occ_fx8, surf_cubes, res)
        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(s_n, cube_fx8, surf_cubes)
        vd, L_dev, vd_gamma, vd_idx_map = self._compute_vd(
            x_nx3, cube_fx8[surf_cubes], surf_edges, s_n, case_ids, beta_fx12, alpha_fx8, gamma_f, idx_map, grad_func)
        vertices, faces, s_edges, edge_indices = self._triangulate(
            s_n, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask, training, grad_func)

        if not output_tetmesh:

            return vertices, faces, L_dev

        else:
            vertices, tets = self._tetrahedralize(
                x_nx3, s_n, cube_fx8, vertices, faces, surf_edges, s_edges, vd_idx_map, case_ids, edge_indices,
                surf_cubes, training)

            return vertices, tets, L_dev

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """
        Regularizer L_dev as in Equation 8
        """

        dist = torch.norm(ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1)

        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = (mean_l2).index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()

        mad = (dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()

        return mad

    def _normalize_weights(self, beta_fx12, alpha_fx8, gamma_f, surf_cubes):
        """
        Normalizes the given weights to be non-negative. If input weights are None, it creates and returns a set of weights of ones.
        """
        n_cubes = surf_cubes.shape[0]

        if beta_fx12 is not None:
            beta_fx12 = (torch.tanh(beta_fx12) * self.weight_scale + 1)

        else:
            beta_fx12 = torch.ones((n_cubes, 12), dtype=torch.float, device=self.device)

        if alpha_fx8 is not None:
            alpha_fx8 = (torch.tanh(alpha_fx8) * self.weight_scale + 1)

        else:
            alpha_fx8 = torch.ones((n_cubes, 8), dtype=torch.float, device=self.device)

        if gamma_f is not None:
            gamma_f = torch.sigmoid(gamma_f) * self.weight_scale + (1 - self.weight_scale)/2

        else:
            gamma_f = torch.ones((n_cubes), dtype=torch.float, device=self.device)

        return beta_fx12[surf_cubes], alpha_fx8[surf_cubes], gamma_f[surf_cubes]

    @torch.no_grad()

    def _get_case_id(self, occ_fx8, surf_cubes, res):
        """Compute the topology case ID for each surface cube."""

        case_ids = (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)

        problem_config = self.check_table.to(self.device)[case_ids]
        to_check = problem_config[..., 0] == 1
        problem_config = problem_config[to_check]

        if not isinstance(res, (list, tuple)):
            res = [res, res, res]

        problem_config_full = torch.zeros(list(res) + [5], device=self.device, dtype=torch.long)

        vol_idx = torch.nonzero(problem_config_full[..., 0] == 0)
        vol_idx_problem = vol_idx[surf_cubes][to_check]
        problem_config_full[
            vol_idx_problem[..., 0],
            vol_idx_problem[..., 1],
            vol_idx_problem[..., 2],
        ] = problem_config
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4]
        within_range = (
            vol_idx_problem_adj[..., 0] >= 0) & (
            vol_idx_problem_adj[..., 0] < res[0]) & (
            vol_idx_problem_adj[..., 1] >= 0) & (
            vol_idx_problem_adj[..., 1] < res[1]) & (
            vol_idx_problem_adj[..., 2] >= 0) & (
            vol_idx_problem_adj[..., 2] < res[2])
        vol_idx_problem = vol_idx_problem[within_range]
        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]
        problem_config_adj = problem_config_full[vol_idx_problem_adj[..., 0],
                                                 vol_idx_problem_adj[..., 1], vol_idx_problem_adj[..., 2]]
        to_invert = (problem_config_adj[..., 0] == 1)

        idx = torch.arange(case_ids.shape[0], device=self.device)[to_check][within_range][to_invert]
        case_ids.index_put_((idx,), problem_config[to_invert][..., -1])

        return case_ids

    @torch.no_grad()

    def _identify_surf_edges(self, s_n, cube_fx8, surf_cubes):
        """Identify edges that cross the surface and assign each one an index."""
        occ_n = s_n < 0
        all_edges = cube_fx8[surf_cubes][:, self.cube_edges].reshape(-1, 2)

        unique_edges, _idx_map, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=cube_fx8.device) * -1

        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=cube_fx8.device)
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]

        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()

    def _identify_surf_cubes(self, s_n, cube_fx8):
        """Identify cubes that intersect the zero level set."""
        occ_n = s_n < 0
        occ_fx8 = occ_n[cube_fx8.reshape(-1)].reshape(-1, 8)

        _occ_sum = torch.sum(occ_fx8, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)

        return surf_cubes, occ_fx8

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2

        assert edges_weight.shape[edge_dim] == 2

        edges_weight = torch.cat([torch.index_select(input=edges_weight, index=torch.tensor(1, device=self.device), dim=edge_dim), -

                                 torch.index_select(input=edges_weight, index=torch.tensor(0, device=self.device), dim=edge_dim)], edge_dim)
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator

        return ue

    def _solve_vd_QEF(self, p_bxnx3, norm_bxnx3, c_bx3=None):
        p_bxnx3 = p_bxnx3.reshape(-1, 7, 3)
        norm_bxnx3 = norm_bxnx3.reshape(-1, 7, 3)
        c_bx3 = c_bx3.reshape(-1, 3)
        A = norm_bxnx3
        B = ((p_bxnx3) * norm_bxnx3).sum(-1, keepdims=True)

        A_reg = (torch.eye(3, device=p_bxnx3.device) * self.qef_reg_scale).unsqueeze(0).repeat(p_bxnx3.shape[0], 1, 1)
        B_reg = (self.qef_reg_scale * c_bx3).unsqueeze(-1)

        A = torch.cat([A, A_reg], 1)

        B = torch.cat([B, B_reg], 1)

        dual_verts = torch.linalg.lstsq(A, B).solution.squeeze(-1)

        return dual_verts

    def _compute_vd(self, x_nx3, surf_cubes_fx8, surf_edges, s_n, case_ids, beta_fx12, alpha_fx8, gamma_f, idx_map, grad_func):
        """Compute the position of dual vertices."""

        alpha_nx12x2 = torch.index_select(input=alpha_fx8, index=self.cube_edges, dim=1).reshape(-1, 12, 2)

        surf_edges_x = torch.index_select(input=x_nx3, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 3)

        surf_edges_s = torch.index_select(input=s_n, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 1)
        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)
        idx_map = idx_map.reshape(-1, 12)

        num_vd = torch.index_select(input=self.num_vd_table, index=case_ids, dim=0)
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges, vd_gamma = [], [], [], [], []
        total_num_vd = 0

        vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.long, device=self.device, requires_grad=False)

        if grad_func is not None:
            normals = torch.nn.functional.normalize(grad_func(zero_crossing), dim=-1)
            vd = []

        for num in torch.unique(num_vd):
            cur_cubes = num_vd == num
            curr_num_vd = cur_cubes.sum() * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)

            curr_edge_group_to_vd = torch.arange(
                curr_num_vd, device=self.device).unsqueeze(-1).repeat(1, 7) + total_num_vd
            total_num_vd += curr_num_vd

            curr_edge_group_to_cube = torch.arange(idx_map.shape[0], device=self.device)[
                cur_cubes].unsqueeze(-1).repeat(1, num * 7).reshape_as(curr_edge_group)
            curr_mask = (curr_edge_group != -1)
            edge_group.append(torch.masked_select(curr_edge_group, curr_mask))
            edge_group_to_vd.append(torch.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask))
            edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask))
            vd_num_edges.append(curr_mask.reshape(-1, 7).sum(-1, keepdims=True))
            vd_gamma.append(torch.masked_select(gamma_f, cur_cubes).unsqueeze(-1).repeat(1, num).reshape(-1))

            if grad_func is not None:
                with torch.no_grad():
                    cube_e_verts_idx = idx_map[cur_cubes]
                    curr_edge_group[~curr_mask] = 0

                    verts_group_idx = torch.gather(input=cube_e_verts_idx, dim=1, index=curr_edge_group)
                    verts_group_idx[verts_group_idx == -1] = 0

                    verts_group_pos = torch.index_select(
                        input=zero_crossing, index=verts_group_idx.reshape(-1), dim=0).reshape(-1, num.item(), 7, 3)
                    v0 = x_nx3[surf_cubes_fx8[cur_cubes][:, 0]].reshape(-1, 1, 1, 3).repeat(1, num.item(), 1, 1)
                    curr_mask = curr_mask.reshape(-1, num.item(), 7, 1)
                    verts_centroid = (verts_group_pos * curr_mask).sum(2) / (curr_mask.sum(2))

                    normals_bx7x3 = torch.index_select(input=normals, index=verts_group_idx.reshape(-1), dim=0).reshape(
                        -1, num.item(), 7,
                        3)
                    curr_mask = curr_mask.squeeze(2)
                    vd.append(self._solve_vd_QEF((verts_group_pos - v0) * curr_mask, normals_bx7x3 * curr_mask,
                                                 verts_centroid - v0.squeeze(2)) + v0.reshape(-1, 3))

        edge_group = torch.cat(edge_group)

        edge_group_to_vd = torch.cat(edge_group_to_vd)

        edge_group_to_cube = torch.cat(edge_group_to_cube)

        vd_num_edges = torch.cat(vd_num_edges)

        vd_gamma = torch.cat(vd_gamma)

        if grad_func is not None:
            vd = torch.cat(vd)

            L_dev = torch.zeros([1], device=self.device)

        else:
            vd = torch.zeros((total_num_vd, 3), device=self.device)

            beta_sum = torch.zeros((total_num_vd, 1), device=self.device)

            idx_group = torch.gather(input=idx_map.reshape(-1), dim=0, index=edge_group_to_cube * 12 + edge_group)

            x_group = torch.index_select(input=surf_edges_x, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 3)

            s_group = torch.index_select(input=surf_edges_s, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 1)

            zero_crossing_group = torch.index_select(
                input=zero_crossing, index=idx_group.reshape(-1), dim=0).reshape(-1, 3)

            alpha_group = torch.index_select(input=alpha_nx12x2.reshape(-1, 2), dim=0,
                                             index=edge_group_to_cube * 12 + edge_group).reshape(-1, 2, 1)
            ue_group = self._linear_interp(s_group * alpha_group, x_group)

            beta_group = torch.gather(input=beta_fx12.reshape(-1), dim=0,
                                      index=edge_group_to_cube * 12 + edge_group).reshape(-1, 1)
            beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)
            vd = vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group) / beta_sum
            L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges)

        v_idx = torch.arange(vd.shape[0], device=self.device)
        vd_idx_map = (vd_idx_map.reshape(-1)).scatter(
            dim=0,
            index=edge_group_to_cube * 12 + edge_group,
            src=v_idx[edge_group_to_vd],
        )

        return vd, L_dev, vd_gamma, vd_idx_map

    def _triangulate(self, s_n, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask, training, grad_func):
        """Create face triangles from neighboring dual vertices."""

        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]

            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4)
            s_edges = s_n[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx[flip_mask] = quad_vd_idx[flip_mask][:, [0, 1, 3, 2]]
            quad_vd_idx[~flip_mask] = quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]

        if grad_func is not None:
            with torch.no_grad():
                vd_gamma = torch.nn.functional.normalize(grad_func(vd), dim=-1)

                quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
                gamma_02 = (quad_gamma[:, 0] * quad_gamma[:, 2]).sum(-1, keepdims=True)
                gamma_13 = (quad_gamma[:, 1] * quad_gamma[:, 3]).sum(-1, keepdims=True)

        else:
            quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4)

            gamma_02 = torch.index_select(input=quad_gamma, index=torch.tensor(

                0, device=self.device), dim=1) * torch.index_select(input=quad_gamma, index=torch.tensor(2, device=self.device), dim=1)

            gamma_13 = torch.index_select(input=quad_gamma, index=torch.tensor(

                1, device=self.device), dim=1) * torch.index_select(input=quad_gamma, index=torch.tensor(3, device=self.device), dim=1)

        if not training:
            mask = (gamma_02 > gamma_13).squeeze(1)

            faces = torch.zeros((quad_gamma.shape[0], 6), dtype=torch.long, device=quad_vd_idx.device)
            faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
            faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
            faces = faces.reshape(-1, 3)

        else:
            vd_quad = torch.index_select(input=vd, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)

            vd_02 = (torch.index_select(input=vd_quad, index=torch.tensor(0, device=self.device), dim=1) +

                     torch.index_select(input=vd_quad, index=torch.tensor(2, device=self.device), dim=1)) / 2

            vd_13 = (torch.index_select(input=vd_quad, index=torch.tensor(1, device=self.device), dim=1) +

                     torch.index_select(input=vd_quad, index=torch.tensor(3, device=self.device), dim=1)) / 2
            weight_sum = (gamma_02 + gamma_13) + 1e-8
            vd_center = ((vd_02 * gamma_02.unsqueeze(-1) + vd_13 * gamma_13.unsqueeze(-1)) /
                         weight_sum.unsqueeze(-1)).squeeze(1)

            vd_center_idx = torch.arange(vd_center.shape[0], device=self.device) + vd.shape[0]

            vd = torch.cat([vd, vd_center])
            faces = quad_vd_idx[:, self.quad_split_train].reshape(-1, 4, 2)

            faces = torch.cat([faces, vd_center_idx.reshape(-1, 1, 1).repeat(1, 4, 1)], -1).reshape(-1, 3)

        return vd, faces, s_edges, edge_indices

    def _tetrahedralize(
            self, x_nx3, s_n, cube_fx8, vertices, faces, surf_edges, s_edges, vd_idx_map, case_ids, edge_indices,
            surf_cubes, training):
        """Build tetrahedra for the interior volume."""
        occ_n = s_n < 0
        occ_fx8 = occ_n[cube_fx8.reshape(-1)].reshape(-1, 8)

        occ_sum = torch.sum(occ_fx8, -1)
        inside_verts = x_nx3[occ_n]

        mapping_inside_verts = torch.ones((occ_n.shape[0]), dtype=torch.long, device=self.device) * -1

        mapping_inside_verts[occ_n] = torch.arange(occ_n.sum(), device=self.device) + vertices.shape[0]
        """ 
        For each grid edge connecting two grid vertices with different
        signs, we first form a four-sided pyramid by connecting one
        of the grid vertices with four mesh vertices that correspond
        to the grid edge and then subdivide the pyramid into two tetrahedra
        """
        inside_verts_idx = mapping_inside_verts[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1, 2)[
            s_edges < 0]]

        if not training:
            inside_verts_idx = inside_verts_idx.unsqueeze(1).expand(-1, 2).reshape(-1)

        else:
            inside_verts_idx = inside_verts_idx.unsqueeze(1).expand(-1, 4).reshape(-1)

        tets_surface = torch.cat([faces, inside_verts_idx.unsqueeze(-1)], -1)
        """ 
        For each grid edge connecting two grid vertices with the
        same sign, the tetrahedron is formed by the two grid vertices
        and two vertices in consecutive adjacent cells
        """
        inside_cubes = (occ_sum == 8)
        inside_cubes_center = x_nx3[cube_fx8[inside_cubes].reshape(-1)].reshape(-1, 8, 3).mean(1)

        inside_cubes_center_idx = torch.arange(
            inside_cubes_center.shape[0], device=inside_cubes.device) + vertices.shape[0] + inside_verts.shape[0]
        surface_n_inside_cubes = surf_cubes | inside_cubes

        edge_center_vertex_idx = torch.ones(((surface_n_inside_cubes).sum(), 13),

                                            dtype=torch.long, device=x_nx3.device) * -1
        surf_cubes = surf_cubes[surface_n_inside_cubes]
        inside_cubes = inside_cubes[surface_n_inside_cubes]
        edge_center_vertex_idx[surf_cubes, :12] = vd_idx_map.reshape(-1, 12)
        edge_center_vertex_idx[inside_cubes, 12] = inside_cubes_center_idx
        all_edges = cube_fx8[surface_n_inside_cubes][:, self.cube_edges].reshape(-1, 2)

        unique_edges, _idx_map, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 2
        mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1

        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=self.device)
        idx_map = mapping[_idx_map]
        group_mask = (counts == 4) & mask
        group = idx_map.reshape(-1)[group_mask]

        edge_indices, indices = torch.sort(group)

        cube_idx = torch.arange((_idx_map.shape[0] // 12), dtype=torch.long,
                                device=self.device).unsqueeze(1).expand(-1, 12).reshape(-1)[group_mask]

        edge_idx = torch.arange((12), dtype=torch.long, device=self.device).unsqueeze(
            0).expand(_idx_map.shape[0] // 12, -1).reshape(-1)[group_mask]
        cube_idx_4 = cube_idx[indices].reshape(-1, 4)
        edge_dir = self.edge_dir_table[edge_idx[indices]].reshape(-1, 4)[..., 0]
        shared_faces_4x2 = self.dir_faces_table[edge_dir].reshape(-1)
        cube_idx_4x2 = cube_idx_4[:, self.adj_pairs].reshape(-1)

        case_ids_expand = torch.ones((surface_n_inside_cubes).sum(), dtype=torch.long, device=x_nx3.device) * 255
        case_ids_expand[surf_cubes] = case_ids
        cases = case_ids_expand[cube_idx_4x2]
        quad_edge = edge_center_vertex_idx[cube_idx_4x2, self.tet_table[cases, shared_faces_4x2]].reshape(-1, 2)
        mask = (quad_edge == -1).sum(-1) == 0
        inside_edge = mapping_inside_verts[unique_edges[mask_edges][edge_indices].reshape(-1)].reshape(-1, 2)

        tets_inside = torch.cat([quad_edge, inside_edge], -1)[mask]

        tets = torch.cat([tets_surface, tets_inside])

        vertices = torch.cat([vertices, inside_verts, inside_cubes_center])

        return vertices, tets
