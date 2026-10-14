import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
from aether.morphology.models import (
    BaseMorphologyModel,
    BipolarModel,
    ToroidalModel,
    EllipsoidalModel,
)
from aether.io.image import to_grayscale

class MorphologicalDeprojector:
    def __init__(self, model: BaseMorphologyModel | None = None) -> None:
        self.model = model or BipolarModel()

    def deproject_volume(
        self,
        rgb: np.ndarray,
        grid_size: int = 64,
        spatial_extent: float = 1.2,
    ) -> pv.ImageData:
        h, w = rgb.shape[:2]
        gray = to_grayscale(rgb)

        coords = np.linspace(-spatial_extent, spatial_extent, grid_size, dtype=np.float32)
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")

        morph_density = self.model.compute_density(xx, yy, zz)

        u = (xx / (2.0 * spatial_extent) + 0.5) * (w - 1)
        v = (-yy / (2.0 * spatial_extent) + 0.5) * (h - 1)

        u_flat = np.clip(u.ravel(), 0.0, float(w - 1))
        v_flat = np.clip(v.ravel(), 0.0, float(h - 1))
        sample_indices = np.array([v_flat, u_flat])

        sample_intensity = map_coordinates(gray, sample_indices, order=1).reshape(xx.shape)
        
        sample_r = map_coordinates(rgb[..., 0], sample_indices, order=1).reshape(xx.shape)
        sample_g = map_coordinates(rgb[..., 1], sample_indices, order=1).reshape(xx.shape)
        sample_b = map_coordinates(rgb[..., 2], sample_indices, order=1).reshape(xx.shape)

        r_norm = np.sqrt(xx**2 + yy**2 + zz**2) / spatial_extent
        boundary_taper = np.clip((1.0 - r_norm) / 0.15, 0.0, 1.0)
        boundary_taper = 0.5 * (1.0 - np.cos(np.pi * boundary_taper))

        density_3d = morph_density * np.power(sample_intensity, 0.8) * boundary_taper

        r_3d = np.clip(sample_r * (morph_density * 1.2), 0.0, 1.0) * 255.0
        g_3d = np.clip(sample_g * (morph_density * 1.2), 0.0, 1.0) * 255.0
        b_3d = np.clip(sample_b * (morph_density * 1.2), 0.0, 1.0) * 255.0

        grid = pv.ImageData()
        grid.dimensions = np.array([grid_size, grid_size, grid_size])
        spacing_val = (2.0 * spatial_extent) / (grid_size - 1) * 100.0
        grid.origin = (-spatial_extent * 100.0, -spatial_extent * 100.0, -spatial_extent * 100.0)
        grid.spacing = (spacing_val, spacing_val, spacing_val)

        grid.point_data["density"] = density_3d.ravel(order="F").astype(np.float32)

        rgb_flat = np.column_stack([
            r_3d.ravel(order="F"),
            g_3d.ravel(order="F"),
            b_3d.ravel(order="F"),
        ]).astype(np.uint8)
        grid.point_data["RGB"] = rgb_flat

        return grid

    def deproject_point_cloud(
        self,
        rgb: np.ndarray,
        num_points: int = 40000,
        spatial_extent: float = 1.2,
        scale: float = 100.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        h, w = rgb.shape[:2]
        gray = to_grayscale(rgb)

        num_candidates = num_points * 6
        cand_x = rng.uniform(-spatial_extent, spatial_extent, num_candidates).astype(np.float32)
        cand_y = rng.uniform(-spatial_extent, spatial_extent, num_candidates).astype(np.float32)
        cand_z = rng.uniform(-spatial_extent, spatial_extent, num_candidates).astype(np.float32)

        densities = self.model.compute_density(cand_x, cand_y, cand_z)

        u = (cand_x / (2.0 * spatial_extent) + 0.5) * (w - 1)
        v = (-cand_y / (2.0 * spatial_extent) + 0.5) * (h - 1)
        u_valid = np.clip(u, 0.0, float(w - 1))
        v_valid = np.clip(v, 0.0, float(h - 1))

        sample_coords = np.array([v_valid, u_valid])
        sampled_intensity = map_coordinates(gray, sample_coords, order=1)

        weights = densities * np.power(sampled_intensity, 0.75)
        weights_sum = np.sum(weights)

        if weights_sum <= 1e-7:
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_clrs = np.zeros((0, 3), dtype=np.float32)
            return empty_pts, empty_clrs

        probabilities = weights / weights_sum
        chosen_indices = rng.choice(num_candidates, size=min(num_points, num_candidates), p=probabilities, replace=False)

        selected_x = cand_x[chosen_indices] * scale
        selected_y = cand_y[chosen_indices] * scale
        selected_z = cand_z[chosen_indices] * scale
        points = np.column_stack([selected_x, selected_y, selected_z]).astype(np.float32)

        chosen_u = u_valid[chosen_indices]
        chosen_v = v_valid[chosen_indices]
        sample_pick = np.array([chosen_v, chosen_u])

        cr = map_coordinates(rgb[..., 0], sample_pick, order=1)
        cg = map_coordinates(rgb[..., 1], sample_pick, order=1)
        cb = map_coordinates(rgb[..., 2], sample_pick, order=1)
        colors = np.clip(np.column_stack([cr, cg, cb]), 0.0, 1.0).astype(np.float32)

        return points, colors

    def deproject_surface_mesh(
        self,
        rgb: np.ndarray,
        grid_size: int = 48,
        iso_value: float = 0.25,
        spatial_extent: float = 1.2,
    ) -> pv.PolyData:
        volume = self.deproject_volume(rgb, grid_size=grid_size, spatial_extent=spatial_extent)
        mesh = volume.contour(isosurfaces=[iso_value], scalars="density")
        
        if mesh.n_points == 0:
            mesh = volume.contour(isosurfaces=[0.08], scalars="density")

        if mesh.n_points == 0:
            return mesh

        try:
            mesh = mesh.smooth_taubin(n_iter=10)
        except Exception:
            mesh = mesh.smooth(n_iter=10)

        pts = mesh.points
        if len(pts) > 0:
            extent_val = spatial_extent * 100.0
            h, w = rgb.shape[:2]
            u_norm = np.clip(pts[:, 0] / (2.0 * extent_val) + 0.5, 0.0, 1.0)
            v_norm = np.clip(-pts[:, 1] / (2.0 * extent_val) + 0.5, 0.0, 1.0)
            u = u_norm * (w - 1)
            v = v_norm * (h - 1)
            coords = np.array([v, u])

            r = map_coordinates(rgb[..., 0], coords, order=1)
            g = map_coordinates(rgb[..., 1], coords, order=1)
            b = map_coordinates(rgb[..., 2], coords, order=1)
            rgb_mesh = (np.clip(np.column_stack([r, g, b]), 0.0, 1.0) * 255.0).astype(np.uint8)
            mesh.point_data["RGB"] = rgb_mesh
            mesh.active_texture_coordinates = np.column_stack([u_norm, v_norm]).astype(np.float32)

        return mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
